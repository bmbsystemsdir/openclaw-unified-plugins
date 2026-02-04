"""Search functionality for vault embeddings."""

from dataclasses import dataclass
from typing import Optional, Iterable
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchText,
)

from .config import VaultConfig
from .embedder import Embedder, get_embedder


@dataclass
class SearchResult:
    """A single search result."""
    
    path: str                   # File path relative to vault
    text: str                   # Chunk text
    score: float               # Similarity score (0-1)
    heading: Optional[str]      # Heading context
    line_start: int            # Starting line
    line_end: int              # Ending line
    chunk_index: int           # Chunk index within file
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "text": self.text,
            "score": self.score,
            "heading": self.heading,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "chunk_index": self.chunk_index,
        }


class VaultSearcher:
    """Searches vault embeddings in Qdrant."""

    def __init__(self, config: VaultConfig, verify_schema: bool = True):
        """Initialize searcher.

        Args:
            config: Vault configuration
            verify_schema: If True, verifies collection exists + vector dimensions match
        """
        self.config = config
        self.verify_schema = verify_schema
        self._embedder: Optional[Embedder] = None
        self._client: Optional[QdrantClient] = None
        self._schema_verified: bool = False
    
    @property
    def embedder(self) -> Embedder:
        """Lazy-load embedder."""
        if self._embedder is None:
            self._embedder = get_embedder(self.config.model_name)
        return self._embedder
    
    @property
    def client(self) -> QdrantClient:
        """Lazy-load Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
            )
        # Verify schema once on first access
        if self.verify_schema and not self._schema_verified:
            self._verify_collection_schema()
            self._schema_verified = True
        return self._client

    def _verify_collection_schema(self) -> None:
        """Verify the collection exists and vector dimensions match config.

        Raises:
            RuntimeError: if collection missing or vector dims mismatch
        """
        try:
            info = self._client.get_collection(self.config.collection_name)
        except Exception as e:
            raise RuntimeError(
                f"Qdrant collection '{self.config.collection_name}' not found. "
                f"Run: vault-embedder index (or create collection) first. ({e})"
            )

        # Qdrant python client returns vectors config in different shapes depending on version
        expected = int(self.config.model_dimensions)
        actual = None
        try:
            # Most common
            actual = info.config.params.vectors.size
        except Exception:
            try:
                # Named vectors
                actual = next(iter(info.config.params.vectors.values())).size
            except Exception:
                actual = None

        if actual is not None and int(actual) != expected:
            raise RuntimeError(
                f"Vector dim mismatch for '{self.config.collection_name}': "
                f"expected {expected}, got {actual}. "
                f"(model_name={self.config.model_name})"
            )

    @staticmethod
    def _build_path_prefix_filter(path_prefix: str) -> FieldCondition:
        """Build a best-effort path prefix filter.

        Qdrant doesn't support true 'startsWith' for keyword fields.
        This uses MatchText, which tokenizes on separators; good enough for folder-level filters
        like 'daily/' or 'projects/'.
        """
        prefix = path_prefix.strip()
        if not prefix:
            return None
        return FieldCondition(key="path", match=MatchText(text=prefix))

    @staticmethod
    def _build_exact_path_filter(path: str) -> FieldCondition:
        """Exact path match filter."""
        return FieldCondition(key="path", match=MatchValue(value=path))

    @staticmethod
    def _combine_must(conds: Iterable[Optional[FieldCondition]]) -> Optional[Filter]:
        must = [c for c in conds if c is not None]
        if not must:
            return None
        return Filter(must=must)
    
    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        path_filter: Optional[str] = None,
        path_prefix: Optional[str] = None,
    ) -> list[SearchResult]:
        """Search for similar content.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            min_score: Minimum similarity score (0-1)
            path_filter: Optional path prefix filter
            
        Returns:
            List of SearchResult objects
        """
        # Embed query
        query_vector = self.embedder.embed(query).tolist()
        
        # Build filter if needed
        search_filter = None
        if path_filter and path_prefix:
            raise ValueError("Provide only one of path_filter (exact) or path_prefix (folder-level).")

        if path_filter:
            search_filter = self._combine_must([self._build_exact_path_filter(path_filter)])
        elif path_prefix:
            search_filter = self._combine_must([self._build_path_prefix_filter(path_prefix)])
        
        # Search Qdrant
        response = self.client.query_points(
            collection_name=self.config.collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=min_score if min_score > 0 else None,
            query_filter=search_filter,
        )
        hits = response.points
        
        # Convert to results
        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(SearchResult(
                path=payload.get("path", ""),
                text=payload.get("text", ""),
                score=hit.score,
                heading=payload.get("heading"),
                line_start=payload.get("line_start", 0),
                line_end=payload.get("line_end", 0),
                chunk_index=payload.get("chunk_index", 0),
            ))
        
        return results
    
    def search_by_path(
        self,
        source_path: str,
        limit: int = 10,
        min_score: float = 0.0,
        exclude_same_file: bool = True,
    ) -> list[SearchResult]:
        """Find content similar to a specific file.
        
        Args:
            source_path: Path to source file (relative to vault)
            limit: Maximum number of results
            min_score: Minimum similarity score
            exclude_same_file: Exclude results from the same file
            
        Returns:
            List of SearchResult objects
        """
        # Get vectors for the source file
        source_points = self.client.scroll(
            collection_name=self.config.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="path",
                        match=MatchValue(value=source_path),
                    )
                ]
            ),
            with_vectors=True,
            limit=100,
        )[0]
        
        if not source_points:
            return []
        
        # Use the first chunk's vector as representative
        # (could also average all vectors or use centroid)
        source_vector = source_points[0].vector
        
        # Build exclusion filter
        search_filter = None
        if exclude_same_file:
            search_filter = Filter(
                must_not=[
                    FieldCondition(
                        key="path",
                        match=MatchValue(value=source_path),
                    )
                ]
            )
        
        # Search
        response = self.client.query_points(
            collection_name=self.config.collection_name,
            query=source_vector,
            limit=limit,
            score_threshold=min_score if min_score > 0 else None,
            query_filter=search_filter,
        )
        hits = response.points
        
        # Convert to results
        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(SearchResult(
                path=payload.get("path", ""),
                text=payload.get("text", ""),
                score=hit.score,
                heading=payload.get("heading"),
                line_start=payload.get("line_start", 0),
                line_end=payload.get("line_end", 0),
                chunk_index=payload.get("chunk_index", 0),
            ))
        
        return results


def search_vault(
    query: str,
    config: VaultConfig,
    limit: int = 10,
    min_score: float = 0.0,
    path_prefix: Optional[str] = None,
) -> list[dict]:
    """Convenience function for searching vault.
    
    Args:
        query: Search query
        config: Vault configuration
        limit: Maximum results
        min_score: Minimum score threshold
        
    Returns:
        List of result dictionaries
    """
    searcher = VaultSearcher(config)
    results = searcher.search(query, limit=limit, min_score=min_score, path_prefix=path_prefix)
    return [r.to_dict() for r in results]
