"""Main indexer that orchestrates embedding and storage."""

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from .config import VaultConfig
from .embedder import Embedder, get_embedder
from .chunker import MarkdownChunker, Chunk
from .walker import VaultWalker, IndexState, FileInfo


@dataclass
class IndexResult:
    """Result of an indexing operation."""
    
    files_processed: int = 0
    files_skipped: int = 0
    files_deleted: int = 0
    chunks_added: int = 0
    chunks_removed: int = 0
    errors: list[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class VaultIndexer:
    """Indexes vault content into Qdrant."""
    
    def __init__(self, config: VaultConfig):
        """Initialize indexer.
        
        Args:
            config: Vault configuration
        """
        self.config = config
        self._embedder: Optional[Embedder] = None
        self._client: Optional[QdrantClient] = None
        self._chunker: Optional[MarkdownChunker] = None
        self._walker: Optional[VaultWalker] = None
        self._state: Optional[IndexState] = None
    
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
        return self._client
    
    @property
    def chunker(self) -> MarkdownChunker:
        """Lazy-load chunker."""
        if self._chunker is None:
            self._chunker = MarkdownChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                min_chunk_size=self.config.min_chunk_size,
            )
        return self._chunker
    
    @property
    def walker(self) -> VaultWalker:
        """Lazy-load walker."""
        if self._walker is None:
            self._walker = VaultWalker(
                vault_path=self.config.vault,
                include_extensions=self.config.include_extensions,
                exclude_patterns=self.config.exclude_patterns,
            )
        return self._walker
    
    @property
    def state(self) -> IndexState:
        """Lazy-load state."""
        if self._state is None:
            self._state = IndexState.load(self.config.state_path)
            self._state.collection_name = self.config.collection_name
            self._state.model_name = self.config.model_name
        return self._state
    
    def ensure_collection(self):
        """Ensure the Qdrant collection exists with correct config."""
        collections = [c.name for c in self.client.get_collections().collections]
        
        if self.config.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.model_dimensions,
                    distance=Distance.COSINE,
                ),
            )
    
    def index(
        self,
        files: Optional[list[str]] = None,
        force: bool = False,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> IndexResult:
        """Index vault content.
        
        Args:
            files: Specific files to index (relative paths). If None, index all.
            force: Force reindex even if unchanged
            progress_callback: Optional callback(file_path, current, total)
            
        Returns:
            IndexResult with statistics
        """
        result = IndexResult()
        
        # Ensure collection exists
        self.ensure_collection()
        
        # Get files to process
        if files:
            file_infos = [
                FileInfo(
                    path=self.config.vault / f,
                    relative_path=f,
                    mtime=(self.config.vault / f).stat().st_mtime,
                    size=(self.config.vault / f).stat().st_size,
                )
                for f in files
                if (self.config.vault / f).exists()
            ]
        else:
            file_infos = list(self.walker.walk())
        
        total_files = len(file_infos)
        
        # Process each file
        for i, file_info in enumerate(file_infos):
            if progress_callback:
                progress_callback(file_info.relative_path, i + 1, total_files)
            
            try:
                # Compute content hash
                file_info.compute_hash()
                
                # Check if needs reindex
                if not force and not self.state.needs_reindex(file_info):
                    result.files_skipped += 1
                    continue
                
                # Remove old chunks if file was previously indexed
                old_chunk_ids = self.state.remove_file(file_info.relative_path)
                if old_chunk_ids:
                    self._delete_chunks(old_chunk_ids)
                    result.chunks_removed += len(old_chunk_ids)
                
                # Index file
                chunk_ids = self._index_file(file_info)
                
                # Update state
                self.state.update_file(file_info, chunk_ids)
                
                result.files_processed += 1
                result.chunks_added += len(chunk_ids)
                
            except Exception as e:
                result.errors.append(f"{file_info.relative_path}: {str(e)}")
        
        # Handle deleted files
        if not files:  # Only check for deletions on full reindex
            deleted = self.walker.find_deleted(self.state)
            for rel_path in deleted:
                chunk_ids = self.state.remove_file(rel_path)
                if chunk_ids:
                    self._delete_chunks(chunk_ids)
                    result.chunks_removed += len(chunk_ids)
                result.files_deleted += 1
        
        # Save state
        self.state.save(self.config.state_path)
        
        return result
    
    def _index_file(self, file_info: FileInfo) -> list[str]:
        """Index a single file.
        
        Args:
            file_info: File information
            
        Returns:
            List of chunk IDs that were created
        """
        # Read content
        content = file_info.path.read_text(encoding='utf-8', errors='replace')
        
        # Chunk content
        chunks = list(self.chunker.chunk(content))
        
        if not chunks:
            return []
        
        # Embed all chunks
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_batch(texts)
        
        # Build points
        points = []
        chunk_ids = []
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)
            
            points.append(PointStruct(
                id=chunk_id,
                vector=embedding.tolist(),
                payload={
                    "path": file_info.relative_path,
                    "text": chunk.text,
                    "heading": chunk.heading,
                    "line_start": chunk.line_start,
                    "line_end": chunk.line_end,
                    "chunk_index": chunk.chunk_index,
                    "content_hash": file_info.content_hash,
                },
            ))
        
        # Upsert to Qdrant
        self.client.upsert(
            collection_name=self.config.collection_name,
            points=points,
        )
        
        return chunk_ids
    
    def _delete_chunks(self, chunk_ids: list[str]):
        """Delete chunks from Qdrant.
        
        Args:
            chunk_ids: List of chunk IDs to delete
        """
        if not chunk_ids:
            return
        
        self.client.delete(
            collection_name=self.config.collection_name,
            points_selector=chunk_ids,
        )
    
    def delete_file(self, relative_path: str) -> int:
        """Delete a file from the index.
        
        Args:
            relative_path: Relative path of file to delete
            
        Returns:
            Number of chunks deleted
        """
        chunk_ids = self.state.remove_file(relative_path)
        if chunk_ids:
            self._delete_chunks(chunk_ids)
            self.state.save(self.config.state_path)
        return len(chunk_ids)
    
    def clear(self):
        """Clear the entire collection and state."""
        try:
            self.client.delete_collection(self.config.collection_name)
        except Exception:
            pass  # Collection might not exist
        
        self._state = IndexState()
        self._state.collection_name = self.config.collection_name
        self._state.model_name = self.config.model_name
        self.state.save(self.config.state_path)
    
    def status(self) -> dict:
        """Get index status.
        
        Returns:
            Dict with status information
        """
        try:
            collection_info = self.client.get_collection(self.config.collection_name)
            point_count = collection_info.points_count
            vector_count = collection_info.vectors_count
        except Exception:
            point_count = 0
            vector_count = 0
        
        return {
            "collection_name": self.config.collection_name,
            "vault_path": str(self.config.vault),
            "model_name": self.config.model_name,
            "indexed_files": len(self.state.files),
            "total_chunks": point_count,
            "last_indexed": self.state.last_indexed,
        }
