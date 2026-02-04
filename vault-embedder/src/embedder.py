"""Embedding model wrapper using sentence-transformers."""

from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Wrapper for bge-micro-v2 embedding model."""
    
    def __init__(
        self,
        model_name: str = "taylorai/bge-micro-v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """Initialize the embedder.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detected if None.
            cache_dir: Directory to cache the model
        """
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._device = device
        self._cache_dir = cache_dir
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model on first use."""
        if self._model is None:
            self._model = SentenceTransformer(
                self.model_name,
                device=self._device,
                cache_folder=self._cache_dir,
            )
        return self._model
    
    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.model.get_sentence_embedding_dimension()
    
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        return self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[np.ndarray]:
        """Embed multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > batch_size,
        )
        return list(embeddings)
    
    def unload(self):
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None


# Singleton instance for CLI use
_embedder: Optional[Embedder] = None


def get_embedder(model_name: str = "taylorai/bge-micro-v2") -> Embedder:
    """Get or create the singleton embedder instance."""
    global _embedder
    if _embedder is None or _embedder.model_name != model_name:
        _embedder = Embedder(model_name)
    return _embedder
