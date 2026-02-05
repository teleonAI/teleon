"""
FastEmbed embedding model - Free, local embeddings.
"""

from typing import List, Optional
import logging

from teleon.cortex.embeddings.base import EmbeddingModel, EMBEDDING_DIMENSION

logger = logging.getLogger("teleon.cortex.embeddings.fastembed")

try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    TextEmbedding = None


class FastEmbedModel(EmbeddingModel):
    """
    FastEmbed embedding model.

    Free, local embeddings using ONNX-optimized models.
    Uses BAAI/bge-small-en-v1.5 (384 dimensions) by default.
    """

    # FastEmbed model dimension
    MODEL_DIMENSION = 384

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize FastEmbed model.

        Args:
            model_name: Model to use (default: bge-small-en-v1.5)
            cache_dir: Optional cache directory for models
        """
        if not FASTEMBED_AVAILABLE:
            raise ImportError(
                "FastEmbed not available. Install with: pip install fastembed"
            )

        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model: Optional[TextEmbedding] = None

        logger.info(f"FastEmbed model initialized: {model_name}")

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self.MODEL_DIMENSION

    def _load_model(self) -> "TextEmbedding":
        """Lazy load model."""
        if self._model is None:
            logger.info(f"Loading FastEmbed model: {self.model_name}")

            kwargs = {}
            if self.cache_dir:
                kwargs["cache_dir"] = self.cache_dir

            self._model = TextEmbedding(
                model_name=self.model_name,
                **kwargs
            )

            logger.info("FastEmbed model loaded successfully")

        return self._model

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        model = self._load_model()

        # FastEmbed returns generator
        embeddings = list(model.embed([text]))
        embedding = embeddings[0].tolist()

        # Pad to standard dimension
        return self.normalize_embedding(embedding, EMBEDDING_DIMENSION)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        model = self._load_model()

        embeddings = list(model.embed(texts))

        # Pad all to standard dimension
        return [
            self.normalize_embedding(emb.tolist(), EMBEDDING_DIMENSION)
            for emb in embeddings
        ]
