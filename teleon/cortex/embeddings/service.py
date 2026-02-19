"""
Embedding service with tier-based model selection.

Priority order:
1. Teleon centralized service (when TELEON_EMBEDDING_URL is set — managed deployments)
2. OpenAI (paid tier, requires OPENAI_API_KEY)
3. FastEmbed (free tier, local model)
"""

from typing import List, Optional, Dict, Any
import logging
import hashlib
import os

from teleon.cortex.embeddings.base import EmbeddingModel, EMBEDDING_DIMENSION

logger = logging.getLogger("teleon.cortex.embeddings.service")


class EmbeddingCache:
    """Simple in-memory cache for embeddings."""

    def __init__(self, max_size: int = 10000):
        self._cache: Dict[str, List[float]] = {}
        self._max_size = max_size

    def _make_key(self, text: str, model_name: str) -> str:
        """Create cache key from text and model."""
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """Get cached embedding."""
        key = self._make_key(text, model_name)
        return self._cache.get(key)

    def set(self, text: str, model_name: str, embedding: List[float]) -> None:
        """Cache embedding."""
        # Simple eviction: clear half when full
        if len(self._cache) >= self._max_size:
            keys = list(self._cache.keys())
            for key in keys[:len(keys) // 2]:
                del self._cache[key]

        key = self._make_key(text, model_name)
        self._cache[key] = embedding

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()


class EmbeddingService:
    """
    Embedding service with tier-based model selection.

    - Free tier: FastEmbed (local, free)
    - Paid tier: OpenAI (API, better quality)
    """

    def __init__(
        self,
        is_paid_tier: bool = False,
        cache_enabled: bool = True,
        cache_max_size: int = 10000
    ):
        """
        Initialize embedding service.

        Args:
            is_paid_tier: Whether user is on paid tier
            cache_enabled: Enable embedding cache
            cache_max_size: Maximum cache entries
        """
        self._is_paid_tier = is_paid_tier
        self._model: Optional[EmbeddingModel] = None
        self._cache = EmbeddingCache(cache_max_size) if cache_enabled else None

        logger.info(f"Embedding service initialized (paid_tier={is_paid_tier})")

    def _get_model(self) -> EmbeddingModel:
        """
        Get or create embedding model.

        Selection priority:
        1. Teleon centralized service (TELEON_EMBEDDING_URL env var)
        2. OpenAI (paid tier + OPENAI_API_KEY)
        3. FastEmbed (local fallback)
        """
        if self._model is None:
            embedding_url = os.getenv("TELEON_EMBEDDING_URL", "")

            if embedding_url:
                try:
                    from teleon.cortex.embeddings.teleon import TeleonEmbedModel
                    self._model = TeleonEmbedModel(embedding_url=embedding_url)
                    logger.info(f"Using Teleon centralized embeddings ({embedding_url})")
                except (ImportError, ValueError) as e:
                    logger.warning(f"Teleon embedding service not available: {e}")

            if self._model is None and self._is_paid_tier:
                try:
                    from teleon.cortex.embeddings.openai import OpenAIEmbedModel
                    self._model = OpenAIEmbedModel()
                    logger.info("Using OpenAI embeddings (paid tier)")
                except (ImportError, ValueError) as e:
                    logger.warning(f"OpenAI not available, falling back to FastEmbed: {e}")

            if self._model is None:
                from teleon.cortex.embeddings.fastembed import FastEmbedModel
                self._model = FastEmbedModel()
                logger.info("Using FastEmbed (local fallback)")

        return self._model

    @property
    def model_name(self) -> str:
        """Get current model name."""
        model = self._get_model()
        if hasattr(model, 'model_name'):
            return model.model_name
        return model.__class__.__name__

    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        Uses caching to avoid re-computing for identical text.
        """
        model = self._get_model()

        # Check cache
        if self._cache:
            cached = self._cache.get(text, self.model_name)
            if cached:
                return cached

        # Generate embedding
        embedding = await model.embed(text)

        # Cache result
        if self._cache:
            self._cache.set(text, self.model_name, embedding)

        return embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        model = self._get_model()

        # Check cache for each text
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        if self._cache:
            for i, text in enumerate(texts):
                cached = self._cache.get(text, self.model_name)
                if cached:
                    results[i] = cached
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        # Generate uncached embeddings
        if uncached_texts:
            new_embeddings = await model.embed_batch(uncached_texts)

            for idx, embedding in zip(uncached_indices, new_embeddings):
                results[idx] = embedding

                # Cache result
                if self._cache:
                    self._cache.set(texts[idx], self.model_name, embedding)

        return results  # type: ignore

    def set_paid_tier(self, is_paid: bool) -> None:
        """
        Update tier setting.

        Note: This will reset the model on next embed call.
        """
        if self._is_paid_tier != is_paid:
            self._is_paid_tier = is_paid
            self._model = None
            logger.info(f"Tier changed to {'paid' if is_paid else 'free'}")


# Global service instance
_global_service: Optional[EmbeddingService] = None


def get_embedding_service(is_paid_tier: bool = False) -> EmbeddingService:
    """Get or create global embedding service."""
    global _global_service

    if _global_service is None:
        _global_service = EmbeddingService(is_paid_tier=is_paid_tier)
    elif _global_service._is_paid_tier != is_paid_tier:
        _global_service.set_paid_tier(is_paid_tier)

    return _global_service
