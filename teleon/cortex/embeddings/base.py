"""
Abstract embedding model interface.
"""

from abc import ABC, abstractmethod
from typing import List
import logging

logger = logging.getLogger("teleon.cortex.embeddings")

# Standard embedding dimension (OpenAI text-embedding-3-small)
EMBEDDING_DIMENSION = 1536


def normalize_embedding(embedding: List[float], target_dim: int = EMBEDDING_DIMENSION) -> List[float]:
    """
    Pad embedding to target dimension with zeros.

    This ensures all embeddings have the same dimension for storage,
    regardless of the model used.

    Args:
        embedding: Original embedding
        target_dim: Target dimension (default: 1536)

    Returns:
        Padded embedding
    """
    if len(embedding) >= target_dim:
        return embedding[:target_dim]
    return embedding + [0.0] * (target_dim - len(embedding))


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    def normalize_embedding(self, embedding: List[float], target_dim: int = EMBEDDING_DIMENSION) -> List[float]:
        """
        Pad embedding to target dimension with zeros.

        This ensures all embeddings have the same dimension for storage,
        regardless of the model used.

        Args:
            embedding: Original embedding
            target_dim: Target dimension (default: 1536)

        Returns:
            Padded embedding
        """
        if len(embedding) >= target_dim:
            return embedding[:target_dim]
        return embedding + [0.0] * (target_dim - len(embedding))
