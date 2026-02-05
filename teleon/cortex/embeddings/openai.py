"""
OpenAI embedding model - High quality paid embeddings.
"""

from typing import List, Optional
import os
import logging

from teleon.cortex.embeddings.base import EmbeddingModel, EMBEDDING_DIMENSION

logger = logging.getLogger("teleon.cortex.embeddings.openai")

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None


class OpenAIEmbedModel(EmbeddingModel):
    """
    OpenAI embedding model.

    Uses text-embedding-3-small (1536 dimensions) by default.
    Requires OPENAI_API_KEY environment variable or explicit api_key.
    """

    # OpenAI model dimension
    MODEL_DIMENSION = 1536

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ):
        """
        Initialize OpenAI embedding model.

        Args:
            model_name: Model to use (default: text-embedding-3-small)
            api_key: OpenAI API key (default: from OPENAI_API_KEY env)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI not available. Install with: pip install openai"
            )

        self.model_name = model_name
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self._client: Optional[AsyncOpenAI] = None

        logger.info(f"OpenAI embedding model initialized: {model_name}")

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self.MODEL_DIMENSION

    def _get_client(self) -> "AsyncOpenAI":
        """Get or create OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        client = self._get_client()

        response = await client.embeddings.create(
            model=self.model_name,
            input=text
        )

        return response.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        client = self._get_client()

        response = await client.embeddings.create(
            model=self.model_name,
            input=texts
        )

        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)

        return [item.embedding for item in sorted_data]
