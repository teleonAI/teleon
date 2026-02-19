"""
Teleon centralized embedding model — calls the platform embedding service over HTTP.

Used automatically for managed deployments when TELEON_EMBEDDING_URL is set.
No local model download, no GPU, no heavy dependencies.
"""

from typing import List, Optional
import os
import logging

from teleon.cortex.embeddings.base import EmbeddingModel

logger = logging.getLogger("teleon.cortex.embeddings.teleon")

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None


class TeleonEmbedModel(EmbeddingModel):
    """
    HTTP client that calls the Teleon centralized embedding service.

    The service runs multilingual-e5-base (768 dims) on a shared ECS task.
    All managed agent deployments use this by default.
    """

    def __init__(
        self,
        embedding_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for TeleonEmbedModel. "
                "Install with: pip install httpx"
            )

        self._base_url = (
            embedding_url
            or os.getenv("TELEON_EMBEDDING_URL", "")
        ).rstrip("/")

        if not self._base_url:
            raise ValueError(
                "Embedding service URL required. "
                "Set TELEON_EMBEDDING_URL environment variable."
            )

        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._dimension: Optional[int] = None
        self.model_name = "teleon-managed"

        logger.info(f"TeleonEmbedModel targeting {self._base_url}")

    def _get_client(self) -> "httpx.AsyncClient":
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._client

    @property
    def dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension
        # Default for multilingual-e5-base; updated on first /info call
        return 768

    async def _fetch_info(self) -> None:
        """Fetch model metadata from the service (called once lazily)."""
        if self._dimension is not None:
            return
        try:
            client = self._get_client()
            resp = await client.get("/info")
            resp.raise_for_status()
            data = resp.json()
            self._dimension = data.get("dimension", 768)
            self.model_name = data.get("model", "teleon-managed")
            logger.info(
                f"Embedding service info: model={self.model_name}, "
                f"dimension={self._dimension}"
            )
        except Exception as e:
            logger.warning(f"Could not fetch embedding service info: {e}")
            self._dimension = 768

    async def embed(self, text: str) -> List[float]:
        await self._fetch_info()
        client = self._get_client()

        resp = await client.post("/embed", json={"text": text})
        resp.raise_for_status()
        data = resp.json()
        return data["embedding"]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        await self._fetch_info()
        client = self._get_client()

        resp = await client.post("/embed_batch", json={"texts": texts})
        resp.raise_for_status()
        data = resp.json()
        return data["embeddings"]

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
