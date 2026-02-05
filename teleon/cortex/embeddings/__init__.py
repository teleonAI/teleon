"""
Cortex Embedding Service.

Provides tier-based embedding model selection:
- Free tier: FastEmbed (local, free, 384 dims padded to 1536)
- Paid tier: OpenAI (API, better quality, 1536 dims)

Example:
    ```python
    from teleon.cortex.embeddings import get_embedding_service, EmbeddingService

    # Get service based on tier
    service = get_embedding_service(is_paid_tier=False)

    # Generate embedding
    embedding = await service.embed("Hello, world!")

    # Batch embeddings
    embeddings = await service.embed_batch(["Hello", "World"])
    ```
"""

from teleon.cortex.embeddings.base import (
    EmbeddingModel,
    EMBEDDING_DIMENSION,
    normalize_embedding,
)
from teleon.cortex.embeddings.service import (
    EmbeddingService,
    EmbeddingCache,
    get_embedding_service,
)

# Optional embedding models
try:
    from teleon.cortex.embeddings.fastembed import FastEmbedModel
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    FastEmbedModel = None

try:
    from teleon.cortex.embeddings.openai import OpenAIEmbedModel
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAIEmbedModel = None


__all__ = [
    # Base
    "EmbeddingModel",
    "EMBEDDING_DIMENSION",
    "normalize_embedding",

    # Service
    "EmbeddingService",
    "EmbeddingCache",
    "get_embedding_service",

    # Models
    "FastEmbedModel",
    "OpenAIEmbedModel",

    # Availability flags
    "FASTEMBED_AVAILABLE",
    "OPENAI_AVAILABLE",
]
