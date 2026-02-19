"""
Cortex Embedding Service.

Provides embedding model selection with the following priority:
1. Teleon centralized service (managed deployments, 768 dims, multilingual)
2. OpenAI (paid tier, 1536 dims)
3. FastEmbed (local fallback, 384 dims padded to EMBEDDING_DIMENSION)

Example:
    ```python
    from teleon.cortex.embeddings import get_embedding_service, EmbeddingService

    # Managed deployments auto-detect TELEON_EMBEDDING_URL
    service = get_embedding_service()

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

try:
    from teleon.cortex.embeddings.teleon import TeleonEmbedModel
    TELEON_EMBED_AVAILABLE = True
except ImportError:
    TELEON_EMBED_AVAILABLE = False
    TeleonEmbedModel = None


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
    "TeleonEmbedModel",

    # Availability flags
    "FASTEMBED_AVAILABLE",
    "OPENAI_AVAILABLE",
    "TELEON_EMBED_AVAILABLE",
]
