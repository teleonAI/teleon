"""
Cortex - Simple Memory System for Teleon Agents.

Cortex provides a simple, powerful memory system with just 6 methods:
store, search, get, update, delete, count.

Example:
    ```python
    @teleon.agent(memory=True)
    async def support_agent(query: str, customer_id: str, memory: Memory):
        # Auto-retrieved context available
        if memory.context:
            print(f"Found {len(memory.context.entries)} relevant memories")

        # Store new information
        await memory.store(
            content=f"Customer asked: {query}",
            customer_id=customer_id,
            type="query"
        )

        # Search semantically
        results = await memory.search(
            query="billing issues",
            filter={"customer_id": customer_id}
        )

        # Get recent history
        history = await memory.get(
            filter={"customer_id": customer_id},
            limit=10
        )

        return response
    ```

Configuration:
    ```python
    @teleon.agent(
        memory={
            "auto": True,           # Auto-save conversations
            "scope": ["customer_id"],  # Multi-tenancy scope
            "fields": ["query", "type"],  # Fields to store
        }
    )
    async def agent(query: str, customer_id: str, memory: Memory):
        ...
    ```
"""

# Core API
from teleon.cortex.memory import Memory
from teleon.cortex.entry import Entry
from teleon.cortex.context import MemoryContext
from teleon.cortex.config import MemoryConfig, AutoContextConfig, LayerConfig, parse_memory_config
from teleon.cortex.layers import MemoryLayer
from teleon.cortex.scope import ScopeEnforcer

# Manager (for framework integration)
from teleon.cortex.manager import (
    MemoryManager,
    get_memory_manager,
    get_storage_backend,
    set_storage_backend,
    clear_memory_managers,
)

# Storage backends
from teleon.cortex.storage.base import StorageBackend
from teleon.cortex.storage.inmemory import InMemoryBackend

# Embedding service
from teleon.cortex.embeddings.service import (
    EmbeddingService,
    EmbeddingCache,
    get_embedding_service,
)
from teleon.cortex.embeddings.base import EmbeddingModel, EMBEDDING_DIMENSION, normalize_embedding

# Optional backends
try:
    from teleon.cortex.storage.postgres import PostgresBackend
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    PostgresBackend = None

try:
    from teleon.cortex.storage.redis import RedisBackend
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    RedisBackend = None

# Optional embedding models
try:
    from teleon.cortex.embeddings.fastembed import FastEmbedModel
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    FastEmbedModel = None

try:
    from teleon.cortex.embeddings.openai import OpenAIEmbedModel
    OPENAI_EMBED_AVAILABLE = True
except ImportError:
    OPENAI_EMBED_AVAILABLE = False
    OpenAIEmbedModel = None


__all__ = [
    # Core API
    "Memory",
    "Entry",
    "MemoryContext",
    "MemoryConfig",
    "AutoContextConfig",
    "LayerConfig",
    "parse_memory_config",
    "MemoryLayer",
    "ScopeEnforcer",

    # Manager
    "MemoryManager",
    "get_memory_manager",
    "get_storage_backend",
    "set_storage_backend",
    "clear_memory_managers",

    # Storage
    "StorageBackend",
    "InMemoryBackend",
    "PostgresBackend",
    "RedisBackend",

    # Embeddings
    "EmbeddingService",
    "EmbeddingCache",
    "EmbeddingModel",
    "get_embedding_service",
    "EMBEDDING_DIMENSION",
    "normalize_embedding",
    "FastEmbedModel",
    "OpenAIEmbedModel",

    # Availability flags
    "POSTGRES_AVAILABLE",
    "REDIS_AVAILABLE",
    "FASTEMBED_AVAILABLE",
    "OPENAI_EMBED_AVAILABLE",
]
