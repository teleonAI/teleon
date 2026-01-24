"""
Cortex Storage Backends.

This module provides storage backends for Cortex memory system.
There are two types of storage backends:

**Key-Value Storage (StorageBackend)**:
- InMemoryStorage: Fast, ephemeral storage for development/testing
- RedisStorage: Distributed, production-ready storage
- PostgresStorage: Relational storage for complex queries

**Vector Storage (VectorStorageBackend)**:
- ChromaDBVectorStorage: Embedded vector database with semantic search

Example:
    ```python
    from teleon.cortex.storage import InMemoryStorage, RedisStorage, PostgresStorage
    
    # Development - Key-Value Storage
    storage = InMemoryStorage()
    await storage.initialize()
    
    # Production (Redis)
    from teleon.cortex.storage import RedisConfig
    storage = RedisStorage(RedisConfig(
        host="localhost",
        password="secret"
    ))
    await storage.initialize()
    
    # Vector Storage (ChromaDB)
    from teleon.cortex.storage import ChromaDBVectorStorage, create_chroma_storage
    vector_storage = create_chroma_storage(deployment_id="my-agent")
    await vector_storage.initialize()
    
    # Store with embeddings
    doc_id = await vector_storage.store(
        content="Python is a programming language",
        embedding=embed_fn("Python is a programming language")
    )
    
    # Semantic search
    results = await vector_storage.search(
        query_embedding=embed_fn("What is Python?"),
        limit=5
    )
    ```
"""

from typing import Optional

# Key-Value Storage
from teleon.cortex.storage.base import (
    StorageBackend,
    StorageConfig,
    StorageMetrics,
    StorageError,
    ConnectionError as KVConnectionError,
    SerializationError,
    TTLError,
)

from teleon.cortex.storage.memory import InMemoryStorage

# Vector Storage
from teleon.cortex.storage.vector_base import (
    VectorStorageBackend,
    VectorStorageConfig,
    VectorSearchResult,
    VectorStorageMetrics,
    VectorStorageError,
    ConnectionError as VectorConnectionError,
    CollectionNotFoundError,
    DocumentNotFoundError,
    EmbeddingDimensionError,
    HealthStatus,
)

# Optional backends (require additional packages)
try:
    from teleon.cortex.storage.redis import RedisStorage, RedisConfig
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    RedisStorage = None
    RedisConfig = None

try:
    from teleon.cortex.storage.postgres import PostgresStorage, PostgresConfig
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    PostgresStorage = None
    PostgresConfig = None

try:
    from teleon.cortex.storage.chroma_storage import (
        ChromaDBVectorStorage,
        ChromaDBStorage,  # Backward compatibility alias
        create_chroma_storage,
    )
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    ChromaDBVectorStorage = None
    ChromaDBStorage = None
    create_chroma_storage = None


__all__ = [
    # Key-Value Storage - Base classes
    "StorageBackend",
    "StorageConfig",
    "StorageMetrics",
    
    # Key-Value Storage - Errors
    "StorageError",
    "KVConnectionError",
    "SerializationError",
    "TTLError",
    
    # Key-Value Storage - Implementations
    "InMemoryStorage",
    "RedisStorage",
    "RedisConfig",
    "PostgresStorage",
    "PostgresConfig",
    
    # Vector Storage - Base classes
    "VectorStorageBackend",
    "VectorStorageConfig",
    "VectorSearchResult",
    "VectorStorageMetrics",
    "HealthStatus",
    
    # Vector Storage - Errors
    "VectorStorageError",
    "VectorConnectionError",
    "CollectionNotFoundError",
    "DocumentNotFoundError",
    "EmbeddingDimensionError",
    
    # Vector Storage - Implementations
    "ChromaDBVectorStorage",
    "ChromaDBStorage",  # Backward compatibility
    "create_chroma_storage",
    
    # Availability flags
    "REDIS_AVAILABLE",
    "POSTGRES_AVAILABLE",
    "CHROMADB_AVAILABLE",
]


def create_storage(
    backend: str = "memory",
    config: Optional[StorageConfig] = None
) -> StorageBackend:
    """
    Factory function to create key-value storage backend.
    
    Args:
        backend: Backend type ("memory", "redis", "postgres")
        config: Backend-specific configuration
    
    Returns:
        Initialized storage backend
    
    Raises:
        ValueError: If backend type is invalid or not available
    
    Example:
        ```python
        # Create memory storage
        storage = create_storage("memory")
        
        # Create Redis storage
        from teleon.cortex.storage import RedisConfig
        storage = create_storage("redis", RedisConfig(host="localhost"))
        ```
    """
    backend = backend.lower()
    
    if backend == "memory":
        return InMemoryStorage(config)
    
    elif backend == "redis":
        if not REDIS_AVAILABLE:
            raise ValueError(
                "Redis storage not available. "
                "Install with: pip install 'redis[asyncio]'"
            )
        return RedisStorage(config)
    
    elif backend == "postgres" or backend == "postgresql":
        if not POSTGRES_AVAILABLE:
            raise ValueError(
                "PostgreSQL storage not available. "
                "Install with: pip install asyncpg"
            )
        return PostgresStorage(config)
    
    else:
        raise ValueError(
            f"Invalid backend type: {backend}. "
            f"Valid types: memory, redis, postgres"
        )


def create_vector_storage(
    backend: str = "chroma",
    deployment_id: Optional[str] = None,
    **kwargs
) -> VectorStorageBackend:
    """
    Factory function to create vector storage backend.
    
    Args:
        backend: Backend type ("chroma", "pinecone", "weaviate")
        deployment_id: Deployment/tenant ID for isolation
        **kwargs: Backend-specific configuration
    
    Returns:
        Initialized vector storage backend
    
    Raises:
        ValueError: If backend type is invalid or not available
    
    Example:
        ```python
        # Create ChromaDB storage
        storage = create_vector_storage("chroma", deployment_id="my-agent")
        await storage.initialize()
        ```
    """
    backend = backend.lower()
    
    if backend == "chroma" or backend == "chromadb":
        if not CHROMADB_AVAILABLE:
            raise ValueError(
                "ChromaDB storage not available. "
                "Install with: pip install chromadb fastembed"
            )
        return create_chroma_storage(deployment_id=deployment_id, **kwargs)
    
    else:
        raise ValueError(
            f"Invalid vector backend type: {backend}. "
            f"Valid types: chroma"
        )


__all__.extend(["create_storage", "create_vector_storage"])
