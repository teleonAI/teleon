"""
Cortex Storage Backends.

This module provides storage backends for Cortex memory system.
Different backends are suited for different use cases:

- InMemoryStorage: Fast, ephemeral storage for development/testing
- RedisStorage: Distributed, production-ready storage
- PostgresStorage: Relational storage for complex queries

Example:
    ```python
    from teleon.cortex.storage import InMemoryStorage, RedisStorage, PostgresStorage
    
    # Development
    storage = InMemoryStorage()
    await storage.initialize()
    
    # Production (Redis)
    from teleon.cortex.storage import RedisConfig
    storage = RedisStorage(RedisConfig(
        host="localhost",
        password="secret"
    ))
    await storage.initialize()
    
    # Production (PostgreSQL)
    from teleon.cortex.storage import PostgresConfig
    storage = PostgresStorage(PostgresConfig(
        host="localhost",
        database="teleon",
        user="teleon",
        password="secret"
    ))
    await storage.initialize()
    ```
"""

from typing import Optional

from teleon.cortex.storage.base import (
    StorageBackend,
    StorageConfig,
    StorageMetrics,
    StorageError,
    ConnectionError,
    SerializationError,
    TTLError,
)

from teleon.cortex.storage.memory import InMemoryStorage

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
    from teleon.cortex.storage.chroma_storage import ChromaDBStorage, create_chroma_storage
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    ChromaDBStorage = None
    create_chroma_storage = None


__all__ = [
    # Base classes
    "StorageBackend",
    "StorageConfig",
    "StorageMetrics",
    
    # Errors
    "StorageError",
    "ConnectionError",
    "SerializationError",
    "TTLError",
    
    # Implementations
    "InMemoryStorage",
    "RedisStorage",
    "RedisConfig",
    "PostgresStorage",
    "PostgresConfig",
    "ChromaDBStorage",
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
    Factory function to create storage backend.
    
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
    from typing import Optional
    
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
    
    elif backend == "chroma" or backend == "chromadb":
        if not CHROMADB_AVAILABLE:
            raise ValueError(
                "ChromaDB storage not available. "
                "Install with: pip install chromadb fastembed"
            )
        # If config is None, create default ChromaDB storage
        if config is None:
            return create_chroma_storage()
        return config  # config is expected to be a ChromaDBStorage instance
    
    else:
        raise ValueError(
            f"Invalid backend type: {backend}. "
            f"Valid types: memory, redis, postgres, chroma"
        )


__all__.append("create_storage")

