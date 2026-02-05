"""
Cortex Storage Backends.

This module provides storage backends for the Cortex memory system.

Available backends:
- InMemoryBackend: Fast, ephemeral storage for development/testing
- PostgresBackend: Production storage with pgvector for similarity search
- RedisBackend: Production storage with RediSearch for similarity search

Example:
    ```python
    from teleon.cortex.storage import InMemoryBackend, PostgresBackend, RedisBackend

    # Development - In-memory storage
    backend = InMemoryBackend()

    # Production with PostgreSQL + pgvector
    backend = PostgresBackend(
        host="localhost",
        port=5432,
        database="teleon",
        user="postgres",
        password="secret"
    )

    # Production with Redis + RediSearch
    backend = RedisBackend(
        host="localhost",
        port=6379,
        password="secret"
    )
    ```
"""

from teleon.cortex.storage.base import StorageBackend, StorageError
from teleon.cortex.storage.inmemory import InMemoryBackend

# Optional backends (require additional packages)
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


__all__ = [
    # Base classes
    "StorageBackend",
    "StorageError",

    # Implementations
    "InMemoryBackend",
    "PostgresBackend",
    "RedisBackend",

    # Availability flags
    "POSTGRES_AVAILABLE",
    "REDIS_AVAILABLE",
]


def create_backend(
    backend_type: str = "memory",
    **kwargs
) -> StorageBackend:
    """
    Factory function to create storage backend.

    Args:
        backend_type: Backend type ("memory", "postgres", "redis")
        **kwargs: Backend-specific configuration

    Returns:
        Storage backend instance

    Raises:
        ValueError: If backend type is invalid or not available

    Example:
        ```python
        # Create memory storage
        backend = create_backend("memory")

        # Create PostgreSQL storage
        backend = create_backend(
            "postgres",
            host="localhost",
            database="teleon"
        )

        # Create Redis storage
        backend = create_backend(
            "redis",
            host="localhost",
            password="secret"
        )
        ```
    """
    backend_type = backend_type.lower()

    if backend_type == "memory" or backend_type == "inmemory":
        return InMemoryBackend()

    elif backend_type == "postgres" or backend_type == "postgresql":
        if not POSTGRES_AVAILABLE:
            raise ValueError(
                "PostgreSQL backend not available. "
                "Install with: pip install asyncpg"
            )
        return PostgresBackend(**kwargs)

    elif backend_type == "redis":
        if not REDIS_AVAILABLE:
            raise ValueError(
                "Redis backend not available. "
                "Install with: pip install 'redis[asyncio]'"
            )
        return RedisBackend(**kwargs)

    else:
        raise ValueError(
            f"Invalid backend type: {backend_type}. "
            f"Valid types: memory, postgres, redis"
        )


__all__.append("create_backend")
