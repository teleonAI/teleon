"""
Base Storage Backend Interface for Cortex Memory System.

This module provides the abstract base class that all storage backends must implement.
Storage backends handle the persistence layer for Cortex memory types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timedelta
from pydantic import BaseModel, Field


class StorageMetrics(BaseModel):
    """Metrics for storage backend performance."""
    
    total_operations: int = 0
    get_operations: int = 0
    set_operations: int = 0
    delete_operations: int = 0
    hits: int = 0
    misses: int = 0
    total_keys: int = 0
    total_size_bytes: int = 0
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


class StorageConfig(BaseModel):
    """Configuration for storage backends."""
    
    default_ttl: Optional[int] = Field(
        None,
        description="Default TTL in seconds for stored items"
    )
    max_size: Optional[int] = Field(
        None,
        description="Maximum number of items to store"
    )
    max_size_bytes: Optional[int] = Field(
        None,
        description="Maximum total size in bytes"
    )
    enable_metrics: bool = Field(
        True,
        description="Enable metrics collection"
    )
    compression: bool = Field(
        False,
        description="Enable compression for stored values"
    )


class StorageBackend(ABC):
    """
    Abstract base class for all storage backends.
    
    Storage backends provide the persistence layer for Cortex memory.
    They handle storing, retrieving, and managing data with optional TTL.
    
    Implementations include:
    - InMemoryStorage: Fast, ephemeral storage for development
    - RedisStorage: Distributed, production-ready storage
    - PostgresStorage: Relational database storage for complex queries
    
    Example:
        ```python
        storage = InMemoryStorage()
        await storage.set("user_123", {"name": "Alice"}, ttl=3600)
        value = await storage.get("user_123")
        ```
    """
    
    def __init__(self, config: Optional[StorageConfig] = None):
        """
        Initialize storage backend.
        
        Args:
            config: Storage configuration
        """
        self.config = config or StorageConfig()
        self.metrics = StorageMetrics() if self.config.enable_metrics else None
        self._initialized = False
    
    async def initialize(self) -> None:
        """
        Initialize the storage backend.
        
        Called once before first use. Subclasses should override
        to set up connections, create tables, etc.
        """
        self._initialized = True
    
    async def shutdown(self) -> None:
        """
        Shutdown the storage backend gracefully.
        
        Called when shutting down. Subclasses should override
        to close connections, flush buffers, etc.
        """
        self._initialized = False
    
    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a value with optional TTL.
        
        Args:
            key: Storage key
            value: Value to store (must be JSON-serializable)
            ttl: Time-to-live in seconds (None = no expiration)
            metadata: Optional metadata to store with value
        
        Returns:
            True if stored successfully
        
        Raises:
            StorageError: If storage operation fails
        """
        pass
    
    @abstractmethod
    async def get(
        self,
        key: str,
        default: Optional[Any] = None
    ) -> Optional[Any]:
        """
        Retrieve a value by key.
        
        Args:
            key: Storage key
            default: Default value if key not found
        
        Returns:
            Stored value or default if not found
        
        Raises:
            StorageError: If retrieval operation fails
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete a value by key.
        
        Args:
            key: Storage key
        
        Returns:
            True if deleted, False if key didn't exist
        
        Raises:
            StorageError: If delete operation fails
        """
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists.
        
        Args:
            key: Storage key
        
        Returns:
            True if key exists
        
        Raises:
            StorageError: If check operation fails
        """
        pass
    
    @abstractmethod
    async def list_keys(
        self,
        pattern: str = "*",
        limit: Optional[int] = None
    ) -> List[str]:
        """
        List keys matching a pattern.
        
        Args:
            pattern: Pattern to match (supports wildcards)
            limit: Maximum number of keys to return
        
        Returns:
            List of matching keys
        
        Raises:
            StorageError: If list operation fails
        """
        pass
    
    @abstractmethod
    async def clear(self, pattern: str = "*") -> int:
        """
        Clear all keys matching a pattern.
        
        Args:
            pattern: Pattern to match (default: all keys)
        
        Returns:
            Number of keys deleted
        
        Raises:
            StorageError: If clear operation fails
        """
        pass
    
    @abstractmethod
    async def get_ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for a key.
        
        Args:
            key: Storage key
        
        Returns:
            Remaining TTL in seconds, None if no TTL or key doesn't exist
        
        Raises:
            StorageError: If TTL check fails
        """
        pass
    
    @abstractmethod
    async def set_ttl(self, key: str, ttl: int) -> bool:
        """
        Set or update TTL for an existing key.
        
        Args:
            key: Storage key
            ttl: Time-to-live in seconds
        
        Returns:
            True if TTL was set
        
        Raises:
            StorageError: If TTL set operation fails
        """
        pass
    
    async def get_many(
        self,
        keys: List[str]
    ) -> Dict[str, Any]:
        """
        Get multiple values at once.
        
        Default implementation calls get() for each key.
        Subclasses should override with batch operations if available.
        
        Args:
            keys: List of storage keys
        
        Returns:
            Dictionary of key-value pairs (missing keys omitted)
        """
        results = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                results[key] = value
        return results
    
    async def set_many(
        self,
        items: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> int:
        """
        Set multiple values at once.
        
        Default implementation calls set() for each item.
        Subclasses should override with batch operations if available.
        
        Args:
            items: Dictionary of key-value pairs to store
            ttl: Time-to-live in seconds for all items
        
        Returns:
            Number of items successfully stored
        """
        count = 0
        for key, value in items.items():
            if await self.set(key, value, ttl=ttl):
                count += 1
        return count
    
    async def delete_many(self, keys: List[str]) -> int:
        """
        Delete multiple keys at once.
        
        Default implementation calls delete() for each key.
        Subclasses should override with batch operations if available.
        
        Args:
            keys: List of storage keys to delete
        
        Returns:
            Number of keys successfully deleted
        """
        count = 0
        for key in keys:
            if await self.delete(key):
                count += 1
        return count
    
    def get_metrics(self) -> Optional[StorageMetrics]:
        """
        Get storage metrics.
        
        Returns:
            StorageMetrics if enabled, None otherwise
        """
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset storage metrics."""
        if self.metrics:
            self.metrics = StorageMetrics()
    
    def _update_metrics(
        self,
        operation: str,
        hit: Optional[bool] = None
    ) -> None:
        """
        Update metrics for an operation.
        
        Args:
            operation: Operation type (get, set, delete)
            hit: Whether operation was a cache hit (for get operations)
        """
        if not self.metrics:
            return
        
        self.metrics.total_operations += 1
        
        if operation == "get":
            self.metrics.get_operations += 1
            if hit is True:
                self.metrics.hits += 1
            elif hit is False:
                self.metrics.misses += 1
        elif operation == "set":
            self.metrics.set_operations += 1
        elif operation == "delete":
            self.metrics.delete_operations += 1


class StorageError(Exception):
    """Base exception for storage backend errors."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        key: Optional[str] = None
    ):
        """
        Initialize storage error.
        
        Args:
            message: Error message
            operation: Operation that failed
            key: Key involved in the operation
        """
        self.message = message
        self.operation = operation
        self.key = key
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format error message."""
        parts = [self.message]
        if self.operation:
            parts.append(f"operation={self.operation}")
        if self.key:
            parts.append(f"key={self.key}")
        return " | ".join(parts)


class ConnectionError(StorageError):
    """Raised when storage backend connection fails."""
    pass


class SerializationError(StorageError):
    """Raised when value serialization/deserialization fails."""
    pass


class TTLError(StorageError):
    """Raised when TTL operation fails."""
    pass

