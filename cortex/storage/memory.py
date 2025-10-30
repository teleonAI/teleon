"""
In-Memory Storage Backend for Cortex.

Fast, ephemeral storage ideal for development, testing, and single-instance deployments.
All data is lost when the process stops.
"""

import asyncio
import fnmatch
import json
import sys
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from collections import OrderedDict
from pydantic import BaseModel

from teleon.cortex.storage.base import (
    StorageBackend,
    StorageConfig,
    StorageError,
    SerializationError,
)


class StoredItem:
    """Represents an item stored in memory."""
    
    def __init__(
        self,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize stored item.
        
        Args:
            value: Stored value
            ttl: Time-to-live in seconds
            metadata: Optional metadata
        """
        self.value = value
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.expires_at = (
            self.created_at + timedelta(seconds=ttl) if ttl else None
        )
        self.size_bytes = self._calculate_size(value)
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        def json_encoder(obj):
            """Custom JSON encoder for datetime and Pydantic models."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, BaseModel):
                return obj.model_dump()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
        
        try:
            return sys.getsizeof(json.dumps(value, default=json_encoder))
        except:
            return sys.getsizeof(str(value))
    
    def is_expired(self) -> bool:
        """Check if item has expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    def get_remaining_ttl(self) -> Optional[int]:
        """Get remaining TTL in seconds."""
        if not self.expires_at:
            return None
        remaining = (self.expires_at - datetime.utcnow()).total_seconds()
        return max(0, int(remaining))


class InMemoryStorage(StorageBackend):
    """
    In-memory storage backend using Python dict.
    
    Features:
    - Extremely fast (no I/O)
    - Automatic TTL expiration
    - LRU eviction when max_size is reached
    - Thread-safe operations
    - Metrics tracking
    
    Limitations:
    - Data lost on process restart
    - Not suitable for multi-instance deployments
    - Memory usage grows with stored data
    
    Example:
        ```python
        storage = InMemoryStorage(StorageConfig(
            max_size=10000,
            default_ttl=3600
        ))
        await storage.initialize()
        
        # Store value with TTL
        await storage.set("user:123", {"name": "Alice"}, ttl=300)
        
        # Retrieve value
        user = await storage.get("user:123")
        
        # Check metrics
        metrics = storage.get_metrics()
        print(f"Hit rate: {metrics.hit_rate():.2f}%")
        ```
    """
    
    def __init__(self, config: Optional[StorageConfig] = None):
        """
        Initialize in-memory storage.
        
        Args:
            config: Storage configuration
        """
        super().__init__(config)
        self._store: OrderedDict[str, StoredItem] = OrderedDict()
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = 60  # seconds
    
    async def initialize(self) -> None:
        """Initialize storage and start cleanup task."""
        await super().initialize()
        
        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired())
    
    async def shutdown(self) -> None:
        """Shutdown storage and stop cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        await super().shutdown()
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a value in memory.
        
        Args:
            key: Storage key
            value: Value to store
            ttl: Time-to-live in seconds
            metadata: Optional metadata
        
        Returns:
            True if stored successfully
        """
        async with self._lock:
            try:
                # Use default TTL if not specified
                effective_ttl = ttl or self.config.default_ttl
                
                # Create stored item
                item = StoredItem(value, effective_ttl, metadata)
                
                # Check if we need to evict items
                if self.config.max_size and len(self._store) >= self.config.max_size:
                    if key not in self._store:
                        # Evict oldest item (LRU)
                        self._store.popitem(last=False)
                
                # Check size limits
                if self.config.max_size_bytes:
                    current_size = sum(item.size_bytes for item in self._store.values())
                    if current_size + item.size_bytes > self.config.max_size_bytes:
                        raise StorageError(
                            f"Storage size limit exceeded: {self.config.max_size_bytes} bytes",
                            operation="set",
                            key=key
                        )
                
                # Store item (move to end for LRU)
                if key in self._store:
                    del self._store[key]
                self._store[key] = item
                
                # Update metrics
                self._update_metrics("set")
                if self.metrics:
                    self.metrics.total_keys = len(self._store)
                    self.metrics.total_size_bytes = sum(
                        item.size_bytes for item in self._store.values()
                    )
                
                return True
                
            except Exception as e:
                if isinstance(e, StorageError):
                    raise
                raise StorageError(
                    f"Failed to store value: {str(e)}",
                    operation="set",
                    key=key
                )
    
    async def get(
        self,
        key: str,
        default: Optional[Any] = None
    ) -> Optional[Any]:
        """
        Retrieve a value from memory.
        
        Args:
            key: Storage key
            default: Default value if not found
        
        Returns:
            Stored value or default
        """
        async with self._lock:
            item = self._store.get(key)
            
            if item is None:
                self._update_metrics("get", hit=False)
                return default
            
            # Check if expired
            if item.is_expired():
                del self._store[key]
                self._update_metrics("get", hit=False)
                return default
            
            # Move to end for LRU
            self._store.move_to_end(key)
            
            self._update_metrics("get", hit=True)
            return item.value
    
    async def delete(self, key: str) -> bool:
        """
        Delete a value from memory.
        
        Args:
            key: Storage key
        
        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if key in self._store:
                del self._store[key]
                self._update_metrics("delete")
                
                if self.metrics:
                    self.metrics.total_keys = len(self._store)
                    self.metrics.total_size_bytes = sum(
                        item.size_bytes for item in self._store.values()
                    )
                
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in memory.
        
        Args:
            key: Storage key
        
        Returns:
            True if key exists and not expired
        """
        async with self._lock:
            item = self._store.get(key)
            if item is None:
                return False
            
            if item.is_expired():
                del self._store[key]
                return False
            
            return True
    
    async def list_keys(
        self,
        pattern: str = "*",
        limit: Optional[int] = None
    ) -> List[str]:
        """
        List keys matching a pattern.
        
        Args:
            pattern: Wildcard pattern (e.g., "user:*")
            limit: Maximum number of keys to return
        
        Returns:
            List of matching keys
        """
        async with self._lock:
            # Remove expired items first (internal method, no lock)
            self._remove_expired_internal()
            
            matching_keys = [
                key for key in self._store.keys()
                if fnmatch.fnmatch(key, pattern)
            ]
            
            if limit:
                matching_keys = matching_keys[:limit]
            
            return matching_keys
    
    async def clear(self, pattern: str = "*") -> int:
        """
        Clear keys matching a pattern.
        
        Args:
            pattern: Wildcard pattern
        
        Returns:
            Number of keys deleted
        """
        async with self._lock:
            keys_to_delete = [
                key for key in self._store.keys()
                if fnmatch.fnmatch(key, pattern)
            ]
            
            for key in keys_to_delete:
                del self._store[key]
            
            if self.metrics:
                self.metrics.total_keys = len(self._store)
                self.metrics.total_size_bytes = sum(
                    item.size_bytes for item in self._store.values()
                )
            
            return len(keys_to_delete)
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for a key.
        
        Args:
            key: Storage key
        
        Returns:
            Remaining TTL in seconds, None if no TTL
        """
        async with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            
            if item.is_expired():
                del self._store[key]
                return None
            
            return item.get_remaining_ttl()
    
    async def set_ttl(self, key: str, ttl: int) -> bool:
        """
        Set TTL for an existing key.
        
        Args:
            key: Storage key
            ttl: Time-to-live in seconds
        
        Returns:
            True if TTL was set
        """
        async with self._lock:
            item = self._store.get(key)
            if item is None:
                return False
            
            if item.is_expired():
                del self._store[key]
                return False
            
            item.expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            return True
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values efficiently.
        
        Args:
            keys: List of storage keys
        
        Returns:
            Dictionary of key-value pairs
        """
        async with self._lock:
            results = {}
            for key in keys:
                item = self._store.get(key)
                if item and not item.is_expired():
                    results[key] = item.value
                    # Move to end for LRU
                    self._store.move_to_end(key)
            return results
    
    async def set_many(
        self,
        items: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> int:
        """
        Set multiple values efficiently.
        
        Args:
            items: Dictionary of key-value pairs
            ttl: Time-to-live for all items
        
        Returns:
            Number of items stored
        """
        count = 0
        for key, value in items.items():
            if await self.set(key, value, ttl=ttl):
                count += 1
        return count
    
    async def _cleanup_expired(self) -> None:
        """Background task to remove expired items."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._remove_expired()
            except asyncio.CancelledError:
                break
            except Exception:
                # Silently continue on errors
                pass
    
    def _remove_expired_internal(self) -> int:
        """Remove all expired items (must be called with lock held)."""
        expired_keys = [
            key for key, item in self._store.items()
            if item.is_expired()
        ]
        
        for key in expired_keys:
            del self._store[key]
        
        if expired_keys and self.metrics:
            self.metrics.total_keys = len(self._store)
            self.metrics.total_size_bytes = sum(
                item.size_bytes for item in self._store.values()
            )
        
        return len(expired_keys)
    
    async def _remove_expired(self) -> int:
        """Remove all expired items."""
        async with self._lock:
            return self._remove_expired_internal()
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get detailed storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        async with self._lock:
            # Remove expired items first (internal method, no lock)
            self._remove_expired_internal()
            
            stats = {
                "total_keys": len(self._store),
                "total_size_bytes": sum(item.size_bytes for item in self._store.values()),
                "items_with_ttl": sum(
                    1 for item in self._store.values()
                    if item.expires_at is not None
                ),
                "items_without_ttl": sum(
                    1 for item in self._store.values()
                    if item.expires_at is None
                ),
            }
            
            if self.metrics:
                stats.update({
                    "total_operations": self.metrics.total_operations,
                    "get_operations": self.metrics.get_operations,
                    "set_operations": self.metrics.set_operations,
                    "delete_operations": self.metrics.delete_operations,
                    "hit_rate": f"{self.metrics.hit_rate():.2f}%",
                })
            
            return stats

