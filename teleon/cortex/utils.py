"""
Utility functions for Cortex memory system.

Provides thread-safe caching, validation, and helper functions.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable
from collections import OrderedDict
from datetime import datetime
import logging

logger = logging.getLogger("teleon.cortex.utils")


class AsyncLRUCache:
    """
    Thread-safe LRU cache with TTL support for async operations.
    
    Features:
    - Async-safe operations with locks
    - TTL (time-to-live) support
    - LRU eviction when cache is full
    - Automatic expiration cleanup
    
    Example:
        ```python
        cache = AsyncLRUCache(max_size=1000, default_ttl=300)
        
        # Set value
        await cache.set("key", "value", ttl=60)
        
        # Get value
        value = await cache.get("key")
        
        # Clear cache
        await cache.clear()
        ```
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = None):
        """
        Initialize async LRU cache.
        
        Args:
            max_size: Maximum number of items in cache
            default_ttl: Default TTL in seconds (None = no expiration)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            if key not in self._cache:
                return None
            
            value, expiry = self._cache[key]
            
            # Check TTL
            if expiry and time.time() > expiry:
                del self._cache[key]
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = use default_ttl)
        """
        async with self._lock:
            expiry = None
            if ttl or self.default_ttl:
                expiry = time.time() + (ttl or self.default_ttl)
            
            # Remove if exists
            if key in self._cache:
                del self._cache[key]
            
            # Add new entry
            self._cache[key] = (value, expiry)
            
            # Evict if over limit
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)  # Remove oldest
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
    
    async def size(self) -> int:
        """Get current cache size."""
        async with self._lock:
            return len(self._cache)
    
    async def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        async with self._lock:
            now = time.time()
            expired_keys = [
                key for key, (_, expiry) in self._cache.items()
                if expiry and expiry < now
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            return len(expired_keys)


def validate_episode(episode: Any) -> None:
    """
    Validate episode data.
    
    Args:
        episode: Episode to validate
        
    Raises:
        ValueError: If episode is invalid
    """
    if episode is None:
        raise ValueError("Episode cannot be None")
    
    if not hasattr(episode, 'input') or not isinstance(episode.input, dict) or not episode.input:
        raise ValueError("Episode input must be a non-empty dictionary")
    
    if not hasattr(episode, 'output') or not isinstance(episode.output, dict) or not episode.output:
        raise ValueError("Episode output must be a non-empty dictionary")
    
    if not hasattr(episode, 'agent_id') or not episode.agent_id:
        raise ValueError("Episode must have a valid agent_id")


def validate_limit(limit: int, max_limit: int = 10000, min_limit: int = 1) -> int:
    """
    Validate and cap limit value.
    
    Args:
        limit: Requested limit
        max_limit: Maximum allowed limit
        min_limit: Minimum allowed limit
        
    Returns:
        Validated limit (capped if necessary)
    """
    if limit < min_limit:
        raise ValueError(f"Limit must be >= {min_limit}")
    
    if limit > max_limit:
        logger.warning(
            f"Requested limit {limit} exceeds max_limit {max_limit}, "
            f"capping to {max_limit}"
        )
        return max_limit
    
    return limit


def validate_content(content: str, max_length: int = 1000000) -> str:
    """
    Validate content string.
    
    Args:
        content: Content to validate
        max_length: Maximum allowed length
        
    Returns:
        Validated content
        
    Raises:
        ValueError: If content is invalid
    """
    if not content or not isinstance(content, str):
        raise ValueError("Content must be a non-empty string")
    
    if len(content) > max_length:
        raise ValueError(f"Content length ({len(content)}) exceeds maximum ({max_length})")
    
    return content.strip()

