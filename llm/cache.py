"""Response caching for LLM Gateway."""

from abc import ABC, abstractmethod
from typing import Optional, Dict
import time
import asyncio

from teleon.llm.types import LLMResponse


class ResponseCache(ABC):
    """Abstract base class for response caches."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[LLMResponse]:
        """
        Get a cached response.
        
        Args:
            key: Cache key
        
        Returns:
            Cached response or None
        """
        pass
    
    @abstractmethod
    async def set(
        self,
        key: str,
        response: LLMResponse,
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache a response.
        
        Args:
            key: Cache key
            response: Response to cache
            ttl: Time-to-live in seconds
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """
        Delete a cached response.
        
        Args:
            key: Cache key
        """
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached responses."""
        pass


class InMemoryCache(ResponseCache):
    """
    In-memory response cache.
    
    Simple implementation using a dictionary.
    For production, consider using Redis or similar.
    """
    
    def __init__(self, default_ttl: int = 3600, max_size: int = 1000):
        """
        Initialize the cache.
        
        Args:
            default_ttl: Default TTL in seconds
            max_size: Maximum number of cached items
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: Dict[str, tuple] = {}  # key -> (response, expiry_time)
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[LLMResponse]:
        """Get a cached response."""
        async with self._lock:
            if key not in self._cache:
                return None
            
            response, expiry_time = self._cache[key]
            
            # Check if expired
            if time.time() > expiry_time:
                del self._cache[key]
                return None
            
            return response
    
    async def set(
        self,
        key: str,
        response: LLMResponse,
        ttl: Optional[int] = None
    ) -> None:
        """Cache a response."""
        async with self._lock:
            # Enforce max size (simple LRU: remove oldest)
            if len(self._cache) >= self.max_size:
                # Remove the oldest entry
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k][1]
                )
                del self._cache[oldest_key]
            
            # Calculate expiry time
            ttl = ttl or self.default_ttl
            expiry_time = time.time() + ttl
            
            # Store response
            self._cache[key] = (response, expiry_time)
    
    async def delete(self, key: str) -> None:
        """Delete a cached response."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
    
    async def clear(self) -> None:
        """Clear all cached responses."""
        async with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get cache statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "default_ttl": self.default_ttl
        }


class RedisCache(ResponseCache):
    """
    Redis-based response cache.
    
    For production use with distributed systems.
    """
    
    def __init__(
        self,
        redis_url: str,
        default_ttl: int = 3600,
        key_prefix: str = "teleon:llm:"
    ):
        """
        Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL
            default_ttl: Default TTL in seconds
            key_prefix: Prefix for cache keys
        """
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self._redis = None
    
    async def _get_redis(self):
        """Get or create Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = await redis.from_url(self.redis_url)
            except ImportError:
                raise ImportError(
                    "Redis package not installed. Install it with: pip install redis"
                )
        return self._redis
    
    async def get(self, key: str) -> Optional[LLMResponse]:
        """Get a cached response from Redis."""
        redis = await self._get_redis()
        
        full_key = f"{self.key_prefix}{key}"
        data = await redis.get(full_key)
        
        if data is None:
            return None
        
        # Deserialize response
        import json
        response_dict = json.loads(data)
        return LLMResponse(**response_dict)
    
    async def set(
        self,
        key: str,
        response: LLMResponse,
        ttl: Optional[int] = None
    ) -> None:
        """Cache a response in Redis."""
        redis = await self._get_redis()
        
        full_key = f"{self.key_prefix}{key}"
        ttl = ttl or self.default_ttl
        
        # Serialize response
        import json
        data = json.dumps(response.dict())
        
        await redis.setex(full_key, ttl, data)
    
    async def delete(self, key: str) -> None:
        """Delete a cached response from Redis."""
        redis = await self._get_redis()
        
        full_key = f"{self.key_prefix}{key}"
        await redis.delete(full_key)
    
    async def clear(self) -> None:
        """Clear all cached responses with the key prefix."""
        redis = await self._get_redis()
        
        # Find all keys with our prefix
        pattern = f"{self.key_prefix}*"
        keys = []
        async for key in redis.scan_iter(match=pattern):
            keys.append(key)
        
        # Delete all keys
        if keys:
            await redis.delete(*keys)

