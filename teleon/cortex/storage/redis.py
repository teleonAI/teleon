"""
Redis Storage Backend for Cortex.

Production-ready, distributed storage using Redis.
Suitable for multi-instance deployments with persistence and high availability.
"""

import json
import fnmatch
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

try:
    import redis.asyncio as aioredis
    from redis.asyncio import Redis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None
    RedisError = Exception
    RedisConnectionError = Exception

from teleon.cortex.storage.base import (
    StorageBackend,
    StorageConfig,
    StorageError,
    ConnectionError,
    SerializationError,
)


class RedisConfig(StorageConfig):
    """Configuration for Redis storage."""
    
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    key_prefix: str = "teleon:cortex:"
    connection_pool_size: int = 10
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    decode_responses: bool = False  # We handle JSON serialization


class RedisStorage(StorageBackend):
    """
    Redis storage backend for distributed deployments.
    
    Features:
    - Distributed storage across multiple instances
    - Native TTL support
    - High performance
    - Persistence options
    - High availability with Redis Cluster/Sentinel
    - Connection pooling
    - Automatic reconnection
    
    Requirements:
    - redis[asyncio] package
    - Redis server 5.0+
    
    Example:
        ```python
        storage = RedisStorage(RedisConfig(
            host="localhost",
            port=6379,
            db=0,
            password="secret",
            key_prefix="myapp:"
        ))
        await storage.initialize()
        
        # Store with TTL
        await storage.set("session:123", {"user_id": 456}, ttl=3600)
        
        # Retrieve
        session = await storage.get("session:123")
        
        # Batch operations
        items = await storage.get_many(["session:123", "session:456"])
        ```
    """
    
    def __init__(self, config: Optional[RedisConfig] = None):
        """
        Initialize Redis storage.
        
        Args:
            config: Redis configuration
        
        Raises:
            ImportError: If redis package not installed
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis package is required for RedisStorage. "
                "Install with: pip install 'redis[asyncio]'"
            )
        
        super().__init__(config)
        self.redis_config: RedisConfig = config or RedisConfig()
        self._client: Optional[Redis] = None
    
    async def initialize(self) -> None:
        """Initialize Redis connection."""
        await super().initialize()
        
        try:
            self._client = await aioredis.from_url(
                f"redis://{self.redis_config.host}:{self.redis_config.port}/{self.redis_config.db}",
                password=self.redis_config.password,
                encoding="utf-8",
                decode_responses=self.redis_config.decode_responses,
                socket_timeout=self.redis_config.socket_timeout,
                socket_connect_timeout=self.redis_config.socket_connect_timeout,
                retry_on_timeout=self.redis_config.retry_on_timeout,
                max_connections=self.redis_config.connection_pool_size,
            )
            
            # Test connection
            await self._client.ping()
            
        except RedisConnectionError as e:
            raise ConnectionError(
                f"Failed to connect to Redis: {str(e)}",
                operation="initialize"
            )
        except Exception as e:
            raise StorageError(
                f"Failed to initialize Redis storage: {str(e)}",
                operation="initialize"
            )
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get detailed storage statistics from Redis.
        
        Returns:
            Dictionary with storage statistics
        """
        if not self._client:
            return {
                "total_keys": 0,
                "total_operations": 0,
                "get_operations": 0,
                "set_operations": 0,
                "delete_operations": 0,
                "hit_rate": "0%"
            }
        
        try:
            # Get key count from Redis
            pattern = f"{self.redis_config.key_prefix}*"
            keys = await self._client.keys(pattern)
            total_keys = len(keys)
            
            # Get Redis info for additional stats
            info = await self._client.info("stats")
            
            stats = {
                "total_keys": total_keys,
                "redis_uptime_seconds": info.get("uptime_in_seconds", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec", 0),
            }
            
            # Add metrics if available
            if self.metrics:
                stats.update({
                    "total_operations": self.metrics.total_operations,
                    "get_operations": self.metrics.get_operations,
                    "set_operations": self.metrics.set_operations,
                    "delete_operations": self.metrics.delete_operations,
                    "hit_rate": f"{self.metrics.hit_rate():.2f}%",
                })
            
            return stats
            
        except RedisError as e:
            # Return basic stats on error
            return {
                "total_keys": 0,
                "error": str(e),
                "total_operations": self.metrics.total_operations if self.metrics else 0,
            }
    
    async def shutdown(self) -> None:
        """Shutdown Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
        
        await super().shutdown()
    
    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.redis_config.key_prefix}{key}"
    
    def _strip_prefix(self, key: str) -> str:
        """Remove prefix from key."""
        prefix = self.redis_config.key_prefix
        if key.startswith(prefix):
            return key[len(prefix):]
        return key
    
    def _serialize(self, value: Any) -> str:
        """Serialize value to JSON with datetime support."""
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
            return json.dumps(value, default=json_encoder)
        except (TypeError, ValueError) as e:
            raise SerializationError(
                f"Failed to serialize value: {str(e)}",
                operation="serialize"
            )
    
    def _deserialize(self, data: str) -> Any:
        """Deserialize JSON to value."""
        try:
            return json.loads(data)
        except (TypeError, ValueError) as e:
            raise SerializationError(
                f"Failed to deserialize value: {str(e)}",
                operation="deserialize"
            )
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a value in Redis.
        
        Args:
            key: Storage key
            value: Value to store
            ttl: Time-to-live in seconds
            metadata: Optional metadata (stored separately)
        
        Returns:
            True if stored successfully
        """
        if not self._client:
            raise StorageError("Redis client not initialized", operation="set", key=key)
        
        try:
            redis_key = self._make_key(key)
            serialized = self._serialize(value)
            
            # Use default TTL if not specified
            effective_ttl = ttl or self.redis_config.default_ttl
            
            # Store value
            if effective_ttl:
                await self._client.setex(
                    redis_key,
                    effective_ttl,
                    serialized
                )
            else:
                await self._client.set(redis_key, serialized)
            
            # Store metadata if provided
            if metadata:
                metadata_key = f"{redis_key}:metadata"
                metadata_serialized = self._serialize(metadata)
                if effective_ttl:
                    await self._client.setex(
                        metadata_key,
                        effective_ttl,
                        metadata_serialized
                    )
                else:
                    await self._client.set(metadata_key, metadata_serialized)
            
            # Update metrics
            self._update_metrics("set")
            
            return True
            
        except RedisError as e:
            raise StorageError(
                f"Redis error: {str(e)}",
                operation="set",
                key=key
            )
        except Exception as e:
            if isinstance(e, (StorageError, SerializationError)):
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
        Retrieve a value from Redis.
        
        Args:
            key: Storage key
            default: Default value if not found
        
        Returns:
            Stored value or default
        """
        if not self._client:
            raise StorageError("Redis client not initialized", operation="get", key=key)
        
        try:
            redis_key = self._make_key(key)
            data = await self._client.get(redis_key)
            
            if data is None:
                self._update_metrics("get", hit=False)
                return default
            
            # Deserialize value
            value = self._deserialize(data)
            
            self._update_metrics("get", hit=True)
            return value
            
        except RedisError as e:
            raise StorageError(
                f"Redis error: {str(e)}",
                operation="get",
                key=key
            )
        except Exception as e:
            if isinstance(e, (StorageError, SerializationError)):
                raise
            raise StorageError(
                f"Failed to retrieve value: {str(e)}",
                operation="get",
                key=key
            )
    
    async def delete(self, key: str) -> bool:
        """
        Delete a value from Redis.
        
        Args:
            key: Storage key
        
        Returns:
            True if deleted, False if not found
        """
        if not self._client:
            raise StorageError("Redis client not initialized", operation="delete", key=key)
        
        try:
            redis_key = self._make_key(key)
            metadata_key = f"{redis_key}:metadata"
            
            # Delete both value and metadata
            deleted = await self._client.delete(redis_key, metadata_key)
            
            self._update_metrics("delete")
            return deleted > 0
            
        except RedisError as e:
            raise StorageError(
                f"Redis error: {str(e)}",
                operation="delete",
                key=key
            )
    
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in Redis.
        
        Args:
            key: Storage key
        
        Returns:
            True if key exists
        """
        if not self._client:
            raise StorageError("Redis client not initialized", operation="exists", key=key)
        
        try:
            redis_key = self._make_key(key)
            exists = await self._client.exists(redis_key)
            return exists > 0
            
        except RedisError as e:
            raise StorageError(
                f"Redis error: {str(e)}",
                operation="exists",
                key=key
            )
    
    async def list_keys(
        self,
        pattern: str = "*",
        limit: Optional[int] = None
    ) -> List[str]:
        """
        List keys matching a pattern.
        
        Args:
            pattern: Wildcard pattern
            limit: Maximum number of keys
        
        Returns:
            List of matching keys (without prefix)
        """
        if not self._client:
            raise StorageError("Redis client not initialized", operation="list_keys")
        
        try:
            redis_pattern = self._make_key(pattern)
            
            # Use SCAN for better performance
            keys = []
            cursor = 0
            while True:
                cursor, batch = await self._client.scan(
                    cursor,
                    match=redis_pattern,
                    count=100
                )
                
                # Filter out metadata keys
                batch_keys = [
                    self._strip_prefix(k.decode() if isinstance(k, bytes) else k)
                    for k in batch
                    if not k.endswith(b":metadata") and not k.decode().endswith(":metadata")
                ]
                keys.extend(batch_keys)
                
                if cursor == 0:
                    break
                if limit and len(keys) >= limit:
                    keys = keys[:limit]
                    break
            
            return keys
            
        except RedisError as e:
            raise StorageError(
                f"Redis error: {str(e)}",
                operation="list_keys"
            )
    
    async def clear(self, pattern: str = "*") -> int:
        """
        Clear keys matching a pattern.
        
        Args:
            pattern: Wildcard pattern
        
        Returns:
            Number of keys deleted
        """
        if not self._client:
            raise StorageError("Redis client not initialized", operation="clear")
        
        try:
            keys = await self.list_keys(pattern)
            if not keys:
                return 0
            
            # Delete keys in batches
            redis_keys = [self._make_key(k) for k in keys]
            deleted = await self._client.delete(*redis_keys)
            
            # Also delete associated metadata
            metadata_keys = [f"{k}:metadata" for k in redis_keys]
            await self._client.delete(*metadata_keys)
            
            return deleted
            
        except RedisError as e:
            raise StorageError(
                f"Redis error: {str(e)}",
                operation="clear"
            )
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for a key.
        
        Args:
            key: Storage key
        
        Returns:
            Remaining TTL in seconds, None if no TTL
        """
        if not self._client:
            raise StorageError("Redis client not initialized", operation="get_ttl", key=key)
        
        try:
            redis_key = self._make_key(key)
            ttl = await self._client.ttl(redis_key)
            
            # Redis returns -1 if key exists but has no TTL
            # Returns -2 if key doesn't exist
            if ttl == -2:
                return None
            if ttl == -1:
                return None
            
            return max(0, ttl)
            
        except RedisError as e:
            raise StorageError(
                f"Redis error: {str(e)}",
                operation="get_ttl",
                key=key
            )
    
    async def set_ttl(self, key: str, ttl: int) -> bool:
        """
        Set TTL for an existing key.
        
        Args:
            key: Storage key
            ttl: Time-to-live in seconds
        
        Returns:
            True if TTL was set
        """
        if not self._client:
            raise StorageError("Redis client not initialized", operation="set_ttl", key=key)
        
        try:
            redis_key = self._make_key(key)
            success = await self._client.expire(redis_key, ttl)
            
            # Also set TTL for metadata if exists
            metadata_key = f"{redis_key}:metadata"
            if await self._client.exists(metadata_key):
                await self._client.expire(metadata_key, ttl)
            
            return bool(success)
            
        except RedisError as e:
            raise StorageError(
                f"Redis error: {str(e)}",
                operation="set_ttl",
                key=key
            )
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values efficiently using MGET.
        
        Args:
            keys: List of storage keys
        
        Returns:
            Dictionary of key-value pairs
        """
        if not self._client:
            raise StorageError("Redis client not initialized", operation="get_many")
        
        if not keys:
            return {}
        
        try:
            redis_keys = [self._make_key(k) for k in keys]
            values = await self._client.mget(redis_keys)
            
            results = {}
            for key, value in zip(keys, values):
                if value is not None:
                    results[key] = self._deserialize(value)
            
            return results
            
        except RedisError as e:
            raise StorageError(
                f"Redis error: {str(e)}",
                operation="get_many"
            )
    
    async def set_many(
        self,
        items: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> int:
        """
        Set multiple values efficiently using pipeline.
        
        Args:
            items: Dictionary of key-value pairs
            ttl: Time-to-live for all items
        
        Returns:
            Number of items stored
        """
        if not self._client:
            raise StorageError("Redis client not initialized", operation="set_many")
        
        if not items:
            return 0
        
        try:
            pipeline = self._client.pipeline()
            
            effective_ttl = ttl or self.redis_config.default_ttl
            
            for key, value in items.items():
                redis_key = self._make_key(key)
                serialized = self._serialize(value)
                
                if effective_ttl:
                    pipeline.setex(redis_key, effective_ttl, serialized)
                else:
                    pipeline.set(redis_key, serialized)
            
            await pipeline.execute()
            return len(items)
            
        except RedisError as e:
            raise StorageError(
                f"Redis error: {str(e)}",
                operation="set_many"
            )
    
    async def delete_many(self, keys: List[str]) -> int:
        """
        Delete multiple keys efficiently.
        
        Args:
            keys: List of storage keys
        
        Returns:
            Number of keys deleted
        """
        if not self._client:
            raise StorageError("Redis client not initialized", operation="delete_many")
        
        if not keys:
            return 0
        
        try:
            redis_keys = [self._make_key(k) for k in keys]
            deleted = await self._client.delete(*redis_keys)
            return deleted
            
        except RedisError as e:
            raise StorageError(
                f"Redis error: {str(e)}",
                operation="delete_many"
            )

