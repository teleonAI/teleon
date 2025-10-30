"""
PostgreSQL Storage Backend for Cortex.

Relational database storage for complex queries and data integrity.
Ideal for episodic memory and structured data storage.
"""

import json
import fnmatch
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

try:
    import asyncpg
    from asyncpg import Pool, Connection
    from asyncpg.exceptions import PostgresError
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    Pool = None
    Connection = None
    PostgresError = Exception

from teleon.cortex.storage.base import (
    StorageBackend,
    StorageConfig,
    StorageError,
    ConnectionError,
    SerializationError,
)


class PostgresConfig(StorageConfig):
    """Configuration for PostgreSQL storage."""
    
    host: str = "localhost"
    port: int = 5432
    database: str = "teleon"
    user: str = "teleon"
    password: Optional[str] = None
    table_name: str = "cortex_storage"
    pool_size_min: int = 5
    pool_size_max: int = 20
    command_timeout: int = 60


class PostgresStorage(StorageBackend):
    """
    PostgreSQL storage backend for structured data.
    
    Features:
    - ACID compliance
    - Complex queries
    - JSON support (JSONB)
    - Full-text search
    - Transactions
    - Connection pooling
    - Automatic schema creation
    
    Requirements:
    - asyncpg package
    - PostgreSQL 10+
    
    Schema:
        CREATE TABLE cortex_storage (
            key VARCHAR(255) PRIMARY KEY,
            value JSONB NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW(),
            expires_at TIMESTAMP,
            size_bytes INTEGER
        );
    
    Example:
        ```python
        storage = PostgresStorage(PostgresConfig(
            host="localhost",
            database="teleon",
            user="teleon",
            password="secret"
        ))
        await storage.initialize()
        
        # Store complex data
        await storage.set("user:123", {
            "name": "Alice",
            "preferences": {"theme": "dark"},
            "history": [...]
        }, ttl=86400)
        
        # Query with pattern
        user_keys = await storage.list_keys("user:*")
        ```
    """
    
    def __init__(self, config: Optional[PostgresConfig] = None):
        """
        Initialize PostgreSQL storage.
        
        Args:
            config: PostgreSQL configuration
        
        Raises:
            ImportError: If asyncpg package not installed
        """
        if not POSTGRES_AVAILABLE:
            raise ImportError(
                "asyncpg package is required for PostgresStorage. "
                "Install with: pip install asyncpg"
            )
        
        super().__init__(config)
        self.pg_config: PostgresConfig = config or PostgresConfig()
        self._pool: Optional[Pool] = None
    
    async def initialize(self) -> None:
        """Initialize PostgreSQL connection pool and create schema."""
        await super().initialize()
        
        try:
            # Create connection pool
            self._pool = await asyncpg.create_pool(
                host=self.pg_config.host,
                port=self.pg_config.port,
                database=self.pg_config.database,
                user=self.pg_config.user,
                password=self.pg_config.password,
                min_size=self.pg_config.pool_size_min,
                max_size=self.pg_config.pool_size_max,
                command_timeout=self.pg_config.command_timeout,
            )
            
            # Create schema if not exists
            await self._create_schema()
            
            # Create index for expiration cleanup
            await self._create_indexes()
            
        except PostgresError as e:
            raise ConnectionError(
                f"Failed to connect to PostgreSQL: {str(e)}",
                operation="initialize"
            )
        except Exception as e:
            raise StorageError(
                f"Failed to initialize PostgreSQL storage: {str(e)}",
                operation="initialize"
            )
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get detailed storage statistics from PostgreSQL.
        
        Returns:
            Dictionary with storage statistics
        """
        if not self._pool:
            return {
                "total_keys": 0,
                "total_operations": 0,
                "get_operations": 0,
                "set_operations": 0,
                "delete_operations": 0,
                "hit_rate": "0%"
            }
        
        try:
            async with self._pool.acquire() as conn:
                # Get total key count
                total_keys = await conn.fetchval(f"""
                    SELECT COUNT(*) 
                    FROM {self.pg_config.table_name}
                    WHERE expires_at IS NULL OR expires_at > NOW()
                """)
                
                # Get table size
                table_size = await conn.fetchval(f"""
                    SELECT pg_total_relation_size('{self.pg_config.table_name}')
                """)
                
                stats = {
                    "total_keys": total_keys,
                    "table_size_bytes": table_size or 0,
                    "table_name": self.pg_config.table_name,
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
                
        except PostgresError as e:
            # Return basic stats on error
            return {
                "total_keys": 0,
                "error": str(e),
                "total_operations": self.metrics.total_operations if self.metrics else 0,
            }
    
    async def shutdown(self) -> None:
        """Shutdown PostgreSQL connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
        
        await super().shutdown()
    
    async def _create_schema(self) -> None:
        """Create storage table if not exists."""
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.pg_config.table_name} (
                    key VARCHAR(255) PRIMARY KEY,
                    value JSONB NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    expires_at TIMESTAMP,
                    size_bytes INTEGER
                )
            """)
    
    async def _create_indexes(self) -> None:
        """Create indexes for better performance."""
        async with self._pool.acquire() as conn:
            # Index for expiration cleanup
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.pg_config.table_name}_expires_at
                ON {self.pg_config.table_name}(expires_at)
                WHERE expires_at IS NOT NULL
            """)
            
            # Index for pattern matching
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.pg_config.table_name}_key_pattern
                ON {self.pg_config.table_name}(key text_pattern_ops)
            """)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes."""
        try:
            return len(self._json_dumps(value).encode('utf-8'))
        except:
            return len(str(value).encode('utf-8'))
    
    def _json_encoder(self, obj):
        """Custom JSON encoder for datetime and Pydantic models."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, BaseModel):
            return obj.model_dump()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    
    def _json_dumps(self, value: Any) -> str:
        """JSON dumps with datetime support."""
        return json.dumps(value, default=self._json_encoder)
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a value in PostgreSQL.
        
        Args:
            key: Storage key
            value: Value to store
            ttl: Time-to-live in seconds
            metadata: Optional metadata
        
        Returns:
            True if stored successfully
        """
        if not self._pool:
            raise StorageError("PostgreSQL pool not initialized", operation="set", key=key)
        
        try:
            # Calculate expiration
            expires_at = None
            effective_ttl = ttl or self.config.default_ttl
            if effective_ttl:
                expires_at = datetime.utcnow() + timedelta(seconds=effective_ttl)
            
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Convert to JSONB with datetime support
            value_json = self._json_dumps(value)
            metadata_json = self._json_dumps(metadata) if metadata else None
            
            async with self._pool.acquire() as conn:
                await conn.execute(f"""
                    INSERT INTO {self.pg_config.table_name}
                    (key, value, metadata, expires_at, size_bytes)
                    VALUES ($1, $2::jsonb, $3::jsonb, $4, $5)
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        metadata = EXCLUDED.metadata,
                        expires_at = EXCLUDED.expires_at,
                        size_bytes = EXCLUDED.size_bytes,
                        created_at = NOW()
                """, key, value_json, metadata_json, expires_at, size_bytes)
            
            # Update metrics
            self._update_metrics("set")
            
            return True
            
        except PostgresError as e:
            raise StorageError(
                f"PostgreSQL error: {str(e)}",
                operation="set",
                key=key
            )
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
        Retrieve a value from PostgreSQL.
        
        Args:
            key: Storage key
            default: Default value if not found
        
        Returns:
            Stored value or default
        """
        if not self._pool:
            raise StorageError("PostgreSQL pool not initialized", operation="get", key=key)
        
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(f"""
                    SELECT value, expires_at
                    FROM {self.pg_config.table_name}
                    WHERE key = $1
                """, key)
                
                if row is None:
                    self._update_metrics("get", hit=False)
                    return default
                
                # Check if expired
                if row['expires_at'] and datetime.utcnow() > row['expires_at']:
                    # Delete expired entry
                    await conn.execute(f"""
                        DELETE FROM {self.pg_config.table_name}
                        WHERE key = $1
                    """, key)
                    self._update_metrics("get", hit=False)
                    return default
                
                # Parse JSONB
                value = json.loads(row['value']) if isinstance(row['value'], str) else row['value']
                
                self._update_metrics("get", hit=True)
                return value
                
        except PostgresError as e:
            raise StorageError(
                f"PostgreSQL error: {str(e)}",
                operation="get",
                key=key
            )
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            raise StorageError(
                f"Failed to retrieve value: {str(e)}",
                operation="get",
                key=key
            )
    
    async def delete(self, key: str) -> bool:
        """
        Delete a value from PostgreSQL.
        
        Args:
            key: Storage key
        
        Returns:
            True if deleted, False if not found
        """
        if not self._pool:
            raise StorageError("PostgreSQL pool not initialized", operation="delete", key=key)
        
        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(f"""
                    DELETE FROM {self.pg_config.table_name}
                    WHERE key = $1
                """, key)
                
                # Result format: "DELETE N" where N is number of rows
                deleted = int(result.split()[-1]) if result else 0
                
                self._update_metrics("delete")
                return deleted > 0
                
        except PostgresError as e:
            raise StorageError(
                f"PostgreSQL error: {str(e)}",
                operation="delete",
                key=key
            )
    
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in PostgreSQL.
        
        Args:
            key: Storage key
        
        Returns:
            True if key exists and not expired
        """
        if not self._pool:
            raise StorageError("PostgreSQL pool not initialized", operation="exists", key=key)
        
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(f"""
                    SELECT expires_at
                    FROM {self.pg_config.table_name}
                    WHERE key = $1
                """, key)
                
                if row is None:
                    return False
                
                # Check if expired
                if row['expires_at'] and datetime.utcnow() > row['expires_at']:
                    return False
                
                return True
                
        except PostgresError as e:
            raise StorageError(
                f"PostgreSQL error: {str(e)}",
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
            List of matching keys
        """
        if not self._pool:
            raise StorageError("PostgreSQL pool not initialized", operation="list_keys")
        
        try:
            # Convert wildcard pattern to SQL LIKE pattern
            sql_pattern = pattern.replace('*', '%').replace('?', '_')
            
            async with self._pool.acquire() as conn:
                # Remove expired entries first
                await self._cleanup_expired(conn)
                
                query = f"""
                    SELECT key
                    FROM {self.pg_config.table_name}
                    WHERE key LIKE $1
                    AND (expires_at IS NULL OR expires_at > NOW())
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                rows = await conn.fetch(query, sql_pattern)
                return [row['key'] for row in rows]
                
        except PostgresError as e:
            raise StorageError(
                f"PostgreSQL error: {str(e)}",
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
        if not self._pool:
            raise StorageError("PostgreSQL pool not initialized", operation="clear")
        
        try:
            sql_pattern = pattern.replace('*', '%').replace('?', '_')
            
            async with self._pool.acquire() as conn:
                result = await conn.execute(f"""
                    DELETE FROM {self.pg_config.table_name}
                    WHERE key LIKE $1
                """, sql_pattern)
                
                deleted = int(result.split()[-1]) if result else 0
                return deleted
                
        except PostgresError as e:
            raise StorageError(
                f"PostgreSQL error: {str(e)}",
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
        if not self._pool:
            raise StorageError("PostgreSQL pool not initialized", operation="get_ttl", key=key)
        
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(f"""
                    SELECT expires_at
                    FROM {self.pg_config.table_name}
                    WHERE key = $1
                """, key)
                
                if row is None or row['expires_at'] is None:
                    return None
                
                remaining = (row['expires_at'] - datetime.utcnow()).total_seconds()
                return max(0, int(remaining))
                
        except PostgresError as e:
            raise StorageError(
                f"PostgreSQL error: {str(e)}",
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
        if not self._pool:
            raise StorageError("PostgreSQL pool not initialized", operation="set_ttl", key=key)
        
        try:
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            
            async with self._pool.acquire() as conn:
                result = await conn.execute(f"""
                    UPDATE {self.pg_config.table_name}
                    SET expires_at = $1
                    WHERE key = $2
                """, expires_at, key)
                
                updated = int(result.split()[-1]) if result else 0
                return updated > 0
                
        except PostgresError as e:
            raise StorageError(
                f"PostgreSQL error: {str(e)}",
                operation="set_ttl",
                key=key
            )
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values efficiently.
        
        Args:
            keys: List of storage keys
        
        Returns:
            Dictionary of key-value pairs
        """
        if not self._pool:
            raise StorageError("PostgreSQL pool not initialized", operation="get_many")
        
        if not keys:
            return {}
        
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(f"""
                    SELECT key, value
                    FROM {self.pg_config.table_name}
                    WHERE key = ANY($1)
                    AND (expires_at IS NULL OR expires_at > NOW())
                """, keys)
                
                results = {}
                for row in rows:
                    value = json.loads(row['value']) if isinstance(row['value'], str) else row['value']
                    results[row['key']] = value
                
                return results
                
        except PostgresError as e:
            raise StorageError(
                f"PostgreSQL error: {str(e)}",
                operation="get_many"
            )
    
    async def set_many(
        self,
        items: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> int:
        """
        Set multiple values efficiently using transaction.
        
        Args:
            items: Dictionary of key-value pairs
            ttl: Time-to-live for all items
        
        Returns:
            Number of items stored
        """
        if not self._pool:
            raise StorageError("PostgreSQL pool not initialized", operation="set_many")
        
        if not items:
            return 0
        
        try:
            effective_ttl = ttl or self.config.default_ttl
            expires_at = None
            if effective_ttl:
                expires_at = datetime.utcnow() + timedelta(seconds=effective_ttl)
            
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    for key, value in items.items():
                        value_json = self._json_dumps(value)
                        size_bytes = self._calculate_size(value)
                        
                        await conn.execute(f"""
                            INSERT INTO {self.pg_config.table_name}
                            (key, value, expires_at, size_bytes)
                            VALUES ($1, $2::jsonb, $3, $4)
                            ON CONFLICT (key) DO UPDATE SET
                                value = EXCLUDED.value,
                                expires_at = EXCLUDED.expires_at,
                                size_bytes = EXCLUDED.size_bytes,
                                created_at = NOW()
                        """, key, value_json, expires_at, size_bytes)
            
            return len(items)
            
        except PostgresError as e:
            raise StorageError(
                f"PostgreSQL error: {str(e)}",
                operation="set_many"
            )
    
    async def _cleanup_expired(self, conn: Connection) -> int:
        """Remove expired entries."""
        result = await conn.execute(f"""
            DELETE FROM {self.pg_config.table_name}
            WHERE expires_at IS NOT NULL AND expires_at < NOW()
        """)
        deleted = int(result.split()[-1]) if result else 0
        return deleted
    
    async def cleanup_expired(self) -> int:
        """
        Public method to manually cleanup expired entries.
        
        Returns:
            Number of entries deleted
        """
        if not self._pool:
            return 0
        
        async with self._pool.acquire() as conn:
            return await self._cleanup_expired(conn)

