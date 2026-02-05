"""
Redis storage backend with RediSearch for vector similarity.
"""

from typing import Optional, Any, List, Dict
from datetime import datetime, timezone
import json
import logging
import uuid
import numpy as np

from teleon.cortex.storage.base import StorageBackend, StorageError
from teleon.cortex.entry import Entry
from teleon.cortex.embeddings.base import EMBEDDING_DIMENSION

logger = logging.getLogger("teleon.cortex.storage.redis")

try:
    import redis.asyncio as aioredis
    from redis.asyncio import Redis
    from redis.commands.search.field import VectorField, TextField, TagField, NumericField
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None
    Redis = None


class RedisBackend(StorageBackend):
    """
    Redis storage backend with RediSearch for vector similarity search.

    Requires:
        - redis[asyncio]: pip install redis[asyncio]
        - Redis Stack (includes RediSearch) or RediSearch module

    Example:
        backend = RedisBackend(
            host="localhost",
            port=6379,
            password="secret"
        )
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "teleon:memory:",
        connection_string: Optional[str] = None
    ):
        """
        Initialize Redis backend.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            key_prefix: Prefix for all keys
            connection_string: Full connection string (overrides other params)
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis not available. Install with: pip install redis[asyncio]"
            )

        self._host = host
        self._port = port
        self._db = db
        self._password = password
        self._key_prefix = key_prefix
        self._connection_string = connection_string

        self._client: Optional[Redis] = None
        self._indexes_created: set = set()

        logger.info(f"RedisBackend initialized for {host}:{port}")

    async def _get_client(self) -> Redis:
        """Get or create Redis client."""
        if self._client is None:
            if self._connection_string:
                self._client = await aioredis.from_url(
                    self._connection_string,
                    decode_responses=False
                )
            else:
                self._client = await aioredis.from_url(
                    f"redis://{self._host}:{self._port}/{self._db}",
                    password=self._password,
                    decode_responses=False
                )

            # Test connection
            await self._client.ping()

        return self._client

    async def _ensure_index(self, memory_name: str) -> None:
        """Ensure RediSearch index exists for memory."""
        if memory_name in self._indexes_created:
            return

        client = await self._get_client()
        index_name = f"{self._key_prefix}{memory_name}:idx"

        try:
            # Check if index exists
            await client.ft(index_name).info()
            self._indexes_created.add(memory_name)
            return
        except Exception:
            pass  # Index doesn't exist, create it

        try:
            # Define schema
            schema = [
                TextField("content"),
                TagField("memory_name"),
                NumericField("created_at", sortable=True),
                NumericField("expires_at"),
                TextField("fields_json"),
                VectorField(
                    "embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": EMBEDDING_DIMENSION,
                        "DISTANCE_METRIC": "COSINE"
                    }
                )
            ]

            # Create index
            await client.ft(index_name).create_index(
                fields=schema,
                definition=IndexDefinition(
                    prefix=[f"{self._key_prefix}{memory_name}:"],
                    index_type=IndexType.HASH
                )
            )

            self._indexes_created.add(memory_name)
            logger.info(f"Created RediSearch index for {memory_name}")

        except Exception as e:
            logger.warning(f"Could not create RediSearch index: {e}")

    def _make_key(self, memory_name: str, entry_id: str) -> str:
        """Generate Redis key for entry."""
        return f"{self._key_prefix}{memory_name}:{entry_id}"

    def _vector_to_bytes(self, embedding: List[float]) -> bytes:
        """Convert embedding to bytes for Redis."""
        return np.array(embedding, dtype=np.float32).tobytes()

    def _bytes_to_vector(self, data: bytes) -> List[float]:
        """Convert bytes back to embedding."""
        return np.frombuffer(data, dtype=np.float32).tolist()

    async def store(
        self,
        memory_name: str,
        content: str,
        embedding: List[float],
        fields: Dict[str, Any],
        expires_at: Optional[datetime] = None,
        upsert: bool = False
    ) -> str:
        """Store entry in Redis."""
        client = await self._get_client()
        await self._ensure_index(memory_name)

        if upsert and fields:
            # Try to find existing entry with matching fields
            existing = await self._find_by_fields(memory_name, fields)
            if existing:
                entry_id = existing.id
                key = self._make_key(memory_name, entry_id)

                # Update existing entry
                now = datetime.now(timezone.utc)
                mapping = {
                    "content": content,
                    "embedding": self._vector_to_bytes(embedding),
                    "fields_json": json.dumps(fields),
                    "updated_at": now.timestamp(),
                }
                if expires_at:
                    mapping["expires_at"] = expires_at.timestamp()

                await client.hset(key, mapping=mapping)

                # Update TTL if needed
                if expires_at:
                    ttl = int((expires_at - now).total_seconds())
                    if ttl > 0:
                        await client.expire(key, ttl)

                logger.debug(f"Updated entry {entry_id}")
                return entry_id

        # Create new entry
        entry_id = str(uuid.uuid4())
        key = self._make_key(memory_name, entry_id)
        now = datetime.now(timezone.utc)

        mapping = {
            "id": entry_id,
            "memory_name": memory_name,
            "content": content,
            "embedding": self._vector_to_bytes(embedding),
            "fields_json": json.dumps(fields),
            "created_at": now.timestamp(),
            "updated_at": now.timestamp(),
        }

        if expires_at:
            mapping["expires_at"] = expires_at.timestamp()

        await client.hset(key, mapping=mapping)

        # Set TTL if expires_at is set
        if expires_at:
            ttl = int((expires_at - now).total_seconds())
            if ttl > 0:
                await client.expire(key, ttl)

        logger.debug(f"Stored entry {entry_id}")
        return entry_id

    async def _find_by_fields(
        self,
        memory_name: str,
        fields: Dict[str, Any]
    ) -> Optional[Entry]:
        """Find entry by exact field match."""
        # Use get() with filter to find matching entry
        entries = await self.get(memory_name, fields, limit=1)
        return entries[0] if entries else None

    async def search(
        self,
        memory_name: str,
        query_embedding: Optional[List[float]],
        filter: Dict[str, Any],
        limit: int = 10
    ) -> List[Entry]:
        """Semantic search with RediSearch."""
        client = await self._get_client()
        await self._ensure_index(memory_name)

        index_name = f"{self._key_prefix}{memory_name}:idx"

        try:
            if query_embedding:
                # Vector similarity search
                query_vector = self._vector_to_bytes(query_embedding)

                # Build query with filter
                query_str = f"@memory_name:{{{memory_name}}}"

                # Add field filters
                for key, value in filter.items():
                    if isinstance(value, list):
                        values = "|".join(str(v) for v in value)
                        query_str += f" @fields_json:*{values}*"
                    else:
                        query_str += f" @fields_json:*{value}*"

                query_str += f"=>[KNN {limit} @embedding $vec AS score]"

                query = (
                    Query(query_str)
                    .sort_by("score")
                    .return_fields("id", "content", "fields_json", "created_at", "updated_at", "expires_at", "score")
                    .dialect(2)
                )

                results = await client.ft(index_name).search(
                    query,
                    query_params={"vec": query_vector}
                )

                entries = []
                for doc in results.docs:
                    entry = self._doc_to_entry(doc, with_score=True)
                    if entry:
                        entries.append(entry)

                return entries

            else:
                # Filter only, get by pattern and filter in memory
                return await self.get(memory_name, filter, limit)

        except Exception as e:
            logger.warning(f"RediSearch query failed, falling back to scan: {e}")
            return await self.get(memory_name, filter, limit)

    async def get(
        self,
        memory_name: str,
        filter: Dict[str, Any],
        limit: int = 10
    ) -> List[Entry]:
        """Get entries by filter using SCAN."""
        client = await self._get_client()

        pattern = f"{self._key_prefix}{memory_name}:*"
        entries = []
        now = datetime.now(timezone.utc).timestamp()

        cursor = 0
        while True:
            cursor, keys = await client.scan(cursor, match=pattern, count=100)

            for key in keys:
                # Skip index keys
                if key.endswith(b":idx") or b":idx:" in key:
                    continue

                data = await client.hgetall(key)
                if not data:
                    continue

                # Check expiration
                expires_at = data.get(b"expires_at")
                if expires_at and float(expires_at) <= now:
                    continue

                # Parse fields and check filter
                fields_json = data.get(b"fields_json", b"{}")
                try:
                    fields = json.loads(fields_json)
                except:
                    fields = {}

                # Check filter match
                if not self._matches_filter(fields, filter):
                    continue

                entry = self._hash_to_entry(data)
                if entry:
                    entries.append(entry)

                if len(entries) >= limit:
                    break

            if cursor == 0 or len(entries) >= limit:
                break

        # Sort by created_at DESC
        entries.sort(key=lambda e: e.created_at.timestamp(), reverse=True)

        return entries[:limit]

    def _matches_filter(self, fields: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """Check if fields match filter."""
        for key, value in filter.items():
            field_value = fields.get(key)
            if isinstance(value, list):
                if field_value not in value:
                    return False
            else:
                if str(field_value) != str(value):
                    return False
        return True

    async def update(
        self,
        memory_name: str,
        filter: Dict[str, Any],
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        fields: Optional[Dict[str, Any]] = None
    ) -> int:
        """Update entries matching filter."""
        if not filter:
            logger.warning("Update called without filter - skipping for safety")
            return 0

        client = await self._get_client()

        # Find matching entries
        entries = await self.get(memory_name, filter, limit=1000)

        count = 0
        for entry in entries:
            key = self._make_key(memory_name, entry.id)

            mapping = {
                "updated_at": datetime.now(timezone.utc).timestamp()
            }

            if content is not None:
                mapping["content"] = content

            if embedding is not None:
                mapping["embedding"] = self._vector_to_bytes(embedding)

            if fields is not None:
                # Merge with existing fields
                existing_fields = entry.fields.copy()
                existing_fields.update(fields)
                mapping["fields_json"] = json.dumps(existing_fields)

            await client.hset(key, mapping=mapping)
            count += 1

        logger.debug(f"Updated {count} entries")
        return count

    async def delete(
        self,
        memory_name: str,
        filter: Dict[str, Any]
    ) -> int:
        """Delete entries matching filter."""
        if not filter:
            logger.warning("Delete called without filter - skipping for safety")
            return 0

        client = await self._get_client()

        # Find matching entries
        entries = await self.get(memory_name, filter, limit=10000)

        count = 0
        for entry in entries:
            key = self._make_key(memory_name, entry.id)
            result = await client.delete(key)
            if result:
                count += 1

        logger.debug(f"Deleted {count} entries")
        return count

    async def count(
        self,
        memory_name: str,
        filter: Dict[str, Any]
    ) -> int:
        """Count entries matching filter."""
        entries = await self.get(memory_name, filter, limit=100000)
        return len(entries)

    async def cleanup_expired(self) -> int:
        """Remove expired entries (Redis handles this with TTL)."""
        # Redis automatically removes expired keys
        return 0

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Redis connection closed")

    def _hash_to_entry(self, data: Dict[bytes, bytes]) -> Optional[Entry]:
        """Convert Redis hash to Entry."""
        try:
            entry_id = data.get(b"id", b"").decode()
            if not entry_id:
                return None

            content = data.get(b"content", b"").decode()

            fields_json = data.get(b"fields_json", b"{}")
            try:
                fields = json.loads(fields_json)
            except:
                fields = {}

            created_at = float(data.get(b"created_at", 0))
            updated_at = float(data.get(b"updated_at", created_at))

            expires_at = data.get(b"expires_at")
            expires_dt = None
            if expires_at:
                expires_dt = datetime.fromtimestamp(float(expires_at), tz=timezone.utc)

            return Entry(
                id=entry_id,
                content=content,
                fields=fields,
                created_at=datetime.fromtimestamp(created_at, tz=timezone.utc),
                updated_at=datetime.fromtimestamp(updated_at, tz=timezone.utc),
                expires_at=expires_dt,
                score=None
            )
        except Exception as e:
            logger.warning(f"Failed to parse entry: {e}")
            return None

    def _doc_to_entry(self, doc: Any, with_score: bool = False) -> Optional[Entry]:
        """Convert RediSearch document to Entry."""
        try:
            entry_id = getattr(doc, "id", "").replace(f"{self._key_prefix}", "").split(":")[-1]

            content = getattr(doc, "content", "")
            if isinstance(content, bytes):
                content = content.decode()

            fields_json = getattr(doc, "fields_json", "{}")
            if isinstance(fields_json, bytes):
                fields_json = fields_json.decode()
            try:
                fields = json.loads(fields_json)
            except:
                fields = {}

            created_at = float(getattr(doc, "created_at", 0))
            updated_at = float(getattr(doc, "updated_at", created_at))

            expires_at = getattr(doc, "expires_at", None)
            expires_dt = None
            if expires_at:
                expires_dt = datetime.fromtimestamp(float(expires_at), tz=timezone.utc)

            score = None
            if with_score:
                score_raw = getattr(doc, "score", None)
                if score_raw is not None:
                    # RediSearch returns distance, convert to similarity
                    score = 1 - float(score_raw)

            return Entry(
                id=entry_id,
                content=content,
                fields=fields,
                created_at=datetime.fromtimestamp(created_at, tz=timezone.utc),
                updated_at=datetime.fromtimestamp(updated_at, tz=timezone.utc),
                expires_at=expires_dt,
                score=score
            )
        except Exception as e:
            logger.warning(f"Failed to parse document: {e}")
            return None
