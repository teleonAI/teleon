"""
PostgreSQL storage backend with pgvector support.
"""

from typing import Optional, Any, List, Dict
from datetime import datetime, timezone
import json
import logging
import uuid

from teleon.cortex.storage.base import StorageBackend, StorageError
from teleon.cortex.entry import Entry
from teleon.cortex.embeddings.base import EMBEDDING_DIMENSION

logger = logging.getLogger("teleon.cortex.storage.postgres")

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    asyncpg = None


class PostgresBackend(StorageBackend):
    """
    PostgreSQL storage backend with pgvector for similarity search.

    Requires:
        - asyncpg: pip install asyncpg
        - pgvector extension in PostgreSQL

    Example:
        backend = PostgresBackend(
            host="localhost",
            port=5432,
            database="teleon",
            user="postgres",
            password="secret"
        )
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "teleon",
        user: str = "postgres",
        password: Optional[str] = None,
        min_connections: int = 2,
        max_connections: int = 10,
        connection_string: Optional[str] = None
    ):
        """
        Initialize PostgreSQL backend.

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            min_connections: Minimum pool connections
            max_connections: Maximum pool connections
            connection_string: Full connection string (overrides other params)
        """
        if not ASYNCPG_AVAILABLE:
            raise ImportError(
                "asyncpg not available. Install with: pip install asyncpg"
            )

        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._min_connections = min_connections
        self._max_connections = max_connections
        self._connection_string = connection_string

        self._pool: Optional[asyncpg.Pool] = None

        logger.info(f"PostgresBackend initialized for {host}:{port}/{database}")

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool."""
        if self._pool is None:
            if self._connection_string:
                self._pool = await asyncpg.create_pool(
                    self._connection_string,
                    min_size=self._min_connections,
                    max_size=self._max_connections
                )
            else:
                self._pool = await asyncpg.create_pool(
                    host=self._host,
                    port=self._port,
                    database=self._database,
                    user=self._user,
                    password=self._password,
                    min_size=self._min_connections,
                    max_size=self._max_connections
                )

            # Initialize schema
            await self._init_schema()

        return self._pool

    async def _init_schema(self) -> None:
        """Initialize database schema."""
        pool = self._pool
        if pool is None:
            return

        async with pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create memories table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS memories (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    memory_name VARCHAR(255) NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({EMBEDDING_DIMENSION}),
                    fields JSONB DEFAULT '{{}}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    expires_at TIMESTAMPTZ
                );
            """)

            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_memory_name
                ON memories(memory_name);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_created_at
                ON memories(created_at DESC);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_expires_at
                ON memories(expires_at)
                WHERE expires_at IS NOT NULL;
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_fields
                ON memories USING GIN(fields);
            """)

            # Create vector index for similarity search (IVFFlat)
            # Check if index exists first to avoid errors
            exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE indexname = 'idx_memories_embedding'
                );
            """)
            if not exists:
                await conn.execute(f"""
                    CREATE INDEX idx_memories_embedding
                    ON memories USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)

        logger.info("PostgreSQL schema initialized")

    def _build_filter_clause(
        self,
        filter: Dict[str, Any],
        param_offset: int = 1
    ) -> tuple[str, List[Any]]:
        """Build WHERE clause from filter dict."""
        if not filter:
            return "", []

        conditions = []
        params = []

        for key, value in filter.items():
            param_idx = param_offset + len(params)
            if isinstance(value, list):
                # Array contains - check if field contains any of the values
                placeholders = ", ".join(f"${param_idx + i}" for i in range(len(value)))
                conditions.append(f"fields->>'{key}' IN ({placeholders})")
                params.extend([str(v) for v in value])
            else:
                # Exact match
                conditions.append(f"fields->>'{key}' = ${param_idx}")
                params.append(str(value))

        return " AND ".join(conditions), params

    async def store(
        self,
        memory_name: str,
        content: str,
        embedding: List[float],
        fields: Dict[str, Any],
        expires_at: Optional[datetime] = None,
        upsert: bool = False
    ) -> str:
        """Store entry in PostgreSQL."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            if upsert and fields:
                # Try to find existing entry with matching fields
                filter_clause, filter_params = self._build_filter_clause(fields, param_offset=2)

                if filter_clause:
                    query = f"""
                        SELECT id FROM memories
                        WHERE memory_name = $1 AND {filter_clause}
                        LIMIT 1
                    """
                    existing = await conn.fetchrow(query, memory_name, *filter_params)

                    if existing:
                        # Update existing entry
                        entry_id = str(existing['id'])
                        await conn.execute(
                            """
                            UPDATE memories
                            SET content = $1, embedding = $2, fields = $3,
                                updated_at = NOW(), expires_at = $4
                            WHERE id = $5
                            """,
                            content,
                            str(embedding),
                            json.dumps(fields),
                            expires_at,
                            existing['id']
                        )
                        logger.debug(f"Updated entry {entry_id}")
                        return entry_id

            # Insert new entry
            entry_id = str(uuid.uuid4())
            await conn.execute(
                """
                INSERT INTO memories (id, memory_name, content, embedding, fields, expires_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                uuid.UUID(entry_id),
                memory_name,
                content,
                str(embedding),
                json.dumps(fields),
                expires_at
            )

            logger.debug(f"Stored entry {entry_id}")
            return entry_id

    async def search(
        self,
        memory_name: str,
        query_embedding: Optional[List[float]],
        filter: Dict[str, Any],
        limit: int = 10
    ) -> List[Entry]:
        """Semantic search with pgvector."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            filter_clause, filter_params = self._build_filter_clause(filter, param_offset=3)

            if query_embedding:
                # Semantic search with cosine similarity
                base_query = f"""
                    SELECT id, content, fields, created_at, updated_at, expires_at,
                           1 - (embedding <=> $1::vector) as score
                    FROM memories
                    WHERE memory_name = $2
                    AND (expires_at IS NULL OR expires_at > NOW())
                """

                if filter_clause:
                    base_query += f" AND {filter_clause}"

                base_query += " ORDER BY embedding <=> $1::vector LIMIT $" + str(3 + len(filter_params))

                rows = await conn.fetch(
                    base_query,
                    str(query_embedding),
                    memory_name,
                    *filter_params,
                    limit
                )
            else:
                # Filter only, order by created_at
                base_query = """
                    SELECT id, content, fields, created_at, updated_at, expires_at,
                           NULL as score
                    FROM memories
                    WHERE memory_name = $1
                    AND (expires_at IS NULL OR expires_at > NOW())
                """

                if filter_clause:
                    base_query += f" AND {filter_clause}"

                base_query += " ORDER BY created_at DESC LIMIT $" + str(2 + len(filter_params))

                rows = await conn.fetch(
                    base_query,
                    memory_name,
                    *filter_params,
                    limit
                )

            return [self._row_to_entry(row) for row in rows]

    async def get(
        self,
        memory_name: str,
        filter: Dict[str, Any],
        limit: int = 10
    ) -> List[Entry]:
        """Get entries by filter."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            filter_clause, filter_params = self._build_filter_clause(filter, param_offset=2)

            base_query = """
                SELECT id, content, fields, created_at, updated_at, expires_at,
                       NULL as score
                FROM memories
                WHERE memory_name = $1
                AND (expires_at IS NULL OR expires_at > NOW())
            """

            if filter_clause:
                base_query += f" AND {filter_clause}"

            base_query += " ORDER BY created_at DESC LIMIT $" + str(2 + len(filter_params))

            rows = await conn.fetch(
                base_query,
                memory_name,
                *filter_params,
                limit
            )

            return [self._row_to_entry(row) for row in rows]

    async def update(
        self,
        memory_name: str,
        filter: Dict[str, Any],
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        fields: Optional[Dict[str, Any]] = None
    ) -> int:
        """Update entries matching filter."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            filter_clause, filter_params = self._build_filter_clause(filter, param_offset=1)

            if not filter_clause:
                logger.warning("Update called without filter - skipping for safety")
                return 0

            # Build SET clause
            set_parts = ["updated_at = NOW()"]
            set_params = []
            param_idx = 1 + len(filter_params)

            if content is not None:
                set_parts.append(f"content = ${param_idx}")
                set_params.append(content)
                param_idx += 1

            if embedding is not None:
                set_parts.append(f"embedding = ${param_idx}::vector")
                set_params.append(str(embedding))
                param_idx += 1

            if fields is not None:
                set_parts.append(f"fields = fields || ${param_idx}::jsonb")
                set_params.append(json.dumps(fields))
                param_idx += 1

            query = f"""
                UPDATE memories
                SET {', '.join(set_parts)}
                WHERE memory_name = ${param_idx} AND {filter_clause}
            """

            result = await conn.execute(
                query,
                *set_params,
                memory_name,
                *filter_params
            )

            # Extract count from result
            count = int(result.split()[-1])
            logger.debug(f"Updated {count} entries")
            return count

    async def delete(
        self,
        memory_name: str,
        filter: Dict[str, Any]
    ) -> int:
        """Delete entries matching filter."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            filter_clause, filter_params = self._build_filter_clause(filter, param_offset=2)

            if not filter_clause:
                logger.warning("Delete called without filter - skipping for safety")
                return 0

            query = f"""
                DELETE FROM memories
                WHERE memory_name = $1 AND {filter_clause}
            """

            result = await conn.execute(query, memory_name, *filter_params)

            # Extract count from result
            count = int(result.split()[-1])
            logger.debug(f"Deleted {count} entries")
            return count

    async def count(
        self,
        memory_name: str,
        filter: Dict[str, Any]
    ) -> int:
        """Count entries matching filter."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            filter_clause, filter_params = self._build_filter_clause(filter, param_offset=2)

            base_query = """
                SELECT COUNT(*) FROM memories
                WHERE memory_name = $1
                AND (expires_at IS NULL OR expires_at > NOW())
            """

            if filter_clause:
                base_query += f" AND {filter_clause}"

            count = await conn.fetchval(base_query, memory_name, *filter_params)
            return count or 0

    async def cleanup_expired(self) -> int:
        """Remove expired entries."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= NOW()"
            )

            count = int(result.split()[-1])
            if count > 0:
                logger.info(f"Cleaned up {count} expired entries")
            return count

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL connection pool closed")

    def _row_to_entry(self, row: asyncpg.Record) -> Entry:
        """Convert database row to Entry."""
        fields = row['fields']
        if isinstance(fields, str):
            fields = json.loads(fields)

        created_at = row['created_at']
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        updated_at = row['updated_at']
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)

        expires_at = row['expires_at']
        if expires_at is not None and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        return Entry(
            id=str(row['id']),
            content=row['content'],
            fields=fields,
            created_at=created_at,
            updated_at=updated_at,
            expires_at=expires_at,
            score=row['score']
        )
