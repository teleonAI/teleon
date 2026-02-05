"""
Memory layers for hierarchical organization.
"""

from typing import Optional, Any, List, Dict, TYPE_CHECKING
from datetime import datetime, timezone, timedelta
import logging

from teleon.cortex.entry import Entry
from teleon.cortex.scope import ScopeEnforcer

if TYPE_CHECKING:
    from teleon.cortex.storage.base import StorageBackend
    from teleon.cortex.embeddings.service import EmbeddingService

logger = logging.getLogger("teleon.cortex.layers")


class MemoryLayer:
    """
    A scoped view into memory.

    Each layer has its own scope enforcement, allowing hierarchical
    memory organization (e.g., company → team → personal).

    Provides the same API as Memory: store, search, get, update, delete, count.
    """

    def __init__(
        self,
        backend: "StorageBackend",
        embedding_service: "EmbeddingService",
        memory_name: str,
        scope: List[str],
        scope_values: Dict[str, Any]
    ):
        """
        Initialize memory layer.

        Args:
            backend: Storage backend instance
            embedding_service: Embedding service instance
            memory_name: Unique name for this layer (e.g., "support:team")
            scope: List of scope field names for this layer
            scope_values: Values for scope fields (from function args)
        """
        self._backend = backend
        self._embedding = embedding_service
        self._memory_name = memory_name
        self._scope_enforcer = ScopeEnforcer(scope, scope_values)

        logger.debug(f"MemoryLayer initialized: {memory_name}, scope={scope}")

    async def store(
        self,
        content: str,
        ttl: Optional[int] = None,
        upsert: bool = False,
        **fields
    ) -> str:
        """Store content with scope enforcement."""
        fields = self._scope_enforcer.enforce_fields(fields)
        embedding = await self._embedding.embed(content)

        expires_at = None
        if ttl:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)

        logger.debug(f"Layer '{self._memory_name}' storing: {len(content)} chars")

        return await self._backend.store(
            memory_name=self._memory_name,
            content=content,
            embedding=embedding,
            fields=fields,
            expires_at=expires_at,
            upsert=upsert
        )

    async def search(
        self,
        query: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Entry]:
        """Semantic search with scope enforcement."""
        filter = self._scope_enforcer.enforce_filter(filter or {})

        query_embedding = None
        if query:
            query_embedding = await self._embedding.embed(query)

        logger.debug(f"Layer '{self._memory_name}' searching: query={bool(query)}")

        return await self._backend.search(
            memory_name=self._memory_name,
            query_embedding=query_embedding,
            filter=filter,
            limit=limit
        )

    async def get(
        self,
        filter: Dict[str, Any],
        limit: int = 10
    ) -> List[Entry]:
        """Get by filter with scope enforcement."""
        filter = self._scope_enforcer.enforce_filter(filter)

        logger.debug(f"Layer '{self._memory_name}' getting with filter")

        return await self._backend.get(
            memory_name=self._memory_name,
            filter=filter,
            limit=limit
        )

    async def update(
        self,
        filter: Dict[str, Any],
        content: Optional[str] = None,
        **fields
    ) -> int:
        """Update entries with scope enforcement."""
        filter = self._scope_enforcer.enforce_filter(filter)

        embedding = None
        if content:
            embedding = await self._embedding.embed(content)

        logger.debug(f"Layer '{self._memory_name}' updating")

        return await self._backend.update(
            memory_name=self._memory_name,
            filter=filter,
            content=content,
            embedding=embedding,
            fields=fields if fields else None
        )

    async def delete(
        self,
        filter: Dict[str, Any]
    ) -> int:
        """Delete entries with scope enforcement."""
        filter = self._scope_enforcer.enforce_filter(filter)

        logger.debug(f"Layer '{self._memory_name}' deleting")

        return await self._backend.delete(
            memory_name=self._memory_name,
            filter=filter
        )

    async def count(
        self,
        filter: Optional[Dict[str, Any]] = None
    ) -> int:
        """Count entries with scope enforcement."""
        filter = self._scope_enforcer.enforce_filter(filter or {})

        return await self._backend.count(
            memory_name=self._memory_name,
            filter=filter
        )
