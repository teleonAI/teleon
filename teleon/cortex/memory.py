"""
Memory class - Main user-facing API.
"""

from typing import Optional, Any, List, Dict, TYPE_CHECKING
from datetime import datetime, timezone, timedelta
import logging

from teleon.cortex.entry import Entry
from teleon.cortex.context import MemoryContext
from teleon.cortex.scope import ScopeEnforcer
from teleon.cortex.layers import MemoryLayer

if TYPE_CHECKING:
    from teleon.cortex.storage.base import StorageBackend
    from teleon.cortex.embeddings.service import EmbeddingService
    from teleon.cortex.config import LayerConfig

logger = logging.getLogger("teleon.cortex.memory")


class Memory:
    """
    User-facing memory API.

    Provides store, search, get, update, delete, count operations
    with automatic scope enforcement and optional layer support.

    Example:
        # Basic usage
        await memory.store(content="User prefers dark mode", user_id="alice")
        results = await memory.search(query="preferences", filter={"user_id": "alice"})

        # With layers
        await memory.team.store(content="Team note", team_id="engineering")
        await memory.personal.store(content="My preference", user_id="alice")
    """

    def __init__(
        self,
        backend: "StorageBackend",
        embedding_service: "EmbeddingService",
        memory_name: str,
        scope: Optional[List[str]] = None,
        scope_values: Optional[Dict[str, Any]] = None,
        layers: Optional[Dict[str, "LayerConfig"]] = None
    ):
        """
        Initialize Memory.

        Args:
            backend: Storage backend instance
            embedding_service: Embedding service instance
            memory_name: Unique name for this memory (usually agent name)
            scope: List of scope field names for multi-tenancy
            scope_values: Values for scope fields (extracted from function args)
            layers: Optional layer configurations for hierarchical memory
        """
        self._backend = backend
        self._embedding = embedding_service
        self._memory_name = memory_name
        self._scope_enforcer = ScopeEnforcer(scope or [], scope_values or {})
        self._context: Optional[MemoryContext] = None

        # Create layer proxies for hierarchical access
        self._layers: Dict[str, MemoryLayer] = {}
        if layers:
            for layer_name, layer_config in layers.items():
                layer_scope = layer_config.scope if hasattr(layer_config, 'scope') else layer_config.get("scope", [])
                self._layers[layer_name] = MemoryLayer(
                    backend=backend,
                    embedding_service=embedding_service,
                    memory_name=f"{memory_name}:{layer_name}",
                    scope=layer_scope,
                    scope_values=scope_values or {}
                )
                logger.debug(f"Created layer '{layer_name}' with scope {layer_scope}")

        logger.debug(f"Memory initialized: {memory_name}, scope={scope}, layers={list(self._layers.keys())}")

    def __getattr__(self, name: str):
        """Allow memory.layer_name access for layers."""
        if name.startswith("_"):
            raise AttributeError(name)
        if "_layers" in self.__dict__ and name in self._layers:
            return self._layers[name]
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'. "
            f"Available layers: {list(self._layers.keys()) if '_layers' in self.__dict__ else []}"
        )

    @property
    def context(self) -> Optional[MemoryContext]:
        """Auto-retrieved context (set by MemoryManager before function execution)."""
        return self._context

    def _set_context(self, context: MemoryContext) -> None:
        """Set context (called by MemoryManager before function execution)."""
        self._context = context

    async def store(
        self,
        content: str,
        ttl: Optional[int] = None,
        upsert: bool = False,
        **fields
    ) -> str:
        """
        Store content with any fields.

        Args:
            content: Text content to store
            ttl: Optional time-to-live in seconds
            upsert: If True, update existing entry matching fields
            **fields: Any additional fields to store

        Returns:
            Entry ID

        Example:
            # Basic
            await memory.store(content="User prefers dark mode")

            # With fields
            await memory.store(
                content="Resolved billing issue",
                customer_id="alice",
                type="resolution",
                tags=["billing", "resolved"]
            )

            # With TTL
            await memory.store(
                content="Session data",
                session_id="abc",
                ttl=3600  # Expires in 1 hour
            )
        """
        # Enforce scope
        fields = self._scope_enforcer.enforce_fields(fields)

        # Generate embedding
        embedding = await self._embedding.embed(content)

        # Calculate expires_at from TTL
        expires_at = None
        if ttl:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)

        logger.debug(f"Storing in '{self._memory_name}': {len(content)} chars, fields={list(fields.keys())}")

        # Delegate to backend
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
        """
        Semantic search with optional filter.

        Args:
            query: Search query for semantic matching
            filter: Field filters (exact match)
            limit: Maximum results to return

        Returns:
            List of matching entries with relevance scores

        Example:
            # Semantic search
            results = await memory.search(query="refund policy")

            # Filter only
            results = await memory.search(filter={"customer_id": "alice"})

            # Combined
            results = await memory.search(
                query="shipping problems",
                filter={"customer_id": "alice", "type": "complaint"}
            )
        """
        # Enforce scope on filter
        filter = self._scope_enforcer.enforce_filter(filter or {})

        # Generate query embedding if query provided
        query_embedding = None
        if query:
            query_embedding = await self._embedding.embed(query)

        logger.debug(f"Searching '{self._memory_name}': query={bool(query)}, filter_keys={list(filter.keys())}")

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
        """
        Get by filter only (no semantic search).

        Faster than search() when you don't need semantic matching.

        Args:
            filter: Field filters (exact match)
            limit: Maximum results to return

        Returns:
            List of matching entries ordered by created_at DESC

        Example:
            history = await memory.get(filter={"customer_id": "alice"}, limit=50)
        """
        filter = self._scope_enforcer.enforce_filter(filter)

        logger.debug(f"Getting from '{self._memory_name}': filter_keys={list(filter.keys())}")

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
        """
        Update entries matching filter.

        Args:
            filter: Filter to match entries
            content: New content (optional)
            **fields: Fields to update

        Returns:
            Number of entries updated

        Example:
            # Update content
            await memory.update(
                filter={"user_id": "alice", "type": "preference"},
                content="Now prefers SMS"
            )

            # Update fields
            await memory.update(
                filter={"lead_id": "123"},
                status="qualified",
                score=85
            )
        """
        filter = self._scope_enforcer.enforce_filter(filter)

        # Re-embed if content changed
        embedding = None
        if content:
            embedding = await self._embedding.embed(content)

        logger.debug(f"Updating in '{self._memory_name}': filter_keys={list(filter.keys())}")

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
        """
        Delete entries matching filter.

        Args:
            filter: Filter to match entries

        Returns:
            Number of entries deleted

        Example:
            # GDPR: Delete all user data
            deleted = await memory.delete(filter={"user_id": "alice"})
        """
        filter = self._scope_enforcer.enforce_filter(filter)

        logger.debug(f"Deleting from '{self._memory_name}': filter_keys={list(filter.keys())}")

        return await self._backend.delete(
            memory_name=self._memory_name,
            filter=filter
        )

    async def count(
        self,
        filter: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Count entries matching filter.

        Args:
            filter: Optional filter (counts all if None)

        Returns:
            Number of matching entries

        Example:
            total = await memory.count(filter={"customer_id": "alice"})
        """
        filter = self._scope_enforcer.enforce_filter(filter or {})

        return await self._backend.count(
            memory_name=self._memory_name,
            filter=filter
        )
