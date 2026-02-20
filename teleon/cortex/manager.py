"""
Memory manager - Handles lifecycle, auto-save, and auto-context injection.
"""

from typing import Optional, Any, Dict, List, Callable, Tuple
import logging
import inspect
import os

from teleon.cortex.memory import Memory
from teleon.cortex.entry import Entry
from teleon.cortex.context import MemoryContext
from teleon.cortex.config import MemoryConfig, parse_memory_config
from teleon.cortex.storage.base import StorageBackend
from teleon.cortex.storage.inmemory import InMemoryBackend
from teleon.cortex.embeddings.service import EmbeddingService, get_embedding_service

logger = logging.getLogger("teleon.cortex.manager")

# Global storage backend (shared across all agents unless overridden)
_global_backend: Optional[StorageBackend] = None


def get_storage_backend() -> StorageBackend:
    """
    Get or create global storage backend.

    Automatically detects and configures backend based on environment variables:
    - POSTGRES_URL or DATABASE_URL: Uses PostgreSQL with pgvector
    - REDIS_URL: Uses Redis with RediSearch

    Falls back to in-memory storage for development.
    """
    global _global_backend
    if _global_backend is None:
        _global_backend = _create_backend_from_env()
    return _global_backend


def _create_backend_from_env() -> StorageBackend:
    """Create storage backend based on environment variables."""

    # Check for PostgreSQL
    postgres_url = os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL")
    if postgres_url:
        try:
            from teleon.cortex.storage.postgres import PostgresBackend
            backend = PostgresBackend(connection_string=postgres_url)
            logger.info("Using PostgreSQL storage backend")
            return backend
        except ImportError:
            logger.warning("PostgreSQL URL found but asyncpg not installed. Run: pip install asyncpg")
        except Exception as e:
            logger.warning(f"Failed to initialize PostgreSQL backend: {e}")

    # Check for Redis
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            from teleon.cortex.storage.redis import RedisBackend
            backend = RedisBackend(connection_string=redis_url)
            logger.info("Using Redis storage backend")
            return backend
        except ImportError:
            logger.warning("Redis URL found but redis not installed. Run: pip install redis")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis backend: {e}")

    # Fallback to in-memory
    logger.info("Using in-memory storage backend (development mode)")
    return InMemoryBackend()


def set_storage_backend(backend: StorageBackend) -> None:
    """Set global storage backend."""
    global _global_backend
    _global_backend = backend
    logger.info(f"Storage backend set to {type(backend).__name__}")


class MemoryManager:
    """
    Manages Memory instances for agents.

    Handles:
    - Creating Memory instances with proper configuration
    - Extracting scope values from function arguments
    - Auto-context retrieval and injection
    - Auto-saving conversations after execution
    """

    def __init__(
        self,
        config: MemoryConfig,
        agent_name: str,
        backend: Optional[StorageBackend] = None,
        embedding_service: Optional[EmbeddingService] = None,
        is_paid_tier: bool = False
    ):
        """
        Initialize MemoryManager.

        Args:
            config: Memory configuration
            agent_name: Name of the agent
            backend: Optional storage backend (uses global if not provided)
            embedding_service: Optional embedding service
            is_paid_tier: Whether user is on paid tier (affects embedding model)
        """
        self.config = config
        self.agent_name = agent_name
        self._backend = backend or get_storage_backend()
        self._embedding = embedding_service or get_embedding_service(is_paid_tier)

        # Memory name (for sharing across agents)
        self.memory_name = config.name or agent_name

        logger.info(f"MemoryManager initialized for '{agent_name}' (memory_name='{self.memory_name}')")

    def extract_scope_values(
        self,
        args: Tuple,
        kwargs: Dict[str, Any],
        func: Callable
    ) -> Dict[str, Any]:
        """
        Extract scope values from function arguments.

        Args:
            args: Positional arguments
            kwargs: Keyword arguments
            func: The function being called

        Returns:
            Dictionary of scope field -> value
        """
        scope_values = {}

        if not self.config.scope:
            return scope_values

        # Get function signature
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Build full kwargs from args + kwargs
        full_kwargs = dict(kwargs)
        for i, arg in enumerate(args):
            if i < len(params):
                full_kwargs[params[i]] = arg

        # Extract scope values
        for scope_field in self.config.scope:
            if scope_field in full_kwargs:
                scope_values[scope_field] = full_kwargs[scope_field]
            else:
                # If the caller didn't supply the scope field, use a default value if the
                # function signature defines one (common in playground usage).
                param = sig.parameters.get(scope_field)
                if param is not None and param.default is not inspect.Parameter.empty:
                    scope_values[scope_field] = param.default
                else:
                    logger.warning(f"Scope field '{scope_field}' not found in function arguments")

        return scope_values

    def extract_field_values(
        self,
        args: Tuple,
        kwargs: Dict[str, Any],
        func: Callable
    ) -> Dict[str, Any]:
        """
        Extract field values to save with memory entry.

        Args:
            args: Positional arguments
            kwargs: Keyword arguments
            func: The function being called

        Returns:
            Dictionary of field -> value
        """
        # Get function signature
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Build full kwargs from args + kwargs
        full_kwargs = dict(kwargs)
        for i, arg in enumerate(args):
            if i < len(params):
                full_kwargs[params[i]] = arg

        # Filter to configured fields (or all if not specified)
        if self.config.fields:
            return {k: v for k, v in full_kwargs.items() if k in self.config.fields}
        else:
            # Exclude 'memory' parameter and internal fields
            excluded = {'memory', 'self', 'cls'}
            return {k: v for k, v in full_kwargs.items() if k not in excluded and not k.startswith('_')}

    def create_memory(
        self,
        scope_values: Dict[str, Any]
    ) -> Memory:
        """
        Create a Memory instance with proper configuration.

        Args:
            scope_values: Scope values extracted from function args

        Returns:
            Configured Memory instance
        """
        return Memory(
            backend=self._backend,
            embedding_service=self._embedding,
            memory_name=self.memory_name,
            scope=self.config.scope,
            scope_values=scope_values,
            layers=self.config.layers
        )

    async def retrieve_context(
        self,
        memory: Memory,
        query: Optional[str],
        filter: Dict[str, Any]
    ) -> MemoryContext:
        """
        Retrieve context for auto-injection.

        Args:
            memory: Memory instance
            query: Query text (usually the user's input)
            filter: Filter for context retrieval (usually scope values)

        Returns:
            MemoryContext with retrieved entries
        """
        if not self.config.auto_context.enabled:
            return MemoryContext.empty()

        entries: List[Entry] = []

        # Get recent history
        if self.config.auto_context.history_limit > 0:
            history = await memory.get(
                filter=filter,
                limit=self.config.auto_context.history_limit
            )
            entries.extend(history)

        # Get semantically relevant entries
        if query and self.config.auto_context.relevant_limit > 0:
            # Merge configured filter with scope filter
            search_filter = {**filter, **self.config.auto_context.filter}
            relevant = await memory.search(
                query=query,
                filter=search_filter,
                limit=self.config.auto_context.relevant_limit
            )

            # Add relevant entries not already in history
            existing_ids = {e.id for e in entries}
            for entry in relevant:
                if entry.id not in existing_ids:
                    entries.append(entry)

        # Sort by relevance (scored entries first) then by date
        entries.sort(
            key=lambda e: (e.score or 0, e.created_at.timestamp()),
            reverse=True
        )

        return MemoryContext.from_entries(
            entries,
            max_tokens=self.config.auto_context.max_tokens
        )

    async def auto_save(
        self,
        memory: Memory,
        query: str,
        response: str,
        fields: Dict[str, Any]
    ) -> Optional[str]:
        """
        Auto-save conversation to memory.

        Args:
            memory: Memory instance
            query: User's query
            response: Agent's response
            fields: Fields to save with the entry

        Returns:
            Entry ID if saved, None if auto-save disabled
        """
        if not self.config.auto:
            return None

        # Format conversation
        content = f"Q: {query}\nA: {response}"

        # Add type field
        fields = {**fields, "type": "conversation"}

        entry_id = await memory.store(
            content=content,
            **fields
        )

        logger.debug(f"Auto-saved conversation: {entry_id}")
        return entry_id


# Registry of memory managers per agent
_managers: Dict[str, MemoryManager] = {}


def get_memory_manager(
    agent_name: str,
    config: Optional[MemoryConfig] = None,
    is_paid_tier: bool = False
) -> Optional[MemoryManager]:
    """
    Get or create MemoryManager for an agent.

    Args:
        agent_name: Name of the agent
        config: Memory configuration (required for creation)
        is_paid_tier: Whether user is on paid tier

    Returns:
        MemoryManager or None if memory not configured
    """
    if agent_name in _managers:
        return _managers[agent_name]

    if config is None:
        return None

    manager = MemoryManager(
        config=config,
        agent_name=agent_name,
        is_paid_tier=is_paid_tier
    )
    _managers[agent_name] = manager
    return manager


def clear_memory_managers() -> None:
    """Clear all memory managers (for testing)."""
    _managers.clear()
