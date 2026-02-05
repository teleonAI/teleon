"""
Abstract storage backend interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from teleon.cortex.entry import Entry

logger = logging.getLogger("teleon.cortex.storage")


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.

    Implementations:
    - InMemoryBackend: For development/testing
    - PostgresBackend: Production with pgvector
    - RedisBackend: Production with RediSearch
    """

    @abstractmethod
    async def store(
        self,
        memory_name: str,
        content: str,
        embedding: List[float],
        fields: Dict[str, Any],
        expires_at: Optional[datetime] = None,
        upsert: bool = False
    ) -> str:
        """
        Store entry, return ID.

        If upsert=True, updates existing entry matching all fields (except content).
        """
        pass

    @abstractmethod
    async def search(
        self,
        memory_name: str,
        query_embedding: Optional[List[float]],
        filter: Dict[str, Any],
        limit: int
    ) -> List[Entry]:
        """
        Semantic search with filter.

        Ordering:
        - If query_embedding is provided: order by similarity score DESC
        - If query_embedding is None: order by created_at DESC

        Returns entries with score set (0.0-1.0) when query_embedding provided.
        """
        pass

    @abstractmethod
    async def get(
        self,
        memory_name: str,
        filter: Dict[str, Any],
        limit: int
    ) -> List[Entry]:
        """
        Get by filter only (no semantic search).

        Always ordered by created_at DESC.
        """
        pass

    @abstractmethod
    async def update(
        self,
        memory_name: str,
        filter: Dict[str, Any],
        content: Optional[str],
        embedding: Optional[List[float]],
        fields: Optional[Dict[str, Any]]
    ) -> int:
        """Update matching entries, return count."""
        pass

    @abstractmethod
    async def delete(
        self,
        memory_name: str,
        filter: Dict[str, Any]
    ) -> int:
        """Delete matching entries, return count."""
        pass

    @abstractmethod
    async def count(
        self,
        memory_name: str,
        filter: Dict[str, Any]
    ) -> int:
        """Count matching entries."""
        pass

    async def cleanup_expired(self) -> int:
        """
        Delete expired entries.

        Override in implementations that support TTL.
        Returns count of deleted entries.
        """
        return 0

    async def close(self) -> None:
        """Close connections. Override in implementations that need cleanup."""
        pass


class StorageError(Exception):
    """Base exception for storage backend errors."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        key: Optional[str] = None
    ):
        self.message = message
        self.operation = operation
        self.key = key
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = [self.message]
        if self.operation:
            parts.append(f"operation={self.operation}")
        if self.key:
            parts.append(f"key={self.key}")
        return " | ".join(parts)
