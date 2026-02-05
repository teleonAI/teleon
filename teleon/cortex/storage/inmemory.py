"""
In-memory storage backend for development and testing.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import uuid
import logging
import math

from teleon.cortex.entry import Entry
from teleon.cortex.storage.base import StorageBackend

logger = logging.getLogger("teleon.cortex.storage.inmemory")


class InMemoryBackend(StorageBackend):
    """
    In-memory storage backend.

    For development and testing only. Data is lost on restart.
    """

    def __init__(self):
        self._entries: Dict[str, Dict[str, Any]] = {}  # id -> entry data
        self._embeddings: Dict[str, List[float]] = {}  # id -> embedding

    async def store(
        self,
        memory_name: str,
        content: str,
        embedding: List[float],
        fields: Dict[str, Any],
        expires_at: Optional[datetime] = None,
        upsert: bool = False
    ) -> str:
        """Store entry."""
        now = datetime.now(timezone.utc)

        # Handle upsert: find existing entry with matching fields
        if upsert:
            for entry_id, entry_data in self._entries.items():
                if entry_data["memory_name"] != memory_name:
                    continue
                # Check if all fields match (excluding content)
                if self._fields_match(entry_data["fields"], fields):
                    # Update existing entry
                    entry_data["content"] = content
                    entry_data["updated_at"] = now
                    entry_data["expires_at"] = expires_at
                    self._embeddings[entry_id] = embedding
                    logger.debug(f"Upserted entry {entry_id}")
                    return entry_id

        # Create new entry
        entry_id = str(uuid.uuid4())
        self._entries[entry_id] = {
            "id": entry_id,
            "memory_name": memory_name,
            "content": content,
            "fields": fields,
            "created_at": now,
            "updated_at": now,
            "expires_at": expires_at,
        }
        self._embeddings[entry_id] = embedding

        logger.debug(f"Stored entry {entry_id} in {memory_name}")
        return entry_id

    def _fields_match(self, stored: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Check if stored fields match query fields."""
        for key, value in query.items():
            if key not in stored:
                return False
            if stored[key] != value:
                return False
        return True

    def _filter_matches(self, fields: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """Check if entry fields match filter."""
        for key, value in filter.items():
            if key not in fields:
                return False

            stored_value = fields[key]

            # Handle list values (OR matching)
            if isinstance(value, list):
                if stored_value not in value:
                    # Also check if stored_value is a list and has intersection
                    if isinstance(stored_value, list):
                        if not set(stored_value).intersection(set(value)):
                            return False
                    else:
                        return False
            # Handle dot notation for nested fields
            elif "." in key:
                parts = key.split(".")
                nested_value = fields
                for part in parts:
                    if isinstance(nested_value, dict) and part in nested_value:
                        nested_value = nested_value[part]
                    else:
                        return False
                if nested_value != value:
                    return False
            # Exact match
            elif stored_value != value:
                return False

        return True

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            # Handle dimension mismatch by padding shorter vector
            max_len = max(len(a), len(b))
            a = a + [0.0] * (max_len - len(a))
            b = b + [0.0] * (max_len - len(b))

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _is_expired(self, entry_data: Dict[str, Any]) -> bool:
        """Check if entry is expired."""
        expires_at = entry_data.get("expires_at")
        if expires_at is None:
            return False
        return datetime.now(timezone.utc) > expires_at

    async def search(
        self,
        memory_name: str,
        query_embedding: Optional[List[float]],
        filter: Dict[str, Any],
        limit: int
    ) -> List[Entry]:
        """Semantic search with filter."""
        results = []

        for entry_id, entry_data in self._entries.items():
            # Check memory name
            if entry_data["memory_name"] != memory_name:
                continue

            # Check expiration
            if self._is_expired(entry_data):
                continue

            # Check filter
            if not self._filter_matches(entry_data["fields"], filter):
                continue

            # Calculate score if query embedding provided
            score = None
            if query_embedding and entry_id in self._embeddings:
                score = self._cosine_similarity(query_embedding, self._embeddings[entry_id])
                # Convert to 0-1 range (cosine can be -1 to 1)
                score = (score + 1) / 2

            results.append((entry_data, score))

        # Sort results
        if query_embedding:
            # Sort by score descending
            results.sort(key=lambda x: x[1] if x[1] is not None else 0, reverse=True)
        else:
            # Sort by created_at descending
            results.sort(key=lambda x: x[0]["created_at"], reverse=True)

        # Limit and convert to Entry objects
        entries = []
        for entry_data, score in results[:limit]:
            entries.append(Entry(
                id=entry_data["id"],
                content=entry_data["content"],
                fields=entry_data["fields"],
                created_at=entry_data["created_at"],
                updated_at=entry_data["updated_at"],
                expires_at=entry_data.get("expires_at"),
                score=score
            ))

        return entries

    async def get(
        self,
        memory_name: str,
        filter: Dict[str, Any],
        limit: int
    ) -> List[Entry]:
        """Get by filter only."""
        results = []

        for entry_id, entry_data in self._entries.items():
            if entry_data["memory_name"] != memory_name:
                continue

            if self._is_expired(entry_data):
                continue

            if not self._filter_matches(entry_data["fields"], filter):
                continue

            results.append(entry_data)

        # Sort by created_at descending
        results.sort(key=lambda x: x["created_at"], reverse=True)

        # Limit and convert to Entry objects
        entries = []
        for entry_data in results[:limit]:
            entries.append(Entry(
                id=entry_data["id"],
                content=entry_data["content"],
                fields=entry_data["fields"],
                created_at=entry_data["created_at"],
                updated_at=entry_data["updated_at"],
                expires_at=entry_data.get("expires_at"),
                score=None
            ))

        return entries

    async def update(
        self,
        memory_name: str,
        filter: Dict[str, Any],
        content: Optional[str],
        embedding: Optional[List[float]],
        fields: Optional[Dict[str, Any]]
    ) -> int:
        """Update matching entries."""
        count = 0
        now = datetime.now(timezone.utc)

        for entry_id, entry_data in self._entries.items():
            if entry_data["memory_name"] != memory_name:
                continue

            if self._is_expired(entry_data):
                continue

            if not self._filter_matches(entry_data["fields"], filter):
                continue

            # Update content
            if content is not None:
                entry_data["content"] = content

            # Update embedding
            if embedding is not None:
                self._embeddings[entry_id] = embedding

            # Update fields
            if fields:
                entry_data["fields"].update(fields)

            entry_data["updated_at"] = now
            count += 1

        logger.debug(f"Updated {count} entries in {memory_name}")
        return count

    async def delete(
        self,
        memory_name: str,
        filter: Dict[str, Any]
    ) -> int:
        """Delete matching entries."""
        to_delete = []

        for entry_id, entry_data in self._entries.items():
            if entry_data["memory_name"] != memory_name:
                continue

            if not self._filter_matches(entry_data["fields"], filter):
                continue

            to_delete.append(entry_id)

        for entry_id in to_delete:
            del self._entries[entry_id]
            if entry_id in self._embeddings:
                del self._embeddings[entry_id]

        logger.debug(f"Deleted {len(to_delete)} entries from {memory_name}")
        return len(to_delete)

    async def count(
        self,
        memory_name: str,
        filter: Dict[str, Any]
    ) -> int:
        """Count matching entries."""
        count = 0

        for entry_id, entry_data in self._entries.items():
            if entry_data["memory_name"] != memory_name:
                continue

            if self._is_expired(entry_data):
                continue

            if not self._filter_matches(entry_data["fields"], filter):
                continue

            count += 1

        return count

    async def cleanup_expired(self) -> int:
        """Delete expired entries."""
        to_delete = []

        for entry_id, entry_data in self._entries.items():
            if self._is_expired(entry_data):
                to_delete.append(entry_id)

        for entry_id in to_delete:
            del self._entries[entry_id]
            if entry_id in self._embeddings:
                del self._embeddings[entry_id]

        if to_delete:
            logger.debug(f"Cleaned up {len(to_delete)} expired entries")

        return len(to_delete)

    async def close(self) -> None:
        """Clear all data."""
        self._entries.clear()
        self._embeddings.clear()
