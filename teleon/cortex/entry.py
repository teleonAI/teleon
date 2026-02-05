"""
Entry dataclass for memory entries.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class Entry:
    """A memory entry."""

    id: str
    content: str
    fields: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    score: Optional[float] = None  # Only set for search results

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "fields": self.fields,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "score": self.score
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entry":
        """Create Entry from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            fields=data.get("fields", {}),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if isinstance(data["created_at"], str)
                else data["created_at"]
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if isinstance(data["updated_at"], str)
                else data["updated_at"]
            ),
            expires_at=(
                datetime.fromisoformat(data["expires_at"])
                if data.get("expires_at") and isinstance(data["expires_at"], str)
                else data.get("expires_at")
            ),
            score=data.get("score")
        )

    def __repr__(self) -> str:
        return f"Entry(id={self.id!r}, content={self.content[:50]!r}...)"
