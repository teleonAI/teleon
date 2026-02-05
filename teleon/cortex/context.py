"""
Memory context for auto-injection.
"""

from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from teleon.cortex.entry import Entry


@dataclass
class MemoryContext:
    """
    Auto-retrieved context injected before agent execution.

    Contains entries retrieved by MemoryManager and formatted text
    ready for LLM injection.
    """

    entries: List["Entry"] = field(default_factory=list)
    text: str = ""

    @classmethod
    def empty(cls) -> "MemoryContext":
        """Create empty context."""
        return cls(entries=[], text="")

    @classmethod
    def from_entries(
        cls,
        entries: List["Entry"],
        max_tokens: int = 2000
    ) -> "MemoryContext":
        """
        Create context from entries with token budget.

        Args:
            entries: List of Entry objects to include
            max_tokens: Maximum tokens for context text

        Returns:
            MemoryContext with formatted text
        """
        if not entries:
            return cls.empty()

        lines = ["Relevant context from memory:", ""]
        token_estimate = 10  # Header tokens
        included_entries = []

        for entry in entries:
            # Rough token estimate: ~4 chars per token
            entry_text = f"- {entry.content}"
            entry_tokens = len(entry_text) // 4

            if token_estimate + entry_tokens > max_tokens:
                break

            lines.append(entry_text)
            included_entries.append(entry)
            token_estimate += entry_tokens

        lines.append("")  # Trailing newline

        return cls(
            entries=included_entries,
            text="\n".join(lines)
        )

    def __bool__(self) -> bool:
        """Return True if context has entries."""
        return len(self.entries) > 0

    def __len__(self) -> int:
        """Return number of entries."""
        return len(self.entries)
