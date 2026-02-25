"""
Backend Protocols for Sentinel Detection Systems.

Defines Protocol interfaces for content moderation, PII detection, and prompt injection.
Heuristic backends implement these protocols by default; ML backends can be added later.
"""

from typing import Any, Dict, List, Optional, Tuple, runtime_checkable
from typing_extensions import Protocol


@runtime_checkable
class ContentModerationBackend(Protocol):
    """Protocol for content moderation backends."""

    def check_toxicity(self, text: str) -> Tuple[bool, float]:
        """Check text for toxic content. Returns (detected, score)."""
        ...

    def check_hate_speech(self, text: str) -> Tuple[bool, float]:
        """Check text for hate speech. Returns (detected, score)."""
        ...

    def check_all(self, text: str) -> Dict[str, Any]:
        """Run all content checks. Returns dict with results per category."""
        ...


@runtime_checkable
class PIIDetectionBackend(Protocol):
    """Protocol for PII detection backends."""

    def detect(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in text. Returns dict mapping PII types to detected values."""
        ...

    def redact(self, text: str, pii_types: Optional[List[str]] = None) -> str:
        """Redact PII from text. Returns redacted string."""
        ...


@runtime_checkable
class PromptInjectionBackend(Protocol):
    """Protocol for prompt injection detection backends."""

    def detect(self, text: str) -> Tuple[bool, float, List[str]]:
        """
        Detect prompt injection attempts.

        Returns:
            Tuple of (detected, confidence_score, matched_categories)
        """
        ...
