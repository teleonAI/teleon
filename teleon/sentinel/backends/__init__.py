"""
Sentinel Detection Backends.

Provides pluggable backends for content moderation, PII detection,
and prompt injection detection via Protocol interfaces.
"""

from teleon.sentinel.backends.protocols import (
    ContentModerationBackend,
    PIIDetectionBackend,
    PromptInjectionBackend,
)


def get_content_backend(
    backend_type: str = "heuristic", **kwargs
) -> ContentModerationBackend:
    """Get a content moderation backend by type."""
    if backend_type == "heuristic":
        from teleon.sentinel.backends.content_heuristic import HeuristicContentModerator
        return HeuristicContentModerator(**kwargs)
    raise ValueError(f"Unknown content backend: {backend_type}")


def get_pii_backend(
    backend_type: str = "heuristic", **kwargs
) -> PIIDetectionBackend:
    """Get a PII detection backend by type."""
    if backend_type == "heuristic":
        from teleon.sentinel.backends.pii_heuristic import HeuristicPIIDetector
        return HeuristicPIIDetector(**kwargs)
    raise ValueError(f"Unknown PII backend: {backend_type}")


def get_injection_backend(
    backend_type: str = "heuristic", **kwargs
) -> PromptInjectionBackend:
    """Get a prompt injection detection backend by type."""
    if backend_type == "heuristic":
        from teleon.sentinel.backends.injection_heuristic import HeuristicInjectionDetector
        return HeuristicInjectionDetector(**kwargs)
    raise ValueError(f"Unknown injection backend: {backend_type}")


__all__ = [
    "ContentModerationBackend",
    "PIIDetectionBackend",
    "PromptInjectionBackend",
    "get_content_backend",
    "get_pii_backend",
    "get_injection_backend",
]
