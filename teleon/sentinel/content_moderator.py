"""
Content Moderator - Backward-compatible wrapper.

Delegates to the HeuristicContentModerator backend.
This module is retained for backward compatibility; new code should use
teleon.sentinel.backends.content_heuristic.HeuristicContentModerator directly.
"""

from typing import Tuple, Dict, Any
from teleon.sentinel.backends.content_heuristic import HeuristicContentModerator


class ContentModerator:
    """
    Content moderation and filtering (backward-compatible wrapper).

    New code should use HeuristicContentModerator directly.
    """

    def __init__(self, threshold: float = 0.8):
        self._backend = HeuristicContentModerator(threshold=threshold)
        self.threshold = threshold

    def check_toxicity(self, text: str) -> Tuple[bool, float]:
        return self._backend.check_toxicity(text)

    def check_hate_speech(self, text: str) -> Tuple[bool, float]:
        return self._backend.check_hate_speech(text)

    def check_profanity(self, text: str) -> Tuple[bool, float]:
        return self._backend.check_profanity(text)

    def check_all(self, text: str) -> Dict[str, Any]:
        return self._backend.check_all(text)
