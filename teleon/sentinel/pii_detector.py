"""
PII Detector - Backward-compatible wrapper.

Delegates to the HeuristicPIIDetector backend.
This module is retained for backward compatibility; new code should use
teleon.sentinel.backends.pii_heuristic.HeuristicPIIDetector directly.
"""

from typing import Dict, List, Optional
from teleon.sentinel.backends.pii_heuristic import HeuristicPIIDetector


class PIIDetector:
    """
    PII detection and redaction (backward-compatible wrapper).

    New code should use HeuristicPIIDetector directly.
    """

    def __init__(self):
        self._backend = HeuristicPIIDetector()

    def detect(self, text: str) -> Dict[str, List[str]]:
        return self._backend.detect(text)

    def redact(self, text: str, pii_types: Optional[List[str]] = None) -> str:
        return self._backend.redact(text, pii_types)

    def _validate_credit_card(self, card_number: str) -> bool:
        """Validate credit card using Luhn algorithm (backward compat)."""
        from teleon.sentinel.backends.pii_heuristic import _luhn_check
        return _luhn_check(card_number)
