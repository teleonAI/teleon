"""
Heuristic Prompt Injection Detection Backend.

Detects prompt injection attempts via pattern categories:
override, role hijacking, system prompt manipulation, jailbreak,
encoding evasion, and delimiter injection.
"""

import base64
import re
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Pattern categories: (compiled_regex, weight, category_name)
# ---------------------------------------------------------------------------

_OVERRIDE_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?|context)", re.I), 0.90),
    (re.compile(r"disregard\s+(?:all\s+)?(?:above|previous|prior|earlier)\b", re.I), 0.85),
    (re.compile(r"forget\s+(?:all\s+)?(?:your|the|previous|prior)\s+(?:rules?|instructions?|constraints?|guidelines?)", re.I), 0.90),
    (re.compile(r"override\s+(?:your|all|the)\s+(?:instructions?|rules?|safety|guidelines?)", re.I), 0.85),
    (re.compile(r"do\s+not\s+follow\s+(?:your|the|any)\s+(?:previous|original|initial)\s+(?:instructions?|rules?)", re.I), 0.85),
    (re.compile(r"new\s+(?:instructions?|rules?|directives?)\s*[:;]\s*", re.I), 0.80),
]

_ROLE_HIJACKING_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"you\s+are\s+now\s+(?:a|an|the)\b", re.I), 0.70),
    (re.compile(r"pretend\s+(?:to\s+be|you\s+are)\b", re.I), 0.65),
    (re.compile(r"act\s+as\s+(?:if\s+you\s+are|though\s+you\s+are|a|an)\b", re.I), 0.65),
    (re.compile(r"(?:roleplay|role\s*-?\s*play)\s+as\b", re.I), 0.65),
    (re.compile(r"from\s+now\s+on\s+you\s+(?:are|will)\b", re.I), 0.70),
    (re.compile(r"assume\s+the\s+(?:role|identity|persona)\s+of\b", re.I), 0.70),
]

_SYSTEM_PROMPT_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"^\s*system\s*:", re.I | re.M), 0.85),
    (re.compile(r"\[system\]", re.I), 0.85),
    (re.compile(r"<\|im_start\|>\s*system", re.I), 0.90),
    (re.compile(r"###\s*(?:instruction|system|assistant)\b", re.I), 0.80),
    (re.compile(r"<\|system\|>", re.I), 0.90),
    (re.compile(r"\bBEGIN\s+SYSTEM\s+(?:PROMPT|MESSAGE)\b", re.I), 0.90),
    (re.compile(r"\bEND\s+(?:OF\s+)?SYSTEM\s+(?:PROMPT|MESSAGE)\b", re.I), 0.75),
]

_JAILBREAK_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"\bDAN\s+(?:mode|prompt|jailbreak)\b", re.I), 0.95),
    (re.compile(r"\bbypass\s+(?:safety|filters?|restrictions?|guardrails?|content\s+policy)\b", re.I), 0.90),
    (re.compile(r"\bdeveloper\s+mode\s+(?:enabled|activated|on)\b", re.I), 0.90),
    (re.compile(r"\b(?:unrestricted|unfiltered|uncensored)\s+mode\b", re.I), 0.90),
    (re.compile(r"\bdo\s+anything\s+now\b", re.I), 0.95),
    (re.compile(r"\bno\s+(?:ethical|moral|safety)\s+(?:guidelines?|constraints?|limitations?)\b", re.I), 0.85),
    (re.compile(r"\bignore\s+(?:content\s+)?(?:policy|policies|guidelines?|safety)\b", re.I), 0.85),
]

_DELIMITER_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"---+\s*(?:NEW|REAL|ACTUAL|TRUE)\s+(?:PROMPT|INSTRUCTIONS?|TASK)\s*---+", re.I), 0.70),
    (re.compile(r"={3,}\s*(?:NEW|REAL|ACTUAL)\s+(?:PROMPT|CONTEXT)", re.I), 0.70),
    (re.compile(r"</?(?:system|prompt|instruction|context|user|assistant)>", re.I), 0.60),
    (re.compile(r"\[/?(?:INST|SYS|SYSTEM)\]", re.I), 0.70),
]


def _check_encoding_evasion(text: str) -> Tuple[float, List[str]]:
    """
    Check for encoding-based evasion attempts.

    Returns (score, list_of_reasons).
    """
    score = 0.0
    reasons: List[str] = []

    # Base64 encoded instruction fragments
    b64_pattern = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
    for match in b64_pattern.finditer(text):
        try:
            decoded = base64.b64decode(match.group(0)).decode("utf-8", errors="ignore")
            suspicious = re.search(
                r"(?:ignore|system|override|bypass|jailbreak)", decoded, re.I
            )
            if suspicious:
                score += 0.60
                reasons.append("base64_encoded_injection")
                break
        except Exception:
            pass

    # Excessive unicode escape sequences (potential homoglyph attack)
    unicode_escapes = re.findall(r"\\u[0-9a-fA-F]{4}", text)
    if len(unicode_escapes) > 10:
        score += 0.30
        reasons.append("excessive_unicode_escapes")

    # Zero-width characters (potential invisible text injection)
    zw_chars = len(re.findall(r"[\u200b\u200c\u200d\u2060\ufeff]", text))
    if zw_chars > 3:
        score += 0.40
        reasons.append("zero_width_characters")

    return min(score, 1.0), reasons


class HeuristicInjectionDetector:
    """
    Heuristic prompt injection detector.

    Implements PromptInjectionBackend protocol.
    """

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def detect(self, text: str) -> Tuple[bool, float, List[str]]:
        """
        Detect prompt injection attempts.

        Returns:
            (is_injection, confidence_score, matched_categories)
        """
        if not text or not isinstance(text, str):
            return False, 0.0, []

        total_score = 0.0
        categories: List[str] = []

        # Check each category, take highest match per category
        for category_name, patterns in [
            ("override", _OVERRIDE_PATTERNS),
            ("role_hijacking", _ROLE_HIJACKING_PATTERNS),
            ("system_prompt_manipulation", _SYSTEM_PROMPT_PATTERNS),
            ("jailbreak", _JAILBREAK_PATTERNS),
            ("delimiter_injection", _DELIMITER_PATTERNS),
        ]:
            best_weight = 0.0
            for pattern, weight in patterns:
                if pattern.search(text):
                    best_weight = max(best_weight, weight)
            if best_weight > 0:
                total_score += best_weight
                categories.append(category_name)

        # Encoding evasion
        enc_score, enc_reasons = _check_encoding_evasion(text)
        if enc_score > 0:
            total_score += enc_score
            categories.append("encoding_evasion")

        # Composite: cap at 1.0, use highest contributing category as primary signal
        confidence = min(total_score, 1.0)
        detected = confidence >= self.threshold

        return detected, confidence, categories
