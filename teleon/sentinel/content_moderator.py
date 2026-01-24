"""
Content Moderator - Toxicity and Hate Speech Detection.

Provides content filtering capabilities for agent inputs and outputs.
"""

import re
from typing import Tuple, List, Dict, Any
from teleon.core import StructuredLogger, LogLevel


class ContentModerator:
    """
    Content moderation and filtering.
    
    Detects toxic content, hate speech, and profanity in text.
    Uses pattern-based detection (extensible to ML models).
    """
    
    # Toxicity patterns (simplified - in production use ML models)
    TOXIC_PATTERNS = [
        r'\b(?:kill|murder|suicide|bomb|terrorist|attack)\b',
        r'\b(?:die|death|dead)\s+(?:you|yourself|them)\b',
        r'\b(?:hate|despise|loathe)\s+(?:you|your|them)\b',
    ]
    
    # Hate speech patterns
    HATE_SPEECH_PATTERNS = [
        r'\b(?:racial|racist|sexist|homophobic)\s+(?:slur|epithet)\b',
        r'\b(?:discriminate|discrimination)\s+(?:against|based)\b',
    ]
    
    # Profanity patterns (basic)
    PROFANITY_PATTERNS = [
        r'\b(?:fuck|shit|damn|hell|asshole|bitch)\b',
    ]
    
    def __init__(self, threshold: float = 0.8):
        """
        Initialize content moderator.
        
        Args:
            threshold: Toxicity threshold (0.0-1.0)
        """
        self.threshold = threshold
        self.logger = StructuredLogger("content_moderator", LogLevel.INFO)
    
    def check_toxicity(self, text: str) -> Tuple[bool, float]:
        """
        Check text for toxic content.
        
        Args:
            text: Text to check
        
        Returns:
            Tuple of (is_toxic, score) where score is 0.0-1.0
        """
        if not text or not isinstance(text, str):
            return False, 0.0
        
        score = 0.0
        matches = []
        
        # Check toxic patterns
        for pattern in self.TOXIC_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern)
                score += 0.3
        
        # Normalize score
        score = min(score, 1.0)
        is_toxic = score >= self.threshold
        
        if is_toxic:
            self.logger.debug(
                "Toxic content detected",
                score=score,
                matches_count=len(matches),
                text_preview=text[:100]
            )
        
        return is_toxic, score
    
    def check_hate_speech(self, text: str) -> Tuple[bool, float]:
        """
        Check for hate speech.
        
        Args:
            text: Text to check
        
        Returns:
            Tuple of (is_hate_speech, score) where score is 0.0-1.0
        """
        if not text or not isinstance(text, str):
            return False, 0.0
        
        score = 0.0
        matches = []
        
        for pattern in self.HATE_SPEECH_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern)
                score += 0.4
        
        # Normalize score
        score = min(score, 1.0)
        is_hate_speech = score >= self.threshold
        
        if is_hate_speech:
            self.logger.debug(
                "Hate speech detected",
                score=score,
                matches_count=len(matches),
                text_preview=text[:100]
            )
        
        return is_hate_speech, score
    
    def check_profanity(self, text: str) -> Tuple[bool, float]:
        """
        Check for profanity.
        
        Args:
            text: Text to check
        
        Returns:
            Tuple of (has_profanity, score) where score is 0.0-1.0
        """
        if not text or not isinstance(text, str):
            return False, 0.0
        
        score = 0.0
        matches = []
        
        for pattern in self.PROFANITY_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern)
                score += 0.2
        
        # Normalize score
        score = min(score, 1.0)
        has_profanity = score >= self.threshold
        
        return has_profanity, score
    
    def check_all(self, text: str) -> Dict[str, Any]:
        """
        Run all content checks.
        
        Args:
            text: Text to check
        
        Returns:
            Dictionary with all check results
        """
        toxicity_result = self.check_toxicity(text)
        hate_speech_result = self.check_hate_speech(text)
        profanity_result = self.check_profanity(text)
        
        return {
            "toxicity": {
                "detected": toxicity_result[0],
                "score": toxicity_result[1]
            },
            "hate_speech": {
                "detected": hate_speech_result[0],
                "score": hate_speech_result[1]
            },
            "profanity": {
                "detected": profanity_result[0],
                "score": profanity_result[1]
            },
            "overall_score": max(
                toxicity_result[1],
                hate_speech_result[1],
                profanity_result[1]
            )
        }

