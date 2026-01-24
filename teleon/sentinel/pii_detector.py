"""
PII Detector - Personally Identifiable Information Detection and Redaction.

Detects and redacts PII from agent inputs and outputs.
"""

import re
from typing import Dict, List, Optional
from teleon.core import StructuredLogger, LogLevel


class PIIDetector:
    """
    PII (Personally Identifiable Information) detection and redaction.
    
    Detects:
    - Email addresses
    - Phone numbers (US and international)
    - Social Security Numbers (SSN)
    - Credit card numbers (with Luhn validation)
    - IP addresses
    """
    
    # Email pattern
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    # Phone pattern (US)
    PHONE_US_PATTERN = r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'
    
    # Phone pattern (international)
    PHONE_INTERNATIONAL_PATTERN = r'\b\+?[1-9]\d{1,14}\b'
    
    # SSN pattern
    SSN_PATTERN = r'\b\d{3}-\d{2}-\d{4}\b'
    
    # Credit card pattern (with Luhn validation)
    CC_PATTERN = r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
    
    # IP address pattern
    IP_PATTERN = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    
    def __init__(self):
        """Initialize PII detector."""
        self.logger = StructuredLogger("pii_detector", LogLevel.INFO)
    
    def detect(self, text: str) -> Dict[str, List[str]]:
        """
        Detect PII in text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary mapping PII types to lists of detected values
        """
        if not text or not isinstance(text, str):
            return {}
        
        detected = {
            'emails': re.findall(self.EMAIL_PATTERN, text),
            'phones': self._find_phones(text),
            'ssns': re.findall(self.SSN_PATTERN, text),
            'credit_cards': [],
            'ip_addresses': self._find_valid_ips(text)
        }
        
        # Validate credit cards with Luhn algorithm
        cc_matches = re.findall(self.CC_PATTERN, text)
        for cc in cc_matches:
            if self._validate_credit_card(cc.replace('-', '').replace(' ', '')):
                detected['credit_cards'].append(cc)
        
        # Filter out empty lists
        result = {k: v for k, v in detected.items() if v}
        
        if result:
            self.logger.debug(
                "PII detected",
                pii_types=list(result.keys()),
                total_detections=sum(len(v) for v in result.values())
            )
        
        return result
    
    def redact(self, text: str, pii_types: Optional[List[str]] = None) -> str:
        """
        Redact PII from text.
        
        Args:
            text: Text to redact
            pii_types: List of PII types to redact (None = all)
        
        Returns:
            Text with PII redacted
        """
        if not text or not isinstance(text, str):
            return text
        
        if pii_types is None:
            pii_types = ['emails', 'phones', 'ssns', 'credit_cards', 'ip_addresses']
        
        redacted = text
        
        if 'emails' in pii_types:
            redacted = re.sub(self.EMAIL_PATTERN, '[EMAIL_REDACTED]', redacted)
        
        if 'phones' in pii_types:
            redacted = re.sub(self.PHONE_US_PATTERN, '[PHONE_REDACTED]', redacted)
            redacted = re.sub(self.PHONE_INTERNATIONAL_PATTERN, '[PHONE_REDACTED]', redacted)
        
        if 'ssns' in pii_types:
            redacted = re.sub(self.SSN_PATTERN, '[SSN_REDACTED]', redacted)
        
        if 'credit_cards' in pii_types:
            # Redact credit cards
            cc_matches = re.finditer(self.CC_PATTERN, redacted)
            for match in reversed(list(cc_matches)):  # Reverse to maintain positions
                cc = match.group(0).replace('-', '').replace(' ', '')
                if self._validate_credit_card(cc):
                    redacted = redacted[:match.start()] + '[CC_REDACTED]' + redacted[match.end():]
        
        if 'ip_addresses' in pii_types:
            redacted = re.sub(self.IP_PATTERN, '[IP_REDACTED]', redacted)
        
        if redacted != text:
            self.logger.debug(
                "PII redacted",
                pii_types=pii_types,
                original_length=len(text),
                redacted_length=len(redacted)
            )
        
        return redacted
    
    def _validate_credit_card(self, card_number: str) -> bool:
        """
        Validate credit card using Luhn algorithm.
        
        Args:
            card_number: Credit card number (digits only)
        
        Returns:
            True if valid credit card number
        """
        if not card_number or not card_number.isdigit():
            return False
        
        # Luhn algorithm
        def luhn_check(card_num: str) -> bool:
            def digits_of(n):
                return [int(d) for d in str(n)]
            
            digits = digits_of(card_num)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d * 2))
            return checksum % 10 == 0
        
        return luhn_check(card_number)

    def _find_valid_ips(self, text: str) -> List[str]:
        """
        Find valid IP addresses in text.

        Args:
            text: Text to search

        Returns:
            List of valid IP addresses
        """
        candidates = re.findall(self.IP_PATTERN, text)
        valid_ips = []
        for ip in candidates:
            parts = ip.split('.')
            if len(parts) == 4:
                try:
                    if all(0 <= int(part) <= 255 for part in parts):
                        # Exclude common false positives like version numbers
                        # but include private IP ranges that could be PII
                        valid_ips.append(ip)
                except ValueError:
                    continue
        return valid_ips

    def _find_phones(self, text: str) -> List[str]:
        """
        Find phone numbers in text, deduplicating US and international patterns.

        Args:
            text: Text to search

        Returns:
            List of phone number strings
        """
        us_matches = re.findall(self.PHONE_US_PATTERN, text)
        # US pattern returns tuple groups, join them
        us_phones = ['-'.join(match) if isinstance(match, tuple) else match for match in us_matches]

        intl_matches = re.findall(self.PHONE_INTERNATIONAL_PATTERN, text)

        # Deduplicate by converting to set
        all_phones = list(set(us_phones + intl_matches))
        return all_phones

