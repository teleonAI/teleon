"""
Heuristic PII Detection Backend.

International PII detection supporting US, UK, EU, and generic formats.
Structured PII (email, phone, SSN, credit card, IBAN, VAT) plus
semi-structured (passport, driver's license, DOB) and contextual (address, name).
"""

import re
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Structured PII patterns (high accuracy)
# ---------------------------------------------------------------------------

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

_PHONE_PATTERNS: Dict[str, re.Pattern] = {
    "us": re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"
    ),
    "uk": re.compile(r"\b(?:\+?44[-.\s]?)?(?:0?[0-9]{2,5})[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}\b"),
    "eu": re.compile(r"\b\+?(?:3[0-9]|4[0-9]|5[0-9]|6[0-9]|7[0-9]|8[0-9]|9[0-9])[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{3,10}\b"),
    "intl": re.compile(r"\b\+[1-9]\d{6,14}\b"),
}

_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

_CREDIT_CARD = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")

_IBAN = re.compile(
    r"\b[A-Z]{2}\d{2}[-\s]?[A-Z0-9]{4}[-\s]?(?:[A-Z0-9]{4}[-\s]?){1,7}[A-Z0-9]{1,4}\b"
)

_EU_VAT = re.compile(
    r"\b(?:AT|BE|BG|CY|CZ|DE|DK|EE|EL|ES|FI|FR|HR|HU|IE|IT|LT|LU|LV|MT|NL|PL|PT|RO|SE|SI|SK)"
    r"[A-Z0-9]{8,12}\b"
)

_IP_ADDR = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# ---------------------------------------------------------------------------
# Semi-structured PII patterns
# ---------------------------------------------------------------------------

_PASSPORT_PATTERNS: Dict[str, re.Pattern] = {
    "us": re.compile(r"\b[A-Z]\d{8}\b"),  # US passport: letter + 8 digits
    "eu": re.compile(r"\b[A-Z]{2}\d{6,7}\b"),  # Common EU format
}

_DRIVERS_LICENSE_PATTERNS: Dict[str, re.Pattern] = {
    "us": re.compile(r"\b[A-Z]\d{3,8}\b"),  # Simplified US DL (varies by state)
}

_DOB = re.compile(
    r"\b(?:"
    r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}"  # MM/DD/YYYY, DD/MM/YYYY, etc.
    r"|"
    r"\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}"  # YYYY-MM-DD
    r")\b"
)

# ---------------------------------------------------------------------------
# Contextual PII patterns
# ---------------------------------------------------------------------------

_ADDRESS_INDICATORS = re.compile(
    r"\b\d{1,5}\s+(?:"
    r"(?:North|South|East|West|N\.|S\.|E\.|W\.)\s+)?"
    r"(?:[A-Z][a-z]+\s+){0,3}"
    r"(?:Street|St\.?|Avenue|Ave\.?|Boulevard|Blvd\.?|Road|Rd\.?|Drive|Dr\.?"
    r"|Lane|Ln\.?|Court|Ct\.?|Place|Pl\.?|Way|Circle|Cir\.?|Terrace|Terr\.?"
    r"|Highway|Hwy\.?|Parkway|Pkwy\.?)\b",
    re.I,
)

_NAME_INDICATORS = re.compile(
    r"\b(?:Mr\.?|Mrs\.?|Ms\.?|Miss|Dr\.?|Prof\.?|Rev\.?)"
    r"\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b"
)


def _luhn_check(card_number: str) -> bool:
    """Validate credit card number using Luhn algorithm."""
    digits = [int(d) for d in card_number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    checksum = sum(odd_digits)
    for d in even_digits:
        doubled = d * 2
        checksum += doubled - 9 if doubled > 9 else doubled
    return checksum % 10 == 0


def _valid_ip(ip: str) -> bool:
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        return all(0 <= int(p) <= 255 for p in parts)
    except ValueError:
        return False


class HeuristicPIIDetector:
    """
    International heuristic PII detector.

    Implements PIIDetectionBackend protocol.
    """

    def __init__(
        self,
        locales: Optional[List[str]] = None,
        pii_types: Optional[List[str]] = None,
    ):
        """
        Args:
            locales: List of locale codes to enable (e.g. ["us", "uk", "eu"]).
                     Defaults to all.
            pii_types: List of PII types to detect. Defaults to all.
        """
        self.locales = locales or ["us", "uk", "eu", "intl"]
        self.pii_types = pii_types  # None = all

    def _should_detect(self, pii_type: str) -> bool:
        return self.pii_types is None or pii_type in self.pii_types

    def detect(self, text: str) -> Dict[str, List[str]]:
        if not text or not isinstance(text, str):
            return {}

        detected: Dict[str, List[str]] = {}

        # Email
        if self._should_detect("email"):
            emails = _EMAIL.findall(text)
            if emails:
                detected["email"] = emails

        # SSN (detect before phone to filter overlaps)
        ssn_spans: list = []
        if self._should_detect("ssn"):
            for m in _SSN.finditer(text):
                ssn_spans.append((m.start(), m.end()))
                detected.setdefault("ssn", []).append(m.group(0))

        # Phone numbers (deduplicated across locales, excluding SSN overlaps)
        if self._should_detect("phone"):
            phone_spans: list = []  # (start, end, text) for dedup
            for locale in self.locales:
                pattern = _PHONE_PATTERNS.get(locale)
                if pattern:
                    for m in pattern.finditer(text):
                        # Skip if this match overlaps with any SSN match
                        overlaps = any(
                            not (m.end() <= ss or m.start() >= se)
                            for ss, se in ssn_spans
                        )
                        if not overlaps:
                            phone_spans.append((m.start(), m.end(), m.group(0)))
            # Deduplicate: keep longest match when spans overlap
            phone_spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
            phones: list = []
            last_end = -1
            for start, end, val in phone_spans:
                if start >= last_end:
                    phones.append(val)
                    last_end = end
            if phones:
                detected["phone"] = phones

        # SSN (already detected above for overlap filtering)

        # Credit card (with Luhn validation)
        if self._should_detect("credit_card"):
            cc_matches = _CREDIT_CARD.findall(text)
            valid_ccs = []
            for cc in cc_matches:
                digits = cc.replace("-", "").replace(" ", "")
                if _luhn_check(digits):
                    valid_ccs.append(cc)
            if valid_ccs:
                detected["credit_card"] = valid_ccs

        # IBAN
        if self._should_detect("iban"):
            ibans = _IBAN.findall(text)
            if ibans:
                detected["iban"] = ibans

        # EU VAT
        if self._should_detect("eu_vat"):
            vats = _EU_VAT.findall(text)
            if vats:
                detected["eu_vat"] = vats

        # IP addresses
        if self._should_detect("ip_address"):
            ip_matches = _IP_ADDR.findall(text)
            valid_ips = [ip for ip in ip_matches if _valid_ip(ip)]
            if valid_ips:
                detected["ip_address"] = valid_ips

        # Passport
        if self._should_detect("passport"):
            passports: set = set()
            for locale in self.locales:
                pattern = _PASSPORT_PATTERNS.get(locale)
                if pattern:
                    for m in pattern.finditer(text):
                        passports.add(m.group(0))
            if passports:
                detected["passport"] = list(passports)

        # Date of birth
        if self._should_detect("date_of_birth"):
            dobs = _DOB.findall(text)
            if dobs:
                detected["date_of_birth"] = dobs

        # Address
        if self._should_detect("address"):
            addresses = [m.group(0) for m in _ADDRESS_INDICATORS.finditer(text)]
            if addresses:
                detected["address"] = addresses

        # Name (contextual)
        if self._should_detect("name"):
            names = [m.group(0) for m in _NAME_INDICATORS.finditer(text)]
            if names:
                detected["name"] = names

        return detected

    def redact(self, text: str, pii_types: Optional[List[str]] = None) -> str:
        if not text or not isinstance(text, str):
            return text

        types_to_redact = pii_types or [
            "email", "phone", "ssn", "credit_card", "iban", "eu_vat",
            "ip_address", "passport", "date_of_birth", "address", "name",
        ]

        redacted = text

        if "email" in types_to_redact:
            redacted = _EMAIL.sub("[EMAIL_REDACTED]", redacted)

        if "phone" in types_to_redact:
            for locale in self.locales:
                pattern = _PHONE_PATTERNS.get(locale)
                if pattern:
                    redacted = pattern.sub("[PHONE_REDACTED]", redacted)

        if "ssn" in types_to_redact:
            redacted = _SSN.sub("[SSN_REDACTED]", redacted)

        if "credit_card" in types_to_redact:
            for m in reversed(list(_CREDIT_CARD.finditer(redacted))):
                digits = m.group(0).replace("-", "").replace(" ", "")
                if _luhn_check(digits):
                    redacted = redacted[: m.start()] + "[CC_REDACTED]" + redacted[m.end():]

        if "iban" in types_to_redact:
            redacted = _IBAN.sub("[IBAN_REDACTED]", redacted)

        if "eu_vat" in types_to_redact:
            redacted = _EU_VAT.sub("[VAT_REDACTED]", redacted)

        if "ip_address" in types_to_redact:
            redacted = _IP_ADDR.sub("[IP_REDACTED]", redacted)

        if "passport" in types_to_redact:
            for locale in self.locales:
                pattern = _PASSPORT_PATTERNS.get(locale)
                if pattern:
                    redacted = pattern.sub("[PASSPORT_REDACTED]", redacted)

        if "date_of_birth" in types_to_redact:
            redacted = _DOB.sub("[DOB_REDACTED]", redacted)

        if "address" in types_to_redact:
            redacted = _ADDRESS_INDICATORS.sub("[ADDRESS_REDACTED]", redacted)

        if "name" in types_to_redact:
            redacted = _NAME_INDICATORS.sub("[NAME_REDACTED]", redacted)

        return redacted
