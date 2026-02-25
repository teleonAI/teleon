"""
Compliance Checker - Regulatory Compliance Enforcement.

Enforces compliance using structured rules per standard.
Accepts PII backend via dependency injection.
"""

from typing import Any, Dict, List, Optional
from teleon.sentinel.config import ComplianceStandard
from teleon.sentinel.backends.protocols import PIIDetectionBackend
from teleon.core import StructuredLogger, LogLevel


class ComplianceRule:
    """A single structured compliance rule."""

    def __init__(
        self,
        rule_id: str,
        description: str,
        severity: str,
        check_fn,
    ):
        self.rule_id = rule_id
        self.description = description
        self.severity = severity
        self.check_fn = check_fn

    def evaluate(self, text: str, pii: Dict[str, List[str]], metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate rule. Returns violation dict or None."""
        if self.check_fn(text, pii, metadata):
            return {
                "type": self.rule_id,
                "message": self.description,
                "severity": self.severity,
            }
        return None


def _has_pii(pii: Dict[str, List[str]]) -> bool:
    return bool(pii)


def _pii_count(pii: Dict[str, List[str]]) -> int:
    return sum(len(v) for v in pii.values())


def _metadata_missing(metadata: Dict[str, Any], key: str) -> bool:
    return key not in metadata or not metadata.get(key)


# ---------------------------------------------------------------------------
# Rules per standard
# ---------------------------------------------------------------------------

_GDPR_RULES = [
    ComplianceRule(
        "gdpr_data_minimization",
        "Excessive PII collection detected (>10 items)",
        "high",
        lambda text, pii, meta: _pii_count(pii) > 10,
    ),
    ComplianceRule(
        "gdpr_pii_encryption",
        "PII present without encryption context in metadata",
        "high",
        lambda text, pii, meta: _has_pii(pii) and _metadata_missing(meta, "encryption"),
    ),
    ComplianceRule(
        "gdpr_security",
        "Potential unencrypted sensitive data detected",
        "high",
        lambda text, pii, meta: ("password" in text.lower() or "secret" in text.lower())
        and _metadata_missing(meta, "encryption"),
    ),
]

_HIPAA_RULES = [
    ComplianceRule(
        "hipaa_phi_encryption",
        "PHI detected without encryption context",
        "critical",
        lambda text, pii, meta: any(
            ind in text.lower()
            for ind in [
                "medical record", "patient id", "diagnosis", "treatment",
                "prescription", "health insurance", "medical history",
                "symptoms", "physician", "hospital",
            ]
        ) and _metadata_missing(meta, "encryption"),
    ),
    ComplianceRule(
        "hipaa_access_control",
        "PHI detected without access_control field",
        "high",
        lambda text, pii, meta: any(
            ind in text.lower()
            for ind in [
                "medical record", "patient id", "diagnosis", "treatment",
                "prescription", "health insurance",
            ]
        ) and _metadata_missing(meta, "access_control"),
    ),
]

_PCI_DSS_RULES = [
    ComplianceRule(
        "pci_dss_card_encryption",
        "Credit card data detected without encryption/tokenization",
        "critical",
        lambda text, pii, meta: "credit_card" in pii
        and _metadata_missing(meta, "tokenization")
        and _metadata_missing(meta, "encryption"),
    ),
]

_SOC2_RULES = [
    ComplianceRule(
        "soc2_plaintext_password",
        "Potential plaintext password storage detected",
        "high",
        lambda text, pii, meta: "password" in text.lower() and "plaintext" in text.lower(),
    ),
]

_CCPA_RULES = [
    ComplianceRule(
        "ccpa_privacy_policy",
        "PII detected without privacy policy/opt-out indicators",
        "medium",
        lambda text, pii, meta: _has_pii(pii)
        and "privacy policy" not in text.lower()
        and "opt-out" not in text.lower(),
    ),
]

_STANDARD_RULES: Dict[ComplianceStandard, List[ComplianceRule]] = {
    ComplianceStandard.GDPR: _GDPR_RULES,
    ComplianceStandard.HIPAA: _HIPAA_RULES,
    ComplianceStandard.PCI_DSS: _PCI_DSS_RULES,
    ComplianceStandard.SOC2: _SOC2_RULES,
    ComplianceStandard.CCPA: _CCPA_RULES,
}


class ComplianceChecker:
    """
    Compliance rule enforcement using structured rules.

    Accepts a PII backend via dependency injection instead of creating its own.
    """

    def __init__(
        self,
        standards: List[ComplianceStandard],
        pii_backend: Optional[PIIDetectionBackend] = None,
    ):
        self.standards = standards
        self.logger = StructuredLogger("compliance_checker", LogLevel.INFO)

        if pii_backend is not None:
            self._pii_backend = pii_backend
        else:
            from teleon.sentinel.backends.pii_heuristic import HeuristicPIIDetector
            self._pii_backend = HeuristicPIIDetector()

    def check_all(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check all enabled compliance standards."""
        text = self._extract_text(data)
        pii = self._pii_backend.detect(text)
        metadata = data if isinstance(data, dict) else {}

        all_violations: List[Dict[str, Any]] = []
        for standard in self.standards:
            rules = _STANDARD_RULES.get(standard, [])
            for rule in rules:
                violation = rule.evaluate(text, pii, metadata)
                if violation:
                    all_violations.append(violation)

        if all_violations:
            self.logger.debug(
                "Compliance violations detected",
                standards=[s.value for s in self.standards],
                violation_count=len(all_violations),
            )

        return all_violations

    # Keep individual check methods for backward compat
    def check_gdpr(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self._check_standard(ComplianceStandard.GDPR, data)

    def check_hipaa(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self._check_standard(ComplianceStandard.HIPAA, data)

    def check_pci_dss(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self._check_standard(ComplianceStandard.PCI_DSS, data)

    def check_soc2(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self._check_standard(ComplianceStandard.SOC2, data)

    def check_ccpa(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self._check_standard(ComplianceStandard.CCPA, data)

    def _check_standard(self, standard: ComplianceStandard, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = self._extract_text(data)
        pii = self._pii_backend.detect(text)
        metadata = data if isinstance(data, dict) else {}
        violations = []
        for rule in _STANDARD_RULES.get(standard, []):
            violation = rule.evaluate(text, pii, metadata)
            if violation:
                violations.append(violation)
        return violations

    def _extract_text(self, data: Any) -> str:
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            texts = []
            for value in data.values():
                texts.append(self._extract_text(value))
            return " ".join(texts)
        elif isinstance(data, list):
            texts = [self._extract_text(item) for item in data]
            return " ".join(texts)
        else:
            return str(data)
