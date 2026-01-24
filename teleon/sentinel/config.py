"""
Sentinel Configuration Models.

Defines configuration structures for Sentinel safety and compliance system.
"""

from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class ComplianceStandard(str, Enum):
    """Supported compliance standards."""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    SOC2 = "soc2"
    CCPA = "ccpa"


class GuardrailAction(str, Enum):
    """Actions to take when guardrail is triggered."""
    BLOCK = "block"  # Block execution
    FLAG = "flag"  # Flag but allow
    REDACT = "redact"  # Redact sensitive content
    ESCALATE = "escalate"  # Escalate to human


class GuardrailResult(BaseModel):
    """Result of guardrail check."""

    passed: bool = Field(..., description="Whether validation passed")
    action: GuardrailAction = Field(GuardrailAction.FLAG, description="Action taken")
    violations: List[Dict[str, Any]] = Field(default_factory=list, description="List of violations found")
    redacted_content: Optional[str] = Field(None, description="Redacted content if action is REDACT")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = ConfigDict()


class SentinelConfig(BaseModel):
    """Configuration for Sentinel safety and compliance system."""
    
    enabled: bool = Field(True, description="Enable Sentinel validation")
    content_filtering: bool = Field(False, description="Enable content moderation")
    pii_detection: bool = Field(False, description="Enable PII detection and redaction")
    compliance: List[ComplianceStandard] = Field(default_factory=list, description="Compliance standards to enforce")
    custom_policies: List[str] = Field(default_factory=list, description="Custom policy names to enforce")
    moderation_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Content moderation threshold (0.0-1.0)")
    action_on_violation: GuardrailAction = Field(GuardrailAction.BLOCK, description="Action to take on violation")
    log_violations: bool = Field(True, description="Log all violations")
    audit_enabled: bool = Field(True, description="Enable audit logging")

    model_config = ConfigDict()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "enabled": self.enabled,
            "content_filtering": self.content_filtering,
            "pii_detection": self.pii_detection,
            "compliance": [c.value for c in self.compliance],
            "custom_policies": self.custom_policies,
            "moderation_threshold": self.moderation_threshold,
            "action_on_violation": self.action_on_violation.value,
            "log_violations": self.log_violations,
            "audit_enabled": self.audit_enabled,
        }

