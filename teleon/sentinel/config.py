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

    # Prompt injection detection
    prompt_injection_detection: bool = Field(False, description="Enable prompt injection detection")
    injection_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Prompt injection detection threshold")

    # Multi-language support
    language: str = Field("en", description="Primary language for detection")
    additional_languages: Optional[List[str]] = Field(None, description="Extra languages to detect")

    # Backend selection (future-proofs ML tier)
    content_backend: Optional[str] = Field(None, description="Content backend: 'heuristic' (default) or 'ml' (future)")
    pii_backend: Optional[str] = Field(None, description="PII backend: 'heuristic' (default) or 'ml' (future)")
    injection_backend: Optional[str] = Field(None, description="Injection backend: 'heuristic' (default) or 'ml' (future)")

    # Policy DSL
    policy_file: Optional[str] = Field(None, description="Path to YAML/JSON policy file")
    policy_definitions: Optional[List[Dict[str, Any]]] = Field(None, description="Inline policy definitions")

    # Tool guardrails
    tool_guardrails: bool = Field(False, description="Enable tool call guardrails")
    allowed_tools: Optional[List[str]] = Field(None, description="Allowlist of tool names")
    blocked_tools: Optional[List[str]] = Field(None, description="Blocklist of tool names")

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
            "prompt_injection_detection": self.prompt_injection_detection,
            "injection_threshold": self.injection_threshold,
            "language": self.language,
            "additional_languages": self.additional_languages,
            "content_backend": self.content_backend,
            "pii_backend": self.pii_backend,
            "injection_backend": self.injection_backend,
            "policy_file": self.policy_file,
            "policy_definitions": self.policy_definitions,
            "tool_guardrails": self.tool_guardrails,
            "allowed_tools": self.allowed_tools,
            "blocked_tools": self.blocked_tools,
        }
