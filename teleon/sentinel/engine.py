"""
Sentinel Engine - Core Safety and Compliance Validation Engine.

Main engine that orchestrates all Sentinel components with pluggable backends.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from teleon.sentinel.config import (
    SentinelConfig,
    GuardrailResult,
    GuardrailAction,
)
from teleon.sentinel.backends.protocols import (
    ContentModerationBackend,
    PIIDetectionBackend,
    PromptInjectionBackend,
)
from teleon.sentinel.compliance_checker import ComplianceChecker
from teleon.sentinel.audit import SentinelAuditLogger
from teleon.core.exceptions import AgentValidationError
from teleon.core import StructuredLogger, LogLevel


class SentinelEngine:
    """
    Main Sentinel enforcement engine.

    Validates inputs and outputs before/after agent execution.
    Orchestrates content moderation, PII detection, prompt injection detection,
    compliance checks, custom policies, and tool guardrails via pluggable backends.
    """

    def __init__(self, config: SentinelConfig):
        self.config = config
        self.logger = StructuredLogger("sentinel", LogLevel.INFO)

        # Initialize pluggable backends
        self._content_backend: Optional[ContentModerationBackend] = None
        if config.content_filtering:
            self._content_backend = self._init_content_backend()

        self._pii_backend: Optional[PIIDetectionBackend] = None
        if config.pii_detection:
            self._pii_backend = self._init_pii_backend()

        self._injection_backend: Optional[PromptInjectionBackend] = None
        if config.prompt_injection_detection:
            self._injection_backend = self._init_injection_backend()

        self.compliance_checker: Optional[ComplianceChecker] = None
        if config.compliance:
            self.compliance_checker = ComplianceChecker(
                config.compliance, pii_backend=self._pii_backend
            )

        # Policy DSL evaluator + parsed policies
        self._policy_evaluator = None
        self._policies: List[Any] = []
        self._init_policy_system()

        # Legacy PolicyEngine for backward compat with custom_policies list
        self._legacy_policy_engine = None
        if config.custom_policies:
            from teleon.sentinel.policy_engine import PolicyEngine
            self._legacy_policy_engine = PolicyEngine()

        # Tool guardrail (lazy: created when get_tool_guardrail() is called)
        self._tool_guardrail = None

        # Audit logger
        self.audit_logger: Optional[SentinelAuditLogger] = None
        if config.audit_enabled:
            self.audit_logger = SentinelAuditLogger()

    # ------------------------------------------------------------------
    # Backend initialization
    # ------------------------------------------------------------------

    def _init_content_backend(self) -> ContentModerationBackend:
        from teleon.sentinel.backends import get_content_backend
        backend_type = self.config.content_backend or "heuristic"
        return get_content_backend(
            backend_type=backend_type,
            threshold=self.config.moderation_threshold,
            language=self.config.language,
            additional_languages=self.config.additional_languages,
        )

    def _init_pii_backend(self) -> PIIDetectionBackend:
        from teleon.sentinel.backends import get_pii_backend
        backend_type = self.config.pii_backend or "heuristic"
        return get_pii_backend(backend_type=backend_type)

    def _init_injection_backend(self) -> PromptInjectionBackend:
        from teleon.sentinel.backends import get_injection_backend
        backend_type = self.config.injection_backend or "heuristic"
        return get_injection_backend(
            backend_type=backend_type,
            threshold=self.config.injection_threshold,
        )

    def _init_policy_system(self) -> None:
        """Initialize the DSL-based policy system."""
        from teleon.sentinel.policy_dsl.evaluator import SafeEvaluator
        from teleon.sentinel.policy_dsl.parser import PolicyParser

        self._policy_evaluator = SafeEvaluator()

        # Load from file
        if self.config.policy_file:
            try:
                self._policies.extend(PolicyParser.parse_file(self.config.policy_file))
            except Exception as e:
                self.logger.warning(f"Failed to load policy file: {e}")

        # Load inline definitions
        if self.config.policy_definitions:
            try:
                self._policies.extend(PolicyParser.parse_dict(self.config.policy_definitions))
            except Exception as e:
                self.logger.warning(f"Failed to parse policy definitions: {e}")

    # ------------------------------------------------------------------
    # Unified validation
    # ------------------------------------------------------------------

    async def _validate(
        self,
        data: Any,
        agent_name: str,
        validation_type: str,
    ) -> GuardrailResult:
        """Unified validation for input, output, and tool_call."""
        if not self.config.enabled:
            return GuardrailResult(passed=True, action=GuardrailAction.FLAG)

        violations: List[Dict[str, Any]] = []
        redacted_content = None
        text = self._extract_text(data)

        # 1. Prompt injection detection (input only)
        if self._injection_backend and validation_type == "input":
            detected, score, categories = self._injection_backend.detect(text)
            if detected:
                violations.append({
                    "type": "prompt_injection",
                    "score": score,
                    "categories": categories,
                    "message": f"Prompt injection detected (score={score:.2f}, categories={categories})",
                    "severity": "critical",
                })

        # 2. Content filtering
        if self._content_backend and self.config.content_filtering:
            checks = self._content_backend.check_all(text)
            for category in ("toxicity", "hate_speech", "profanity", "threat", "sexual"):
                result = checks.get(category, {})
                if result.get("detected"):
                    violations.append({
                        "type": category,
                        "score": result.get("score", 0.0),
                        "message": f"{category.replace('_', ' ').title()} detected in {validation_type}",
                    })

        # 3. PII detection
        if self._pii_backend and self.config.pii_detection:
            detected_pii = self._pii_backend.detect(text)
            if detected_pii:
                violations.append({
                    "type": "pii_detection",
                    "pii_types": list(detected_pii.keys()),
                    "message": f"PII detected in {validation_type}: {list(detected_pii.keys())}",
                })
                if self.config.action_on_violation == GuardrailAction.REDACT:
                    redacted_content = self._pii_backend.redact(text)

        # 4. Compliance checks
        if self.compliance_checker and self.config.compliance:
            compliance_violations = self.compliance_checker.check_all(
                {"text": text, validation_type: data}
            )
            violations.extend(compliance_violations)

        # 5. DSL policies
        if self._policy_evaluator and self._policies:
            from teleon.sentinel.policy_dsl.models import EvaluationContext
            ctx = EvaluationContext(
                text=text,
                data=data if isinstance(data, dict) else {"raw": data},
                agent_name=agent_name,
                validation_type=validation_type,
            )
            policy_violations = self._policy_evaluator.evaluate_policies(self._policies, ctx)
            violations.extend(policy_violations)

        # 6. Legacy custom policies
        if self._legacy_policy_engine and self.config.custom_policies:
            for policy_name in self.config.custom_policies:
                policy_violations = self._legacy_policy_engine.evaluate_policy(policy_name, data)
                violations.extend(policy_violations)

        # Determine action
        passed = len(violations) == 0
        action = GuardrailAction.FLAG if passed else self.config.action_on_violation

        result = GuardrailResult(
            passed=passed,
            action=action,
            violations=violations,
            redacted_content=redacted_content,
            metadata={
                "agent_name": agent_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "validation_type": validation_type,
            },
        )

        # Log violations
        if violations and self.config.log_violations:
            self.logger.warning(
                f"Guardrail violations detected in {validation_type}",
                agent_name=agent_name,
                violations=violations,
                action=action.value,
            )

        # Audit logging
        if violations and self.audit_logger:
            for violation in violations:
                self.audit_logger.log_violation(
                    agent_id=agent_name,
                    violation_type=violation.get("type", "unknown"),
                    action_taken=action.value,
                    details=violation,
                    validation_type=validation_type,
                )

        # Block if action is BLOCK
        if not passed and action == GuardrailAction.BLOCK:
            raise AgentValidationError(
                f"Guardrail violation in {validation_type}: {violations[0]['message']}",
                {"violations": violations, "agent": agent_name},
            )

        return result

    async def validate_input(self, input_data: Any, agent_name: str) -> GuardrailResult:
        """Validate agent input before execution."""
        return await self._validate(input_data, agent_name, "input")

    async def validate_output(self, output_data: Any, agent_name: str) -> GuardrailResult:
        """Validate agent output after execution."""
        return await self._validate(output_data, agent_name, "output")

    async def validate_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        agent_name: str,
    ) -> GuardrailResult:
        """Validate a tool call before invocation."""
        if not self.config.enabled:
            return GuardrailResult(passed=True, action=GuardrailAction.FLAG)

        violations: List[Dict[str, Any]] = []

        # Check tool allowlist/blocklist from config
        if self.config.allowed_tools is not None:
            if tool_name not in self.config.allowed_tools:
                violations.append({
                    "type": "tool_not_allowed",
                    "tool_name": tool_name,
                    "message": f"Tool '{tool_name}' is not in the allowlist",
                    "severity": "critical",
                })

        if self.config.blocked_tools is not None:
            if tool_name in self.config.blocked_tools:
                violations.append({
                    "type": "tool_blocked",
                    "tool_name": tool_name,
                    "message": f"Tool '{tool_name}' is blocked",
                    "severity": "critical",
                })

        # Evaluate DSL policies targeting tool_call
        if self._policy_evaluator and self._policies:
            from teleon.sentinel.policy_dsl.models import EvaluationContext
            ctx = EvaluationContext(
                text=str(tool_args),
                data=tool_args,
                agent_name=agent_name,
                validation_type="tool_call",
                tool_name=tool_name,
                tool_args=tool_args,
            )
            policy_violations = self._policy_evaluator.evaluate_policies(self._policies, ctx)
            violations.extend(policy_violations)

        passed = len(violations) == 0
        action = GuardrailAction.FLAG if passed else self.config.action_on_violation

        result = GuardrailResult(
            passed=passed,
            action=action,
            violations=violations,
            metadata={
                "agent_name": agent_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "validation_type": "tool_call",
                "tool_name": tool_name,
            },
        )

        if violations and self.config.log_violations:
            self.logger.warning(
                "Tool call guardrail violation",
                agent_name=agent_name,
                tool_name=tool_name,
                violations=violations,
                action=action.value,
            )

        if violations and self.audit_logger:
            for violation in violations:
                self.audit_logger.log_violation(
                    agent_id=agent_name,
                    violation_type=violation.get("type", "unknown"),
                    action_taken=action.value,
                    details={**violation, "tool_name": tool_name, "tool_args": tool_args},
                    validation_type="tool_call",
                )

        if not passed and action == GuardrailAction.BLOCK:
            raise AgentValidationError(
                f"Tool guardrail violation: {violations[0]['message']}",
                {"violations": violations, "agent": agent_name, "tool": tool_name},
            )

        return result

    def get_tool_guardrail(self):
        """Get tool guardrail wrapper if tool guardrails are enabled."""
        if not self.config.tool_guardrails:
            return None
        if self._tool_guardrail is None:
            from teleon.sentinel.tool_guardrails import ToolGuardrail
            self._tool_guardrail = ToolGuardrail(self)
        return self._tool_guardrail

    def _extract_text(self, data: Any) -> str:
        """Extract text from various data types."""
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

    def get_audit_logger(self) -> Optional[SentinelAuditLogger]:
        """Get audit logger instance."""
        return self.audit_logger
