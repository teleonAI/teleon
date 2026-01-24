"""
Sentinel Engine - Core Safety and Compliance Validation Engine.

Main engine that orchestrates all Sentinel components.
"""

from typing import Any, Optional
from datetime import datetime, timezone
from teleon.sentinel.config import (
    SentinelConfig,
    GuardrailResult,
    GuardrailAction,
)
from teleon.sentinel.content_moderator import ContentModerator
from teleon.sentinel.pii_detector import PIIDetector
from teleon.sentinel.compliance_checker import ComplianceChecker
from teleon.sentinel.policy_engine import PolicyEngine
from teleon.sentinel.audit import SentinelAuditLogger
from teleon.core.exceptions import AgentValidationError
from teleon.core import StructuredLogger, LogLevel


class SentinelEngine:
    """
    Main Sentinel enforcement engine.
    
    Validates inputs and outputs before/after agent execution.
    Orchestrates content moderation, PII detection, compliance checks, and custom policies.
    """
    
    def __init__(self, config: SentinelConfig):
        """
        Initialize Sentinel engine.
        
        Args:
            config: Sentinel configuration
        """
        self.config = config
        self.logger = StructuredLogger("sentinel", LogLevel.INFO)
        
        # Initialize components (lazy loading)
        self.content_moderator: Optional[ContentModerator] = None
        if config.content_filtering:
            self.content_moderator = ContentModerator(
                threshold=config.moderation_threshold
            )
        
        self.pii_detector: Optional[PIIDetector] = None
        if config.pii_detection:
            self.pii_detector = PIIDetector()
        
        self.compliance_checker: Optional[ComplianceChecker] = None
        if config.compliance:
            self.compliance_checker = ComplianceChecker(config.compliance)
        
        self.policy_engine: Optional[PolicyEngine] = None
        if config.custom_policies:
            self.policy_engine = PolicyEngine()
        
        self.audit_logger: Optional[SentinelAuditLogger] = None
        if config.audit_enabled:
            self.audit_logger = SentinelAuditLogger()
    
    async def validate_input(
        self,
        input_data: Any,
        agent_name: str
    ) -> GuardrailResult:
        """
        Validate agent input before execution.
        
        Args:
            input_data: Input to validate
            agent_name: Agent name for logging
        
        Returns:
            GuardrailResult with validation status
        
        Raises:
            AgentValidationError: If validation fails and action is BLOCK
        """
        if not self.config.enabled:
            return GuardrailResult(passed=True, action=GuardrailAction.FLAG)
        
        violations = []
        redacted_content = None
        
        # Convert input to string for analysis
        input_text = self._extract_text(input_data)
        
        # Content filtering
        if self.content_moderator and self.config.content_filtering:
            checks = self.content_moderator.check_all(input_text)
            
            if checks['toxicity']['detected']:
                violations.append({
                    'type': 'toxicity',
                    'score': checks['toxicity']['score'],
                    'message': 'Toxic content detected in input'
                })
            
            if checks['hate_speech']['detected']:
                violations.append({
                    'type': 'hate_speech',
                    'score': checks['hate_speech']['score'],
                    'message': 'Hate speech detected in input'
                })
        
        # PII detection
        if self.pii_detector and self.config.pii_detection:
            detected_pii = self.pii_detector.detect(input_text)
            if detected_pii:
                violations.append({
                    'type': 'pii_detection',
                    'pii_types': list(detected_pii.keys()),
                    'message': f'PII detected: {list(detected_pii.keys())}'
                })
                
                # Redact if action is REDACT
                if self.config.action_on_violation == GuardrailAction.REDACT:
                    redacted_content = self.pii_detector.redact(input_text)
        
        # Compliance checks
        if self.compliance_checker and self.config.compliance:
            compliance_violations = self.compliance_checker.check_all(
                {'input': input_data, 'text': input_text}
            )
            violations.extend(compliance_violations)
        
        # Custom policies
        if self.policy_engine and self.config.custom_policies:
            for policy_name in self.config.custom_policies:
                policy_violations = self.policy_engine.evaluate_policy(
                    policy_name,
                    input_data
                )
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
                'agent_name': agent_name,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'validation_type': 'input'
            }
        )
        
        # Log violations
        if violations and self.config.log_violations:
            self.logger.warning(
                "Guardrail violations detected in input",
                agent_name=agent_name,
                violations=violations,
                action=action.value
            )
        
        # Audit logging
        if violations and self.audit_logger:
            for violation in violations:
                self.audit_logger.log_violation(
                    agent_id=agent_name,
                    violation_type=violation.get('type', 'unknown'),
                    action_taken=action.value,
                    details=violation
                )
        
        # Block if action is BLOCK
        if not passed and action == GuardrailAction.BLOCK:
            raise AgentValidationError(
                f"Guardrail violation: {violations[0]['message']}",
                {"violations": violations, "agent": agent_name}
            )
        
        return result
    
    async def validate_output(
        self,
        output_data: Any,
        agent_name: str
    ) -> GuardrailResult:
        """
        Validate agent output after execution.
        
        Args:
            output_data: Output to validate
            agent_name: Agent name for logging
        
        Returns:
            GuardrailResult with validation status
        
        Raises:
            AgentValidationError: If validation fails and action is BLOCK
        """
        if not self.config.enabled:
            return GuardrailResult(passed=True, action=GuardrailAction.FLAG)
        
        violations = []
        redacted_content = None
        
        # Convert output to string for analysis
        output_text = self._extract_text(output_data)
        
        # Content filtering
        if self.content_moderator and self.config.content_filtering:
            checks = self.content_moderator.check_all(output_text)
            
            if checks['toxicity']['detected']:
                violations.append({
                    'type': 'toxicity',
                    'score': checks['toxicity']['score'],
                    'message': 'Toxic content detected in output'
                })
            
            if checks['hate_speech']['detected']:
                violations.append({
                    'type': 'hate_speech',
                    'score': checks['hate_speech']['score'],
                    'message': 'Hate speech detected in output'
                })
        
        # PII detection
        if self.pii_detector and self.config.pii_detection:
            detected_pii = self.pii_detector.detect(output_text)
            if detected_pii:
                violations.append({
                    'type': 'pii_detection',
                    'pii_types': list(detected_pii.keys()),
                    'message': f'PII detected in output: {list(detected_pii.keys())}'
                })
                
                # Redact if action is REDACT
                if self.config.action_on_violation == GuardrailAction.REDACT:
                    redacted_content = self.pii_detector.redact(output_text)
        
        # Compliance checks
        if self.compliance_checker and self.config.compliance:
            compliance_violations = self.compliance_checker.check_all(
                {'output': output_data, 'text': output_text}
            )
            violations.extend(compliance_violations)
        
        # Custom policies
        if self.policy_engine and self.config.custom_policies:
            for policy_name in self.config.custom_policies:
                policy_violations = self.policy_engine.evaluate_policy(
                    policy_name,
                    output_data
                )
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
                'agent_name': agent_name,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'validation_type': 'output'
            }
        )
        
        # Log violations
        if violations and self.config.log_violations:
            self.logger.warning(
                "Guardrail violations in output",
                agent_name=agent_name,
                violations=violations,
                action=action.value
            )
        
        # Audit logging
        if violations and self.audit_logger:
            for violation in violations:
                self.audit_logger.log_violation(
                    agent_id=agent_name,
                    violation_type=violation.get('type', 'unknown'),
                    action_taken=action.value,
                    details=violation
                )
        
        # Block if action is BLOCK
        if not passed and action == GuardrailAction.BLOCK:
            raise AgentValidationError(
                f"Guardrail violation in output: {violations[0]['message']}",
                {"violations": violations, "agent": agent_name}
            )
        
        return result
    
    def _extract_text(self, data: Any) -> str:
        """
        Extract text from various data types.
        
        Args:
            data: Data to extract text from
        
        Returns:
            Extracted text string
        """
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            # Recursively extract text from dict
            texts = []
            for value in data.values():
                texts.append(self._extract_text(value))
            return ' '.join(texts)
        elif isinstance(data, list):
            texts = [self._extract_text(item) for item in data]
            return ' '.join(texts)
        else:
            return str(data)
    
    def get_audit_logger(self) -> Optional[SentinelAuditLogger]:
        """Get audit logger instance."""
        return self.audit_logger

