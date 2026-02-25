"""
Sentinel Integration Layer - Bridges TeleonClient and SentinelEngine.

This module provides the integration between user-facing agent decorators
and the Sentinel safety and compliance system.
"""

from typing import Any, Dict, Optional
from teleon.sentinel.config import SentinelConfig, GuardrailAction
from teleon.sentinel.engine import SentinelEngine
from teleon.sentinel.registry import get_sentinel_registry
from teleon.core import StructuredLogger, LogLevel


class SentinelConfigDict(Dict):
    """
    Configuration for Sentinel safety features.

    Used in @agent decorator to enable safety features:

    Example:
        ```python
        @client.agent(
            name="my-agent",
            sentinel={
                'enabled': True,
                'content_filtering': True,
                'pii_detection': True,
                'prompt_injection_detection': True,
                'compliance': ['gdpr', 'hipaa'],
                'action_on_violation': 'block',
                'tool_guardrails': True,
                'language': 'en',
                'additional_languages': ['es', 'fr'],
                'policy_file': 'policies.yaml',
            }
        )
        async def my_agent(input: str):
            return process(input)
        ```
    """
    pass


async def register_agent_with_sentinel(
    agent_id: str,
    sentinel_config: Dict[str, Any],
    agent_name: Optional[str] = None
) -> SentinelEngine:
    """Register an agent with Sentinel safety system."""
    parsed_config = parse_sentinel_config(sentinel_config)
    config = SentinelConfig(**parsed_config)
    engine = SentinelEngine(config)

    registry = await get_sentinel_registry()
    await registry.register(agent_id, engine)

    logger = StructuredLogger("sentinel.integration", LogLevel.INFO)
    logger.info(
        "Sentinel registered for agent",
        agent_id=agent_id,
        agent_name=agent_name,
        enabled=config.enabled
    )

    return engine


def parse_sentinel_config(sentinel_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse and validate Sentinel configuration from decorator."""
    if not sentinel_config:
        return {}

    # Handle boolean shorthand
    if isinstance(sentinel_config, bool):
        return {"enabled": sentinel_config}

    normalized = {}

    for key, value in sentinel_config.items():
        normalized[key] = value

    # Normalize compliance standards
    if "compliance" in normalized:
        compliance = normalized["compliance"]
        if isinstance(compliance, list):
            from teleon.sentinel.config import ComplianceStandard
            normalized_compliance = []
            for item in compliance:
                if isinstance(item, str):
                    try:
                        normalized_compliance.append(ComplianceStandard(item.lower()))
                    except ValueError:
                        compliance_logger = StructuredLogger("sentinel.integration", LogLevel.WARNING)
                        compliance_logger.warning(f"Invalid compliance standard: {item}")
                else:
                    normalized_compliance.append(item)
            normalized["compliance"] = normalized_compliance

    # Normalize action_on_violation
    if "action_on_violation" in normalized:
        action = normalized["action_on_violation"]
        if isinstance(action, str):
            try:
                normalized["action_on_violation"] = GuardrailAction(action.lower())
            except ValueError:
                action_logger = StructuredLogger("sentinel.integration", LogLevel.WARNING)
                action_logger.warning(f"Invalid action: {action}, using default BLOCK")
                normalized["action_on_violation"] = GuardrailAction.BLOCK

    # Set defaults
    defaults = {
        "enabled": True,
        "content_filtering": False,
        "pii_detection": False,
        "compliance": [],
        "custom_policies": [],
        "moderation_threshold": 0.8,
        "action_on_violation": GuardrailAction.BLOCK,
        "log_violations": True,
        "audit_enabled": True,
        "prompt_injection_detection": False,
        "injection_threshold": 0.8,
        "language": "en",
        "additional_languages": None,
        "content_backend": None,
        "pii_backend": None,
        "injection_backend": None,
        "policy_file": None,
        "policy_definitions": None,
        "tool_guardrails": False,
        "allowed_tools": None,
        "blocked_tools": None,
    }

    for key, default_value in defaults.items():
        if key not in normalized:
            normalized[key] = default_value

    # Validate ranges
    if "moderation_threshold" in normalized:
        threshold = normalized["moderation_threshold"]
        if not isinstance(threshold, (int, float)) or threshold < 0.0 or threshold > 1.0:
            threshold_logger = StructuredLogger("sentinel.integration", LogLevel.WARNING)
            threshold_logger.warning(f"Invalid moderation_threshold: {threshold}, using default 0.8")
            normalized["moderation_threshold"] = 0.8

    if "injection_threshold" in normalized:
        threshold = normalized["injection_threshold"]
        if not isinstance(threshold, (int, float)) or threshold < 0.0 or threshold > 1.0:
            normalized["injection_threshold"] = 0.8

    return normalized


def create_sentinel_engine(
    sentinel_config: Optional[Dict[str, Any]] = None,
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None
) -> Optional[SentinelEngine]:
    """Create a Sentinel engine from configuration."""
    if not sentinel_config:
        return None

    parsed_config = parse_sentinel_config(sentinel_config)
    config = SentinelConfig(**parsed_config)

    if not config.enabled:
        return None

    # Create dedicated Sentinel violation persistence if agent info provided
    violation_persistence = None
    if agent_id and config.audit_enabled:
        try:
            from teleon.sentinel.persistence import SentinelViolationPersistence
            violation_persistence = SentinelViolationPersistence(
                agent_id=agent_id,
                agent_name=agent_name or agent_id
            )
        except Exception as e:
            logger = StructuredLogger("sentinel.integration", LogLevel.WARNING)
            logger.warning(f"Failed to create Sentinel violation persistence: {e}")

    # Create SentinelAuditLogger with dedicated persistence layer
    from teleon.sentinel.audit import SentinelAuditLogger
    sentinel_audit_logger = SentinelAuditLogger(violation_persistence=violation_persistence)

    # Create engine and inject the audit logger
    engine = SentinelEngine(config)
    engine.audit_logger = sentinel_audit_logger

    return engine
