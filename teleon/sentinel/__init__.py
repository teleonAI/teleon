"""
Sentinel - Safety and Compliance System for Teleon Agents.

Sentinel provides production-grade safety controls for AI agents:
- Content filtering and moderation (multi-language, 6+ languages)
- PII detection and redaction (international formats)
- Prompt injection detection
- Compliance enforcement (GDPR, HIPAA, PCI_DSS, etc.)
- Custom policy enforcement via YAML/JSON DSL (no eval())
- Tool call guardrails
- Real-time blocking/flagging/redaction

Example:
    ```python
    from teleon.sentinel import SentinelEngine, SentinelConfig, ComplianceStandard, GuardrailAction

    config = SentinelConfig(
        enabled=True,
        content_filtering=True,
        pii_detection=True,
        prompt_injection_detection=True,
        compliance=[ComplianceStandard.GDPR, ComplianceStandard.HIPAA],
        action_on_violation=GuardrailAction.BLOCK,
        tool_guardrails=True,
        language="en",
        additional_languages=["es", "fr"],
        policy_file="policies.yaml",
    )

    engine = SentinelEngine(config)

    result = await engine.validate_input(data, agent_name="my-agent")
    if not result.passed:
        pass  # Handle violation
    ```
"""

from teleon.sentinel.config import (
    SentinelConfig,
    GuardrailResult,
    ComplianceStandard,
    GuardrailAction,
)
from teleon.sentinel.engine import SentinelEngine
from teleon.sentinel.integration import (
    register_agent_with_sentinel,
    parse_sentinel_config,
    create_sentinel_engine,
)
from teleon.sentinel.registry import (
    SentinelRegistry,
    get_sentinel_registry,
)
from teleon.sentinel.tool_guardrails import ToolGuardrail
from teleon.sentinel.policy_dsl.models import PolicyDefinition, PolicyRule, RuleType, EvaluationContext
from teleon.sentinel.policy_dsl.parser import PolicyParser
from teleon.sentinel.policy_dsl.evaluator import SafeEvaluator

__all__ = [
    # Configuration
    "SentinelConfig",
    "GuardrailResult",
    "ComplianceStandard",
    "GuardrailAction",

    # Core Engine
    "SentinelEngine",

    # Integration
    "register_agent_with_sentinel",
    "parse_sentinel_config",
    "create_sentinel_engine",

    # Registry
    "SentinelRegistry",
    "get_sentinel_registry",

    # Tool Guardrails
    "ToolGuardrail",

    # Policy DSL
    "PolicyDefinition",
    "PolicyRule",
    "PolicyParser",
    "SafeEvaluator",
    "RuleType",
    "EvaluationContext",
]
