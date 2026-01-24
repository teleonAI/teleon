"""
Sentinel - Safety and Compliance System for Teleon Agents.

Sentinel provides production-grade safety controls for AI agents:
- Content filtering and moderation
- PII detection and redaction
- Compliance enforcement (GDPR, HIPAA, PCI_DSS, etc.)
- Custom policy enforcement
- Real-time blocking/flagging/redaction

Example:
    ```python
    from teleon.sentinel import SentinelEngine, SentinelConfig, ComplianceStandard, GuardrailAction
    
    # Create Sentinel engine
    config = SentinelConfig(
        enabled=True,
        content_filtering=True,
        pii_detection=True,
        compliance=[ComplianceStandard.GDPR, ComplianceStandard.HIPAA],
        action_on_violation=GuardrailAction.BLOCK
    )
    
    engine = SentinelEngine(config)
    
    # Validate input
    result = await engine.validate_input(data, agent_name="my-agent")
    if not result.passed:
        # Handle violation
        pass
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
]

