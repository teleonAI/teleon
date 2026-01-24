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
                'compliance': ['gdpr', 'hipaa'],
                'action_on_violation': 'block'
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
    """
    Register an agent with Sentinel safety system.
    
    This creates the bridge between a decorated agent function
    and the Sentinel safety system.
    
    Args:
        agent_id: Unique agent identifier
        sentinel_config: Sentinel configuration from decorator
        agent_name: Optional agent name for logging
    
    Returns:
        SentinelEngine instance
    
    Example:
        ```python
        engine = await register_agent_with_sentinel(
            agent_id="agent-123",
            sentinel_config={
                'enabled': True,
                'content_filtering': True,
                'pii_detection': True
            }
        )
        
        # Agent is now protected by Sentinel
        ```
    """
    # Parse and normalize config
    parsed_config = parse_sentinel_config(sentinel_config)
    
    # Create Sentinel config
    config = SentinelConfig(**parsed_config)
    
    # Create Sentinel engine
    engine = SentinelEngine(config)
    
    # Register with global registry
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
    """
    Parse and validate Sentinel configuration from decorator.
    
    Args:
        sentinel_config: Raw Sentinel config from @agent decorator
    
    Returns:
        Validated and normalized configuration
    
    Example:
        ```python
        # Shorthand
        config = parse_sentinel_config({'content_filtering': True})
        # Returns: {'enabled': True, 'content_filtering': True, ...}
        
        # Full form
        config = parse_sentinel_config({
            'enabled': True,
            'content_filtering': True,
            'pii_detection': True,
            'compliance': ['gdpr', 'hipaa']
        })
        ```
    """
    if not sentinel_config:
        return {}
    
    # Normalize config
    normalized = {}
    
    # Handle boolean flags
    if isinstance(sentinel_config, bool):
        # If just True/False, enable basic Sentinel
        normalized['enabled'] = sentinel_config
        return normalized
    
    # Copy all keys
    for key, value in sentinel_config.items():
        normalized[key] = value
    
    # Normalize compliance standards (convert strings to enum values)
    if 'compliance' in normalized:
        compliance = normalized['compliance']
        if isinstance(compliance, list):
            # Convert string values to enum values if needed
            from teleon.sentinel.config import ComplianceStandard
            normalized_compliance = []
            for item in compliance:
                if isinstance(item, str):
                    # Try to match enum value
                    try:
                        normalized_compliance.append(ComplianceStandard(item.lower()))
                    except ValueError:
                        # Invalid standard, skip
                        compliance_logger = StructuredLogger("sentinel.integration", LogLevel.WARNING)
                        compliance_logger.warning(f"Invalid compliance standard: {item}")
                else:
                    normalized_compliance.append(item)
            normalized['compliance'] = normalized_compliance
    
    # Normalize action_on_violation
    if 'action_on_violation' in normalized:
        action = normalized['action_on_violation']
        if isinstance(action, str):
            try:
                normalized['action_on_violation'] = GuardrailAction(action.lower())
            except ValueError:
                action_logger = StructuredLogger("sentinel.integration", LogLevel.WARNING)
                action_logger.warning(f"Invalid action: {action}, using default BLOCK")
                normalized['action_on_violation'] = GuardrailAction.BLOCK
    
    # Set defaults
    defaults = {
        'enabled': True,
        'content_filtering': False,
        'pii_detection': False,
        'compliance': [],
        'custom_policies': [],
        'moderation_threshold': 0.8,
        'action_on_violation': GuardrailAction.BLOCK,
        'log_violations': True,
        'audit_enabled': True
    }
    
    for key, default_value in defaults.items():
        if key not in normalized:
            normalized[key] = default_value
    
    # Validate ranges
    if 'moderation_threshold' in normalized:
        threshold = normalized['moderation_threshold']
        if not isinstance(threshold, (int, float)) or threshold < 0.0 or threshold > 1.0:
            threshold_logger = StructuredLogger("sentinel.integration", LogLevel.WARNING)
            threshold_logger.warning(f"Invalid moderation_threshold: {threshold}, using default 0.8")
            normalized['moderation_threshold'] = 0.8
    
    return normalized


def create_sentinel_engine(
    sentinel_config: Optional[Dict[str, Any]] = None,
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None
) -> Optional[SentinelEngine]:
    """
    Create a Sentinel engine from configuration.
    
    Convenience function for creating Sentinel engines.
    
    Args:
        sentinel_config: Sentinel configuration dict
        agent_id: Optional agent ID for audit logging
        agent_name: Optional agent name for audit logging
    
    Returns:
        SentinelEngine instance or None if disabled
    
    Example:
        ```python
        engine = create_sentinel_engine({
            'enabled': True,
            'content_filtering': True
        }, agent_id="agent-123", agent_name="my-agent")
        ```
    """
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
            # If persistence creation fails, continue without it
            logger = StructuredLogger("sentinel.integration", LogLevel.WARNING)
            logger.warning(f"Failed to create Sentinel violation persistence: {e}")
    
    # Create SentinelAuditLogger with dedicated persistence layer
    from teleon.sentinel.audit import SentinelAuditLogger
    sentinel_audit_logger = SentinelAuditLogger(violation_persistence=violation_persistence)
    
    # Create engine and inject the audit logger
    engine = SentinelEngine(config)
    engine.audit_logger = sentinel_audit_logger
    
    return engine

