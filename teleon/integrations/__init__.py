"""
Teleon Integrations - External Service Integrations.

Provides pre-built integrations for popular services with automatic
authentication, rate limiting, and error handling.
"""

from teleon.integrations.base import (
    BaseIntegration,
    IntegrationConfig,
    IntegrationRegistry,
    get_registry,
)


__all__ = [
    "BaseIntegration",
    "IntegrationConfig",
    "IntegrationRegistry",
    "get_registry",
]

