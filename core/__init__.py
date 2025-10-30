"""
Teleon Core - Production-grade foundation.

This module provides enterprise-level infrastructure:
- Exception hierarchy with error codes
- Configuration management with validation
- Input/output validation with security
- Comprehensive observability (metrics, logging, tracing)
- Health checking
- Performance monitoring
"""

from teleon.core.exceptions import (
    TeleonError,
    ErrorCode,
    ConfigurationError,
    AgentError,
    AgentNotFoundError,
    AgentExecutionError,
    AgentValidationError,
    LLMError,
    LLMProviderError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMContextLengthError,
    ToolError,
    ToolNotFoundError,
    ToolExecutionError,
    ToolValidationError,
    ToolTimeoutError,
    ToolPermissionError,
    MemoryError,
    SecurityError,
    AuthenticationError,
    AuthorizationError,
    InvalidAPIKeyError,
    ResourceError,
    QuotaExceededError,
    RateLimitExceededError,
)

from teleon.core.config import (
    TeleonConfig,
    LLMConfig,
    MemoryConfig,
    ToolConfig,
    ObservabilityConfig,
    SecretsConfig,
    get_config,
    reset_config,
)

from teleon.core.validation import (
    InputValidator,
    SchemaValidator,
    SecurityValidator,
    StringValidation,
    NumericValidation,
    CollectionValidation,
)

from teleon.core.observability import (
    StructuredLogger,
    MetricsCollector,
    PerformanceMonitor,
    HealthChecker,
    LogLevel,
    get_metrics,
    get_monitor,
    get_health_checker,
)

__all__ = [
    # Exceptions
    "TeleonError",
    "ErrorCode",
    "ConfigurationError",
    "AgentError",
    "AgentNotFoundError",
    "AgentExecutionError",
    "AgentValidationError",
    "LLMError",
    "LLMProviderError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMContextLengthError",
    "ToolError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "ToolValidationError",
    "ToolTimeoutError",
    "ToolPermissionError",
    "MemoryError",
    "SecurityError",
    "AuthenticationError",
    "AuthorizationError",
    "InvalidAPIKeyError",
    "ResourceError",
    "QuotaExceededError",
    "RateLimitExceededError",
    
    # Configuration
    "TeleonConfig",
    "LLMConfig",
    "MemoryConfig",
    "ToolConfig",
    "ObservabilityConfig",
    "SecretsConfig",
    "get_config",
    "reset_config",
    
    # Validation
    "InputValidator",
    "SchemaValidator",
    "SecurityValidator",
    "StringValidation",
    "NumericValidation",
    "CollectionValidation",
    
    # Observability
    "StructuredLogger",
    "MetricsCollector",
    "PerformanceMonitor",
    "HealthChecker",
    "LogLevel",
    "get_metrics",
    "get_monitor",
    "get_health_checker",
]

