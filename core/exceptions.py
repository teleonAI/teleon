"""
Production-grade exception hierarchy for Teleon.

This module defines a comprehensive exception hierarchy following best practices:
- Clear exception hierarchy
- Detailed error context
- Error codes for monitoring
- Proper inheritance structure
"""

from typing import Optional, Dict, Any
from enum import Enum


class ErrorCode(str, Enum):
    """Error codes for monitoring and alerting."""
    
    # Configuration errors (1xxx)
    CONFIG_INVALID = "E1001"
    CONFIG_MISSING = "E1002"
    CONFIG_VALIDATION_FAILED = "E1003"
    
    # Agent errors (2xxx)
    AGENT_NOT_FOUND = "E2001"
    AGENT_INITIALIZATION_FAILED = "E2002"
    AGENT_EXECUTION_FAILED = "E2003"
    AGENT_VALIDATION_FAILED = "E2004"
    
    # LLM errors (3xxx)
    LLM_PROVIDER_ERROR = "E3001"
    LLM_TIMEOUT = "E3002"
    LLM_RATE_LIMIT = "E3003"
    LLM_INVALID_RESPONSE = "E3004"
    LLM_CONTEXT_LENGTH_EXCEEDED = "E3005"
    
    # Tool errors (4xxx)
    TOOL_NOT_FOUND = "E4001"
    TOOL_EXECUTION_FAILED = "E4002"
    TOOL_VALIDATION_FAILED = "E4003"
    TOOL_TIMEOUT = "E4004"
    TOOL_PERMISSION_DENIED = "E4005"
    
    # Memory errors (5xxx)
    MEMORY_OPERATION_FAILED = "E5001"
    MEMORY_QUOTA_EXCEEDED = "E5002"
    MEMORY_CORRUPTION = "E5003"
    
    # Security errors (6xxx)
    AUTHENTICATION_FAILED = "E6001"
    AUTHORIZATION_FAILED = "E6002"
    INVALID_API_KEY = "E6003"
    SECURITY_VIOLATION = "E6004"
    
    # Resource errors (7xxx)
    RESOURCE_EXHAUSTED = "E7001"
    QUOTA_EXCEEDED = "E7002"
    RATE_LIMIT_EXCEEDED = "E7003"
    
    # Network errors (8xxx)
    NETWORK_ERROR = "E8001"
    CONNECTION_FAILED = "E8002"
    TIMEOUT = "E8003"
    
    # Internal errors (9xxx)
    INTERNAL_ERROR = "E9001"
    NOT_IMPLEMENTED = "E9002"
    INVALID_STATE = "E9003"


class TeleonError(Exception):
    """
    Base exception for all Teleon errors.
    
    Provides:
    - Error code for monitoring
    - Context for debugging
    - Proper logging
    - Error categorization
    """
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize Teleon error.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            context: Additional context for debugging
            cause: Original exception if this is a wrapped error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.cause = cause
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "code": self.error_code.value,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        """String representation with error code."""
        return f"[{self.error_code.value}] {self.message}"


# Configuration Errors
class ConfigurationError(TeleonError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCode.CONFIG_INVALID, context)


class ConfigValidationError(TeleonError):
    """Configuration validation errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCode.CONFIG_VALIDATION_FAILED, context)


# Agent Errors
class AgentError(TeleonError):
    """Base class for agent-related errors."""
    pass


class AgentNotFoundError(AgentError):
    """Agent not found."""
    
    def __init__(self, agent_name: str):
        super().__init__(
            f"Agent '{agent_name}' not found",
            ErrorCode.AGENT_NOT_FOUND,
            {"agent_name": agent_name}
        )


class AgentExecutionError(AgentError):
    """Agent execution failed."""
    
    def __init__(
        self,
        agent_name: str,
        message: str,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Agent '{agent_name}' execution failed: {message}",
            ErrorCode.AGENT_EXECUTION_FAILED,
            {"agent_name": agent_name},
            cause
        )


class AgentValidationError(AgentError):
    """Agent input/output validation failed."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message,
            ErrorCode.AGENT_VALIDATION_FAILED,
            context
        )


# LLM Errors
class LLMError(TeleonError):
    """Base class for LLM-related errors."""
    pass


class LLMProviderError(LLMError):
    """LLM provider error."""
    
    def __init__(
        self,
        provider: str,
        message: str,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"LLM provider '{provider}' error: {message}",
            ErrorCode.LLM_PROVIDER_ERROR,
            {"provider": provider},
            cause
        )


class LLMTimeoutError(LLMError):
    """LLM request timeout."""
    
    def __init__(self, timeout: float, model: str):
        super().__init__(
            f"LLM request timed out after {timeout}s",
            ErrorCode.LLM_TIMEOUT,
            {"timeout": timeout, "model": model}
        )


class LLMRateLimitError(LLMError):
    """LLM rate limit exceeded."""
    
    def __init__(self, provider: str, retry_after: Optional[int] = None):
        super().__init__(
            f"Rate limit exceeded for provider '{provider}'",
            ErrorCode.LLM_RATE_LIMIT,
            {"provider": provider, "retry_after": retry_after}
        )


class LLMContextLengthError(LLMError):
    """Context length exceeded."""
    
    def __init__(self, tokens: int, max_tokens: int):
        super().__init__(
            f"Context length {tokens} exceeds maximum {max_tokens}",
            ErrorCode.LLM_CONTEXT_LENGTH_EXCEEDED,
            {"tokens": tokens, "max_tokens": max_tokens}
        )


# Tool Errors
class ToolError(TeleonError):
    """Base class for tool-related errors."""
    pass


class ToolNotFoundError(ToolError):
    """Tool not found."""
    
    def __init__(self, tool_name: str):
        super().__init__(
            f"Tool '{tool_name}' not found",
            ErrorCode.TOOL_NOT_FOUND,
            {"tool_name": tool_name}
        )


class ToolExecutionError(ToolError):
    """Tool execution failed."""
    
    def __init__(
        self,
        tool_name: str,
        message: str,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Tool '{tool_name}' execution failed: {message}",
            ErrorCode.TOOL_EXECUTION_FAILED,
            {"tool_name": tool_name},
            cause
        )


class ToolTimeoutError(ToolError):
    """Tool execution timeout."""
    
    def __init__(self, tool_name: str, timeout: float):
        super().__init__(
            f"Tool '{tool_name}' timed out after {timeout}s",
            ErrorCode.TOOL_TIMEOUT,
            {"tool_name": tool_name, "timeout": timeout}
        )


class ToolValidationError(ToolError):
    """Tool validation failed."""
    
    def __init__(self, tool_name: str, message: str):
        super().__init__(
            f"Tool '{tool_name}' validation failed: {message}",
            ErrorCode.TOOL_VALIDATION_FAILED,
            {"tool_name": tool_name}
        )


class ToolPermissionError(ToolError):
    """Tool permission denied."""
    
    def __init__(self, tool_name: str, reason: str):
        super().__init__(
            f"Permission denied for tool '{tool_name}': {reason}",
            ErrorCode.TOOL_PERMISSION_DENIED,
            {"tool_name": tool_name, "reason": reason}
        )


# Memory Errors
class MemoryError(TeleonError):
    """Base class for memory-related errors."""
    pass


class MemoryOperationError(MemoryError):
    """Memory operation failed."""
    
    def __init__(
        self,
        operation: str,
        message: str,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Memory operation '{operation}' failed: {message}",
            ErrorCode.MEMORY_OPERATION_FAILED,
            {"operation": operation},
            cause
        )


class MemoryQuotaExceededError(MemoryError):
    """Memory quota exceeded."""
    
    def __init__(self, used: int, limit: int):
        super().__init__(
            f"Memory quota exceeded: {used}/{limit}",
            ErrorCode.MEMORY_QUOTA_EXCEEDED,
            {"used": used, "limit": limit}
        )


# Security Errors
class SecurityError(TeleonError):
    """Base class for security-related errors."""
    pass


class AuthenticationError(SecurityError):
    """Authentication failed."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, ErrorCode.AUTHENTICATION_FAILED)


class AuthorizationError(SecurityError):
    """Authorization failed."""
    
    def __init__(self, resource: str, action: str):
        super().__init__(
            f"Not authorized to {action} {resource}",
            ErrorCode.AUTHORIZATION_FAILED,
            {"resource": resource, "action": action}
        )


class InvalidAPIKeyError(SecurityError):
    """Invalid API key."""
    
    def __init__(self, provider: Optional[str] = None):
        super().__init__(
            f"Invalid API key{f' for {provider}' if provider else ''}",
            ErrorCode.INVALID_API_KEY,
            {"provider": provider}
        )


# Resource Errors
class ResourceError(TeleonError):
    """Base class for resource-related errors."""
    pass


class ResourceExhaustedError(ResourceError):
    """Resources exhausted."""
    
    def __init__(self, resource: str):
        super().__init__(
            f"Resource exhausted: {resource}",
            ErrorCode.RESOURCE_EXHAUSTED,
            {"resource": resource}
        )


class QuotaExceededError(ResourceError):
    """Quota exceeded."""
    
    def __init__(self, quota_type: str, used: Any, limit: Any):
        super().__init__(
            f"{quota_type} quota exceeded: {used}/{limit}",
            ErrorCode.QUOTA_EXCEEDED,
            {"quota_type": quota_type, "used": used, "limit": limit}
        )


class RateLimitExceededError(ResourceError):
    """Rate limit exceeded."""
    
    def __init__(self, limit: int, window: str, retry_after: Optional[int] = None):
        super().__init__(
            f"Rate limit exceeded: {limit} requests per {window}",
            ErrorCode.RATE_LIMIT_EXCEEDED,
            {"limit": limit, "window": window, "retry_after": retry_after}
        )


# Network Errors
class NetworkError(TeleonError):
    """Base class for network-related errors."""
    pass


class ConnectionError(NetworkError):
    """Connection failed."""
    
    def __init__(self, host: str, cause: Optional[Exception] = None):
        super().__init__(
            f"Failed to connect to {host}",
            ErrorCode.CONNECTION_FAILED,
            {"host": host},
            cause
        )


class TimeoutError(NetworkError):
    """Operation timeout."""
    
    def __init__(self, operation: str, timeout: float):
        super().__init__(
            f"Operation '{operation}' timed out after {timeout}s",
            ErrorCode.TIMEOUT,
            {"operation": operation, "timeout": timeout}
        )

