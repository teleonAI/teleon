"""Structured logging for Teleon."""

import logging
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from enum import Enum


class LogLevel(str, Enum):
    """Log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class StructuredLogger:
    """
    Structured logger for Teleon agents.
    
    Outputs logs in JSON format for easy parsing and analysis.
    """
    
    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        extra_fields: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            level: Logging level
            extra_fields: Additional fields to include in all logs
        """
        self.name = name
        self.level = level
        self.extra_fields = extra_fields or {}
        self._logger = logging.getLogger(name)
        self._configure_logger()
    
    def _configure_logger(self) -> None:
        """Configure the underlying logger."""
        level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL
        }
        
        self._logger.setLevel(level_map[self.level])
        
        # Remove existing handlers
        self._logger.handlers = []
        
        # Add console handler with JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        self._logger.addHandler(handler)
    
    def _log(
        self,
        level: LogLevel,
        message: str,
        **kwargs: Any
    ) -> None:
        """
        Internal logging method.
        
        Args:
            level: Log level
            message: Log message
            **kwargs: Additional fields
        """
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.value,
            "logger": self.name,
            "message": message,
            **self.extra_fields,
            **kwargs
        }
        
        level_map = {
            LogLevel.DEBUG: self._logger.debug,
            LogLevel.INFO: self._logger.info,
            LogLevel.WARNING: self._logger.warning,
            LogLevel.ERROR: self._logger.error,
            LogLevel.CRITICAL: self._logger.critical
        }
        
        level_map[level](json.dumps(log_data))
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, **kwargs)
    
    def log_execution(
        self,
        agent_name: str,
        execution_id: str,
        duration_ms: float,
        status: str,
        **kwargs: Any
    ) -> None:
        """
        Log agent execution.
        
        Args:
            agent_name: Name of the agent
            execution_id: Execution ID
            duration_ms: Execution duration in milliseconds
            status: Execution status
            **kwargs: Additional fields
        """
        self.info(
            "Agent execution",
            agent_name=agent_name,
            execution_id=execution_id,
            duration_ms=duration_ms,
            status=status,
            **kwargs
        )
    
    def log_llm_call(
        self,
        model: str,
        provider: str,
        tokens: int,
        cost: float,
        latency_ms: float,
        **kwargs: Any
    ) -> None:
        """
        Log LLM API call.
        
        Args:
            model: Model name
            provider: Provider name
            tokens: Total tokens used
            cost: Cost in USD
            latency_ms: Latency in milliseconds
            **kwargs: Additional fields
        """
        self.info(
            "LLM API call",
            model=model,
            provider=provider,
            tokens=tokens,
            cost=cost,
            latency_ms=latency_ms,
            **kwargs
        )


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # The message is already JSON from StructuredLogger
        return record.getMessage()


def get_logger(
    name: str,
    level: LogLevel = LogLevel.INFO,
    **extra_fields: Any
) -> StructuredLogger:
    """
    Get or create a structured logger.
    
    Args:
        name: Logger name
        level: Logging level
        **extra_fields: Additional fields to include in all logs
    
    Returns:
        Structured logger instance
    """
    return StructuredLogger(name, level, extra_fields)

