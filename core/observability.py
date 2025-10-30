"""
Production-grade observability for Teleon.

Features:
- Structured logging with context
- Prometheus metrics
- Distributed tracing
- Health checks
- Performance monitoring
- Error tracking
"""

from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
from functools import wraps
import time
import logging
import json
from datetime import datetime
from enum import Enum
import threading
from collections import defaultdict

# Try to import prometheus_client, but don't fail if not available
try:
    from prometheus_client import Counter, Histogram, Gauge, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class LogLevel(str, Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StructuredLogger:
    """
    Production-grade structured logger.
    
    Features:
    - JSON formatting
    - Context propagation
    - Correlation IDs
    - Performance tracking
    """
    
    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        format_json: bool = True
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            level: Logging level
            format_json: Use JSON formatting
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.value))
        self.format_json = format_json
        self._context = threading.local()
    
    def _format_message(
        self,
        level: str,
        message: str,
        **kwargs
    ) -> str:
        """Format log message."""
        if self.format_json:
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": level,
                "logger": self.logger.name,
                "message": message,
                **kwargs
            }
            
            # Add context if available
            if hasattr(self._context, 'data'):
                log_data.update(self._context.data)
            
            return json.dumps(log_data)
        else:
            extras = " ".join(f"{k}={v}" for k, v in kwargs.items())
            return f"[{level}] {message} {extras}"
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(self._format_message("DEBUG", message, **kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(self._format_message("INFO", message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(self._format_message("WARNING", message, **kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(self._format_message("ERROR", message, **kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(self._format_message("CRITICAL", message, **kwargs))
    
    def set_context(self, **kwargs):
        """Set logging context."""
        if not hasattr(self._context, 'data'):
            self._context.data = {}
        self._context.data.update(kwargs)
    
    def clear_context(self):
        """Clear logging context."""
        if hasattr(self._context, 'data'):
            self._context.data = {}


class MetricsCollector:
    """
    Production-grade metrics collector.
    
    Integrates with Prometheus for monitoring.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize metrics collector.
        
        Args:
            enabled: Enable metrics collection
        """
        self.enabled = enabled and PROMETHEUS_AVAILABLE
        
        if self.enabled:
            # Request metrics
            self.request_count = Counter(
                'teleon_requests_total',
                'Total requests',
                ['component', 'operation', 'status']
            )
            
            self.request_duration = Histogram(
                'teleon_request_duration_seconds',
                'Request duration',
                ['component', 'operation']
            )
            
            # LLM metrics
            self.llm_requests = Counter(
                'teleon_llm_requests_total',
                'Total LLM requests',
                ['provider', 'model', 'status']
            )
            
            self.llm_tokens = Counter(
                'teleon_llm_tokens_total',
                'Total tokens used',
                ['provider', 'model', 'type']
            )
            
            self.llm_cost = Counter(
                'teleon_llm_cost_total',
                'Total LLM cost in USD',
                ['provider', 'model']
            )
            
            # Tool metrics
            self.tool_executions = Counter(
                'teleon_tool_executions_total',
                'Total tool executions',
                ['tool_name', 'status']
            )
            
            self.tool_duration = Histogram(
                'teleon_tool_duration_seconds',
                'Tool execution duration',
                ['tool_name']
            )
            
            # Memory metrics
            self.memory_operations = Counter(
                'teleon_memory_operations_total',
                'Total memory operations',
                ['memory_type', 'operation']
            )
            
            self.memory_size = Gauge(
                'teleon_memory_size_bytes',
                'Memory size in bytes',
                ['memory_type']
            )
            
            # Error metrics
            self.errors = Counter(
                'teleon_errors_total',
                'Total errors',
                ['component', 'error_type']
            )
            
            # System info
            self.system_info = Info(
                'teleon_system',
                'System information'
            )
            
        # Always create fallback counters if prometheus failed
        if not hasattr(self, '_counters'):
            self._counters = defaultdict(int)
        if not hasattr(self, '_histograms'):
            self._histograms = defaultdict(list)
    
    def increment_counter(self, name: str, labels: Dict[str, str], value: float = 1):
        """Increment counter metric."""
        if not self.enabled:
            return
        
        metric = getattr(self, name, None)
        if metric and hasattr(metric, 'labels'):
            metric.labels(**labels).inc(value)
        else:
            key = f"{name}_{labels}"
            self._counters[key] += value
    
    def observe_histogram(self, name: str, labels: Dict[str, str], value: float):
        """Observe histogram metric."""
        if not self.enabled:
            return
        
        metric = getattr(self, name, None)
        if metric and hasattr(metric, 'labels'):
            metric.labels(**labels).observe(value)
        else:
            key = f"{name}_{labels}"
            self._histograms[key].append(value)
    
    def set_gauge(self, name: str, labels: Dict[str, str], value: float):
        """Set gauge metric."""
        if not self.enabled:
            return
        
        metric = getattr(self, name, None)
        if metric and hasattr(metric, 'labels'):
            metric.labels(**labels).set(value)
    
    def record_request(
        self,
        component: str,
        operation: str,
        duration: float,
        status: str = "success"
    ):
        """Record request metrics."""
        self.increment_counter(
            'request_count',
            {'component': component, 'operation': operation, 'status': status}
        )
        self.observe_histogram(
            'request_duration',
            {'component': component, 'operation': operation},
            duration
        )
    
    def record_llm_request(
        self,
        provider: str,
        model: str,
        tokens: int,
        cost: float,
        status: str = "success"
    ):
        """Record LLM request metrics."""
        self.increment_counter(
            'llm_requests',
            {'provider': provider, 'model': model, 'status': status}
        )
        self.increment_counter(
            'llm_tokens',
            {'provider': provider, 'model': model, 'type': 'total'},
            tokens
        )
        self.increment_counter(
            'llm_cost',
            {'provider': provider, 'model': model},
            cost
        )
    
    def record_tool_execution(
        self,
        tool_name: str,
        duration: float,
        status: str = "success"
    ):
        """Record tool execution metrics."""
        self.increment_counter(
            'tool_executions',
            {'tool_name': tool_name, 'status': status}
        )
        self.observe_histogram(
            'tool_duration',
            {'tool_name': tool_name},
            duration
        )
    
    def record_error(self, component: str, error_type: str):
        """Record error metric."""
        self.increment_counter(
            'errors',
            {'component': component, 'error_type': error_type}
        )


class PerformanceMonitor:
    """Monitor and track performance."""
    
    def __init__(self, metrics: Optional[MetricsCollector] = None):
        """
        Initialize performance monitor.
        
        Args:
            metrics: Metrics collector
        """
        self.metrics = metrics or MetricsCollector()
        self.logger = StructuredLogger("performance")
    
    @contextmanager
    def track(self, component: str, operation: str):
        """
        Track operation performance.
        
        Usage:
            with monitor.track("agent", "execute"):
                # operation code
        """
        start_time = time.time()
        status = "success"
        
        try:
            yield
        except Exception as e:
            status = "error"
            self.logger.error(
                f"Operation failed: {operation}",
                component=component,
                error=str(e)
            )
            raise
        finally:
            duration = time.time() - start_time
            self.metrics.record_request(component, operation, duration, status)
            
            if duration > 1.0:  # Slow operation warning
                self.logger.warning(
                    f"Slow operation detected: {operation}",
                    component=component,
                    duration=duration
                )
    
    def track_async(self, component: str, operation: str):
        """
        Decorator to track async function performance.
        
        Usage:
            @monitor.track_async("agent", "execute")
            async def execute_agent(...):
                ...
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                status = "success"
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = "error"
                    self.logger.error(
                        f"Operation failed: {operation}",
                        component=component,
                        error=str(e)
                    )
                    self.metrics.record_error(component, type(e).__name__)
                    raise
                finally:
                    duration = time.time() - start_time
                    self.metrics.record_request(component, operation, duration, status)
            
            return wrapper
        return decorator


class HealthChecker:
    """
    Production-grade health checking.
    
    Monitors system health and dependencies.
    """
    
    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, Callable] = {}
        self.logger = StructuredLogger("health")
    
    def register_check(self, name: str, check_func: Callable[[], bool]):
        """
        Register health check.
        
        Args:
            name: Check name
            check_func: Function that returns True if healthy
        """
        self.checks[name] = check_func
    
    async def check_health(self) -> Dict[str, Any]:
        """
        Run all health checks.
        
        Returns:
            Health check results
        """
        results = {}
        overall_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                healthy = check_func()
                results[name] = {
                    "status": "healthy" if healthy else "unhealthy",
                    "timestamp": datetime.utcnow().isoformat()
                }
                if not healthy:
                    overall_healthy = False
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                overall_healthy = False
                self.logger.error(f"Health check failed: {name}", error=str(e))
        
        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "checks": results,
            "timestamp": datetime.utcnow().isoformat()
        }


# Global instances
_metrics = None
_monitor = None
_health_checker = None


def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


def get_monitor() -> PerformanceMonitor:
    """Get global performance monitor."""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor(get_metrics())
    return _monitor


def get_health_checker() -> HealthChecker:
    """Get global health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker

