"""
Health Endpoints - HTTP endpoints for deployed agent health monitoring.

Provides standard health check endpoints for Kubernetes/container orchestration:
- /health - Overall health status
- /ready - Readiness probe
- /live - Liveness probe
- /metrics - Prometheus-compatible metrics
"""

import os
import time
import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

try:
    from fastapi import FastAPI, APIRouter, Response
    from fastapi.responses import PlainTextResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from teleon.helix.agent_reporter import get_agent_reporter


logger = logging.getLogger("teleon.health_endpoints")


class HealthStatus(str, Enum):
    """Health status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_percent: float = 0.0
    open_files: int = 0
    active_connections: int = 0


class HealthEndpointManager:
    """
    Manages health check endpoints for deployed agents.
    
    Provides:
    - Configurable health checks
    - Automatic system metrics collection
    - Prometheus metrics export
    - Integration with agent metrics reporter
    
    Example:
        ```python
        from fastapi import FastAPI
        from teleon.helix.health_endpoints import HealthEndpointManager
        
        app = FastAPI()
        health_manager = HealthEndpointManager()
        
        # Add custom health check
        async def check_database():
            # Your database check logic
            return True, "Database connected"
        
        health_manager.add_check("database", check_database)
        
        # Mount health endpoints
        health_manager.mount(app)
        ```
    """
    
    def __init__(
        self,
        service_name: Optional[str] = None,
        version: Optional[str] = None,
    ):
        """
        Initialize health endpoint manager.
        
        Args:
            service_name: Name of the service
            version: Service version
        """
        self.service_name = service_name or os.getenv("TELEON_SERVICE_NAME", "teleon-agent")
        self.version = version or os.getenv("TELEON_SERVICE_VERSION", "1.0.0")
        self.start_time = time.time()
        
        # Custom health checks
        self._checks: Dict[str, Callable] = {}
        
        # Readiness state
        self._ready = False
        self._ready_reason = "Starting up"
        
        # Metrics storage
        self._request_count = 0
        self._error_count = 0
        self._last_request_time: Optional[float] = None
        
        if FASTAPI_AVAILABLE:
            self.router = APIRouter(tags=["Health"])
            self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.router.get("/health")
        async def health_check():
            """
            Overall health check endpoint.
            
            Returns 200 if healthy, 503 if unhealthy.
            """
            result = await self.check_health()
            status_code = 200 if result["status"] == "healthy" else 503
            return Response(
                content=self._to_json(result),
                media_type="application/json",
                status_code=status_code
            )
        
        @self.router.get("/ready")
        async def readiness_check():
            """
            Kubernetes readiness probe.
            
            Returns 200 if ready to accept traffic, 503 otherwise.
            """
            result = await self.check_readiness()
            status_code = 200 if result["ready"] else 503
            return Response(
                content=self._to_json(result),
                media_type="application/json",
                status_code=status_code
            )
        
        @self.router.get("/live")
        async def liveness_check():
            """
            Kubernetes liveness probe.
            
            Returns 200 if process is alive and should not be restarted.
            """
            result = await self.check_liveness()
            return Response(
                content=self._to_json(result),
                media_type="application/json",
                status_code=200
            )
        
        @self.router.get("/metrics", response_class=PlainTextResponse)
        async def prometheus_metrics():
            """
            Prometheus-compatible metrics endpoint.
            
            Returns metrics in Prometheus text format.
            """
            metrics = await self.get_prometheus_metrics()
            return PlainTextResponse(content=metrics, media_type="text/plain")
    
    def add_check(
        self,
        name: str,
        check_fn: Callable,
        critical: bool = False
    ):
        """
        Add a custom health check.
        
        Args:
            name: Check name
            check_fn: Async function returning (healthy: bool, message: str)
            critical: If True, failure marks system unhealthy
        """
        self._checks[name] = {
            "fn": check_fn,
            "critical": critical
        }
        logger.info(f"Added health check: {name} (critical={critical})")
    
    def remove_check(self, name: str):
        """Remove a health check."""
        if name in self._checks:
            del self._checks[name]
    
    def set_ready(self, ready: bool = True, reason: str = ""):
        """
        Set readiness state.
        
        Args:
            ready: Whether service is ready
            reason: Reason for state
        """
        self._ready = ready
        self._ready_reason = reason or ("Ready" if ready else "Not ready")
        logger.info(f"Readiness state changed: ready={ready}, reason={self._ready_reason}")
    
    def record_request(self, success: bool = True):
        """Record a request for metrics."""
        self._request_count += 1
        self._last_request_time = time.time()
        if not success:
            self._error_count += 1
    
    async def check_health(self) -> Dict[str, Any]:
        """
        Run all health checks and return overall status.
        
        Returns:
            Health check result dictionary
        """
        checks_results: List[HealthCheckResult] = []
        overall_status = HealthStatus.HEALTHY
        
        # System check
        system_result = await self._check_system()
        checks_results.append(system_result)
        
        # Custom checks
        for name, check_config in self._checks.items():
            result = await self._run_check(name, check_config)
            checks_results.append(result)
            
            if result.status == HealthStatus.UNHEALTHY and check_config.get("critical", False):
                overall_status = HealthStatus.UNHEALTHY
            elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
        
        # Report health to metrics reporter
        reporter = get_agent_reporter()
        await reporter.report_health(
            status=overall_status.value,
            active_requests=self._request_count,
            queue_depth=0  # Could be tracked separately
        )
        
        return {
            "status": overall_status.value,
            "service": self.service_name,
            "version": self.version,
            "uptime_seconds": time.time() - self.start_time,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "message": r.message,
                    "latency_ms": r.latency_ms,
                    "details": r.details
                }
                for r in checks_results
            ]
        }
    
    async def check_readiness(self) -> Dict[str, Any]:
        """
        Check if service is ready to accept traffic.
        
        Returns:
            Readiness check result
        """
        return {
            "ready": self._ready,
            "reason": self._ready_reason,
            "service": self.service_name,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def check_liveness(self) -> Dict[str, Any]:
        """
        Check if service is alive.
        
        Returns:
            Liveness check result
        """
        return {
            "alive": True,
            "service": self.service_name,
            "uptime_seconds": time.time() - self.start_time,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def get_prometheus_metrics(self) -> str:
        """
        Generate Prometheus-compatible metrics.
        
        Returns:
            Metrics in Prometheus text format
        """
        lines = []
        labels = f'service="{self.service_name}",version="{self.version}"'
        
        # Uptime
        uptime = time.time() - self.start_time
        lines.append(f"# HELP teleon_uptime_seconds Service uptime in seconds")
        lines.append(f"# TYPE teleon_uptime_seconds gauge")
        lines.append(f"teleon_uptime_seconds{{{labels}}} {uptime:.2f}")
        
        # Request counts
        lines.append(f"# HELP teleon_requests_total Total number of requests")
        lines.append(f"# TYPE teleon_requests_total counter")
        lines.append(f"teleon_requests_total{{{labels}}} {self._request_count}")
        
        lines.append(f"# HELP teleon_errors_total Total number of errors")
        lines.append(f"# TYPE teleon_errors_total counter")
        lines.append(f"teleon_errors_total{{{labels}}} {self._error_count}")
        
        # System metrics
        system = self._get_system_metrics()
        
        lines.append(f"# HELP teleon_cpu_percent CPU usage percentage")
        lines.append(f"# TYPE teleon_cpu_percent gauge")
        lines.append(f"teleon_cpu_percent{{{labels}}} {system.cpu_percent:.2f}")
        
        lines.append(f"# HELP teleon_memory_percent Memory usage percentage")
        lines.append(f"# TYPE teleon_memory_percent gauge")
        lines.append(f"teleon_memory_percent{{{labels}}} {system.memory_percent:.2f}")
        
        lines.append(f"# HELP teleon_memory_used_bytes Memory used in bytes")
        lines.append(f"# TYPE teleon_memory_used_bytes gauge")
        lines.append(f"teleon_memory_used_bytes{{{labels}}} {system.memory_used_mb * 1024 * 1024:.0f}")
        
        # Get stats from reporter
        reporter = get_agent_reporter()
        stats = reporter.get_local_stats()
        period_stats = stats.get("period_stats", {})
        
        lines.append(f"# HELP teleon_llm_requests_total Total LLM requests in period")
        lines.append(f"# TYPE teleon_llm_requests_total counter")
        lines.append(f"teleon_llm_requests_total{{{labels}}} {period_stats.get('total_requests', 0)}")
        
        lines.append(f"# HELP teleon_llm_tokens_total Total tokens processed")
        lines.append(f"# TYPE teleon_llm_tokens_total counter")
        total_tokens = period_stats.get("total_input_tokens", 0) + period_stats.get("total_output_tokens", 0)
        lines.append(f"teleon_llm_tokens_total{{{labels}}} {total_tokens}")
        
        lines.append(f"# HELP teleon_llm_cost_total Total cost in USD")
        lines.append(f"# TYPE teleon_llm_cost_total counter")
        lines.append(f"teleon_llm_cost_total{{{labels}}} {period_stats.get('total_cost', 0):.6f}")
        
        # Ready state
        lines.append(f"# HELP teleon_ready Readiness state (1=ready, 0=not ready)")
        lines.append(f"# TYPE teleon_ready gauge")
        lines.append(f"teleon_ready{{{labels}}} {1 if self._ready else 0}")
        
        return "\n".join(lines) + "\n"
    
    async def _check_system(self) -> HealthCheckResult:
        """Run system health check."""
        start = time.time()
        metrics = self._get_system_metrics()
        latency = (time.time() - start) * 1000
        
        # Determine status based on resource usage
        status = HealthStatus.HEALTHY
        messages = []
        
        if metrics.cpu_percent > 90:
            status = HealthStatus.DEGRADED
            messages.append(f"High CPU: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > 90:
            status = HealthStatus.DEGRADED
            messages.append(f"High memory: {metrics.memory_percent:.1f}%")
        
        if metrics.memory_percent > 95:
            status = HealthStatus.UNHEALTHY
        
        return HealthCheckResult(
            name="system",
            status=status,
            message="; ".join(messages) if messages else "System resources OK",
            latency_ms=latency,
            details={
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "memory_used_mb": metrics.memory_used_mb,
            }
        )
    
    async def _run_check(
        self,
        name: str,
        check_config: Dict
    ) -> HealthCheckResult:
        """Run a custom health check."""
        start = time.time()
        
        try:
            check_fn = check_config["fn"]
            
            if asyncio.iscoroutinefunction(check_fn):
                result = await check_fn()
            else:
                result = check_fn()
            
            latency = (time.time() - start) * 1000
            
            if isinstance(result, tuple):
                healthy, message = result
            else:
                healthy = bool(result)
                message = ""
            
            return HealthCheckResult(
                name=name,
                status=HealthStatus.HEALTHY if healthy else HealthStatus.UNHEALTHY,
                message=message,
                latency_ms=latency
            )
            
        except Exception as e:
            latency = (time.time() - start) * 1000
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=latency
            )
    
    def _get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        metrics = SystemMetrics()
        
        if PSUTIL_AVAILABLE:
            try:
                metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
                
                mem = psutil.virtual_memory()
                metrics.memory_percent = mem.percent
                metrics.memory_used_mb = mem.used / (1024 * 1024)
                metrics.memory_available_mb = mem.available / (1024 * 1024)
                
                try:
                    disk = psutil.disk_usage("/")
                    metrics.disk_percent = disk.percent
                except Exception:
                    pass
                
                try:
                    process = psutil.Process()
                    metrics.open_files = len(process.open_files())
                    metrics.active_connections = len(process.connections())
                except Exception:
                    pass
                    
            except Exception as e:
                logger.warning(f"Error collecting system metrics: {e}")
        
        return metrics
    
    def _to_json(self, data: Dict) -> str:
        """Convert dict to JSON string."""
        import json
        return json.dumps(data, indent=2)
    
    def mount(self, app, prefix: str = ""):
        """
        Mount health endpoints to a FastAPI app.
        
        Args:
            app: FastAPI application instance
            prefix: URL prefix for endpoints
        """
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI is required for health endpoints")
        
        app.include_router(self.router, prefix=prefix)
        logger.info(f"Mounted health endpoints at prefix: {prefix or '/'}")


# Global health manager
_global_health_manager: Optional[HealthEndpointManager] = None


def get_health_manager() -> HealthEndpointManager:
    """Get or create the global health endpoint manager."""
    global _global_health_manager
    if _global_health_manager is None:
        _global_health_manager = HealthEndpointManager()
    return _global_health_manager


def setup_health_endpoints(
    app,
    service_name: Optional[str] = None,
    version: Optional[str] = None,
    prefix: str = "",
) -> HealthEndpointManager:
    """
    Setup health endpoints on a FastAPI app.
    
    Args:
        app: FastAPI application instance
        service_name: Service name
        version: Service version
        prefix: URL prefix
    
    Returns:
        Configured HealthEndpointManager
    """
    global _global_health_manager
    _global_health_manager = HealthEndpointManager(
        service_name=service_name,
        version=version
    )
    _global_health_manager.mount(app, prefix)
    return _global_health_manager



