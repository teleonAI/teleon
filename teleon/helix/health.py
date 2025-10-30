"""
Health Checker - Production-grade health monitoring.

Features:
- Readiness checks
- Liveness checks
- Custom health checks
- Automatic failure detection
- Health status aggregation
"""

from typing import Dict, Optional, Callable, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import asyncio

from teleon.core import (
    get_metrics,
    StructuredLogger,
    LogLevel,
)


class HealthStatus(str, Enum):
    """Health status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class CheckType(str, Enum):
    """Health check type."""
    READINESS = "readiness"  # Ready to receive traffic
    LIVENESS = "liveness"    # Process is alive
    CUSTOM = "custom"        # Custom check


class HealthCheck(BaseModel):
    """Health check configuration."""
    
    name: str = Field(..., description="Check name")
    check_type: CheckType = Field(CheckType.READINESS, description="Check type")
    
    # Check function (callable)
    check_fn: Optional[Callable] = Field(None, description="Check function")
    
    # Timing
    interval: int = Field(30, ge=5, description="Check interval (seconds)")
    timeout: int = Field(10, ge=1, description="Check timeout (seconds)")
    
    # Thresholds
    failure_threshold: int = Field(3, ge=1, description="Failures before unhealthy")
    success_threshold: int = Field(1, ge=1, description="Successes before healthy")
    
    # Delays
    initial_delay: int = Field(0, ge=0, description="Initial delay (seconds)")
    
    class Config:
        arbitrary_types_allowed = True


class HealthResult(BaseModel):
    """Health check result."""
    
    status: HealthStatus = Field(..., description="Health status")
    message: Optional[str] = Field(None, description="Status message")
    
    # Metadata
    checked_at: datetime = Field(default_factory=datetime.utcnow)
    response_time_ms: float = Field(0.0, description="Check response time")
    
    # Counters
    consecutive_failures: int = Field(0, description="Consecutive failures")
    consecutive_successes: int = Field(0, description="Consecutive successes")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthChecker:
    """
    Production-grade health checker.
    
    Features:
    - Multiple check types
    - Configurable thresholds
    - Automatic monitoring
    - Status aggregation
    """
    
    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, HealthCheck] = {}
        self.results: Dict[str, HealthResult] = {}
        
        self.lock = asyncio.Lock()
        self.logger = StructuredLogger("health_checker", LogLevel.INFO)
        
        # Monitoring tasks
        self.monitor_tasks: Dict[str, asyncio.Task] = {}
        self.running = False
    
    async def register_check(
        self,
        target_id: str,
        health_check: HealthCheck
    ):
        """
        Register a health check.
        
        Args:
            target_id: Target identifier (agent/process)
            health_check: Health check configuration
        """
        async with self.lock:
            self.checks[target_id] = health_check
            self.results[target_id] = HealthResult(status=HealthStatus.UNKNOWN)
        
        self.logger.info(
            "Health check registered",
            target_id=target_id,
            check_name=health_check.name,
            check_type=health_check.check_type.value
        )
        
        # Start monitoring if running
        if self.running:
            await self._start_check_monitor(target_id)
    
    async def unregister_check(self, target_id: str):
        """
        Unregister a health check.
        
        Args:
            target_id: Target identifier
        """
        async with self.lock:
            if target_id in self.checks:
                del self.checks[target_id]
            if target_id in self.results:
                del self.results[target_id]
        
        # Stop monitoring task
        if target_id in self.monitor_tasks:
            self.monitor_tasks[target_id].cancel()
            del self.monitor_tasks[target_id]
        
        self.logger.info("Health check unregistered", target_id=target_id)
    
    async def check_health(self, target_id: str) -> Dict[str, Any]:
        """
        Check health status.
        
        Args:
            target_id: Target identifier
        
        Returns:
            Health status dictionary
        """
        async with self.lock:
            result = self.results.get(target_id)
            if not result:
                return {"status": HealthStatus.UNKNOWN.value}
            
            return {
                "status": result.status.value,
                "message": result.message,
                "checked_at": result.checked_at.isoformat(),
                "response_time_ms": result.response_time_ms,
                "consecutive_failures": result.consecutive_failures,
                "consecutive_successes": result.consecutive_successes
            }
    
    async def _perform_check(self, target_id: str) -> HealthResult:
        """Perform a single health check."""
        check = self.checks.get(target_id)
        if not check:
            return HealthResult(status=HealthStatus.UNKNOWN)
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Execute check function
            if check.check_fn:
                if asyncio.iscoroutinefunction(check.check_fn):
                    success = await asyncio.wait_for(
                        check.check_fn(),
                        timeout=check.timeout
                    )
                else:
                    success = check.check_fn()
            else:
                # Default check (always healthy)
                success = True
            
            # Calculate response time
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Get previous result
            prev_result = self.results.get(target_id, HealthResult(status=HealthStatus.UNKNOWN))
            
            if success:
                consecutive_successes = prev_result.consecutive_successes + 1
                consecutive_failures = 0
                
                # Determine status based on thresholds
                if consecutive_successes >= check.success_threshold:
                    status = HealthStatus.HEALTHY
                    message = "Health check passed"
                else:
                    status = prev_result.status
                    message = f"Recovering ({consecutive_successes}/{check.success_threshold})"
            
            else:
                consecutive_failures = prev_result.consecutive_failures + 1
                consecutive_successes = 0
                
                # Determine status based on thresholds
                if consecutive_failures >= check.failure_threshold:
                    status = HealthStatus.UNHEALTHY
                    message = "Health check failed"
                else:
                    status = HealthStatus.DEGRADED
                    message = f"Degraded ({consecutive_failures}/{check.failure_threshold})"
            
            return HealthResult(
                status=status,
                message=message,
                response_time_ms=response_time,
                consecutive_failures=consecutive_failures,
                consecutive_successes=consecutive_successes
            )
        
        except asyncio.TimeoutError:
            prev_result = self.results.get(target_id, HealthResult(status=HealthStatus.UNKNOWN))
            consecutive_failures = prev_result.consecutive_failures + 1
            
            return HealthResult(
                status=HealthStatus.UNHEALTHY if consecutive_failures >= check.failure_threshold else HealthStatus.DEGRADED,
                message=f"Health check timeout after {check.timeout}s",
                consecutive_failures=consecutive_failures,
                consecutive_successes=0
            )
        
        except Exception as e:
            prev_result = self.results.get(target_id, HealthResult(status=HealthStatus.UNKNOWN))
            consecutive_failures = prev_result.consecutive_failures + 1
            
            return HealthResult(
                status=HealthStatus.UNHEALTHY if consecutive_failures >= check.failure_threshold else HealthStatus.DEGRADED,
                message=f"Health check error: {str(e)}",
                consecutive_failures=consecutive_failures,
                consecutive_successes=0
            )
    
    async def _check_loop(self, target_id: str):
        """Health check monitoring loop."""
        check = self.checks.get(target_id)
        if not check:
            return
        
        # Initial delay
        if check.initial_delay > 0:
            await asyncio.sleep(check.initial_delay)
        
        while True:
            try:
                # Perform check
                result = await self._perform_check(target_id)
                
                # Update result
                async with self.lock:
                    self.results[target_id] = result
                
                # Log status changes
                prev_status = self.results.get(target_id)
                if not prev_status or prev_status.status != result.status:
                    self.logger.info(
                        "Health status changed",
                        target_id=target_id,
                        status=result.status.value,
                        message=result.message
                    )
                
                # Record metrics
                get_metrics().increment_counter(
                    'memory_operations',
                    {'memory_type': 'health_checks', 'operation': result.status.value},
                    1
                )
                
                # Wait for next check
                await asyncio.sleep(check.interval)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    "Health check loop error",
                    target_id=target_id,
                    error=str(e)
                )
                await asyncio.sleep(check.interval)
    
    async def _start_check_monitor(self, target_id: str):
        """Start health check monitor for target."""
        if target_id in self.monitor_tasks:
            return
        
        task = asyncio.create_task(self._check_loop(target_id))
        self.monitor_tasks[target_id] = task
    
    async def start_monitoring(self):
        """Start monitoring all registered checks."""
        if self.running:
            return
        
        self.running = True
        self.logger.info("Health monitoring started")
        
        # Start monitors for all checks
        for target_id in self.checks.keys():
            await self._start_check_monitor(target_id)
    
    async def stop_monitoring(self):
        """Stop all monitoring."""
        self.running = False
        
        # Cancel all tasks
        for task in self.monitor_tasks.values():
            task.cancel()
        
        await asyncio.gather(*self.monitor_tasks.values(), return_exceptions=True)
        self.monitor_tasks.clear()
        
        self.logger.info("Health monitoring stopped")
    
    async def get_overall_health(self) -> Dict[str, Any]:
        """
        Get overall health status.
        
        Returns:
            Aggregated health status
        """
        async with self.lock:
            if not self.results:
                return {
                    "status": HealthStatus.UNKNOWN.value,
                    "checks": 0
                }
            
            # Count statuses
            status_counts = {
                HealthStatus.HEALTHY: 0,
                HealthStatus.DEGRADED: 0,
                HealthStatus.UNHEALTHY: 0,
                HealthStatus.UNKNOWN: 0
            }
            
            for result in self.results.values():
                status_counts[result.status] += 1
            
            # Determine overall status
            if status_counts[HealthStatus.UNHEALTHY] > 0:
                overall_status = HealthStatus.UNHEALTHY
            elif status_counts[HealthStatus.DEGRADED] > 0:
                overall_status = HealthStatus.DEGRADED
            elif status_counts[HealthStatus.HEALTHY] > 0:
                overall_status = HealthStatus.HEALTHY
            else:
                overall_status = HealthStatus.UNKNOWN
            
            return {
                "status": overall_status.value,
                "checks": len(self.results),
                "healthy": status_counts[HealthStatus.HEALTHY],
                "degraded": status_counts[HealthStatus.DEGRADED],
                "unhealthy": status_counts[HealthStatus.UNHEALTHY],
                "unknown": status_counts[HealthStatus.UNKNOWN]
            }
    
    async def shutdown(self):
        """Shutdown health checker."""
        await self.stop_monitoring()
        self.logger.info("Health checker shutdown complete")

