"""
Agent Metrics Reporter - Reports metrics from deployed agents to Teleon Platform.

This module provides a lightweight metrics reporting system that runs inside
deployed agents and sends telemetry data back to the Teleon Platform for
monitoring, billing, and analytics.
"""

import os
import asyncio
import time
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass, field
from collections import deque

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


logger = logging.getLogger("teleon.agent_reporter")


@dataclass
class RequestMetric:
    """Single request metric record."""
    timestamp: float
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    success: bool
    cost: float = 0.0
    error_type: Optional[str] = None


@dataclass
class HealthMetric:
    """Health metric record."""
    timestamp: float
    status: str  # healthy, degraded, unhealthy
    cpu_percent: float
    memory_percent: float
    active_requests: int
    queue_depth: int


@dataclass
class MetricsBatch:
    """Batch of metrics to send to platform."""
    deployment_id: str
    agent_id: str
    requests: List[Dict[str, Any]] = field(default_factory=list)
    health: Optional[Dict[str, Any]] = None
    aggregated: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AgentMetricsReporter:
    """
    Reports metrics from deployed agents back to Teleon Platform.
    
    Features:
    - Batched metric reporting for efficiency
    - Automatic retry with exponential backoff
    - Local buffering when platform is unreachable
    - Token, latency, and cost tracking
    - Health status reporting
    
    Environment Variables:
    - TELEON_DEPLOYMENT_ID: Unique deployment identifier
    - TELEON_API_KEY: API key for authentication
    - TELEON_PLATFORM_URL: Platform API URL (default: https://api.teleon.ai)
    - TELEON_METRICS_INTERVAL: Reporting interval in seconds (default: 10)
    - TELEON_METRICS_BATCH_SIZE: Max batch size before flush (default: 100)
    
    Example:
        ```python
        reporter = AgentMetricsReporter()
        await reporter.start()
        
        # Report a request
        await reporter.report_request(
            input_tokens=100,
            output_tokens=50,
            latency_ms=250.5,
            model="gpt-4",
            success=True
        )
        
        # Stop and flush remaining metrics
        await reporter.stop()
        ```
    """
    
    def __init__(
        self,
        deployment_id: Optional[str] = None,
        api_key: Optional[str] = None,
        platform_url: Optional[str] = None,
        flush_interval: float = 10.0,
        batch_size: int = 100,
        max_buffer_size: int = 10000,
    ):
        """
        Initialize the metrics reporter.
        
        Args:
            deployment_id: Deployment ID (from env if not provided)
            api_key: API key (from env if not provided)
            platform_url: Platform URL (from env if not provided)
            flush_interval: Seconds between automatic flushes
            batch_size: Number of requests before automatic flush
            max_buffer_size: Maximum buffered metrics before dropping
        """
        self.deployment_id = deployment_id or os.getenv("TELEON_DEPLOYMENT_ID", "")
        self.api_key = api_key or os.getenv("TELEON_API_KEY", "")
        self.platform_url = platform_url or os.getenv(
            "TELEON_PLATFORM_URL", 
            "https://api.teleon.ai"
        )
        self.agent_id = os.getenv("TELEON_AGENT_ID", "default")
        
        self.flush_interval = float(os.getenv("TELEON_METRICS_INTERVAL", str(flush_interval)))
        self.batch_size = int(os.getenv("TELEON_METRICS_BATCH_SIZE", str(batch_size)))
        self.max_buffer_size = max_buffer_size
        
        # Metrics buffer
        self._request_buffer: deque = deque(maxlen=max_buffer_size)
        self._last_health: Optional[HealthMetric] = None
        
        # Aggregated stats (for the current reporting period)
        self._period_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_latency_ms": 0.0,
            "total_cost": 0.0,
            "min_latency_ms": float("inf"),
            "max_latency_ms": 0.0,
            "models": {},
        }
        
        # State
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        
        # Retry settings
        self._max_retries = 3
        self._retry_delay = 1.0
        
        # Track failed sends for later retry
        self._failed_batches: deque = deque(maxlen=100)
        
        # Circuit breaker state
        self._circuit_open = False
        self._circuit_failures = 0
        self._circuit_failure_threshold = 5
        self._circuit_reset_time = 60.0  # seconds
        self._circuit_last_failure: Optional[float] = None
        
        # Request size limits (10MB)
        self._max_payload_size = 10 * 1024 * 1024  # 10MB
        
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available - metrics will be logged locally only")
    
    @property
    def is_configured(self) -> bool:
        """Check if reporter is properly configured."""
        return bool(self.deployment_id and self.api_key)
    
    async def start(self):
        """Start the metrics reporter background task."""
        if self._running:
            return
        
        if not self.is_configured:
            logger.warning(
                "AgentMetricsReporter not configured - "
                "set TELEON_DEPLOYMENT_ID and TELEON_API_KEY"
            )
            return
        
        self._running = True
        
        if AIOHTTP_AVAILABLE:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "X-Teleon-Deployment-ID": self.deployment_id,
                }
            )
        
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info(
            f"AgentMetricsReporter started - "
            f"deployment={self.deployment_id}, interval={self.flush_interval}s"
        )
    
    async def stop(self):
        """Stop the reporter and flush remaining metrics."""
        if not self._running:
            return
        
        self._running = False
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self.flush()
        
        if self._session:
            await self._session.close()
            self._session = None
        
        logger.info("AgentMetricsReporter stopped")
    
    async def report_request(
        self,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        model: str,
        success: bool,
        cost: Optional[float] = None,
        error_type: Optional[str] = None,
    ):
        """
        Report a single LLM request.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            latency_ms: Request latency in milliseconds
            model: Model name used
            success: Whether request was successful
            cost: Optional cost of request
            error_type: Error type if request failed
        """
        metric = RequestMetric(
            timestamp=time.time(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model=model,
            success=success,
            cost=cost or 0.0,
            error_type=error_type,
        )
        
        async with self._lock:
            # Check buffer size limit
            if len(self._request_buffer) >= self.max_buffer_size:
                # Drop oldest metric to prevent unbounded growth
                self._request_buffer.popleft()
                logger.warning("Request buffer full, dropping oldest metric")
            
            self._request_buffer.append(metric)
            self._update_period_stats(metric)
        
        # Auto-flush if batch size reached
        if len(self._request_buffer) >= self.batch_size:
            asyncio.create_task(self.flush())
    
    async def report_health(
        self,
        status: str = "healthy",
        active_requests: int = 0,
        queue_depth: int = 0,
    ):
        """
        Report agent health status.
        
        Args:
            status: Health status (healthy, degraded, unhealthy)
            active_requests: Number of active requests
            queue_depth: Number of queued requests
        """
        cpu_percent = 0.0
        memory_percent = 0.0
        
        if PSUTIL_AVAILABLE:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
            except Exception:
                pass
        
        self._last_health = HealthMetric(
            timestamp=time.time(),
            status=status,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            active_requests=active_requests,
            queue_depth=queue_depth,
        )
    
    async def flush(self):
        """Flush buffered metrics to the platform."""
        # Use a separate lock for building the batch to minimize lock time
        batch = None
        
        async with self._lock:
            if not self._request_buffer and not self._last_health:
                return
            
            # Build the batch
            batch = MetricsBatch(
                deployment_id=self.deployment_id,
                agent_id=self.agent_id,
            )
            
            # Add request metrics (with size limit protection)
            batch_size = 0
            while self._request_buffer:
                metric = self._request_buffer.popleft()
                metric_dict = {
                    "timestamp": metric.timestamp,
                    "input_tokens": metric.input_tokens,
                    "output_tokens": metric.output_tokens,
                    "latency_ms": metric.latency_ms,
                    "model": metric.model,
                    "success": metric.success,
                    "cost": metric.cost,
                    "error_type": metric.error_type,
                }
                
                # Estimate payload size (rough estimate)
                estimated_size = len(str(metric_dict)) * 2  # Rough bytes estimate
                if batch_size + estimated_size > self._max_payload_size:
                    # Put metric back and flush what we have
                    self._request_buffer.appendleft(metric)
                    break
                
                batch.requests.append(metric_dict)
                batch_size += estimated_size
            
            # Add health metrics
            if self._last_health:
                batch.health = {
                    "timestamp": self._last_health.timestamp,
                    "status": self._last_health.status,
                    "cpu_percent": self._last_health.cpu_percent,
                    "memory_percent": self._last_health.memory_percent,
                    "active_requests": self._last_health.active_requests,
                    "queue_depth": self._last_health.queue_depth,
                }
            
            # Add aggregated stats
            batch.aggregated = self._get_and_reset_period_stats()
        
        # Send to platform (outside lock to avoid blocking)
        if batch:
            await self._send_batch(batch)
    
    async def _send_batch(self, batch: MetricsBatch):
        """Send a metrics batch to the platform."""
        if not self._session:
            # Log locally if no session
            logger.info(f"Metrics batch (local): {len(batch.requests)} requests")
            return
        
        # Check circuit breaker
        if self._circuit_open:
            current_time = time.time()
            if self._circuit_last_failure and (current_time - self._circuit_last_failure) > self._circuit_reset_time:
                # Try to reset circuit
                self._circuit_open = False
                self._circuit_failures = 0
                logger.info("Circuit breaker reset - attempting to send metrics")
            else:
                # Circuit still open, buffer for later
                logger.debug("Circuit breaker open - buffering batch")
                payload = {
                    "deployment_id": batch.deployment_id,
                    "agent_id": batch.agent_id,
                    "requests": batch.requests,
                    "health": batch.health,
                    "aggregated": batch.aggregated,
                    "timestamp": batch.timestamp,
                }
                if len(self._failed_batches) < 100:
                    self._failed_batches.append(payload)
                return
        
        url = f"{self.platform_url}/api/v1/agents/{self.deployment_id}/metrics"
        payload = {
            "deployment_id": batch.deployment_id,
            "agent_id": batch.agent_id,
            "requests": batch.requests,
            "health": batch.health,
            "aggregated": batch.aggregated,
            "timestamp": batch.timestamp,
        }
        
        # Check payload size
        import json
        payload_size = len(json.dumps(payload).encode('utf-8'))
        if payload_size > self._max_payload_size:
            logger.warning(
                f"Payload too large ({payload_size} bytes), splitting batch"
            )
            # Split into smaller batches
            chunk_size = len(batch.requests) // 2
            if chunk_size > 0:
                batch1 = MetricsBatch(
                    deployment_id=batch.deployment_id,
                    agent_id=batch.agent_id,
                    requests=batch.requests[:chunk_size],
                    health=batch.health,
                    aggregated=batch.aggregated,
                    timestamp=batch.timestamp,
                )
                batch2 = MetricsBatch(
                    deployment_id=batch.deployment_id,
                    agent_id=batch.agent_id,
                    requests=batch.requests[chunk_size:],
                    health=None,
                    aggregated={},
                    timestamp=batch.timestamp,
                )
                await self._send_batch(batch1)
                await self._send_batch(batch2)
            return
        
        # Create timeout
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        
        for attempt in range(self._max_retries):
            try:
                async with self._session.post(url, json=payload, timeout=timeout) as response:
                    if response.status == 200:
                        logger.debug(f"Sent metrics batch: {len(batch.requests)} requests")
                        # Reset circuit breaker on success
                        self._circuit_open = False
                        self._circuit_failures = 0
                        return
                    elif response.status == 429:
                        # Rate limited - wait longer
                        await asyncio.sleep(self._retry_delay * (2 ** attempt) * 2)
                    else:
                        error_text = await response.text()
                        logger.warning(
                            f"Failed to send metrics (status={response.status}): {error_text}"
                        )
                        self._circuit_failures += 1
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout sending metrics (attempt {attempt + 1}/{self._max_retries})")
                self._circuit_failures += 1
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay * (2 ** attempt))
            except aiohttp.ClientError as e:
                logger.warning(f"Network error sending metrics: {e}")
                self._circuit_failures += 1
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay * (2 ** attempt))
            except Exception as e:
                logger.error(f"Unexpected error sending metrics: {e}")
                self._circuit_failures += 1
                break
        
        # Check if we should open circuit breaker
        if self._circuit_failures >= self._circuit_failure_threshold:
            self._circuit_open = True
            self._circuit_last_failure = time.time()
            logger.warning(
                f"Circuit breaker opened after {self._circuit_failures} failures"
            )
        
        # Store for later retry
        if len(self._failed_batches) < 100:
            self._failed_batches.append(payload)
            logger.warning(f"Buffered failed batch for retry ({len(self._failed_batches)} pending)")
    
    async def _flush_loop(self):
        """Background task that periodically flushes metrics."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)

                # Collect current health metrics before flushing
                await self.report_health(status="healthy")

                await self.flush()

                # Also retry failed batches
                await self._retry_failed_batches()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")
    
    async def _retry_failed_batches(self):
        """Retry sending previously failed batches."""
        if not self._failed_batches or not self._session:
            return
        
        url = f"{self.platform_url}/api/v1/agents/{self.deployment_id}/metrics"
        
        # Try to send up to 5 failed batches per cycle
        for _ in range(min(5, len(self._failed_batches))):
            if not self._failed_batches:
                break
            
            payload = self._failed_batches.popleft()
            
            try:
                async with self._session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.debug("Successfully sent previously failed batch")
                    else:
                        # Put it back for later
                        self._failed_batches.append(payload)
                        break
            except Exception:
                self._failed_batches.append(payload)
                break
    
    def _update_period_stats(self, metric: RequestMetric):
        """Update aggregated stats for the period."""
        self._period_stats["total_requests"] += 1
        
        if metric.success:
            self._period_stats["successful_requests"] += 1
        else:
            self._period_stats["failed_requests"] += 1
        
        self._period_stats["total_input_tokens"] += metric.input_tokens
        self._period_stats["total_output_tokens"] += metric.output_tokens
        self._period_stats["total_latency_ms"] += metric.latency_ms
        self._period_stats["total_cost"] += metric.cost
        
        self._period_stats["min_latency_ms"] = min(
            self._period_stats["min_latency_ms"], 
            metric.latency_ms
        )
        self._period_stats["max_latency_ms"] = max(
            self._period_stats["max_latency_ms"], 
            metric.latency_ms
        )
        
        # Track by model
        if metric.model not in self._period_stats["models"]:
            self._period_stats["models"][metric.model] = {
                "requests": 0,
                "tokens": 0,
                "cost": 0.0,
            }
        
        self._period_stats["models"][metric.model]["requests"] += 1
        self._period_stats["models"][metric.model]["tokens"] += (
            metric.input_tokens + metric.output_tokens
        )
        self._period_stats["models"][metric.model]["cost"] += metric.cost
    
    def _get_and_reset_period_stats(self) -> Dict[str, Any]:
        """Get current period stats and reset for next period."""
        stats = dict(self._period_stats)
        
        # Calculate averages
        if stats["total_requests"] > 0:
            stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["total_requests"]
            stats["avg_tokens_per_request"] = (
                (stats["total_input_tokens"] + stats["total_output_tokens"]) 
                / stats["total_requests"]
            )
        else:
            stats["avg_latency_ms"] = 0.0
            stats["avg_tokens_per_request"] = 0.0
        
        # Fix infinity
        if stats["min_latency_ms"] == float("inf"):
            stats["min_latency_ms"] = 0.0
        
        # Reset
        self._period_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_latency_ms": 0.0,
            "total_cost": 0.0,
            "min_latency_ms": float("inf"),
            "max_latency_ms": 0.0,
            "models": {},
        }
        
        return stats
    
    def get_local_stats(self) -> Dict[str, Any]:
        """Get current local statistics without sending."""
        return {
            "deployment_id": self.deployment_id,
            "agent_id": self.agent_id,
            "buffered_requests": len(self._request_buffer),
            "failed_batches": len(self._failed_batches),
            "period_stats": dict(self._period_stats),
            "last_health": self._last_health.__dict__ if self._last_health else None,
            "is_running": self._running,
            "is_configured": self.is_configured,
        }


# Global reporter instance
_global_reporter: Optional[AgentMetricsReporter] = None


def get_agent_reporter() -> AgentMetricsReporter:
    """
    Get or create the global agent metrics reporter.
    
    Returns:
        Global AgentMetricsReporter instance
    """
    global _global_reporter
    if _global_reporter is None:
        _global_reporter = AgentMetricsReporter()
    return _global_reporter


async def init_agent_reporter(**kwargs) -> AgentMetricsReporter:
    """
    Initialize and start the global agent metrics reporter.
    
    Args:
        **kwargs: Arguments passed to AgentMetricsReporter
    
    Returns:
        Started AgentMetricsReporter instance
    """
    global _global_reporter
    _global_reporter = AgentMetricsReporter(**kwargs)
    await _global_reporter.start()
    return _global_reporter


async def shutdown_agent_reporter():
    """Shutdown the global agent metrics reporter."""
    global _global_reporter
    if _global_reporter:
        await _global_reporter.stop()
        _global_reporter = None



