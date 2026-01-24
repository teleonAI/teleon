"""
Monitoring Integration for Cortex Memory System.

Provides metrics collection, health checks, and observability for production deployments.
"""

import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("teleon.cortex.monitoring")


@dataclass
class OperationMetrics:
    """Metrics for a single operation type."""
    count: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    latency_samples: List[float] = field(default_factory=list)
    
    def record(self, latency_ms: float, error: bool = False) -> None:
        """Record an operation."""
        self.count += 1
        if error:
            self.errors += 1
        else:
            self.total_latency_ms += latency_ms
            self.latency_samples.append(latency_ms)
            self.min_latency_ms = min(self.min_latency_ms, latency_ms)
            self.max_latency_ms = max(self.max_latency_ms, latency_ms)
            
            # Keep only last 1000 samples for percentile calculation
            if len(self.latency_samples) > 1000:
                self.latency_samples = self.latency_samples[-1000:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for this operation."""
        if self.count == 0:
            return {
                "count": 0,
                "errors": 0,
                "error_rate": 0.0,
                "avg_latency_ms": 0.0,
                "min_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
            }
        
        error_rate = (self.errors / self.count) * 100
        
        if self.latency_samples:
            sorted_samples = sorted(self.latency_samples)
            p50 = sorted_samples[len(sorted_samples) // 2]
            p95 = sorted_samples[int(len(sorted_samples) * 0.95)]
            p99 = sorted_samples[int(len(sorted_samples) * 0.99)]
            avg = sum(self.latency_samples) / len(self.latency_samples)
        else:
            p50 = p95 = p99 = avg = 0.0
        
        return {
            "count": self.count,
            "errors": self.errors,
            "error_rate": round(error_rate, 2),
            "avg_latency_ms": round(avg, 2),
            "min_latency_ms": round(self.min_latency_ms if self.min_latency_ms != float('inf') else 0.0, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
            "p99_latency_ms": round(p99, 2),
        }


class CortexMetrics:
    """
    Metrics collector for Cortex memory operations.
    
    Tracks:
    - Operation counts and error rates
    - Latency percentiles (p50, p95, p99)
    - Memory sizes
    - Cache hit rates
    - Storage backend health
    
    Example:
        ```python
        metrics = CortexMetrics(agent_id="agent-123")
        
        # Record operations
        with metrics.record_operation("episodic", "store"):
            await episodic.store(episode)
        
        # Get metrics
        stats = metrics.get_metrics()
        ```
    """
    
    def __init__(self, agent_id: str):
        """
        Initialize metrics collector.
        
        Args:
            agent_id: Agent ID for this metrics instance
        """
        self.agent_id = agent_id
        self._operation_metrics: Dict[str, Dict[str, OperationMetrics]] = defaultdict(
            lambda: defaultdict(OperationMetrics)
        )
        self._start_time = time.time()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def record_operation(
        self,
        memory_type: str,
        operation: str,
        latency_ms: float,
        error: bool = False
    ) -> None:
        """
        Record an operation.
        
        Args:
            memory_type: Memory type (episodic, semantic, procedural, working)
            operation: Operation name (store, get, search, delete, etc.)
            latency_ms: Operation latency in milliseconds
            error: Whether operation failed
        """
        self._operation_metrics[memory_type][operation].record(latency_ms, error)
    
    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self._cache_hits += 1
    
    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        self._cache_misses += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics.
        
        Returns:
            Dictionary with comprehensive metrics
        """
        metrics = {
            "agent_id": self.agent_id,
            "uptime_seconds": round(time.time() - self._start_time, 2),
            "operations": {},
            "cache": {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "hit_rate": round(
                    (self._cache_hits / (self._cache_hits + self._cache_misses) * 100)
                    if (self._cache_hits + self._cache_misses) > 0 else 0.0,
                    2
                )
            }
        }
        
        # Aggregate operation metrics
        for memory_type, operations in self._operation_metrics.items():
            metrics["operations"][memory_type] = {
                op: metrics_obj.get_stats()
                for op, metrics_obj in operations.items()
            }
        
        return metrics
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._operation_metrics.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._start_time = time.time()


class PerformanceProfiler:
    """
    Performance profiler for Cortex operations.
    
    Tracks slow operations, identifies bottlenecks, and provides recommendations.
    
    Example:
        ```python
        profiler = PerformanceProfiler()
        
        @profiler.profile("episodic.store")
        async def store_episode(episode):
            # ... operation ...
        
        # Get performance report
        report = profiler.get_performance_report()
        ```
    """
    
    def __init__(self, slow_threshold_ms: float = 100.0):
        """
        Initialize performance profiler.
        
        Args:
            slow_threshold_ms: Threshold for slow operations (ms)
        """
        self.slow_threshold_ms = slow_threshold_ms
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.slow_operations: List[Dict[str, Any]] = []
        self._lock = None  # Will be set if needed
    
    def profile(self, operation_name: str):
        """
        Decorator to profile async operations.
        
        Args:
            operation_name: Name of the operation being profiled
        """
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration_ms = (time.perf_counter() - start) * 1000
                    self.operation_times[operation_name].append(duration_ms)
                    self.operation_counts[operation_name] += 1
                    
                    # Track slow operations
                    if duration_ms > self.slow_threshold_ms:
                        self.slow_operations.append({
                            'operation': operation_name,
                            'duration_ms': duration_ms,
                            'timestamp': time.time(),
                            'args_preview': str(args)[:100] if args else ""
                        })
                        # Keep only last 100 slow operations
                        if len(self.slow_operations) > 100:
                            self.slow_operations = self.slow_operations[-100:]
            return wrapper
        return decorator
    
    def record_operation(self, operation_name: str, duration_ms: float) -> None:
        """
        Record an operation manually.
        
        Args:
            operation_name: Name of the operation
            duration_ms: Duration in milliseconds
        """
        self.operation_times[operation_name].append(duration_ms)
        self.operation_counts[operation_name] += 1
        
        if duration_ms > self.slow_threshold_ms:
            self.slow_operations.append({
                'operation': operation_name,
                'duration_ms': duration_ms,
                'timestamp': time.time()
            })
            if len(self.slow_operations) > 100:
                self.slow_operations = self.slow_operations[-100:]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance stats
        """
        stats = {}
        for op_name, times in self.operation_times.items():
            if times:
                sorted_times = sorted(times)
                stats[op_name] = {
                    'count': len(times),
                    'total_ms': sum(times),
                    'avg_ms': sum(times) / len(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                    'p50_ms': sorted_times[len(times) // 2],
                    'p95_ms': sorted_times[int(len(times) * 0.95)],
                    'p99_ms': sorted_times[int(len(times) * 0.99)],
                }
        return stats
    
    def get_slow_operations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get slowest operations.
        
        Args:
            limit: Maximum number of operations to return
        
        Returns:
            List of slow operations sorted by duration
        """
        return sorted(
            self.slow_operations,
            key=lambda x: x['duration_ms'],
            reverse=True
        )[:limit]
    
    def get_recommendations(self) -> List[str]:
        """
        Get performance optimization recommendations.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Analyze operation stats
        for op_name, times in self.operation_times.items():
            if not times:
                continue
            
            avg_ms = sum(times) / len(times)
            p95_ms = sorted(times)[int(len(times) * 0.95)]
            
            # Check for slow operations
            if "search" in op_name.lower() and avg_ms > 50:
                recommendations.append(
                    f"Consider enabling caching for {op_name} "
                    f"(avg latency: {avg_ms:.1f}ms)"
                )
            
            if "batch" not in op_name.lower() and len(times) > 100:
                recommendations.append(
                    f"Consider using batch operations for {op_name} "
                    f"({len(times)} individual operations)"
                )
            
            if p95_ms > 200:
                recommendations.append(
                    f"{op_name} has high p95 latency ({p95_ms:.1f}ms). "
                    f"Consider optimization."
                )
        
        return recommendations
    
    def reset(self) -> None:
        """Reset profiler data."""
        self.operation_times.clear()
        self.operation_counts.clear()
        self.slow_operations.clear()

