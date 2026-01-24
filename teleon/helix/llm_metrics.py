"""
LLM-Specific Metrics Tracking and Collection.

This module provides comprehensive metrics tracking for LLM agent workloads,
including token throughput, latency, cost, and queue depth monitoring.
"""

from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field, ConfigDict, field_serializer
import asyncio
import time
from collections import deque

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from teleon.core import (
    get_metrics,
    StructuredLogger,
    LogLevel,
)


class LLMMetrics(BaseModel):
    """
    LLM-specific metrics snapshot.
    
    Captures key metrics for LLM agent performance monitoring and scaling decisions.
    """
    
    # Throughput metrics
    tokens_per_second: float = Field(0.0, description="Current token processing rate")
    requests_per_minute: float = Field(0.0, description="Request rate")
    input_tokens_per_second: float = Field(0.0, description="Input token rate")
    output_tokens_per_second: float = Field(0.0, description="Output token rate")
    
    # Queue metrics
    queue_depth: int = Field(0, description="Number of queued requests")
    avg_wait_time_ms: float = Field(0.0, description="Average queue wait time")
    
    # Latency metrics
    avg_ttft_ms: float = Field(0.0, description="Avg time to first token")
    avg_completion_time_ms: float = Field(0.0, description="Avg completion time")
    p50_latency_ms: float = Field(0.0, description="P50 latency")
    p95_latency_ms: float = Field(0.0, description="P95 latency")
    p99_latency_ms: float = Field(0.0, description="P99 latency")
    
    # Concurrency
    concurrent_llm_calls: int = Field(0, description="Active LLM calls")
    max_concurrent_calls: int = Field(10, description="Max concurrent calls")
    
    # Cost metrics
    cost_per_hour: float = Field(0.0, description="Current hourly cost")
    cost_per_request: float = Field(0.0, description="Average cost per request")
    total_cost: float = Field(0.0, description="Total cost accumulated")
    
    # Context metrics
    avg_context_tokens: int = Field(0, description="Average context size")
    max_context_tokens: int = Field(0, description="Max context seen")
    avg_response_tokens: int = Field(0, description="Average response size")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict()

    @field_serializer('timestamp')
    def serialize_datetime(self, value: datetime) -> str:
        return value.isoformat() if value else None
    
    def get_utilization(self) -> float:
        """Calculate utilization percentage (0-1)."""
        if self.max_concurrent_calls == 0:
            return 0.0
        return self.concurrent_llm_calls / self.max_concurrent_calls
    
    def is_overloaded(self) -> bool:
        """Check if system is overloaded."""
        return (
            self.queue_depth > 10 or
            self.avg_wait_time_ms > 5000 or
            self.p95_latency_ms > 10000 or
            self.get_utilization() > 0.9
        )
    
    def is_underutilized(self) -> bool:
        """Check if system is underutilized."""
        return (
            self.queue_depth == 0 and
            self.get_utilization() < 0.3 and
            self.requests_per_minute < 1.0
        )


class TokenCounter:
    """
    Token counting utility using tiktoken.
    
    Provides accurate token counting for different models.
    """
    
    def __init__(self):
        """Initialize token counter."""
        self.encoders: Dict[str, Any] = {}
        self.logger = StructuredLogger("token_counter", LogLevel.INFO)
        
        if not TIKTOKEN_AVAILABLE:
            self.logger.warning("tiktoken not available, using fallback counter")
    
    def get_encoder(self, model: str):
        """Get encoder for model."""
        if not TIKTOKEN_AVAILABLE:
            return None
        
        if model not in self.encoders:
            try:
                # Try to get model-specific encoder
                self.encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base (GPT-4, GPT-3.5-turbo)
                self.encoders[model] = tiktoken.get_encoding("cl100k_base")
        
        return self.encoders[model]
    
    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """
        Count tokens in text for given model.
        
        Args:
            text: Text to count
            model: Model name
        
        Returns:
            Token count
        """
        if not text:
            return 0
        
        encoder = self.get_encoder(model)
        
        if encoder:
            return len(encoder.encode(text))
        else:
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4
    
    def count_messages_tokens(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4"
    ) -> int:
        """
        Count tokens in message list (for chat models).
        
        Args:
            messages: List of message dicts
            model: Model name
        
        Returns:
            Total token count
        """
        # Per-message overhead
        tokens_per_message = 3  # Every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = 1  # If there's a name, the role is omitted
        
        total_tokens = 0
        
        for message in messages:
            total_tokens += tokens_per_message
            
            for key, value in message.items():
                total_tokens += self.count_tokens(str(value), model)
                
                if key == "name":
                    total_tokens += tokens_per_name
        
        total_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>
        
        return total_tokens


class LLMResourceTracker:
    """
    Track LLM-specific resource usage.
    
    Monitors token throughput, costs, latency, and queue metrics
    for LLM agent instances.
    """
    
    def __init__(
        self,
        agent_id: str,
        model: str = "gpt-4",
        window_size: int = 300  # 5 minutes
    ):
        """
        Initialize LLM resource tracker.
        
        Args:
            agent_id: Agent identifier
            model: LLM model name
            window_size: Time window for metrics (seconds)
        """
        self.agent_id = agent_id
        self.model = model
        self.window_size = window_size
        
        # Token counter
        self.token_counter = TokenCounter()
        
        # Time-series data (use deque for efficient windowing)
        self.request_times = deque(maxlen=1000)
        self.latencies = deque(maxlen=1000)
        self.ttft_times = deque(maxlen=1000)
        self.costs = deque(maxlen=1000)
        self.input_tokens = deque(maxlen=1000)
        self.output_tokens = deque(maxlen=1000)
        self.context_sizes = deque(maxlen=1000)
        
        # Queue tracking
        self.queue_depth = 0
        self.wait_times = deque(maxlen=1000)
        
        # Concurrency tracking
        self.concurrent_calls = 0
        self.max_concurrent = 10
        
        # Cost tracking
        self.total_cost = 0.0
        
        self.lock = asyncio.Lock()
        self.logger = StructuredLogger(f"llm_tracker.{agent_id}", LogLevel.INFO)
    
    async def record_request(
        self,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        ttft_ms: Optional[float] = None,
        cost: Optional[float] = None,
        wait_time_ms: float = 0.0
    ):
        """
        Record a completed LLM request.
        
        Args:
            input_tokens: Input token count
            output_tokens: Output token count
            latency_ms: Total latency
            ttft_ms: Time to first token
            cost: Request cost
            wait_time_ms: Time spent in queue
        """
        async with self.lock:
            now = time.time()
            
            self.request_times.append(now)
            self.latencies.append(latency_ms)
            self.input_tokens.append(input_tokens)
            self.output_tokens.append(output_tokens)
            self.context_sizes.append(input_tokens)
            
            if ttft_ms:
                self.ttft_times.append(ttft_ms)
            
            if cost:
                self.costs.append(cost)
                self.total_cost += cost
            
            if wait_time_ms > 0:
                self.wait_times.append(wait_time_ms)
            
            # Record to global metrics
            get_metrics().increment_counter(
                'llm_requests_total',
                {'agent_id': self.agent_id, 'model': self.model},
                1
            )
            
            get_metrics().record_histogram(
                'llm_latency_ms',
                {'agent_id': self.agent_id, 'model': self.model},
                latency_ms
            )
    
    async def increment_queue(self):
        """Increment queue depth."""
        async with self.lock:
            self.queue_depth += 1
    
    async def decrement_queue(self):
        """Decrement queue depth."""
        async with self.lock:
            self.queue_depth = max(0, self.queue_depth - 1)
    
    async def increment_concurrent(self):
        """Increment concurrent call counter."""
        async with self.lock:
            self.concurrent_calls += 1
    
    async def decrement_concurrent(self):
        """Decrement concurrent call counter."""
        async with self.lock:
            self.concurrent_calls = max(0, self.concurrent_calls - 1)
    
    async def get_metrics(self) -> LLMMetrics:
        """
        Get current LLM metrics.
        
        Returns:
            LLMMetrics snapshot
        """
        async with self.lock:
            now = time.time()
            cutoff = now - self.window_size
            
            # Filter to time window
            recent_requests = [t for t in self.request_times if t > cutoff]
            recent_latencies = list(self.latencies)[-len(recent_requests):]
            recent_input = list(self.input_tokens)[-len(recent_requests):]
            recent_output = list(self.output_tokens)[-len(recent_requests):]
            recent_costs = list(self.costs)[-len(recent_requests):]
            recent_contexts = list(self.context_sizes)[-len(recent_requests):]
            
            # Calculate throughput
            if recent_requests:
                time_span = now - recent_requests[0] if len(recent_requests) > 1 else 1.0
                requests_per_minute = (len(recent_requests) / time_span) * 60 if time_span > 0 else 0
                
                total_input = sum(recent_input)
                total_output = sum(recent_output)
                
                tokens_per_second = (total_input + total_output) / time_span if time_span > 0 else 0
                input_tps = total_input / time_span if time_span > 0 else 0
                output_tps = total_output / time_span if time_span > 0 else 0
            else:
                requests_per_minute = 0.0
                tokens_per_second = 0.0
                input_tps = 0.0
                output_tps = 0.0
            
            # Calculate latency percentiles
            if recent_latencies and NUMPY_AVAILABLE:
                latencies_array = np.array(recent_latencies)
                p50 = float(np.percentile(latencies_array, 50))
                p95 = float(np.percentile(latencies_array, 95))
                p99 = float(np.percentile(latencies_array, 99))
                avg_latency = float(np.mean(latencies_array))
            elif recent_latencies:
                sorted_lat = sorted(recent_latencies)
                p50 = sorted_lat[len(sorted_lat) // 2]
                p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
                p99 = sorted_lat[int(len(sorted_lat) * 0.99)]
                avg_latency = sum(sorted_lat) / len(sorted_lat)
            else:
                p50 = p95 = p99 = avg_latency = 0.0
            
            # Calculate TTFT
            if self.ttft_times:
                avg_ttft = sum(self.ttft_times) / len(self.ttft_times)
            else:
                avg_ttft = 0.0
            
            # Calculate wait time
            if self.wait_times:
                avg_wait = sum(self.wait_times) / len(self.wait_times)
            else:
                avg_wait = 0.0
            
            # Calculate costs
            if recent_costs:
                avg_cost_per_request = sum(recent_costs) / len(recent_costs)
                cost_per_hour = (sum(recent_costs) / time_span) * 3600 if time_span > 0 else 0
            else:
                avg_cost_per_request = 0.0
                cost_per_hour = 0.0
            
            # Calculate context stats
            if recent_contexts:
                avg_context = sum(recent_contexts) / len(recent_contexts)
                max_context = max(recent_contexts)
            else:
                avg_context = 0
                max_context = 0
            
            # Calculate response tokens
            if recent_output:
                avg_response = sum(recent_output) / len(recent_output)
            else:
                avg_response = 0
            
            return LLMMetrics(
                tokens_per_second=tokens_per_second,
                requests_per_minute=requests_per_minute,
                input_tokens_per_second=input_tps,
                output_tokens_per_second=output_tps,
                queue_depth=self.queue_depth,
                avg_wait_time_ms=avg_wait,
                avg_ttft_ms=avg_ttft,
                avg_completion_time_ms=avg_latency,
                p50_latency_ms=p50,
                p95_latency_ms=p95,
                p99_latency_ms=p99,
                concurrent_llm_calls=self.concurrent_calls,
                max_concurrent_calls=self.max_concurrent,
                cost_per_hour=cost_per_hour,
                cost_per_request=avg_cost_per_request,
                total_cost=self.total_cost,
                avg_context_tokens=int(avg_context),
                max_context_tokens=max_context,
                avg_response_tokens=int(avg_response),
            )
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics."""
        metrics = await self.get_metrics()
        
        return {
            "agent_id": self.agent_id,
            "model": self.model,
            "metrics": metrics.dict(),
            "utilization": metrics.get_utilization(),
            "is_overloaded": metrics.is_overloaded(),
            "is_underutilized": metrics.is_underutilized(),
            "total_requests": len(self.request_times),
            "window_size": self.window_size,
        }


class TokenThroughputMonitor:
    """
    Monitor token throughput across multiple agents.
    
    Aggregates metrics from multiple LLM resource trackers for
    cluster-level monitoring and scaling decisions.
    """
    
    def __init__(self):
        """Initialize throughput monitor."""
        self.trackers: Dict[str, LLMResourceTracker] = {}
        self.lock = asyncio.Lock()
        self.logger = StructuredLogger("throughput_monitor", LogLevel.INFO)
    
    async def register_tracker(
        self,
        agent_id: str,
        tracker: LLMResourceTracker
    ):
        """Register a resource tracker."""
        async with self.lock:
            self.trackers[agent_id] = tracker
            self.logger.info(f"Registered tracker for agent: {agent_id}")
    
    async def unregister_tracker(self, agent_id: str):
        """Unregister a resource tracker."""
        async with self.lock:
            if agent_id in self.trackers:
                del self.trackers[agent_id]
                self.logger.info(f"Unregistered tracker for agent: {agent_id}")
    
    async def get_aggregate_metrics(self) -> LLMMetrics:
        """
        Get aggregated metrics across all trackers.
        
        Returns:
            Aggregated LLMMetrics
        """
        async with self.lock:
            if not self.trackers:
                return LLMMetrics()
            
            # Collect metrics from all trackers
            all_metrics = []
            for tracker in self.trackers.values():
                metrics = await tracker.get_metrics()
                all_metrics.append(metrics)
            
            # Aggregate
            total_tps = sum(m.tokens_per_second for m in all_metrics)
            total_rpm = sum(m.requests_per_minute for m in all_metrics)
            total_queue = sum(m.queue_depth for m in all_metrics)
            total_concurrent = sum(m.concurrent_llm_calls for m in all_metrics)
            total_max_concurrent = sum(m.max_concurrent_calls for m in all_metrics)
            
            # Average latencies
            avg_ttft = sum(m.avg_ttft_ms for m in all_metrics) / len(all_metrics)
            avg_completion = sum(m.avg_completion_time_ms for m in all_metrics) / len(all_metrics)
            avg_p95 = sum(m.p95_latency_ms for m in all_metrics) / len(all_metrics)
            
            # Sum costs
            total_cost_per_hour = sum(m.cost_per_hour for m in all_metrics)
            total_cost = sum(m.total_cost for m in all_metrics)
            
            return LLMMetrics(
                tokens_per_second=total_tps,
                requests_per_minute=total_rpm,
                queue_depth=total_queue,
                avg_ttft_ms=avg_ttft,
                avg_completion_time_ms=avg_completion,
                p95_latency_ms=avg_p95,
                concurrent_llm_calls=total_concurrent,
                max_concurrent_calls=total_max_concurrent,
                cost_per_hour=total_cost_per_hour,
                total_cost=total_cost,
            )
    
    async def get_per_agent_metrics(self) -> Dict[str, LLMMetrics]:
        """Get metrics for each agent."""
        async with self.lock:
            result = {}
            for agent_id, tracker in self.trackers.items():
                result[agent_id] = await tracker.get_metrics()
            return result


# Global throughput monitor
_global_monitor: Optional[TokenThroughputMonitor] = None


def get_throughput_monitor() -> TokenThroughputMonitor:
    """Get global throughput monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = TokenThroughputMonitor()
    return _global_monitor

