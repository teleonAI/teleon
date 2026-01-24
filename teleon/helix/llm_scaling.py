"""
Token-Aware and Cost-Aware Scaling Policies.

This module provides intelligent scaling policies specifically designed for
LLM workloads, considering token throughput, cost budgets, and queue depths.
"""

from typing import Dict, Optional, Any
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field
import asyncio

from teleon.helix.scaling import ScalingPolicy, ScalingMetrics
from teleon.helix.llm_metrics import LLMMetrics, LLMResourceTracker
from teleon.core import (
    get_metrics,
    StructuredLogger,
    LogLevel,
)


class TokenAwareScalingPolicy(BaseModel):
    """
    Scaling policy based on token throughput.
    
    Scales instances based on tokens/second targets rather than
    traditional CPU/memory metrics.
    """
    
    # Base scaling config
    min_instances: int = Field(1, ge=1, description="Minimum instances")
    max_instances: int = Field(10, ge=1, description="Maximum instances")
    
    # Token-based thresholds
    target_tokens_per_second: float = Field(1000.0, gt=0, description="Target tokens/sec per instance")
    tokens_per_second_buffer: float = Field(0.2, ge=0, le=1, description="Buffer percentage (20% = scale at 80%)")
    
    # Queue-based thresholds
    max_queue_depth: int = Field(10, ge=1, description="Max queue depth before scaling")
    queue_depth_threshold: int = Field(5, ge=1, description="Queue depth to trigger scale up")
    
    # Latency-based thresholds
    target_p95_latency_ms: float = Field(2000.0, gt=0, description="Target P95 latency")
    target_p99_latency_ms: float = Field(5000.0, gt=0, description="Target P99 latency")
    
    # Cooldown periods
    scale_up_cooldown: int = Field(60, ge=10, description="Scale up cooldown (seconds)")
    scale_down_cooldown: int = Field(300, ge=30, description="Scale down cooldown (seconds)")
    
    # Step sizes
    scale_up_step: int = Field(1, ge=1, description="Instances to add on scale up")
    scale_down_step: int = Field(1, ge=1, description="Instances to remove on scale down")
    
    def evaluate_scaling(
        self,
        llm_metrics: LLMMetrics,
        current_instances: int,
        last_scale_time: Optional[datetime] = None
    ) -> Optional[int]:
        """
        Evaluate if scaling is needed based on LLM metrics.
        
        Args:
            llm_metrics: Current LLM metrics
            current_instances: Current number of instances
            last_scale_time: Last scaling action time
        
        Returns:
            Desired instance count or None if no scaling needed
        """
        # Check cooldown
        if last_scale_time:
            time_since_last = (datetime.now(timezone.utc) - last_scale_time).total_seconds()
        else:
            time_since_last = float('inf')
        
        # Calculate per-instance metrics
        if current_instances > 0:
            tps_per_instance = llm_metrics.tokens_per_second / current_instances
        else:
            tps_per_instance = 0
        
        # Determine scaling need
        scale_up = False
        scale_down = False
        reason = ""
        
        # Check token throughput
        target_with_buffer = self.target_tokens_per_second * (1 - self.tokens_per_second_buffer)
        if tps_per_instance > target_with_buffer:
            scale_up = True
            reason = f"Token throughput {tps_per_instance:.0f} > target {target_with_buffer:.0f}"
        elif tps_per_instance < target_with_buffer * 0.5:  # Less than 50% of target
            scale_down = True
            reason = f"Token throughput {tps_per_instance:.0f} < 50% of target"
        
        # Check queue depth
        if llm_metrics.queue_depth >= self.queue_depth_threshold:
            scale_up = True
            reason = f"Queue depth {llm_metrics.queue_depth} >= threshold {self.queue_depth_threshold}"
        elif llm_metrics.queue_depth == 0 and scale_down:
            reason += " and queue empty"
        
        # Check latency
        if llm_metrics.p95_latency_ms > self.target_p95_latency_ms:
            scale_up = True
            reason = f"P95 latency {llm_metrics.p95_latency_ms:.0f}ms > target {self.target_p95_latency_ms:.0f}ms"
        elif llm_metrics.p99_latency_ms > self.target_p99_latency_ms:
            scale_up = True
            reason = f"P99 latency {llm_metrics.p99_latency_ms:.0f}ms > target {self.target_p99_latency_ms:.0f}ms"
        
        # Apply cooldown
        if scale_up and time_since_last < self.scale_up_cooldown:
            return None  # Still in cooldown
        elif scale_down and time_since_last < self.scale_down_cooldown:
            return None  # Still in cooldown
        
        # Calculate desired instances
        if scale_up:
            desired = min(
                current_instances + self.scale_up_step,
                self.max_instances
            )
            if desired > current_instances:
                return desired
        
        elif scale_down:
            desired = max(
                current_instances - self.scale_down_step,
                self.min_instances
            )
            if desired < current_instances:
                return desired
        
        return None


class CostAwareScalingPolicy(BaseModel):
    """
    Scaling policy with cost budget constraints.
    
    Prevents scaling up if it would exceed cost budgets.
    Scales down proactively to stay within budget.
    """
    
    # Base scaling config
    min_instances: int = Field(1, ge=1, description="Minimum instances")
    max_instances: int = Field(10, ge=1, description="Maximum instances")
    
    # Cost budgets
    max_cost_per_hour: Optional[float] = Field(None, gt=0, description="Max hourly cost")
    max_cost_per_day: Optional[float] = Field(None, gt=0, description="Max daily cost")
    max_cost_per_month: Optional[float] = Field(None, gt=0, description="Max monthly cost")
    
    # Cost thresholds
    cost_warning_threshold: float = Field(0.8, ge=0, le=1, description="Warn at this % of budget")
    cost_critical_threshold: float = Field(0.95, ge=0, le=1, description="Critical at this % of budget")
    
    # Performance config
    base_scaling_policy: Optional[TokenAwareScalingPolicy] = Field(None, description="Base policy to extend")
    
    def evaluate_scaling(
        self,
        llm_metrics: LLMMetrics,
        current_instances: int,
        last_scale_time: Optional[datetime] = None
    ) -> Optional[int]:
        """
        Evaluate scaling with cost constraints.
        
        Args:
            llm_metrics: Current LLM metrics
            current_instances: Current number of instances
            last_scale_time: Last scaling action time
        
        Returns:
            Desired instance count or None if no scaling needed
        """
        # Check cost budget
        current_cost_per_hour = llm_metrics.cost_per_hour
        
        # Check if we're over budget
        if self.max_cost_per_hour:
            cost_ratio = current_cost_per_hour / self.max_cost_per_hour
            
            # If over critical threshold, force scale down
            if cost_ratio >= self.cost_critical_threshold:
                return max(current_instances - 1, self.min_instances)
            
            # If near budget, don't scale up
            if cost_ratio >= self.cost_warning_threshold:
                # Can only scale down or stay same
                if self.base_scaling_policy:
                    desired = self.base_scaling_policy.evaluate_scaling(
                        llm_metrics,
                        current_instances,
                        last_scale_time
                    )
                    if desired and desired < current_instances:
                        return desired
                return None
        
        # If within budget, use base policy
        if self.base_scaling_policy:
            desired = self.base_scaling_policy.evaluate_scaling(
                llm_metrics,
                current_instances,
                last_scale_time
            )
            
            # Check if scaling up would exceed budget
            if desired and desired > current_instances:
                projected_cost = current_cost_per_hour * (desired / current_instances)
                
                if self.max_cost_per_hour and projected_cost > self.max_cost_per_hour:
                    # Scale up partially to stay within budget
                    max_affordable_instances = int(
                        (self.max_cost_per_hour / current_cost_per_hour) * current_instances
                    )
                    return min(max_affordable_instances, desired)
            
            return desired
        
        return None
    
    def get_budget_status(self, llm_metrics: LLMMetrics) -> Dict[str, Any]:
        """
        Get budget status.
        
        Returns:
            Budget status information
        """
        status = {
            "current_cost_per_hour": llm_metrics.cost_per_hour,
            "budgets": {},
            "warnings": []
        }
        
        if self.max_cost_per_hour:
            ratio = llm_metrics.cost_per_hour / self.max_cost_per_hour
            status["budgets"]["hourly"] = {
                "limit": self.max_cost_per_hour,
                "current": llm_metrics.cost_per_hour,
                "utilization": ratio,
                "remaining": self.max_cost_per_hour - llm_metrics.cost_per_hour
            }
            
            if ratio >= self.cost_critical_threshold:
                status["warnings"].append("CRITICAL: Hourly cost at critical threshold")
            elif ratio >= self.cost_warning_threshold:
                status["warnings"].append("WARNING: Hourly cost approaching limit")
        
        return status


class LLMScaler:
    """
    Specialized scaler for LLM workloads.
    
    Combines token-aware and cost-aware scaling with intelligent
    decision making for LLM agent instances.
    """
    
    def __init__(
        self,
        token_policy: Optional[TokenAwareScalingPolicy] = None,
        cost_policy: Optional[CostAwareScalingPolicy] = None
    ):
        """
        Initialize LLM scaler.
        
        Args:
            token_policy: Token-aware scaling policy
            cost_policy: Cost-aware scaling policy
        """
        self.token_policy = token_policy or TokenAwareScalingPolicy()
        self.cost_policy = cost_policy
        
        # Scaling history
        self.last_scale_action: Dict[str, datetime] = {}
        self.scaling_history: Dict[str, list] = {}
        
        self.lock = asyncio.Lock()
        self.logger = StructuredLogger("llm_scaler", LogLevel.INFO)
    
    async def evaluate_scaling(
        self,
        agent_id: str,
        llm_metrics: LLMMetrics,
        current_instances: int
    ) -> Optional[int]:
        """
        Evaluate scaling decision for agent.
        
        Args:
            agent_id: Agent identifier
            llm_metrics: Current LLM metrics
            current_instances: Current instance count
        
        Returns:
            Desired instance count or None if no scaling needed
        """
        async with self.lock:
            last_scale_time = self.last_scale_action.get(agent_id)
            
            # If we have cost policy, use it (it may delegate to token policy)
            if self.cost_policy:
                desired = self.cost_policy.evaluate_scaling(
                    llm_metrics,
                    current_instances,
                    last_scale_time
                )
            else:
                # Use token policy directly
                desired = self.token_policy.evaluate_scaling(
                    llm_metrics,
                    current_instances,
                    last_scale_time
                )
            
            if desired and desired != current_instances:
                # Record scaling decision
                self.last_scale_action[agent_id] = datetime.now(timezone.utc)
                
                if agent_id not in self.scaling_history:
                    self.scaling_history[agent_id] = []
                
                self.scaling_history[agent_id].append({
                    "timestamp": datetime.now(timezone.utc),
                    "from_instances": current_instances,
                    "to_instances": desired,
                    "metrics": llm_metrics.dict()
                })
                
                # Keep only recent history
                if len(self.scaling_history[agent_id]) > 100:
                    self.scaling_history[agent_id] = self.scaling_history[agent_id][-100:]
                
                self.logger.info(
                    "Scaling decision",
                    agent_id=agent_id,
                    from_instances=current_instances,
                    to_instances=desired,
                    tokens_per_second=llm_metrics.tokens_per_second,
                    queue_depth=llm_metrics.queue_depth,
                    cost_per_hour=llm_metrics.cost_per_hour
                )
                
                # Record metrics
                get_metrics().increment_counter(
                    'llm_scaling_events',
                    {
                        'agent_id': agent_id,
                        'direction': 'up' if desired > current_instances else 'down'
                    },
                    1
                )
            
            return desired
    
    async def get_scaling_history(
        self,
        agent_id: str,
        limit: int = 10
    ) -> list:
        """Get recent scaling history for agent."""
        async with self.lock:
            history = self.scaling_history.get(agent_id, [])
            return history[-limit:]
    
    async def reset_cooldown(self, agent_id: str):
        """Reset cooldown timer for agent."""
        async with self.lock:
            if agent_id in self.last_scale_action:
                del self.last_scale_action[agent_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scaler statistics."""
        total_events = sum(len(h) for h in self.scaling_history.values())
        
        return {
            "tracked_agents": len(self.scaling_history),
            "total_scaling_events": total_events,
            "token_policy": self.token_policy.dict() if self.token_policy else None,
            "cost_policy": self.cost_policy.dict() if self.cost_policy else None,
        }


# Global LLM scaler
_global_llm_scaler: Optional[LLMScaler] = None


def get_llm_scaler() -> LLMScaler:
    """Get global LLM scaler."""
    global _global_llm_scaler
    if _global_llm_scaler is None:
        _global_llm_scaler = LLMScaler()
    return _global_llm_scaler


def create_llm_scaler(
    target_tokens_per_second: float = 1000.0,
    max_cost_per_hour: Optional[float] = None,
    min_instances: int = 1,
    max_instances: int = 10,
    **kwargs
) -> LLMScaler:
    """
    Create LLM scaler with custom configuration.
    
    Args:
        target_tokens_per_second: Target tokens/sec per instance
        max_cost_per_hour: Maximum cost per hour
        min_instances: Minimum instances
        max_instances: Maximum instances
        **kwargs: Additional policy configuration
    
    Returns:
        Configured LLM scaler
    """
    token_policy = TokenAwareScalingPolicy(
        target_tokens_per_second=target_tokens_per_second,
        min_instances=min_instances,
        max_instances=max_instances,
        **{k: v for k, v in kwargs.items() if k in TokenAwareScalingPolicy.__fields__}
    )
    
    cost_policy = None
    if max_cost_per_hour:
        cost_policy = CostAwareScalingPolicy(
            max_cost_per_hour=max_cost_per_hour,
            min_instances=min_instances,
            max_instances=max_instances,
            base_scaling_policy=token_policy,
            **{k: v for k, v in kwargs.items() if k in CostAwareScalingPolicy.__fields__}
        )
    
    return LLMScaler(token_policy=token_policy, cost_policy=cost_policy)

