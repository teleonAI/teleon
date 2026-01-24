"""
Token Tracking and Budget Management for LLM Agents.

This module provides comprehensive token tracking, budget management,
and token usage optimization recommendations for LLM workloads.
"""

from typing import Dict, Optional, List, Any, Callable
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field, ConfigDict, field_serializer
from enum import Enum
import asyncio
from collections import defaultdict

from teleon.core import (
    get_metrics,
    StructuredLogger,
    LogLevel,
)


class TokenPeriod(str, Enum):
    """Token tracking periods."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class TokenAlert(BaseModel):
    """Token budget alert."""
    
    alert_id: str = Field(..., description="Alert identifier")
    level: str = Field(..., description="Alert level (warning, critical)")
    message: str = Field(..., description="Alert message")
    agent_id: Optional[str] = Field(None, description="Related agent ID")
    current_tokens: int = Field(..., description="Current token count")
    budget_limit: int = Field(..., description="Budget limit in tokens")
    utilization: float = Field(..., description="Budget utilization (0-1)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict()

    @field_serializer('timestamp')
    def serialize_datetime(self, value: datetime) -> str:
        return value.isoformat() if value else None


class TokenBreakdown(BaseModel):
    """Detailed token breakdown."""
    
    total_tokens: int = Field(0, description="Total tokens")
    by_agent: Dict[str, int] = Field(default_factory=dict, description="Tokens by agent")
    by_model: Dict[str, int] = Field(default_factory=dict, description="Tokens by model")
    by_operation: Dict[str, int] = Field(default_factory=dict, description="Tokens by operation")
    input_tokens: int = Field(0, description="Total input tokens")
    output_tokens: int = Field(0, description="Total output tokens")
    
    period_start: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    period_end: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict()

    @field_serializer('period_start', 'period_end')
    def serialize_datetime(self, value: datetime) -> str:
        return value.isoformat() if value else None


class TokenTracker:
    """
    Real-time token tracking for LLM operations.
    
    Tracks tokens at multiple granularities:
    - Per agent
    - Per model
    - Per operation type
    - Over time periods (hour, day, month)
    """
    
    def __init__(self):
        """Initialize token tracker."""
        # Token accumulation
        self.total_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.tokens_by_agent: Dict[str, int] = defaultdict(int)
        self.tokens_by_model: Dict[str, int] = defaultdict(int)
        self.tokens_by_operation: Dict[str, int] = defaultdict(int)
        
        # Separate input/output tracking
        self.input_tokens_by_agent: Dict[str, int] = defaultdict(int)
        self.output_tokens_by_agent: Dict[str, int] = defaultdict(int)
        self.input_tokens_by_model: Dict[str, int] = defaultdict(int)
        self.output_tokens_by_model: Dict[str, int] = defaultdict(int)
        
        # Time-series data
        self.hourly_tokens: Dict[str, int] = {}  # timestamp -> tokens
        self.daily_tokens: Dict[str, int] = {}
        self.monthly_tokens: Dict[str, int] = {}
        
        # Token events (for detailed analysis)
        self.token_events: List[Dict[str, Any]] = []
        self.max_events = 10000  # Keep last N events
        
        self.lock = asyncio.Lock()
        self.logger = StructuredLogger("token_tracker", LogLevel.INFO)
    
    async def record_tokens(
        self,
        agent_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation: str = "completion",
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Record tokens for an LLM operation.
        
        Args:
            agent_id: Agent identifier
            model: Model name
            input_tokens: Input token count
            output_tokens: Output token count
            operation: Operation type
            metadata: Additional metadata
        
        Returns:
            Total tokens recorded
        """
        total_tokens = input_tokens + output_tokens
        
        async with self.lock:
            # Accumulate tokens
            self.total_tokens += total_tokens
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.tokens_by_agent[agent_id] += total_tokens
            self.tokens_by_model[model] += total_tokens
            self.tokens_by_operation[operation] += total_tokens
            
            # Track input/output separately
            self.input_tokens_by_agent[agent_id] += input_tokens
            self.output_tokens_by_agent[agent_id] += output_tokens
            self.input_tokens_by_model[model] += input_tokens
            self.output_tokens_by_model[model] += output_tokens
            
            # Track by time periods
            now = datetime.now(timezone.utc)
            hour_key = now.strftime("%Y-%m-%d-%H")
            day_key = now.strftime("%Y-%m-%d")
            month_key = now.strftime("%Y-%m")
            
            self.hourly_tokens[hour_key] = self.hourly_tokens.get(hour_key, 0) + total_tokens
            self.daily_tokens[day_key] = self.daily_tokens.get(day_key, 0) + total_tokens
            self.monthly_tokens[month_key] = self.monthly_tokens.get(month_key, 0) + total_tokens
            
            # Record event
            event = {
                "timestamp": now,
                "agent_id": agent_id,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "operation": operation,
                "metadata": metadata or {}
            }
            self.token_events.append(event)
            
            # Trim events if needed
            if len(self.token_events) > self.max_events:
                self.token_events = self.token_events[-self.max_events:]
            
            # Record to global metrics
            get_metrics().increment_counter(
                'llm_tokens_total',
                {'agent_id': agent_id, 'model': model},
                total_tokens
            )
            
            get_metrics().increment_counter(
                'llm_input_tokens_total',
                {'agent_id': agent_id, 'model': model},
                input_tokens
            )
            
            get_metrics().increment_counter(
                'llm_output_tokens_total',
                {'agent_id': agent_id, 'model': model},
                output_tokens
            )
        
        return total_tokens
    
    async def get_tokens(
        self,
        agent_id: Optional[str] = None,
        period: Optional[TokenPeriod] = None,
        period_key: Optional[str] = None
    ) -> int:
        """
        Get token count for agent or time period.
        
        Args:
            agent_id: Optional agent filter
            period: Optional time period
            period_key: Optional specific period key
        
        Returns:
            Token count
        """
        async with self.lock:
            if agent_id:
                return self.tokens_by_agent.get(agent_id, 0)
            
            if period and period_key:
                if period == TokenPeriod.HOURLY:
                    return self.hourly_tokens.get(period_key, 0)
                elif period == TokenPeriod.DAILY:
                    return self.daily_tokens.get(period_key, 0)
                elif period == TokenPeriod.MONTHLY:
                    return self.monthly_tokens.get(period_key, 0)
            
            return self.total_tokens
    
    async def get_breakdown(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> TokenBreakdown:
        """
        Get detailed token breakdown.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
        
        Returns:
            Token breakdown
        """
        async with self.lock:
            # Filter events by time if specified
            if start_time or end_time:
                filtered_events = []
                for event in self.token_events:
                    ts = event["timestamp"]
                    if start_time and ts < start_time:
                        continue
                    if end_time and ts > end_time:
                        continue
                    filtered_events.append(event)
            else:
                filtered_events = self.token_events
            
            # Calculate breakdown
            total = sum(e["total_tokens"] for e in filtered_events)
            input_total = sum(e["input_tokens"] for e in filtered_events)
            output_total = sum(e["output_tokens"] for e in filtered_events)
            
            by_agent = defaultdict(int)
            by_model = defaultdict(int)
            by_operation = defaultdict(int)
            
            for event in filtered_events:
                by_agent[event["agent_id"]] += event["total_tokens"]
                by_model[event["model"]] += event["total_tokens"]
                by_operation[event["operation"]] += event["total_tokens"]
            
            return TokenBreakdown(
                total_tokens=total,
                by_agent=dict(by_agent),
                by_model=dict(by_model),
                by_operation=dict(by_operation),
                input_tokens=input_total,
                output_tokens=output_total,
                period_start=start_time or datetime.now(timezone.utc) - timedelta(hours=1),
                period_end=end_time or datetime.now(timezone.utc)
            )
    
    async def get_tokens_per_hour(self, agent_id: Optional[str] = None) -> int:
        """Get current tokens per hour rate."""
        now = datetime.now(timezone.utc)
        hour_key = now.strftime("%Y-%m-%d-%H")
        
        async with self.lock:
            if agent_id:
                # Calculate from recent events
                hour_start = now.replace(minute=0, second=0, microsecond=0)
                hour_tokens = sum(
                    e["total_tokens"] for e in self.token_events
                    if e["agent_id"] == agent_id and e["timestamp"] >= hour_start
                )
                return hour_tokens
            else:
                return self.hourly_tokens.get(hour_key, 0)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive token statistics."""
        async with self.lock:
            breakdown = await self.get_breakdown()
            
            return {
                "total_tokens": self.total_tokens,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "by_agent": dict(self.tokens_by_agent),
                "by_model": dict(self.tokens_by_model),
                "by_operation": dict(self.tokens_by_operation),
                "input_by_agent": dict(self.input_tokens_by_agent),
                "output_by_agent": dict(self.output_tokens_by_agent),
                "input_by_model": dict(self.input_tokens_by_model),
                "output_by_model": dict(self.output_tokens_by_model),
                "recent_breakdown": breakdown.dict(),
                "total_events": len(self.token_events),
            }
    
    async def reset(self):
        """Reset all token tracking."""
        async with self.lock:
            self.total_tokens = 0
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.tokens_by_agent.clear()
            self.tokens_by_model.clear()
            self.tokens_by_operation.clear()
            self.input_tokens_by_agent.clear()
            self.output_tokens_by_agent.clear()
            self.input_tokens_by_model.clear()
            self.output_tokens_by_model.clear()
            self.hourly_tokens.clear()
            self.daily_tokens.clear()
            self.monthly_tokens.clear()
            self.token_events.clear()
        
        self.logger.info("Token tracking reset")


class TokenBudgetManager:
    """
    Manage token budgets and alerts.
    
    Features:
    - Set budgets per agent or globally
    - Multiple time periods (hour, day, month)
    - Automatic alerts at thresholds
    - Budget enforcement
    """
    
    def __init__(self, token_tracker: TokenTracker):
        """
        Initialize budget manager.
        
        Args:
            token_tracker: Token tracker instance
        """
        self.token_tracker = token_tracker
        
        # Budgets (in tokens)
        self.global_budgets: Dict[TokenPeriod, int] = {}
        self.agent_budgets: Dict[str, Dict[TokenPeriod, int]] = defaultdict(dict)
        
        # Alert thresholds
        self.warning_threshold = 0.8  # 80%
        self.critical_threshold = 0.95  # 95%
        
        # Alert tracking
        self.alerts: List[TokenAlert] = []
        self.alert_callbacks: List[Callable] = []
        
        self.lock = asyncio.Lock()
        self.logger = StructuredLogger("token_budget_manager", LogLevel.INFO)
    
    async def set_budget(
        self,
        amount: int,
        period: TokenPeriod,
        agent_id: Optional[str] = None
    ):
        """
        Set budget for period.
        
        Args:
            amount: Budget amount in tokens
            period: Time period
            agent_id: Optional agent-specific budget
        """
        async with self.lock:
            if agent_id:
                self.agent_budgets[agent_id][period] = amount
                self.logger.info(
                    f"Set {period.value} token budget for agent {agent_id}: {amount:,} tokens"
                )
            else:
                self.global_budgets[period] = amount
                self.logger.info(
                    f"Set global {period.value} token budget: {amount:,} tokens"
                )
    
    async def check_budget(
        self,
        agent_id: Optional[str] = None,
        period: Optional[TokenPeriod] = None
    ) -> Dict[str, Any]:
        """
        Check budget status.
        
        Args:
            agent_id: Optional agent filter
            period: Optional period filter
        
        Returns:
            Budget status
        """
        async with self.lock:
            status = {}
            
            # Determine which budgets to check
            if agent_id and period:
                budgets_to_check = [(agent_id, period)]
            elif agent_id:
                budgets_to_check = [
                    (agent_id, p) for p in self.agent_budgets.get(agent_id, {}).keys()
                ]
            elif period:
                budgets_to_check = [(None, period)]
            else:
                # Check all
                budgets_to_check = [(None, p) for p in self.global_budgets.keys()]
                for aid in self.agent_budgets:
                    budgets_to_check.extend([
                        (aid, p) for p in self.agent_budgets[aid].keys()
                    ])
            
            # Check each budget
            for aid, p in budgets_to_check:
                # Get budget limit
                if aid:
                    budget_limit = self.agent_budgets[aid].get(p)
                else:
                    budget_limit = self.global_budgets.get(p)
                
                if not budget_limit:
                    continue
                
                # Get current tokens
                if p == TokenPeriod.HOURLY:
                    current_tokens = await self.token_tracker.get_tokens_per_hour(aid)
                else:
                    # For daily/monthly, get from breakdown
                    now = datetime.now(timezone.utc)
                    if p == TokenPeriod.DAILY:
                        start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
                    else:  # MONTHLY
                        start_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                    
                    breakdown = await self.token_tracker.get_breakdown(
                        start_time=start_time,
                        end_time=now
                    )
                    
                    if aid:
                        current_tokens = breakdown.by_agent.get(aid, 0)
                    else:
                        current_tokens = breakdown.total_tokens
                
                # Calculate utilization
                utilization = current_tokens / budget_limit if budget_limit > 0 else 0.0
                
                # Check thresholds
                alert_level = None
                if utilization >= self.critical_threshold:
                    alert_level = "critical"
                elif utilization >= self.warning_threshold:
                    alert_level = "warning"
                
                key = f"{aid or 'global'}_{p.value}"
                status[key] = {
                    "budget_limit": budget_limit,
                    "current_tokens": current_tokens,
                    "utilization": utilization,
                    "remaining": budget_limit - current_tokens,
                    "alert_level": alert_level
                }
                
                # Create alert if needed
                if alert_level:
                    await self._create_alert(
                        level=alert_level,
                        agent_id=aid,
                        period=p,
                        current_tokens=current_tokens,
                        budget_limit=budget_limit,
                        utilization=utilization
                    )
            
            return status
    
    async def _create_alert(
        self,
        level: str,
        agent_id: Optional[str],
        period: TokenPeriod,
        current_tokens: int,
        budget_limit: int,
        utilization: float
    ):
        """Create token budget alert."""
        import uuid
        
        alert = TokenAlert(
            alert_id=str(uuid.uuid4()),
            level=level,
            message=f"{level.upper()}: {period.value} token budget at {utilization*100:.1f}% "
                   f"({current_tokens:,} of {budget_limit:,} tokens)",
            agent_id=agent_id,
            current_tokens=current_tokens,
            budget_limit=budget_limit,
            utilization=utilization
        )
        
        self.alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        self.logger.warning(
            alert.message,
            alert_level=level,
            agent_id=agent_id,
            period=period.value
        )
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for token budget alerts."""
        self.alert_callbacks.append(callback)
    
    async def get_alerts(
        self,
        level: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 100
    ) -> List[TokenAlert]:
        """Get recent alerts."""
        async with self.lock:
            filtered = self.alerts
            
            if level:
                filtered = [a for a in filtered if a.level == level]
            
            if agent_id:
                filtered = [a for a in filtered if a.agent_id == agent_id]
            
            return filtered[-limit:]


# Global token tracker
_global_token_tracker: Optional[TokenTracker] = None
_global_token_budget_manager: Optional[TokenBudgetManager] = None


def get_token_tracker() -> TokenTracker:
    """Get global token tracker."""
    global _global_token_tracker
    if _global_token_tracker is None:
        _global_token_tracker = TokenTracker()
    return _global_token_tracker


def get_token_budget_manager() -> TokenBudgetManager:
    """Get global token budget manager."""
    global _global_token_budget_manager, _global_token_tracker
    if _global_token_budget_manager is None:
        if _global_token_tracker is None:
            _global_token_tracker = TokenTracker()
        _global_token_budget_manager = TokenBudgetManager(_global_token_tracker)
    return _global_token_budget_manager


# Backward compatibility alias (for migration period only)
get_cost_tracker = get_token_tracker
