"""
Cost Management - Enterprise cost tracking and optimization.

Features:
- Real-time cost tracking
- Budget controls
- Cost optimization
- Billing integration
- Cost allocation
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field
from enum import Enum
from collections import defaultdict

from teleon.core import StructuredLogger, LogLevel, get_metrics


class CostCategory(str, Enum):
    """Cost categories."""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    LLM = "llm"
    TOOLS = "tools"
    MEMORY = "memory"


class CostLimit(BaseModel):
    """Cost limit configuration."""
    
    daily_limit: Optional[float] = Field(None, description="Daily cost limit ($)")
    monthly_limit: Optional[float] = Field(None, description="Monthly cost limit ($)")
    
    # Actions
    notify_at_percent: float = Field(80.0, ge=0, le=100, description="Notify at % of limit")
    halt_at_percent: float = Field(100.0, ge=0, le=100, description="Halt at % of limit")


class CostTracker:
    """
    Cost tracker for real-time cost monitoring.
    
    Features:
    - Real-time tracking
    - Category-based costs
    - Time-series data
    - Cost attribution
    """
    
    def __init__(self):
        """Initialize cost tracker."""
        self.costs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.daily_totals: Dict[str, float] = {}
        self.monthly_totals: Dict[str, float] = {}
        
        self.logger = StructuredLogger("cost_tracker", LogLevel.INFO)
    
    def record_cost(
        self,
        agent_id: str,
        category: CostCategory,
        amount: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record cost.
        
        Args:
            agent_id: Agent ID
            category: Cost category
            amount: Cost amount ($)
            metadata: Additional metadata
        """
        cost_entry = {
            "timestamp": datetime.now(timezone.utc),
            "category": category.value,
            "amount": amount,
            "metadata": metadata or {}
        }
        
        self.costs[agent_id].append(cost_entry)
        
        # Update totals
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        
        self.daily_totals[today] = self.daily_totals.get(today, 0) + amount
        self.monthly_totals[month] = self.monthly_totals.get(month, 0) + amount
        
        # Record metrics
        get_metrics().increment_counter(
            'costs',
            {'agent_id': agent_id, 'category': category.value},
            amount
        )
    
    def get_agent_costs(
        self,
        agent_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get agent costs.
        
        Args:
            agent_id: Agent ID
            start_time: Start time
            end_time: End time
        
        Returns:
            Cost summary
        """
        costs = self.costs.get(agent_id, [])
        
        # Filter by time
        if start_time:
            costs = [c for c in costs if c["timestamp"] >= start_time]
        if end_time:
            costs = [c for c in costs if c["timestamp"] <= end_time]
        
        # Calculate totals by category
        category_totals = defaultdict(float)
        for cost in costs:
            category_totals[cost["category"]] += cost["amount"]
        
        total = sum(category_totals.values())
        
        return {
            "agent_id": agent_id,
            "total_cost": total,
            "by_category": dict(category_totals),
            "num_entries": len(costs)
        }
    
    def get_daily_cost(self, date: Optional[str] = None) -> float:
        """
        Get daily cost.
        
        Args:
            date: Date (YYYY-MM-DD), defaults to today
        
        Returns:
            Daily cost ($)
        """
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        return self.daily_totals.get(date, 0.0)
    
    def get_monthly_cost(self, month: Optional[str] = None) -> float:
        """
        Get monthly cost.
        
        Args:
            month: Month (YYYY-MM), defaults to current month
        
        Returns:
            Monthly cost ($)
        """
        if month is None:
            month = datetime.now(timezone.utc).strftime("%Y-%m")
        
        return self.monthly_totals.get(month, 0.0)


class CostOptimizer:
    """
    Cost optimizer for automated cost reduction.
    
    Features:
    - Model downgrading
    - Caching recommendations
    - Resource right-sizing
    - Batch processing
    """
    
    def __init__(self, cost_tracker: CostTracker):
        """
        Initialize cost optimizer.
        
        Args:
            cost_tracker: Cost tracker instance
        """
        self.cost_tracker = cost_tracker
        self.logger = StructuredLogger("cost_optimizer", LogLevel.INFO)
    
    def analyze_costs(self, agent_id: str) -> Dict[str, Any]:
        """
        Analyze costs and provide optimization recommendations.
        
        Args:
            agent_id: Agent ID
        
        Returns:
            Analysis and recommendations
        """
        costs = self.cost_tracker.get_agent_costs(agent_id)
        recommendations = []
        
        # Check LLM costs
        llm_cost = costs["by_category"].get(CostCategory.LLM.value, 0)
        if llm_cost > costs["total_cost"] * 0.7:  # >70% of costs
            recommendations.append({
                "type": "llm_optimization",
                "severity": "high",
                "message": "LLM costs are high. Consider using cheaper models or caching.",
                "potential_savings": llm_cost * 0.3  # Estimate 30% savings
            })
        
        # Check storage costs
        storage_cost = costs["by_category"].get(CostCategory.STORAGE.value, 0)
        if storage_cost > costs["total_cost"] * 0.3:  # >30% of costs
            recommendations.append({
                "type": "storage_optimization",
                "severity": "medium",
                "message": "Storage costs are high. Consider data retention policies.",
                "potential_savings": storage_cost * 0.2
            })
        
        # Check compute costs
        compute_cost = costs["by_category"].get(CostCategory.COMPUTE.value, 0)
        if compute_cost > costs["total_cost"] * 0.4:  # >40% of costs
            recommendations.append({
                "type": "compute_optimization",
                "severity": "medium",
                "message": "Compute costs are high. Consider right-sizing instances.",
                "potential_savings": compute_cost * 0.25
            })
        
        total_potential_savings = sum(r["potential_savings"] for r in recommendations)
        
        return {
            "agent_id": agent_id,
            "current_costs": costs,
            "recommendations": recommendations,
            "potential_savings": total_potential_savings,
            "analyzed_at": datetime.now(timezone.utc).isoformat()
        }
    
    def suggest_model_alternatives(
        self,
        current_model: str,
        quality_threshold: float = 0.9
    ) -> List[Dict[str, Any]]:
        """
        Suggest cheaper model alternatives.
        
        Args:
            current_model: Current model
            quality_threshold: Minimum quality ratio
        
        Returns:
            List of alternatives
        """
        # Model alternatives with cost and quality ratios
        alternatives_map = {
            "gpt-4": [
                {"model": "gpt-3.5-turbo", "cost_ratio": 0.1, "quality_ratio": 0.85},
                {"model": "claude-2", "cost_ratio": 0.5, "quality_ratio": 0.95},
            ],
            "claude-3-opus": [
                {"model": "claude-3-sonnet", "cost_ratio": 0.2, "quality_ratio": 0.9},
                {"model": "claude-3-haiku", "cost_ratio": 0.05, "quality_ratio": 0.8},
            ]
        }
        
        alternatives = alternatives_map.get(current_model, [])
        filtered = [a for a in alternatives if a["quality_ratio"] >= quality_threshold]
        
        return filtered


class BudgetManager:
    """
    Budget manager for cost control.
    
    Features:
    - Budget limits
    - Automated alerts
    - Spend tracking
    - Budget forecasting
    """
    
    def __init__(
        self,
        cost_tracker: CostTracker,
        limits: Optional[CostLimit] = None
    ):
        """
        Initialize budget manager.
        
        Args:
            cost_tracker: Cost tracker instance
            limits: Cost limits
        """
        self.cost_tracker = cost_tracker
        self.limits = limits or CostLimit()
        
        self.alerts_sent: Dict[str, datetime] = {}
        self.halted_agents: set = set()
        
        self.logger = StructuredLogger("budget_manager", LogLevel.INFO)
    
    def check_daily_budget(self) -> Dict[str, Any]:
        """
        Check daily budget status.
        
        Returns:
            Budget status
        """
        if not self.limits.daily_limit:
            return {"status": "no_limit"}
        
        current_cost = self.cost_tracker.get_daily_cost()
        percent_used = (current_cost / self.limits.daily_limit) * 100
        
        status = {
            "current_cost": current_cost,
            "limit": self.limits.daily_limit,
            "percent_used": percent_used,
            "remaining": self.limits.daily_limit - current_cost
        }
        
        # Check for alerts
        if percent_used >= self.limits.halt_at_percent:
            status["action"] = "halt"
            self.logger.error(
                "Daily budget exceeded - halting operations",
                current_cost=current_cost,
                limit=self.limits.daily_limit
            )
        
        elif percent_used >= self.limits.notify_at_percent:
            status["action"] = "notify"
            self.logger.warning(
                "Daily budget alert",
                current_cost=current_cost,
                limit=self.limits.daily_limit,
                percent=percent_used
            )
        
        else:
            status["action"] = "none"
        
        return status
    
    def check_monthly_budget(self) -> Dict[str, Any]:
        """
        Check monthly budget status.
        
        Returns:
            Budget status
        """
        if not self.limits.monthly_limit:
            return {"status": "no_limit"}
        
        current_cost = self.cost_tracker.get_monthly_cost()
        percent_used = (current_cost / self.limits.monthly_limit) * 100
        
        status = {
            "current_cost": current_cost,
            "limit": self.limits.monthly_limit,
            "percent_used": percent_used,
            "remaining": self.limits.monthly_limit - current_cost
        }
        
        # Check for alerts
        if percent_used >= self.limits.halt_at_percent:
            status["action"] = "halt"
            self.logger.error(
                "Monthly budget exceeded - halting operations",
                current_cost=current_cost,
                limit=self.limits.monthly_limit
            )
        
        elif percent_used >= self.limits.notify_at_percent:
            status["action"] = "notify"
            self.logger.warning(
                "Monthly budget alert",
                current_cost=current_cost,
                limit=self.limits.monthly_limit,
                percent=percent_used
            )
        
        else:
            status["action"] = "none"
        
        return status
    
    def forecast_monthly_cost(self) -> float:
        """
        Forecast end-of-month cost based on current spend rate.
        
        Returns:
            Forecasted monthly cost
        """
        current_cost = self.cost_tracker.get_monthly_cost()
        
        # Calculate days elapsed and remaining
        now = datetime.now(timezone.utc)
        days_in_month = 30  # Simplified
        days_elapsed = now.day
        days_remaining = days_in_month - days_elapsed
        
        if days_elapsed == 0:
            return current_cost
        
        # Calculate daily average and forecast
        daily_average = current_cost / days_elapsed
        forecast = current_cost + (daily_average * days_remaining)
        
        return forecast

