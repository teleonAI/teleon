"""
Cost Optimizer - Model Selection and Resource Optimization.

Optimizes agent performance by analyzing costs, latencies, and success rates
to recommend better model selections, caching strategies, and configurations.
"""

from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field

from teleon.cortex.memory.episodic import Episode


class OptimizationStrategy(str, Enum):
    """Optimization strategy types."""
    COST = "cost"
    LATENCY = "latency"
    QUALITY = "quality"
    BALANCED = "balanced"


class ModelRecommendation(BaseModel):
    """Recommendation for model selection."""
    
    model_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    expected_cost: float
    expected_latency_ms: float
    expected_success_rate: float
    reasoning: str
    alternatives: List[str] = Field(default_factory=list)


class CachingRecommendation(BaseModel):
    """Recommendation for caching strategy."""
    
    cache_enabled: bool
    ttl_seconds: Optional[int] = None
    cache_hit_potential: float = Field(..., ge=0.0, le=1.0)
    estimated_savings: float
    reasoning: str


class OptimizationReport(BaseModel):
    """Comprehensive optimization report."""
    
    current_metrics: Dict[str, float]
    opportunities: List[Dict[str, Any]]
    model_recommendations: List[ModelRecommendation]
    caching_recommendations: List[CachingRecommendation]
    estimated_savings: Dict[str, float]
    priority_actions: List[str]


class CostOptimizer:
    """
    Cost and performance optimizer for agents.
    
    Features:
    - Model selection optimization
    - Cost reduction strategies
    - Latency optimization
    - Caching recommendations
    - A/B testing suggestions
    - Resource allocation
    
    Example:
        ```python
        optimizer = CostOptimizer(
            strategy=OptimizationStrategy.BALANCED,
            cost_weight=0.4,
            latency_weight=0.3,
            quality_weight=0.3
        )
        
        # Analyze episodes
        report = await optimizer.analyze(episodes)
        
        # Get model recommendation
        recommendation = await optimizer.recommend_model(
            query_type="simple",
            current_model="gpt-4"
        )
        
        # Check caching potential
        caching = await optimizer.recommend_caching(episodes)
        ```
    """
    
    def __init__(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        cost_weight: float = 0.33,
        latency_weight: float = 0.33,
        quality_weight: float = 0.34
    ):
        """
        Initialize cost optimizer.
        
        Args:
            strategy: Optimization strategy
            cost_weight: Weight for cost optimization (0-1)
            latency_weight: Weight for latency optimization (0-1)
            quality_weight: Weight for quality optimization (0-1)
        """
        self.strategy = strategy
        self.cost_weight = cost_weight
        self.latency_weight = latency_weight
        self.quality_weight = quality_weight
        
        # Normalize weights
        total = cost_weight + latency_weight + quality_weight
        self.cost_weight /= total
        self.latency_weight /= total
        self.quality_weight /= total
        
        # Model pricing (example - update with real pricing)
        self.model_pricing = {
            "gpt-4": {"cost_per_1k": 0.03, "avg_latency_ms": 2000},
            "gpt-4-turbo": {"cost_per_1k": 0.01, "avg_latency_ms": 1500},
            "gpt-3.5-turbo": {"cost_per_1k": 0.002, "avg_latency_ms": 800},
            "claude-3-opus": {"cost_per_1k": 0.015, "avg_latency_ms": 1800},
            "claude-3-sonnet": {"cost_per_1k": 0.003, "avg_latency_ms": 1000},
            "claude-3-haiku": {"cost_per_1k": 0.00025, "avg_latency_ms": 500},
        }
    
    async def analyze(
        self,
        episodes: List[Episode],
        time_window_hours: int = 24
    ) -> OptimizationReport:
        """
        Analyze episodes and generate optimization report.
        
        Args:
            episodes: Episodes to analyze
            time_window_hours: Time window for analysis
        
        Returns:
            Comprehensive optimization report
        """
        # Filter recent episodes
        cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
        recent_episodes = [e for e in episodes if e.timestamp >= cutoff]
        
        if not recent_episodes:
            return OptimizationReport(
                current_metrics={},
                opportunities=[],
                model_recommendations=[],
                caching_recommendations=[],
                estimated_savings={},
                priority_actions=["Collect more data"]
            )
        
        # Calculate current metrics
        current_metrics = self._calculate_metrics(recent_episodes)
        
        # Identify opportunities
        opportunities = self._identify_opportunities(recent_episodes, current_metrics)
        
        # Model recommendations
        model_recs = await self._generate_model_recommendations(recent_episodes)
        
        # Caching recommendations
        caching_recs = await self._generate_caching_recommendations(recent_episodes)
        
        # Calculate potential savings
        savings = self._calculate_savings(
            recent_episodes,
            model_recs,
            caching_recs
        )
        
        # Priority actions
        priority = self._prioritize_actions(opportunities, savings)
        
        return OptimizationReport(
            current_metrics=current_metrics,
            opportunities=opportunities,
            model_recommendations=model_recs,
            caching_recommendations=caching_recs,
            estimated_savings=savings,
            priority_actions=priority
        )
    
    def _calculate_metrics(self, episodes: List[Episode]) -> Dict[str, float]:
        """Calculate current performance metrics."""
        costs = [e.cost for e in episodes if e.cost]
        latencies = [e.duration_ms for e in episodes if e.duration_ms]
        successes = sum(1 for e in episodes if e.success)
        
        return {
            "total_interactions": len(episodes),
            "avg_cost": sum(costs) / len(costs) if costs else 0,
            "total_cost": sum(costs) if costs else 0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "p95_latency_ms": self._percentile(latencies, 95) if latencies else 0,
            "success_rate": (successes / len(episodes) * 100) if episodes else 0,
        }
    
    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _identify_opportunities(
        self,
        episodes: List[Episode],
        metrics: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Identify optimization opportunities."""
        opportunities = []
        
        # High cost interactions
        if metrics["avg_cost"] > 0.01:  # $0.01 per interaction
            high_cost = [e for e in episodes if e.cost and e.cost > metrics["avg_cost"] * 1.5]
            if high_cost:
                opportunities.append({
                    "type": "cost_reduction",
                    "severity": "high",
                    "description": f"Found {len(high_cost)} high-cost interactions",
                    "current": metrics["avg_cost"],
                    "potential_savings": len(high_cost) * metrics["avg_cost"] * 0.5,
                    "recommendation": "Consider cheaper models or caching"
                })
        
        # High latency interactions
        if metrics["avg_latency_ms"] > 1000:  # > 1 second
            slow = [e for e in episodes if e.duration_ms and e.duration_ms > metrics["avg_latency_ms"] * 1.5]
            if slow:
                opportunities.append({
                    "type": "latency_reduction",
                    "severity": "medium",
                    "description": f"Found {len(slow)} slow interactions",
                    "current": metrics["avg_latency_ms"],
                    "potential_improvement": "30-50% faster",
                    "recommendation": "Use faster models or optimize processing"
                })
        
        # Low success rate
        if metrics["success_rate"] < 95.0:
            failures = [e for e in episodes if not e.success]
            opportunities.append({
                "type": "quality_improvement",
                "severity": "high",
                "description": f"{len(failures)} failed interactions",
                "current": metrics["success_rate"],
                "target": 95.0,
                "recommendation": "Analyze failures and improve prompts or models"
            })
        
        # Repetitive queries (caching opportunity)
        query_counts = {}
        for ep in episodes:
            query = str(ep.input.get("query", ""))[:100]
            query_counts[query] = query_counts.get(query, 0) + 1
        
        repeated = {q: c for q, c in query_counts.items() if c >= 3}
        if repeated:
            opportunities.append({
                "type": "caching",
                "severity": "medium",
                "description": f"Found {len(repeated)} repeated queries",
                "cache_hit_potential": sum(repeated.values()) / len(episodes),
                "estimated_savings": sum(repeated.values()) * metrics["avg_cost"] * 0.9,
                "recommendation": "Enable response caching"
            })
        
        return opportunities
    
    async def _generate_model_recommendations(
        self,
        episodes: List[Episode]
    ) -> List[ModelRecommendation]:
        """Generate model selection recommendations."""
        recommendations = []
        
        # Categorize queries by complexity
        simple_queries = []
        complex_queries = []
        
        for ep in episodes:
            query = str(ep.input.get("query", ""))
            query_length = len(query.split())
            
            if query_length < 20:
                simple_queries.append(ep)
            else:
                complex_queries.append(ep)
        
        # Recommend for simple queries
        if simple_queries:
            avg_cost = sum(e.cost for e in simple_queries if e.cost) / len(simple_queries)
            
            if avg_cost > 0.005:  # Using expensive model
                recommendations.append(ModelRecommendation(
                    model_name="gpt-3.5-turbo",
                    confidence=0.85,
                    expected_cost=0.002,
                    expected_latency_ms=800,
                    expected_success_rate=92.0,
                    reasoning="Simple queries can use cheaper, faster models",
                    alternatives=["claude-3-haiku"]
                ))
        
        # Recommend for complex queries
        if complex_queries:
            success_rate = sum(1 for e in complex_queries if e.success) / len(complex_queries) * 100
            
            if success_rate < 90.0:
                recommendations.append(ModelRecommendation(
                    model_name="gpt-4-turbo",
                    confidence=0.75,
                    expected_cost=0.01,
                    expected_latency_ms=1500,
                    expected_success_rate=96.0,
                    reasoning="Complex queries benefit from more capable models",
                    alternatives=["claude-3-opus"]
                ))
        
        return recommendations
    
    async def _generate_caching_recommendations(
        self,
        episodes: List[Episode]
    ) -> List[CachingRecommendation]:
        """Generate caching recommendations."""
        recommendations = []
        
        # Analyze query repetition
        query_counts = {}
        for ep in episodes:
            query = str(ep.input.get("query", ""))[:100]
            query_counts[query] = query_counts.get(query, 0) + 1
        
        repeated = {q: c for q, c in query_counts.items() if c >= 2}
        
        if repeated:
            total_repeated = sum(repeated.values())
            cache_hit_potential = total_repeated / len(episodes)
            
            avg_cost = sum(e.cost for e in episodes if e.cost) / len(episodes)
            estimated_savings = (total_repeated - len(repeated)) * avg_cost * 0.9
            
            # Recommend caching if significant potential
            if cache_hit_potential > 0.2:  # > 20% cache hit rate
                recommendations.append(CachingRecommendation(
                    cache_enabled=True,
                    ttl_seconds=3600,  # 1 hour
                    cache_hit_potential=cache_hit_potential,
                    estimated_savings=estimated_savings,
                    reasoning=f"{len(repeated)} queries repeated, potential {cache_hit_potential*100:.1f}% cache hit rate"
                ))
        
        return recommendations
    
    def _calculate_savings(
        self,
        episodes: List[Episode],
        model_recs: List[ModelRecommendation],
        caching_recs: List[CachingRecommendation]
    ) -> Dict[str, float]:
        """Calculate potential savings."""
        savings = {
            "cost_savings": 0.0,
            "latency_improvement": 0.0,
            "success_rate_improvement": 0.0
        }
        
        current_cost = sum(e.cost for e in episodes if e.cost)
        current_latency = sum(e.duration_ms for e in episodes if e.duration_ms) / len(episodes) if episodes else 0
        
        # Model switching savings
        for rec in model_recs:
            potential_cost = rec.expected_cost * len(episodes)
            savings["cost_savings"] += max(0, current_cost - potential_cost)
        
        # Caching savings
        for rec in caching_recs:
            savings["cost_savings"] += rec.estimated_savings
        
        # Latency improvements
        for rec in model_recs:
            if rec.expected_latency_ms < current_latency:
                savings["latency_improvement"] += (
                    (current_latency - rec.expected_latency_ms) / current_latency * 100
                )
        
        return savings
    
    def _prioritize_actions(
        self,
        opportunities: List[Dict[str, Any]],
        savings: Dict[str, float]
    ) -> List[str]:
        """Prioritize optimization actions."""
        actions = []
        
        # Sort opportunities by severity and impact
        high_priority = [o for o in opportunities if o.get("severity") == "high"]
        medium_priority = [o for o in opportunities if o.get("severity") == "medium"]
        
        # Add high priority actions first
        for opp in high_priority:
            actions.append(f"[HIGH] {opp['recommendation']}")
        
        # Add significant savings opportunities
        if savings["cost_savings"] > 10.0:  # > $10 savings
            actions.append(f"[HIGH] Implement model optimization (save ${savings['cost_savings']:.2f})")
        
        # Add medium priority actions
        for opp in medium_priority[:3]:  # Top 3
            actions.append(f"[MEDIUM] {opp['recommendation']}")
        
        # Add quick wins
        if not actions:
            actions.append("[LOW] Continue monitoring for optimization opportunities")
        
        return actions
    
    async def recommend_model(
        self,
        query_type: str,
        current_model: Optional[str] = None,
        budget: Optional[float] = None,
        latency_requirement_ms: Optional[int] = None
    ) -> ModelRecommendation:
        """
        Recommend best model for a query type.
        
        Args:
            query_type: Type of query (simple, complex, etc.)
            current_model: Currently used model
            budget: Cost budget per request
            latency_requirement_ms: Maximum acceptable latency
        
        Returns:
            Model recommendation
        """
        candidates = []
        
        for model, specs in self.model_pricing.items():
            # Check budget constraint
            if budget and specs["cost_per_1k"] > budget:
                continue
            
            # Check latency constraint
            if latency_requirement_ms and specs["avg_latency_ms"] > latency_requirement_ms:
                continue
            
            # Calculate score based on strategy
            if self.strategy == OptimizationStrategy.COST:
                score = 1.0 / (specs["cost_per_1k"] + 0.001)
            elif self.strategy == OptimizationStrategy.LATENCY:
                score = 1.0 / (specs["avg_latency_ms"] + 1)
            elif self.strategy == OptimizationStrategy.QUALITY:
                # Assume more expensive = better quality (simplified)
                score = specs["cost_per_1k"]
            else:  # BALANCED
                cost_score = 1.0 / (specs["cost_per_1k"] + 0.001)
                latency_score = 1.0 / (specs["avg_latency_ms"] + 1)
                score = (cost_score * self.cost_weight + latency_score * self.latency_weight)
            
            candidates.append((model, score, specs))
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if not candidates:
            # Fallback
            return ModelRecommendation(
                model_name="gpt-3.5-turbo",
                confidence=0.5,
                expected_cost=0.002,
                expected_latency_ms=800,
                expected_success_rate=90.0,
                reasoning="Default recommendation"
            )
        
        best_model, _, specs = candidates[0]
        alternatives = [m for m, _, _ in candidates[1:3]]
        
        return ModelRecommendation(
            model_name=best_model,
            confidence=0.8,
            expected_cost=specs["cost_per_1k"],
            expected_latency_ms=specs["avg_latency_ms"],
            expected_success_rate=92.0,
            reasoning=f"Best match for {self.strategy.value} strategy",
            alternatives=alternatives
        )

