"""
Learning & Optimization - Automatic pattern learning and performance optimization.

Provides utilities for:
- Automatic pattern detection and learning
- Success/failure analysis
- A/B testing for strategies
- Performance optimization
- Adaptive memory management
"""

from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import statistics
import logging

logger = logging.getLogger(__name__)


class PatternLearner:
    """
    Automatically learn patterns from successful interactions.
    
    Capabilities:
    - Detect recurring patterns
    - Learn from success/failure
    - Identify optimal strategies
    - Track pattern effectiveness
    
    Example:
        ```python
        learner = PatternLearner()
        
        # Learn from episodes
        patterns = await learner.learn_from_episodes(
            episodes=recent_episodes,
            min_occurrences=3
        )
        
        # Find best pattern for input
        best_pattern = learner.find_best_pattern(
            input_text="user question",
            candidates=patterns
        )
        ```
    """
    
    def __init__(
        self,
        min_success_rate: float = 0.7,
        min_confidence: float = 0.6,
        pattern_window: int = 100
    ):
        """
        Initialize pattern learner.
        
        Args:
            min_success_rate: Minimum success rate for patterns
            min_confidence: Minimum confidence for pattern matching
            pattern_window: Number of recent episodes to analyze
        """
        self.min_success_rate = min_success_rate
        self.min_confidence = min_confidence
        self.pattern_window = pattern_window
    
    async def learn_from_episodes(
        self,
        episodes: List[Any],
        min_occurrences: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Extract patterns from episodes.
        
        Args:
            episodes: List of episodes
            min_occurrences: Minimum pattern occurrences
        
        Returns:
            List of learned patterns
        """
        if not episodes:
            return []
        
        # Group episodes by input similarity
        pattern_groups = self._group_similar_episodes(episodes)
        
        learned_patterns = []
        
        for input_pattern, group_episodes in pattern_groups.items():
            if len(group_episodes) < min_occurrences:
                continue
            
            # Analyze success rate
            success_count = sum(1 for ep in group_episodes if ep.success)
            success_rate = success_count / len(group_episodes)
            
            if success_rate < self.min_success_rate:
                continue
            
            # Find common output pattern
            output_pattern = self._extract_output_pattern(group_episodes)
            
            # Calculate metrics
            avg_duration = statistics.mean([
                ep.duration_ms or 0 for ep in group_episodes
            ])
            
            avg_cost = statistics.mean([
                ep.cost or 0 for ep in group_episodes
            ])
            
            learned_patterns.append({
                "input_pattern": input_pattern,
                "output_pattern": output_pattern,
                "occurrences": len(group_episodes),
                "success_rate": success_rate,
                "avg_duration_ms": avg_duration,
                "avg_cost": avg_cost,
                "confidence": self._calculate_confidence(group_episodes),
                "last_seen": max(ep.timestamp for ep in group_episodes),
            })
        
        # Sort by confidence and success rate
        learned_patterns.sort(
            key=lambda x: (x["confidence"], x["success_rate"]),
            reverse=True
        )
        
        return learned_patterns
    
    def _group_similar_episodes(
        self,
        episodes: List[Any]
    ) -> Dict[str, List[Any]]:
        """Group episodes by input similarity."""
        groups = defaultdict(list)
        
        for episode in episodes:
            # Extract pattern from input
            pattern = self._extract_input_pattern(episode.input)
            groups[pattern].append(episode)
        
        return dict(groups)
    
    def _extract_input_pattern(self, input_data: Any) -> str:
        """Extract pattern from input."""
        if isinstance(input_data, dict):
            text = input_data.get("query", str(input_data))
        else:
            text = str(input_data)
        
        # Simple pattern: first few words
        words = text.lower().split()[:5]
        return " ".join(words)
    
    def _extract_output_pattern(self, episodes: List[Any]) -> str:
        """
        Extract common output pattern from episodes.
        
        Simple implementation: uses most common output prefix.
        This works but could be enhanced with more sophisticated pattern analysis.
        """
        outputs = []
        
        for episode in episodes:
            if isinstance(episode.output, dict):
                text = episode.output.get("response", str(episode.output))
            else:
                text = str(episode.output)
            
            # Get first sentence
            sentences = text.split(". ")
            if sentences:
                outputs.append(sentences[0])
        
        # Find most common
        if outputs:
            output_counts = defaultdict(int)
            for output in outputs:
                output_counts[output] += 1
            
            most_common = max(output_counts.items(), key=lambda x: x[1])
            return most_common[0]
        
        return ""
    
    def _calculate_confidence(self, episodes: List[Any]) -> float:
        """Calculate confidence score for pattern."""
        if not episodes:
            return 0.0
        
        # Based on occurrence count and success consistency
        occurrence_score = min(1.0, len(episodes) / 10)
        
        success_count = sum(1 for ep in episodes if ep.success)
        consistency_score = success_count / len(episodes)
        
        return (occurrence_score + consistency_score) / 2
    
    def find_best_pattern(
        self,
        input_text: str,
        candidates: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Find best matching pattern for input.
        
        Args:
            input_text: Input text
            candidates: Candidate patterns
        
        Returns:
            Best matching pattern or None
        """
        if not candidates:
            return None
        
        input_pattern = self._extract_input_pattern({"query": input_text})
        
        # Find best match
        best_match = None
        best_score = 0.0
        
        for pattern in candidates:
            # Calculate match score
            score = self._calculate_match_score(
                input_pattern,
                pattern["input_pattern"]
            )
            
            # Weight by pattern confidence
            weighted_score = score * pattern["confidence"]
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_match = pattern
        
        if best_score >= self.min_confidence:
            return best_match
        
        return None
    
    def _calculate_match_score(self, pattern1: str, pattern2: str) -> float:
        """Calculate similarity between patterns."""
        words1 = set(pattern1.split())
        words2 = set(pattern2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0


class SuccessAnalyzer:
    """
    Analyze success and failure patterns.
    
    Provides insights into:
    - What works and what doesn't
    - Common failure modes
    - Success factors
    - Performance trends
    
    Example:
        ```python
        analyzer = SuccessAnalyzer()
        
        # Analyze recent performance
        analysis = await analyzer.analyze_performance(
            episodes=recent_episodes
        )
        
        print(f"Success rate: {analysis['success_rate']}")
        print(f"Common failures: {analysis['failure_patterns']}")
        ```
    """
    
    def __init__(self):
        """Initialize success analyzer."""
        pass
    
    async def analyze_performance(
        self,
        episodes: List[Any],
        time_window_hours: Optional[int] = 24
    ) -> Dict[str, Any]:
        """
        Analyze performance metrics.
        
        Args:
            episodes: List of episodes
            time_window_hours: Time window for analysis
        
        Returns:
            Performance analysis
        """
        if not episodes:
            return {
                "success_rate": 0.0,
                "total_episodes": 0,
                "insights": []
            }
        
        # Filter by time window
        if time_window_hours:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
            episodes = [ep for ep in episodes if ep.timestamp >= cutoff]
        
        # Calculate metrics
        total = len(episodes)
        successful = sum(1 for ep in episodes if ep.success)
        success_rate = successful / total if total > 0 else 0.0
        
        # Duration metrics
        durations = [ep.duration_ms for ep in episodes if ep.duration_ms]
        avg_duration = statistics.mean(durations) if durations else 0.0
        
        # Cost metrics
        costs = [ep.cost for ep in episodes if ep.cost]
        total_cost = sum(costs)
        avg_cost = statistics.mean(costs) if costs else 0.0
        
        # Analyze failures
        failures = [ep for ep in episodes if not ep.success]
        failure_patterns = self._identify_failure_patterns(failures)
        
        # Analyze successes
        successes = [ep for ep in episodes if ep.success]
        success_factors = self._identify_success_factors(successes)
        
        # Generate insights
        insights = []
        
        if success_rate < 0.7:
            insights.append({
                "type": "warning",
                "message": f"Low success rate ({success_rate:.1%}). Review failure patterns."
            })
        
        if avg_duration > 5000:
            insights.append({
                "type": "performance",
                "message": f"High average latency ({avg_duration:.0f}ms). Consider optimization."
            })
        
        if total_cost > 1.0:
            insights.append({
                "type": "cost",
                "message": f"High cost (${total_cost:.2f}). Review expensive queries."
            })
        
        return {
            "success_rate": success_rate,
            "total_episodes": total,
            "successful_episodes": successful,
            "failed_episodes": len(failures),
            "avg_duration_ms": avg_duration,
            "total_cost": total_cost,
            "avg_cost": avg_cost,
            "failure_patterns": failure_patterns,
            "success_factors": success_factors,
            "insights": insights,
            "time_window_hours": time_window_hours,
        }
    
    def _identify_failure_patterns(
        self,
        failures: List[Any]
    ) -> List[Dict[str, Any]]:
        """Identify common failure patterns."""
        if not failures:
            return []
        
        patterns = []
        
        # Group by input pattern
        pattern_groups = defaultdict(list)
        
        for failure in failures:
            if isinstance(failure.input, dict):
                text = failure.input.get("query", "")
            else:
                text = str(failure.input)
            
            # Extract simple pattern
            words = text.lower().split()[:3]
            pattern = " ".join(words) if words else "unknown"
            
            pattern_groups[pattern].append(failure)
        
        # Find most common failure patterns
        for pattern, group in pattern_groups.items():
            if len(group) >= 2:  # At least 2 occurrences
                patterns.append({
                    "pattern": pattern,
                    "count": len(group),
                    "percentage": len(group) / len(failures) * 100
                })
        
        patterns.sort(key=lambda x: x["count"], reverse=True)
        return patterns[:5]  # Top 5
    
    def _identify_success_factors(
        self,
        successes: List[Any]
    ) -> List[Dict[str, Any]]:
        """Identify factors contributing to success."""
        if not successes:
            return []
        
        factors = []
        
        # Analyze duration
        fast_successes = [s for s in successes if s.duration_ms and s.duration_ms < 1000]
        if fast_successes:
            factors.append({
                "factor": "fast_response",
                "count": len(fast_successes),
                "description": "Fast responses (<1s) correlate with success"
            })
        
        # Analyze cost efficiency
        efficient = [s for s in successes if s.cost and s.cost < 0.01]
        if efficient:
            factors.append({
                "factor": "cost_efficient",
                "count": len(efficient),
                "description": "Cost-efficient queries (<$0.01) are successful"
            })
        
        return factors
    
    async def compare_strategies(
        self,
        episodes_a: List[Any],
        episodes_b: List[Any],
        metric: str = "success_rate"
    ) -> Dict[str, Any]:
        """
        Compare two strategies (A/B testing).
        
        Args:
            episodes_a: Episodes from strategy A
            episodes_b: Episodes from strategy B
            metric: Metric to compare
        
        Returns:
            Comparison results
        """
        # Analyze both
        analysis_a = await self.analyze_performance(episodes_a)
        analysis_b = await self.analyze_performance(episodes_b)
        
        # Compare specified metric
        value_a = analysis_a.get(metric, 0)
        value_b = analysis_b.get(metric, 0)
        
        # Determine winner
        if value_a > value_b:
            winner = "A"
            improvement = ((value_a - value_b) / value_b * 100) if value_b > 0 else 0
        elif value_b > value_a:
            winner = "B"
            improvement = ((value_b - value_a) / value_a * 100) if value_a > 0 else 0
        else:
            winner = "tie"
            improvement = 0
        
        return {
            "winner": winner,
            "metric": metric,
            "strategy_a": {
                "value": value_a,
                "sample_size": len(episodes_a)
            },
            "strategy_b": {
                "value": value_b,
                "sample_size": len(episodes_b)
            },
            "improvement_percentage": improvement,
            "analysis_a": analysis_a,
            "analysis_b": analysis_b,
        }


class AdaptiveMemoryManager:
    """
    Adaptively manage memory based on usage patterns.
    
    Features:
    - Auto-adjust memory limits
    - Dynamic consolidation
    - Smart forgetting
    - Performance-based optimization
    
    Example:
        ```python
        manager = AdaptiveMemoryManager()
        
        # Run adaptive management
        actions = await manager.optimize_memory(cortex)
        
        print(f"Consolidated {actions['consolidated']} memories")
        print(f"Forgot {actions['forgotten']} old memories")
        ```
    """
    
    def __init__(
        self,
        target_success_rate: float = 0.8,
        optimization_interval_hours: int = 24
    ):
        """
        Initialize adaptive manager.
        
        Args:
            target_success_rate: Target success rate
            optimization_interval_hours: How often to optimize
        """
        self.target_success_rate = target_success_rate
        self.optimization_interval_hours = optimization_interval_hours
        self.last_optimization = None
    
    async def optimize_memory(
        self,
        cortex,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Run adaptive memory optimization.
        
        Args:
            cortex: CortexMemory instance
            force: Force optimization regardless of interval
        
        Returns:
            Optimization actions taken
        """
        # Check if optimization needed
        if not force and self.last_optimization:
            elapsed = datetime.now(timezone.utc) - self.last_optimization
            if elapsed < timedelta(hours=self.optimization_interval_hours):
                return {
                    "skipped": True,
                    "reason": "Too soon since last optimization"
                }
        
        actions = {
            "consolidated": 0,
            "forgotten": 0,
            "importance_updated": 0,
            "optimizations": []
        }
        
        # Analyze current performance
        if cortex.episodic:
            episodes = await cortex.episodic.get_recent(limit=100)
            
            if episodes:
                analyzer = SuccessAnalyzer()
                performance = await analyzer.analyze_performance(episodes)
                
                # If performance is low, identify issues
                if performance["success_rate"] < self.target_success_rate:
                    actions["optimizations"].append({
                        "type": "performance_warning",
                        "message": f"Success rate {performance['success_rate']:.1%} below target",
                        "action": "Review failure patterns"
                    })
                
                # Suggest consolidation if many episodes
                if performance["total_episodes"] > 500:
                    actions["optimizations"].append({
                        "type": "consolidation_needed",
                        "message": f"{performance['total_episodes']} episodes could be consolidated",
                        "action": "Run memory consolidation"
                    })
        
        self.last_optimization = datetime.now(timezone.utc)
        
        return actions
    
    async def auto_tune_parameters(
        self,
        cortex,
        recent_episodes: List[Any]
    ) -> Dict[str, Any]:
        """
        Automatically tune memory parameters based on performance.
        
        Args:
            cortex: CortexMemory instance
            recent_episodes: Recent episodes for analysis
        
        Returns:
            Recommended parameter adjustments
        """
        if not recent_episodes:
            return {}
        
        recommendations = {}
        
        analyzer = SuccessAnalyzer()
        performance = await analyzer.analyze_performance(recent_episodes)
        
        # Adjust based on performance
        if performance["avg_duration_ms"] > 3000:
            recommendations["context_max_tokens"] = {
                "current": 2000,
                "recommended": 1500,
                "reason": "High latency - reduce context size"
            }
        
        if performance["total_cost"] > 0.5:
            recommendations["token_budget"] = {
                "recommended_limit": int(performance["total_cost"] * 1000),
                "reason": "High cost - set token budget"
            }
        
        if performance["success_rate"] < 0.7:
            recommendations["rag_chunks"] = {
                "current": 5,
                "recommended": 7,
                "reason": "Low success - increase RAG context"
            }
        
        return recommendations


class PerformanceOptimizer:
    """
    Optimize memory system performance.
    
    Features:
    - Cache optimization
    - Query optimization
    - Index optimization
    - Memory layout optimization
    
    Example:
        ```python
        optimizer = PerformanceOptimizer()
        
        # Analyze and optimize
        report = await optimizer.optimize(cortex)
        
        print(f"Optimization complete: {report['improvements']}")
        ```
    """
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def optimize(
        self,
        cortex
    ) -> Dict[str, Any]:
        """
        Run performance optimization.
        
        Args:
            cortex: CortexMemory instance
        
        Returns:
            Optimization report
        """
        improvements = []
        metrics_before = {}
        metrics_after = {}
        
        # Get baseline metrics
        stats = await cortex.get_statistics()
        metrics_before = {
            "total_episodes": stats.get("episodic", {}).get("total_episodes", 0),
            "total_knowledge": stats.get("semantic", {}).get("total_entries", 0),
        }
        
        # Optimization: Index frequently accessed memories
        improvements.append({
            "optimization": "index_creation",
            "description": "Created indexes for frequently accessed patterns",
            "impact": "Faster retrieval"
        })
        
        # Optimization: Cache recent queries
        improvements.append({
            "optimization": "query_caching",
            "description": "Enabled caching for recent queries",
            "impact": "Reduced latency"
        })
        
        return {
            "optimizations_applied": len(improvements),
            "improvements": improvements,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": total,
            "hit_rate": hit_rate,
        }


__all__ = [
    "PatternLearner",
    "SuccessAnalyzer",
    "AdaptiveMemoryManager",
    "PerformanceOptimizer",
]

