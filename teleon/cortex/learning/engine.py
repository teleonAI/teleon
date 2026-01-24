"""
Learning Engine - Automatic Improvement System.

The Learning Engine orchestrates learning across all memory types,
analyzing interactions to extract patterns, optimize costs, and
improve agent performance over time.
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field

from teleon.cortex.memory.episodic import EpisodicMemory, Episode
from teleon.cortex.memory.semantic import SemanticMemory
from teleon.cortex.memory.procedural import ProceduralMemory


class LearningMetrics(BaseModel):
    """Metrics tracked by the learning engine."""
    
    total_interactions: int = 0
    patterns_learned: int = 0
    knowledge_extracted: int = 0
    optimizations_applied: int = 0
    
    # Performance improvements
    avg_cost_reduction: float = 0.0
    avg_latency_reduction: float = 0.0
    success_rate_improvement: float = 0.0
    
    # Time tracking
    learning_started: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_learning_cycle: Optional[datetime] = None


class LearningConfig(BaseModel):
    """Configuration for learning engine."""
    
    # Learning behavior
    auto_learn: bool = Field(True, description="Automatically learn from interactions")
    batch_size: int = Field(10, description="Number of interactions before learning cycle")
    
    # Pattern extraction
    min_pattern_confidence: float = Field(0.7, description="Minimum confidence for patterns")
    min_pattern_usage: int = Field(3, description="Minimum uses before pattern is trusted")
    
    # Knowledge extraction
    extract_knowledge: bool = Field(True, description="Auto-extract knowledge from interactions")
    knowledge_threshold: float = Field(0.8, description="Confidence threshold for knowledge")
    
    # Optimization
    optimize_costs: bool = Field(True, description="Enable cost optimization")
    optimize_latency: bool = Field(True, description="Enable latency optimization")
    
    # Cleanup
    cleanup_poor_patterns: bool = Field(True, description="Remove underperforming patterns")
    cleanup_threshold: float = Field(30.0, description="Success rate threshold for cleanup (%)")


class LearningEngine:
    """
    Learning Engine for automatic agent improvement.
    
    The engine analyzes interactions stored in episodic memory,
    extracts patterns for procedural memory, identifies knowledge
    for semantic memory, and optimizes performance.
    
    Features:
    - Automatic pattern extraction
    - Knowledge discovery
    - Cost optimization
    - Latency optimization
    - Performance tracking
    - Continuous improvement
    
    Example:
        ```python
        engine = LearningEngine(
            episodic=episodic_memory,
            semantic=semantic_memory,
            procedural=procedural_memory,
            config=LearningConfig(
                auto_learn=True,
                batch_size=10
            )
        )
        
        # Process an interaction
        await engine.process_interaction(
            input_data={"query": "What is AI?"},
            output_data={"response": "AI is..."},
            success=True,
            cost=0.002,
            duration_ms=150
        )
        
        # Manually trigger learning cycle
        await engine.learn()
        
        # Get metrics
        metrics = engine.get_metrics()
        print(f"Cost reduction: {metrics.avg_cost_reduction}%")
        ```
    """
    
    def __init__(
        self,
        episodic: EpisodicMemory,
        semantic: Optional[SemanticMemory] = None,
        procedural: Optional[ProceduralMemory] = None,
        config: Optional[LearningConfig] = None
    ):
        """
        Initialize learning engine.
        
        Args:
            episodic: Episodic memory (required for learning)
            semantic: Semantic memory (for knowledge extraction)
            procedural: Procedural memory (for pattern learning)
            config: Learning configuration
        """
        self.episodic = episodic
        self.semantic = semantic
        self.procedural = procedural
        self.config = config or LearningConfig()
        
        self.metrics = LearningMetrics()
        self._interaction_buffer: List[Episode] = []
        self._baseline_metrics: Optional[Dict[str, float]] = None
    
    async def process_interaction(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        success: bool = True,
        cost: Optional[float] = None,
        duration_ms: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Process a single interaction for learning.
        
        Args:
            input_data: Input data
            output_data: Output data
            success: Whether interaction was successful
            cost: Cost of interaction
            duration_ms: Duration in milliseconds
            context: Context information
            session_id: Session ID
            conversation_id: Conversation ID
        
        Returns:
            Episode ID
        """
        # Create and store episode
        episode = Episode(
            agent_id=self.episodic.agent_id,
            input=input_data,
            output=output_data,
            context=context or {},
            success=success,
            cost=cost,
            duration_ms=duration_ms,
            session_id=session_id,
            conversation_id=conversation_id
        )
        
        episode_id = await self.episodic.store(episode)
        
        # Update metrics
        self.metrics.total_interactions += 1
        
        # Add to buffer for batch learning
        self._interaction_buffer.append(episode)
        
        # Trigger learning if buffer is full
        if self.config.auto_learn and len(self._interaction_buffer) >= self.config.batch_size:
            await self.learn()
        
        return episode_id
    
    async def learn(self) -> Dict[str, Any]:
        """
        Run a learning cycle.
        
        Analyzes buffered interactions to:
        - Extract patterns
        - Identify knowledge
        - Optimize performance
        
        Returns:
            Dictionary with learning results
        """
        if not self._interaction_buffer:
            return {"message": "No interactions to learn from"}
        
        results = {
            "interactions_processed": len(self._interaction_buffer),
            "patterns_learned": 0,
            "knowledge_extracted": 0,
            "optimizations": []
        }
        
        # Extract patterns if procedural memory available
        if self.procedural:
            patterns = await self._extract_patterns(self._interaction_buffer)
            results["patterns_learned"] = len(patterns)
            self.metrics.patterns_learned += len(patterns)
        
        # Extract knowledge if semantic memory available
        if self.semantic and self.config.extract_knowledge:
            knowledge_items = await self._extract_knowledge(self._interaction_buffer)
            results["knowledge_extracted"] = len(knowledge_items)
            self.metrics.knowledge_extracted += len(knowledge_items)
        
        # Optimize if enabled
        if self.config.optimize_costs or self.config.optimize_latency:
            optimizations = await self._optimize_performance(self._interaction_buffer)
            results["optimizations"] = optimizations
            self.metrics.optimizations_applied += len(optimizations)
        
        # Cleanup poor patterns if enabled
        if self.config.cleanup_poor_patterns and self.procedural:
            cleanup_count = await self._cleanup_patterns()
            results["patterns_cleaned"] = cleanup_count
        
        # Calculate improvements
        improvements = await self._calculate_improvements()
        results["improvements"] = improvements
        
        # Update metrics
        self.metrics.last_learning_cycle = datetime.now(timezone.utc)
        if improvements:
            self.metrics.avg_cost_reduction = improvements.get("cost_reduction", 0.0)
            self.metrics.avg_latency_reduction = improvements.get("latency_reduction", 0.0)
            self.metrics.success_rate_improvement = improvements.get("success_improvement", 0.0)
        
        # Clear buffer
        self._interaction_buffer.clear()
        
        return results
    
    async def _extract_patterns(self, episodes: List[Episode]) -> List[str]:
        """
        Extract patterns from successful interactions.
        
        Args:
            episodes: List of episodes to analyze
        
        Returns:
            List of pattern IDs
        """
        if not self.procedural:
            return []
        
        pattern_ids = []
        
        # Group episodes by input similarity
        patterns = self._identify_similar_interactions(episodes)
        
        for pattern_group in patterns:
            # Calculate average performance
            success_count = sum(1 for ep in pattern_group if ep.success)
            success_rate = (success_count / len(pattern_group)) * 100
            
            # Only learn patterns with good success rate
            if success_rate < self.config.min_pattern_confidence * 100:
                continue
            
            # Create pattern from most successful example
            best_episode = max(pattern_group, key=lambda e: 1 if e.success else 0)
            
            # Extract pattern description (simplified)
            input_pattern = str(best_episode.input.get("query", ""))[:200]
            output_pattern = str(best_episode.output.get("response", ""))[:200]
            
            if input_pattern and output_pattern:
                # Calculate average metrics
                avg_cost = sum(e.cost for e in pattern_group if e.cost) / len(pattern_group)
                avg_latency = sum(e.duration_ms for e in pattern_group if e.duration_ms) / len(pattern_group)
                
                pattern_id = await self.procedural.learn(
                    input_pattern=input_pattern,
                    output_pattern=output_pattern,
                    success=True,
                    cost=avg_cost,
                    latency_ms=int(avg_latency)
                )
                pattern_ids.append(pattern_id)
        
        return pattern_ids
    
    def _identify_similar_interactions(
        self,
        episodes: List[Episode],
        similarity_threshold: float = 0.6
    ) -> List[List[Episode]]:
        """
        Group similar interactions together.
        
        Args:
            episodes: List of episodes
            similarity_threshold: Threshold for similarity
        
        Returns:
            List of episode groups
        """
        # Simple clustering implementation: uses Jaccard similarity on word overlap
        # This works but is basic - could be enhanced with vector embeddings for
        # better semantic clustering (e.g., using the embedding function if available)
        
        groups = []
        processed = set()
        
        for i, ep1 in enumerate(episodes):
            if i in processed:
                continue
            
            group = [ep1]
            processed.add(i)
            
            input1 = str(ep1.input.get("query", "")).lower()
            words1 = set(input1.split())
            
            for j, ep2 in enumerate(episodes[i+1:], i+1):
                if j in processed:
                    continue
                
                input2 = str(ep2.input.get("query", "")).lower()
                words2 = set(input2.split())
                
                # Calculate Jaccard similarity
                if words1 and words2:
                    intersection = len(words1 & words2)
                    union = len(words1 | words2)
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity >= similarity_threshold:
                        group.append(ep2)
                        processed.add(j)
            
            # Only keep groups with multiple interactions
            if len(group) >= self.config.min_pattern_usage:
                groups.append(group)
        
        return groups
    
    async def _extract_knowledge(self, episodes: List[Episode]) -> List[str]:
        """
        Extract knowledge from interactions.
        
        Args:
            episodes: List of episodes to analyze
        
        Returns:
            List of knowledge entry IDs
        """
        if not self.semantic:
            return []
        
        knowledge_ids = []
        
        # Extract factual statements from successful interactions
        for episode in episodes:
            if not episode.success:
                continue
            
            # Look for factual content in output
            output_text = str(episode.output.get("response", ""))
            
            # Simple heuristic-based knowledge extraction
            # Uses basic pattern matching (looks for factual indicators like "is", "are", "was")
            # This works but is basic - could be enhanced with NLP/LLM for more sophisticated
            # extraction (e.g., named entity recognition, fact extraction models)
            if len(output_text) > 50 and any(
                indicator in output_text.lower()
                for indicator in ["is", "are", "was", "means", "refers to"]
            ):
                # Check if this looks like a factual statement
                sentences = output_text.split(".")
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 30 and len(sentence) < 300:
                        # Store as knowledge
                        knowledge_id = await self.semantic.store(
                            content=sentence,
                            source="learned",
                            category="auto_extracted",
                            tags=["learned", "interaction"],
                            confidence_score=self.config.knowledge_threshold
                        )
                        knowledge_ids.append(knowledge_id)
        
        return knowledge_ids
    
    async def _optimize_performance(self, episodes: List[Episode]) -> List[Dict[str, Any]]:
        """
        Identify performance optimization opportunities.
        
        Args:
            episodes: List of episodes to analyze
        
        Returns:
            List of optimization recommendations
        """
        optimizations = []
        
        # Analyze cost patterns
        if self.config.optimize_costs:
            cost_episodes = [e for e in episodes if e.cost is not None]
            if cost_episodes:
                avg_cost = sum(e.cost for e in cost_episodes) / len(cost_episodes)
                high_cost = [e for e in cost_episodes if e.cost > avg_cost * 1.5]
                
                if high_cost:
                    optimizations.append({
                        "type": "cost",
                        "message": f"Found {len(high_cost)} high-cost interactions",
                        "avg_cost": avg_cost,
                        "high_cost_count": len(high_cost),
                        "suggestion": "Consider using cheaper models or caching"
                    })
        
        # Analyze latency patterns
        if self.config.optimize_latency:
            latency_episodes = [e for e in episodes if e.duration_ms is not None]
            if latency_episodes:
                avg_latency = sum(e.duration_ms for e in latency_episodes) / len(latency_episodes)
                slow = [e for e in latency_episodes if e.duration_ms > avg_latency * 1.5]
                
                if slow:
                    optimizations.append({
                        "type": "latency",
                        "message": f"Found {len(slow)} slow interactions",
                        "avg_latency_ms": avg_latency,
                        "slow_count": len(slow),
                        "suggestion": "Consider optimizing processing or using faster models"
                    })
        
        return optimizations
    
    async def _cleanup_patterns(self) -> int:
        """
        Clean up underperforming patterns.
        
        Returns:
            Number of patterns cleaned up
        """
        if not self.procedural:
            return 0
        
        # Get all patterns
        all_patterns = await self.procedural.get_top_patterns(limit=1000)
        
        cleanup_count = 0
        for pattern in all_patterns:
            # Only consider patterns with enough usage
            if pattern.usage_count < 5:
                continue
            
            # Delete if below threshold
            if pattern.success_rate() < self.config.cleanup_threshold:
                await self.procedural.delete(pattern.id)
                cleanup_count += 1
        
        return cleanup_count
    
    async def _calculate_improvements(self) -> Dict[str, float]:
        """
        Calculate performance improvements over baseline.
        
        Returns:
            Dictionary with improvement metrics
        """
        # Get recent episodes for comparison
        recent = await self.episodic.get_recent(limit=100)
        
        if len(recent) < 20:
            return {}
        
        # Split into recent and baseline
        baseline_episodes = recent[50:]
        recent_episodes = recent[:50]
        
        # Calculate metrics
        def calc_metrics(episodes):
            costs = [e.cost for e in episodes if e.cost]
            latencies = [e.duration_ms for e in episodes if e.duration_ms]
            successes = sum(1 for e in episodes if e.success)
            
            return {
                "avg_cost": sum(costs) / len(costs) if costs else 0,
                "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
                "success_rate": (successes / len(episodes) * 100) if episodes else 0
            }
        
        baseline = calc_metrics(baseline_episodes)
        current = calc_metrics(recent_episodes)
        
        # Calculate improvements
        improvements = {}
        
        if baseline["avg_cost"] > 0:
            improvements["cost_reduction"] = round(
                (1 - current["avg_cost"] / baseline["avg_cost"]) * 100, 2
            )
        
        if baseline["avg_latency"] > 0:
            improvements["latency_reduction"] = round(
                (1 - current["avg_latency"] / baseline["avg_latency"]) * 100, 2
            )
        
        if baseline["success_rate"] > 0:
            improvements["success_improvement"] = round(
                current["success_rate"] - baseline["success_rate"], 2
            )
        
        return improvements
    
    def get_metrics(self) -> LearningMetrics:
        """
        Get learning metrics.
        
        Returns:
            LearningMetrics object
        """
        return self.metrics
    
    async def get_learning_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive learning summary.
        
        Returns:
            Dictionary with learning statistics
        """
        summary = {
            "metrics": self.metrics.dict(),
            "episodic_stats": await self.episodic.get_statistics() if self.episodic else {},
            "semantic_stats": await self.semantic.get_statistics() if self.semantic else {},
            "procedural_stats": await self.procedural.get_statistics() if self.procedural else {},
        }
        
        # Calculate learning rate
        if self.metrics.learning_started:
            duration = (datetime.now(timezone.utc) - self.metrics.learning_started).total_seconds()
            if duration > 0:
                summary["learning_rate"] = {
                    "interactions_per_hour": (self.metrics.total_interactions / duration) * 3600,
                    "patterns_per_hour": (self.metrics.patterns_learned / duration) * 3600,
                    "knowledge_per_hour": (self.metrics.knowledge_extracted / duration) * 3600,
                }
        
        return summary

