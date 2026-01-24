"""
Memory Consolidation - Automatic memory optimization and cleanup.

Provides utilities for:
- Automatic forgetting of old/unimportant memories
- Memory consolidation (merging similar memories)
- Importance scoring and decay
- Memory health monitoring
- Embedding-based clustering
"""

from typing import List, Dict, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta, timezone
import logging
import hashlib
import asyncio

logger = logging.getLogger(__name__)


class ImportanceScorer:
    """
    Calculate and update importance scores for memories.
    
    Factors:
    - Access frequency
    - Recency
    - Success rate
    - User feedback
    - Content uniqueness
    
    Example:
        ```python
        scorer = ImportanceScorer()
        
        # Calculate importance for episode
        score = scorer.score_episode(
            episode=episode,
            access_count=5,
            age_days=3
        )
        
        # Decay importance over time
        new_score = scorer.decay_score(
            current_score=0.8,
            age_days=30
        )
        ```
    """
    
    def __init__(
        self,
        access_weight: float = 0.3,
        recency_weight: float = 0.3,
        success_weight: float = 0.2,
        uniqueness_weight: float = 0.2,
        decay_half_life_days: int = 30
    ):
        """
        Initialize importance scorer.
        
        Args:
            access_weight: Weight for access frequency
            recency_weight: Weight for recency
            success_weight: Weight for success rate
            uniqueness_weight: Weight for uniqueness
            decay_half_life_days: Half-life for importance decay
        """
        self.access_weight = access_weight
        self.recency_weight = recency_weight
        self.success_weight = success_weight
        self.uniqueness_weight = uniqueness_weight
        self.decay_half_life_days = decay_half_life_days
    
    def score_episode(
        self,
        episode: Any,
        access_count: int = 0,
        age_days: Optional[float] = None
    ) -> float:
        """
        Calculate importance score for episode.
        
        Args:
            episode: Episode to score
            access_count: Number of times accessed
            age_days: Age in days (calculated if None)
        
        Returns:
            Importance score (0-1)
        """
        score = 0.0
        
        # Access frequency score
        access_score = min(1.0, access_count / 10)  # Normalize to 10 accesses
        score += access_score * self.access_weight
        
        # Recency score
        if age_days is None:
            age = datetime.now(timezone.utc) - episode.timestamp
            age_days = age.days + (age.seconds / 86400)
        
        recency_score = self.decay_score(1.0, age_days)
        score += recency_score * self.recency_weight
        
        # Success score
        success_score = 1.0 if episode.success else 0.0
        score += success_score * self.success_weight
        
        # Uniqueness score (based on metadata)
        uniqueness_score = getattr(episode, 'metadata', {}).get("uniqueness", 0.5)
        score += uniqueness_score * self.uniqueness_weight
        
        return min(1.0, score)
    
    def score_knowledge(
        self,
        entry: Any,
        access_count: int = 0,
        age_days: Optional[float] = None
    ) -> float:
        """
        Calculate importance score for knowledge entry.
        
        Args:
            entry: Knowledge entry to score
            access_count: Number of times accessed
            age_days: Age in days
        
        Returns:
            Importance score (0-1)
        """
        score = 0.0
        
        # Access frequency
        access_score = min(1.0, access_count / 10)
        score += access_score * 0.4
        
        # Existing importance
        existing_importance = getattr(entry, 'importance_score', 0.5)
        score += existing_importance * 0.3
        
        # Confidence
        confidence = getattr(entry, 'confidence_score', 0.5)
        score += confidence * 0.3
        
        return min(1.0, score)
    
    def decay_score(self, current_score: float, age_days: float) -> float:
        """
        Apply exponential decay to importance score.
        
        Args:
            current_score: Current importance score
            age_days: Age in days
        
        Returns:
            Decayed score
        """
        decay_factor = 0.5 ** (age_days / self.decay_half_life_days)
        return current_score * decay_factor
    
    def boost_score(
        self,
        current_score: float,
        boost_factor: float = 0.1
    ) -> float:
        """
        Boost importance score (e.g., after access).
        
        Args:
            current_score: Current score
            boost_factor: Amount to boost (0-1)
        
        Returns:
            Boosted score (capped at 1.0)
        """
        return min(1.0, current_score + boost_factor)


class MemoryConsolidator:
    """
    Consolidate similar memories to reduce redundancy.
    
    Uses embedding-based clustering when embedding function is provided,
    otherwise falls back to text similarity.
    
    Strategies:
    - Merge similar episodes into summaries
    - Deduplicate knowledge entries
    - Combine related patterns
    - Create hierarchical summaries
    
    Example:
        ```python
        consolidator = MemoryConsolidator(
            embedding_function=embed_fn,
            similarity_threshold=0.8
        )
        
        # Find similar episodes using embeddings
        clusters = consolidator.cluster_episodes(episodes)
        
        # Merge clusters
        for cluster in clusters:
            summary = await consolidator.merge_episodes(cluster)
        ```
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.8,
        embedding_function: Optional[Callable[[str], List[float]]] = None
    ):
        """
        Initialize consolidator.
        
        Args:
            similarity_threshold: Threshold for considering memories similar
            embedding_function: Optional function for embedding-based clustering
        """
        self.similarity_threshold = similarity_threshold
        self.embedding_function = embedding_function
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using word overlap (fallback)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _get_episode_text(self, episode: Any) -> str:
        """Extract text content from episode for similarity calculation."""
        if hasattr(episode, 'get_content_for_embedding'):
            return episode.get_content_for_embedding()
        
        parts = []
        if isinstance(episode.input, dict):
            parts.append(str(episode.input.get("query", episode.input)))
        else:
            parts.append(str(episode.input))
        
        if isinstance(episode.output, dict):
            parts.append(str(episode.output.get("response", episode.output)))
        else:
            parts.append(str(episode.output))
        
        return " ".join(parts)
    
    def _calculate_episode_similarity(
        self,
        ep1: Any,
        ep2: Any,
        embeddings: Optional[Dict[str, List[float]]] = None
    ) -> float:
        """
        Calculate similarity between two episodes.
        
        Uses embeddings if available, otherwise falls back to text similarity.
        """
        # Try embedding-based similarity first
        if embeddings:
            ep1_id = ep1.id if hasattr(ep1, 'id') else str(id(ep1))
            ep2_id = ep2.id if hasattr(ep2, 'id') else str(id(ep2))
            
            if ep1_id in embeddings and ep2_id in embeddings:
                return self._cosine_similarity(embeddings[ep1_id], embeddings[ep2_id])
        
        # Check if episodes have embeddings
        if hasattr(ep1, 'embedding') and ep1.embedding and hasattr(ep2, 'embedding') and ep2.embedding:
            return self._cosine_similarity(ep1.embedding, ep2.embedding)
        
        # Fall back to text similarity
        text1 = self._get_episode_text(ep1)
        text2 = self._get_episode_text(ep2)
        
        return self._text_similarity(text1, text2)
    
    def cluster_episodes(
        self,
        episodes: List[Any],
        max_cluster_size: int = 5
    ) -> List[List[Any]]:
        """
        Cluster similar episodes together using embeddings when available.
        
        Args:
            episodes: List of episodes
            max_cluster_size: Maximum episodes per cluster
        
        Returns:
            List of episode clusters
        """
        if not episodes:
            return []
        
        # Generate embeddings if function is available
        embeddings: Dict[str, List[float]] = {}
        
        if self.embedding_function:
            for episode in episodes:
                ep_id = episode.id if hasattr(episode, 'id') else str(id(episode))
                
                # Use existing embedding or generate new one
                if hasattr(episode, 'embedding') and episode.embedding:
                    embeddings[ep_id] = episode.embedding
                else:
                    try:
                        text = self._get_episode_text(episode)
                        embeddings[ep_id] = self.embedding_function(text)
                    except Exception as e:
                        logger.debug(f"Failed to generate embedding: {e}")
        
        # Cluster using similarity
        clusters = []
        used = set()
        
        for i, ep1 in enumerate(episodes):
            ep1_id = ep1.id if hasattr(ep1, 'id') else str(id(ep1))
            
            if ep1_id in used:
                continue
            
            cluster = [ep1]
            used.add(ep1_id)
            
            for j, ep2 in enumerate(episodes[i+1:], start=i+1):
                ep2_id = ep2.id if hasattr(ep2, 'id') else str(id(ep2))
                
                if ep2_id in used:
                    continue
                
                if len(cluster) >= max_cluster_size:
                    break
                
                # Check similarity
                similarity = self._calculate_episode_similarity(ep1, ep2, embeddings)
                
                if similarity >= self.similarity_threshold:
                    cluster.append(ep2)
                    used.add(ep2_id)
            
            if len(cluster) > 1:  # Only clusters with multiple episodes
                clusters.append(cluster)
        
        logger.debug(f"Created {len(clusters)} episode clusters from {len(episodes)} episodes")
        return clusters
    
    async def merge_episodes(
        self,
        episodes: List[Any],
        keep_first: bool = True
    ) -> Dict[str, Any]:
        """
        Merge similar episodes into a consolidated memory.
        
        Args:
            episodes: Episodes to merge
            keep_first: Keep first episode as base
        
        Returns:
            Merged episode data
        """
        if not episodes:
            return {}
        
        if len(episodes) == 1:
            return {
                "input": episodes[0].input,
                "output": episodes[0].output,
                "timestamp": episodes[0].timestamp,
                "count": 1
            }
        
        # Use first as base
        base = episodes[0]
        
        # Aggregate information
        merged = {
            "input": base.input,
            "output": base.output,
            "timestamp": base.timestamp,
            "count": len(episodes),
            "timestamps": [ep.timestamp for ep in episodes],
            "success_rate": sum(ep.success for ep in episodes) / len(episodes),
            "consolidated": True,
            "original_ids": [ep.id for ep in episodes if hasattr(ep, 'id')]
        }
        
        # Calculate average metrics
        costs = [ep.cost for ep in episodes if ep.cost is not None]
        if costs:
            merged["avg_cost"] = sum(costs) / len(costs)
        
        durations = [ep.duration_ms for ep in episodes if ep.duration_ms is not None]
        if durations:
            merged["avg_duration_ms"] = sum(durations) / len(durations)
        
        return merged
    
    def deduplicate_knowledge(
        self,
        entries: List[Any],
        merge_strategy: str = "keep_highest_confidence"
    ) -> List[Any]:
        """
        Deduplicate knowledge entries using embeddings when available.
        
        Args:
            entries: Knowledge entries
            merge_strategy: How to merge duplicates
        
        Returns:
            Deduplicated entries
        """
        if not entries:
            return []
        
        # Generate embeddings if available
        embeddings: Dict[str, List[float]] = {}
        
        if self.embedding_function:
            for entry in entries:
                entry_id = entry.id if hasattr(entry, 'id') else str(id(entry))
                
                if hasattr(entry, 'embedding') and entry.embedding:
                    embeddings[entry_id] = entry.embedding
                else:
                    try:
                        embeddings[entry_id] = self.embedding_function(entry.content)
                    except Exception:
                        pass
        
        # Group by similarity
        groups: Dict[str, List[Any]] = {}
        used = set()
        
        for i, entry1 in enumerate(entries):
            entry1_id = entry1.id if hasattr(entry1, 'id') else str(id(entry1))
            
            if entry1_id in used:
                continue
            
            # Create new group
            group_key = entry1_id
            groups[group_key] = [entry1]
            used.add(entry1_id)
            
            for j, entry2 in enumerate(entries[i+1:], start=i+1):
                entry2_id = entry2.id if hasattr(entry2, 'id') else str(id(entry2))
                
                if entry2_id in used:
                    continue
                
                # Calculate similarity
                if entry1_id in embeddings and entry2_id in embeddings:
                    similarity = self._cosine_similarity(
                        embeddings[entry1_id],
                        embeddings[entry2_id]
                    )
                else:
                    similarity = self._text_similarity(entry1.content, entry2.content)
                
                if similarity >= self.similarity_threshold:
                    groups[group_key].append(entry2)
                    used.add(entry2_id)
        
        # Keep best from each group
        deduplicated = []
        
        for group in groups.values():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Merge duplicates
                if merge_strategy == "keep_highest_confidence":
                    best = max(
                        group,
                        key=lambda x: getattr(x, 'confidence_score', 0.5)
                    )
                    deduplicated.append(best)
                elif merge_strategy == "keep_newest":
                    best = max(group, key=lambda x: getattr(x, 'created_at', datetime.min))
                    deduplicated.append(best)
                elif merge_strategy == "keep_most_accessed":
                    best = max(group, key=lambda x: getattr(x, 'access_count', 0))
                    deduplicated.append(best)
                else:
                    deduplicated.append(group[0])
        
        logger.debug(
            f"Deduplicated {len(entries)} entries to {len(deduplicated)} "
            f"(removed {len(entries) - len(deduplicated)} duplicates)"
        )
        
        return deduplicated
    
    def _create_content_fingerprint(self, content: str) -> str:
        """Create fingerprint for content."""
        # Normalize and hash
        normalized = content.lower().strip()
        normalized = " ".join(normalized.split())  # Normalize whitespace
        
        return hashlib.md5(normalized.encode()).hexdigest()[:16]


class AutomaticForgetfulness:
    """
    Automatically forget old or unimportant memories.
    
    Strategies:
    - Age-based forgetting
    - Importance-based forgetting
    - Capacity-based forgetting (LRU)
    - Selective forgetting based on criteria
    
    Example:
        ```python
        forgetfulness = AutomaticForgetfulness(
            max_age_days=90,
            min_importance=0.3
        )
        
        # Get candidates for forgetting
        to_forget = await forgetfulness.identify_forgettable_episodes(
            episodes=all_episodes
        )
        
        # Archive or delete
        for episode_id in to_forget:
            await memory.episodic.delete(episode_id)
        ```
    """
    
    def __init__(
        self,
        max_age_days: Optional[int] = 90,
        min_importance: float = 0.2,
        max_capacity: Optional[int] = None,
        preserve_successful: bool = True
    ):
        """
        Initialize automatic forgetfulness.
        
        Args:
            max_age_days: Maximum age before forgetting (None = no limit)
            min_importance: Minimum importance to keep
            max_capacity: Maximum number of memories (None = unlimited)
            preserve_successful: Always preserve successful interactions
        """
        self.max_age_days = max_age_days
        self.min_importance = min_importance
        self.max_capacity = max_capacity
        self.preserve_successful = preserve_successful
        self.scorer = ImportanceScorer()
    
    async def identify_forgettable_episodes(
        self,
        episodes: List[Any],
        access_counts: Optional[Dict[str, int]] = None
    ) -> List[str]:
        """
        Identify episodes that can be forgotten.
        
        Args:
            episodes: List of episodes
            access_counts: Access count per episode ID
        
        Returns:
            List of episode IDs to forget
        """
        if not episodes:
            return []
        
        access_counts = access_counts or {}
        forgettable = []
        now = datetime.now(timezone.utc)
        
        for episode in episodes:
            episode_id = episode.id if hasattr(episode, 'id') else str(episode)
            
            # Always preserve successful if configured
            if self.preserve_successful and getattr(episode, 'success', True):
                continue
            
            # Check age
            timestamp = getattr(episode, 'timestamp', now)
            age = now - timestamp
            age_days = age.days + (age.seconds / 86400)
            
            if self.max_age_days and age_days > self.max_age_days:
                # Calculate importance score
                importance = self.scorer.score_episode(
                    episode,
                    access_count=access_counts.get(episode_id, 0),
                    age_days=age_days
                )
                
                # Forget if not important
                if importance < self.min_importance:
                    forgettable.append(episode_id)
        
        # Apply capacity limit
        if self.max_capacity and len(episodes) > self.max_capacity:
            # Score all remaining episodes
            episodes_with_scores = []
            
            for episode in episodes:
                episode_id = episode.id if hasattr(episode, 'id') else str(episode)
                
                if episode_id in forgettable:
                    continue  # Already marked for forgetting
                
                timestamp = getattr(episode, 'timestamp', now)
                age = now - timestamp
                age_days = age.days + (age.seconds / 86400)
                
                importance = self.scorer.score_episode(
                    episode,
                    access_count=access_counts.get(episode_id, 0),
                    age_days=age_days
                )
                
                episodes_with_scores.append((episode_id, importance))
            
            # Sort by importance (ascending - lowest first)
            episodes_with_scores.sort(key=lambda x: x[1])
            
            # Calculate how many to remove
            excess = len(episodes) - self.max_capacity
            
            # Mark excess for forgetting
            for i in range(min(excess, len(episodes_with_scores))):
                episode_id, _ = episodes_with_scores[i]
                if episode_id not in forgettable:
                    forgettable.append(episode_id)
        
        logger.debug(f"Identified {len(forgettable)} forgettable episodes")
        return forgettable
    
    async def identify_forgettable_knowledge(
        self,
        entries: List[Any],
        access_counts: Optional[Dict[str, int]] = None
    ) -> List[str]:
        """
        Identify knowledge entries that can be forgotten.
        
        Args:
            entries: Knowledge entries
            access_counts: Access counts per entry ID
        
        Returns:
            List of entry IDs to forget
        """
        if not entries:
            return []
        
        access_counts = access_counts or {}
        forgettable = []
        now = datetime.now(timezone.utc)
        
        for entry in entries:
            entry_id = entry.id if hasattr(entry, 'id') else str(entry)
            
            # Calculate importance
            timestamp = getattr(entry, 'created_at', now)
            age = now - timestamp
            age_days = age.days + (age.seconds / 86400)
            
            importance = self.scorer.score_knowledge(
                entry,
                access_count=access_counts.get(entry_id, 0),
                age_days=age_days
            )
            
            # Check age
            if self.max_age_days and age_days > self.max_age_days:
                if importance < self.min_importance:
                    forgettable.append(entry_id)
            
            # Check importance alone (very low importance)
            elif importance < self.min_importance * 0.5:
                forgettable.append(entry_id)
        
        logger.debug(f"Identified {len(forgettable)} forgettable knowledge entries")
        return forgettable


class MemoryHealthMonitor:
    """
    Monitor memory system health and provide recommendations.
    
    Tracks:
    - Memory usage and growth rate
    - Access patterns
    - Consolidation opportunities
    - Performance metrics
    
    Example:
        ```python
        monitor = MemoryHealthMonitor(embedding_function=embed_fn)
        
        # Check health
        health = await monitor.check_health(cortex)
        
        if health['status'] == 'warning':
            print(f"Issues: {health['issues']}")
            print(f"Recommendations: {health['recommendations']}")
        ```
    """
    
    def __init__(
        self,
        embedding_function: Optional[Callable[[str], List[float]]] = None
    ):
        """
        Initialize health monitor.
        
        Args:
            embedding_function: Optional embedding function for analysis
        """
        self.consolidator = MemoryConsolidator(
            embedding_function=embedding_function
        )
        self.scorer = ImportanceScorer()
    
    async def check_health(
        self,
        cortex,
        check_consolidation: bool = True
    ) -> Dict[str, Any]:
        """
        Check memory system health.
        
        Args:
            cortex: CortexMemory instance
            check_consolidation: Check for consolidation opportunities
        
        Returns:
            Health report with status, issues, and recommendations
        """
        issues = []
        recommendations = []
        metrics = {}
        
        # Get statistics
        stats = await cortex.get_statistics()
        
        # Check episodic memory
        if cortex.episodic:
            ep_stats = stats.get("episodic", {})
            total_episodes = ep_stats.get("total_episodes", 0)
            
            metrics["total_episodes"] = total_episodes
            metrics["success_rate"] = ep_stats.get("success_rate", 0)
            
            # Check for excessive growth
            if total_episodes > 10000:
                issues.append("High episode count (>10k)")
                recommendations.append("Enable automatic forgetting or consolidation")
            
            # Check success rate
            if ep_stats.get("success_rate", 100) < 70:
                issues.append(f"Low success rate ({ep_stats.get('success_rate', 0):.1f}%)")
                recommendations.append("Review failure patterns and improve agent behavior")
            
            # Check for consolidation opportunities
            if check_consolidation and total_episodes > 100:
                recent = await cortex.episodic.get_recent(limit=100)
                clusters = self.consolidator.cluster_episodes(recent)
                
                if len(clusters) > 0:
                    metrics["consolidation_opportunities"] = len(clusters)
                    recommendations.append(
                        f"Found {len(clusters)} clusters of similar episodes that could be consolidated"
                    )
        
        # Check semantic memory
        if cortex.semantic:
            sem_stats = stats.get("semantic", {})
            total_entries = sem_stats.get("total_entries", 0)
            
            metrics["total_knowledge_entries"] = total_entries
            
            if total_entries > 50000:
                issues.append("High knowledge entry count (>50k)")
                recommendations.append("Consider deduplication or archiving old entries")
        
        # Check procedural memory
        if cortex.procedural:
            proc_stats = stats.get("procedural", {})
            total_patterns = proc_stats.get("total_patterns", 0)
            avg_success_rate = proc_stats.get("avg_success_rate", 0)
            
            metrics["total_patterns"] = total_patterns
            metrics["pattern_success_rate"] = avg_success_rate
            
            if avg_success_rate < 50:
                issues.append(f"Low pattern success rate ({avg_success_rate:.1f}%)")
                recommendations.append("Review and prune poorly performing patterns")
        
        # Check storage health
        if hasattr(cortex.storage, 'health_check'):
            try:
                storage_health = await cortex.storage.health_check()
                metrics["storage_health"] = storage_health.get("status", "unknown")
                
                if storage_health.get("status") == "unhealthy":
                    issues.append("Storage backend is unhealthy")
                    recommendations.append("Check storage connection and configuration")
            except Exception as e:
                issues.append(f"Failed to check storage health: {e}")
        
        # Determine overall status
        if len(issues) == 0:
            status = "healthy"
        elif len(issues) <= 2:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "status": status,
            "issues": issues,
            "recommendations": recommendations,
            "metrics": metrics,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def suggest_optimizations(
        self,
        cortex
    ) -> List[Dict[str, Any]]:
        """
        Suggest memory optimizations.
        
        Args:
            cortex: CortexMemory instance
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        stats = await cortex.get_statistics()
        
        # Analyze episodic memory
        if cortex.episodic:
            ep_stats = stats.get("episodic", {})
            total = ep_stats.get("total_episodes", 0)
            
            if total > 1000:
                suggestions.append({
                    "type": "consolidation",
                    "priority": "medium",
                    "description": "Consolidate similar episodes to reduce memory usage",
                    "estimated_reduction": f"{int(total * 0.2)} episodes (~20%)",
                    "action": "cortex.consolidate_memories()"
                })
            
            if total > 5000:
                suggestions.append({
                    "type": "forgetting",
                    "priority": "high",
                    "description": "Enable automatic forgetting for old/unimportant memories",
                    "estimated_reduction": f"{int(total * 0.3)} episodes (~30%)",
                    "action": "cortex.cleanup_old_memories(max_age_days=90)"
                })
        
        # Analyze semantic memory
        if cortex.semantic:
            sem_stats = stats.get("semantic", {})
            total = sem_stats.get("total_entries", 0)
            
            if total > 10000:
                suggestions.append({
                    "type": "deduplication",
                    "priority": "medium",
                    "description": "Deduplicate knowledge entries using embedding similarity",
                    "estimated_reduction": f"{int(total * 0.15)} entries (~15%)",
                    "action": "Run deduplication on semantic memory"
                })
        
        # Analyze procedural memory
        if cortex.procedural:
            proc_stats = stats.get("procedural", {})
            avg_success = proc_stats.get("avg_success_rate", 100)
            
            if avg_success < 70:
                suggestions.append({
                    "type": "pattern_pruning",
                    "priority": "high",
                    "description": "Remove poorly performing patterns",
                    "action": "Review and delete patterns with <50% success rate"
                })
        
        return suggestions
    
    async def get_memory_metrics(self, cortex) -> Dict[str, Any]:
        """
        Get detailed memory metrics for monitoring.
        
        Args:
            cortex: CortexMemory instance
        
        Returns:
            Detailed metrics dictionary
        """
        stats = await cortex.get_statistics()
        
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {},
            "episodic": {},
            "semantic": {},
            "procedural": {},
            "storage": {}
        }
        
        # Episodic metrics
        if "episodic" in stats:
            metrics["episodic"] = stats["episodic"]
            metrics["summary"]["episodes"] = stats["episodic"].get("total_episodes", 0)
        
        # Semantic metrics
        if "semantic" in stats:
            metrics["semantic"] = stats["semantic"]
            metrics["summary"]["knowledge"] = stats["semantic"].get("total_entries", 0)
        
        # Procedural metrics
        if "procedural" in stats:
            metrics["procedural"] = stats["procedural"]
            metrics["summary"]["patterns"] = stats["procedural"].get("total_patterns", 0)
        
        # Storage metrics
        if "storage" in stats:
            metrics["storage"] = stats["storage"]
        
        return metrics


__all__ = [
    "ImportanceScorer",
    "MemoryConsolidator",
    "AutomaticForgetfulness",
    "MemoryHealthMonitor",
]
