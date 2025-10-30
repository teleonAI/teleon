"""
Pattern Recognizer - Advanced Pattern Extraction.

Identifies patterns in agent interactions using clustering,
similarity analysis, and statistical methods.
"""

from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict
from datetime import datetime
from pydantic import BaseModel

from teleon.cortex.memory.episodic import Episode


class PatternSignature(BaseModel):
    """Signature of a behavioral pattern."""
    
    input_features: List[str]
    output_features: List[str]
    context_features: Dict[str, Any]
    frequency: int = 1
    success_rate: float = 100.0
    avg_cost: float = 0.0
    avg_latency_ms: float = 0.0
    first_seen: datetime
    last_seen: datetime
    
    def matches(self, other: "PatternSignature", threshold: float = 0.7) -> bool:
        """Check if this pattern matches another."""
        input_sim = self._jaccard_similarity(
            set(self.input_features),
            set(other.input_features)
        )
        output_sim = self._jaccard_similarity(
            set(self.output_features),
            set(other.output_features)
        )
        
        # Combined similarity
        similarity = (input_sim + output_sim) / 2
        return similarity >= threshold
    
    @staticmethod
    def _jaccard_similarity(set1: Set, set2: Set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0


class PatternRecognizer:
    """
    Advanced pattern recognition for agent interactions.
    
    Features:
    - Automatic pattern discovery
    - Similarity-based clustering
    - Feature extraction
    - Statistical analysis
    - Pattern evolution tracking
    
    Example:
        ```python
        recognizer = PatternRecognizer(
            min_pattern_size=3,
            similarity_threshold=0.7
        )
        
        # Analyze episodes
        patterns = await recognizer.recognize_patterns(episodes)
        
        # Find similar patterns
        similar = recognizer.find_similar(pattern_signature, all_patterns)
        
        # Get pattern insights
        insights = recognizer.get_insights(patterns)
        ```
    """
    
    def __init__(
        self,
        min_pattern_size: int = 3,
        similarity_threshold: float = 0.7,
        min_confidence: float = 0.6
    ):
        """
        Initialize pattern recognizer.
        
        Args:
            min_pattern_size: Minimum number of instances to form a pattern
            similarity_threshold: Threshold for pattern similarity (0-1)
            min_confidence: Minimum confidence for pattern acceptance
        """
        self.min_pattern_size = min_pattern_size
        self.similarity_threshold = similarity_threshold
        self.min_confidence = min_confidence
    
    async def recognize_patterns(
        self,
        episodes: List[Episode]
    ) -> List[PatternSignature]:
        """
        Recognize patterns from a list of episodes.
        
        Args:
            episodes: List of episodes to analyze
        
        Returns:
            List of recognized patterns
        """
        if len(episodes) < self.min_pattern_size:
            return []
        
        # Extract features from each episode
        feature_vectors = [self._extract_features(ep) for ep in episodes]
        
        # Cluster similar interactions
        clusters = self._cluster_by_similarity(feature_vectors, episodes)
        
        # Create pattern signatures from clusters
        patterns = []
        for cluster_episodes in clusters:
            if len(cluster_episodes) >= self.min_pattern_size:
                signature = self._create_pattern_signature(cluster_episodes)
                if signature and signature.success_rate >= self.min_confidence * 100:
                    patterns.append(signature)
        
        return patterns
    
    def _extract_features(self, episode: Episode) -> Dict[str, Any]:
        """
        Extract features from an episode.
        
        Args:
            episode: Episode to analyze
        
        Returns:
            Feature dictionary
        """
        features = {
            "input_words": set(),
            "output_words": set(),
            "input_length": 0,
            "output_length": 0,
            "has_cost": episode.cost is not None,
            "has_duration": episode.duration_ms is not None,
            "success": episode.success,
        }
        
        # Extract input features
        if "query" in episode.input:
            input_text = str(episode.input["query"]).lower()
            features["input_words"] = set(input_text.split())
            features["input_length"] = len(input_text)
        
        # Extract output features
        if "response" in episode.output:
            output_text = str(episode.output["response"]).lower()
            features["output_words"] = set(output_text.split())
            features["output_length"] = len(output_text)
        
        # Context features
        if episode.context:
            features["context_keys"] = set(episode.context.keys())
        
        return features
    
    def _cluster_by_similarity(
        self,
        feature_vectors: List[Dict[str, Any]],
        episodes: List[Episode]
    ) -> List[List[Episode]]:
        """
        Cluster episodes by feature similarity.
        
        Args:
            feature_vectors: Feature vectors for each episode
            episodes: Corresponding episodes
        
        Returns:
            List of episode clusters
        """
        clusters = []
        assigned = set()
        
        for i, (features1, episode1) in enumerate(zip(feature_vectors, episodes)):
            if i in assigned:
                continue
            
            cluster = [episode1]
            assigned.add(i)
            
            # Find similar episodes
            for j, (features2, episode2) in enumerate(zip(feature_vectors, episodes)):
                if j in assigned or j <= i:
                    continue
                
                similarity = self._calculate_feature_similarity(features1, features2)
                if similarity >= self.similarity_threshold:
                    cluster.append(episode2)
                    assigned.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _calculate_feature_similarity(
        self,
        features1: Dict[str, Any],
        features2: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity between two feature vectors.
        
        Args:
            features1: First feature vector
            features2: Second feature vector
        
        Returns:
            Similarity score (0-1)
        """
        scores = []
        
        # Input word similarity
        input_sim = self._jaccard_similarity(
            features1.get("input_words", set()),
            features2.get("input_words", set())
        )
        scores.append(input_sim * 0.4)  # Weight: 40%
        
        # Output word similarity
        output_sim = self._jaccard_similarity(
            features1.get("output_words", set()),
            features2.get("output_words", set())
        )
        scores.append(output_sim * 0.3)  # Weight: 30%
        
        # Length similarity
        input_len_sim = 1.0 - abs(
            features1.get("input_length", 0) - features2.get("input_length", 0)
        ) / max(features1.get("input_length", 1), features2.get("input_length", 1))
        scores.append(input_len_sim * 0.15)  # Weight: 15%
        
        output_len_sim = 1.0 - abs(
            features1.get("output_length", 0) - features2.get("output_length", 0)
        ) / max(features1.get("output_length", 1), features2.get("output_length", 1))
        scores.append(output_len_sim * 0.15)  # Weight: 15%
        
        return sum(scores)
    
    @staticmethod
    def _jaccard_similarity(set1: Set, set2: Set) -> float:
        """Calculate Jaccard similarity."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _create_pattern_signature(
        self,
        episodes: List[Episode]
    ) -> Optional[PatternSignature]:
        """
        Create a pattern signature from a cluster of episodes.
        
        Args:
            episodes: Episodes in the cluster
        
        Returns:
            Pattern signature or None
        """
        if not episodes:
            return None
        
        # Extract common features
        all_input_words = []
        all_output_words = []
        context_features = defaultdict(int)
        
        for ep in episodes:
            if "query" in ep.input:
                words = str(ep.input["query"]).lower().split()
                all_input_words.extend(words)
            
            if "response" in ep.output:
                words = str(ep.output["response"]).lower().split()
                all_output_words.extend(words)
            
            for key in ep.context.keys():
                context_features[key] += 1
        
        # Find most common words (pattern features)
        word_counts_input = defaultdict(int)
        for word in all_input_words:
            word_counts_input[word] += 1
        
        word_counts_output = defaultdict(int)
        for word in all_output_words:
            word_counts_output[word] += 1
        
        # Get top features
        top_input = sorted(
            word_counts_input.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        top_output = sorted(
            word_counts_output.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Calculate metrics
        success_count = sum(1 for ep in episodes if ep.success)
        success_rate = (success_count / len(episodes)) * 100
        
        costs = [ep.cost for ep in episodes if ep.cost]
        avg_cost = sum(costs) / len(costs) if costs else 0.0
        
        latencies = [ep.duration_ms for ep in episodes if ep.duration_ms]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        
        # Create signature
        return PatternSignature(
            input_features=[word for word, _ in top_input],
            output_features=[word for word, _ in top_output],
            context_features={k: v for k, v in context_features.items()},
            frequency=len(episodes),
            success_rate=success_rate,
            avg_cost=avg_cost,
            avg_latency_ms=avg_latency,
            first_seen=min(ep.timestamp for ep in episodes),
            last_seen=max(ep.timestamp for ep in episodes)
        )
    
    def find_similar(
        self,
        pattern: PatternSignature,
        candidates: List[PatternSignature],
        threshold: Optional[float] = None
    ) -> List[Tuple[PatternSignature, float]]:
        """
        Find patterns similar to a given pattern.
        
        Args:
            pattern: Pattern to match
            candidates: Candidate patterns
            threshold: Similarity threshold (uses instance threshold if None)
        
        Returns:
            List of (pattern, similarity) tuples
        """
        threshold = threshold or self.similarity_threshold
        similar = []
        
        for candidate in candidates:
            if candidate == pattern:
                continue
            
            # Calculate similarity
            input_sim = self._jaccard_similarity(
                set(pattern.input_features),
                set(candidate.input_features)
            )
            output_sim = self._jaccard_similarity(
                set(pattern.output_features),
                set(candidate.output_features)
            )
            
            similarity = (input_sim + output_sim) / 2
            
            if similarity >= threshold:
                similar.append((candidate, similarity))
        
        # Sort by similarity
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar
    
    def get_insights(
        self,
        patterns: List[PatternSignature]
    ) -> Dict[str, Any]:
        """
        Get insights from recognized patterns.
        
        Args:
            patterns: List of patterns to analyze
        
        Returns:
            Dictionary with insights
        """
        if not patterns:
            return {
                "total_patterns": 0,
                "insights": []
            }
        
        insights = {
            "total_patterns": len(patterns),
            "total_interactions": sum(p.frequency for p in patterns),
            "avg_success_rate": sum(p.success_rate for p in patterns) / len(patterns),
            "patterns": []
        }
        
        # Sort patterns by frequency
        sorted_patterns = sorted(patterns, key=lambda p: p.frequency, reverse=True)
        
        # Analyze top patterns
        for pattern in sorted_patterns[:5]:
            insights["patterns"].append({
                "frequency": pattern.frequency,
                "success_rate": round(pattern.success_rate, 2),
                "avg_cost": round(pattern.avg_cost, 6),
                "avg_latency_ms": round(pattern.avg_latency_ms, 2),
                "input_features": pattern.input_features[:5],
                "output_features": pattern.output_features[:5],
            })
        
        # Identify most successful patterns
        successful = [p for p in patterns if p.success_rate >= 90.0]
        if successful:
            insights["most_successful"] = {
                "count": len(successful),
                "avg_frequency": sum(p.frequency for p in successful) / len(successful),
                "example_features": successful[0].input_features[:3]
            }
        
        # Identify cost-efficient patterns
        cost_patterns = [p for p in patterns if p.avg_cost > 0]
        if cost_patterns:
            cheapest = min(cost_patterns, key=lambda p: p.avg_cost)
            insights["most_cost_efficient"] = {
                "avg_cost": round(cheapest.avg_cost, 6),
                "success_rate": round(cheapest.success_rate, 2),
                "features": cheapest.input_features[:3]
            }
        
        # Identify fast patterns
        latency_patterns = [p for p in patterns if p.avg_latency_ms > 0]
        if latency_patterns:
            fastest = min(latency_patterns, key=lambda p: p.avg_latency_ms)
            insights["fastest"] = {
                "avg_latency_ms": round(fastest.avg_latency_ms, 2),
                "success_rate": round(fastest.success_rate, 2),
                "features": fastest.input_features[:3]
            }
        
        return insights
    
    def detect_anomalies(
        self,
        episodes: List[Episode],
        patterns: List[PatternSignature]
    ) -> List[Episode]:
        """
        Detect anomalous episodes that don't match known patterns.
        
        Args:
            episodes: Episodes to check
            patterns: Known patterns
        
        Returns:
            List of anomalous episodes
        """
        if not patterns:
            return []
        
        anomalies = []
        
        for episode in episodes:
            features = self._extract_features(episode)
            
            # Check if episode matches any pattern
            matches_pattern = False
            for pattern in patterns:
                input_sim = self._jaccard_similarity(
                    features.get("input_words", set()),
                    set(pattern.input_features)
                )
                
                if input_sim >= self.similarity_threshold:
                    matches_pattern = True
                    break
            
            if not matches_pattern:
                anomalies.append(episode)
        
        return anomalies

