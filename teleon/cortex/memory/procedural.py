"""
Procedural Memory - Learned Patterns and Successful Strategies.

Procedural memory stores patterns of successful interactions and strategies.
It enables agents to learn from experience and reuse successful approaches.

Enhanced with:
- Embedding-based pattern matching for semantic similarity
- Batch operations for efficiency
- Improved pattern scoring algorithms
"""

import uuid
import hashlib
import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, Tuple
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict, field_serializer
import logging

from teleon.cortex.storage.base import StorageBackend
from teleon.cortex.utils import AsyncLRUCache, validate_limit

logger = logging.getLogger("teleon.cortex.procedural")


class Pattern(BaseModel):
    """
    A learned pattern representing a successful strategy.
    
    Patterns capture input-output relationships with performance metrics,
    enabling agents to reuse successful approaches.
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Pattern definition
    input_pattern: str = Field(..., description="Pattern description for matching inputs")
    output_pattern: str = Field(..., description="Successful output strategy")
    
    # Embeddings for semantic matching
    input_embedding: Optional[List[float]] = Field(None, description="Embedding of input pattern")
    output_embedding: Optional[List[float]] = Field(None, description="Embedding of output pattern")
    
    # Context
    context_requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context conditions when this pattern applies"
    )
    
    # Performance metrics
    usage_count: int = Field(0, description="Number of times this pattern was used")
    success_count: int = Field(0, description="Number of successful applications")
    failure_count: int = Field(0, description="Number of failed applications")
    
    # Cost metrics
    total_cost: float = Field(0.0, description="Total cost of using this pattern")
    total_latency_ms: int = Field(0, description="Total latency in milliseconds")
    
    # Temporal information
    first_used: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Categorization
    category: Optional[str] = Field(None, description="Pattern category")
    tags: List[str] = Field(default_factory=list, description="Pattern tags")

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict()

    @field_serializer('first_used', 'last_used', 'last_updated')
    def serialize_datetime(self, value: datetime) -> str:
        return value.isoformat() if value else None
    
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return (self.success_count / total) * 100
    
    def avg_cost(self) -> float:
        """Calculate average cost per use."""
        if self.usage_count == 0:
            return 0.0
        return self.total_cost / self.usage_count
    
    def avg_latency_ms(self) -> float:
        """Calculate average latency in milliseconds."""
        if self.usage_count == 0:
            return 0.0
        return self.total_latency_ms / self.usage_count
    
    def pattern_hash(self) -> str:
        """Generate hash for pattern matching."""
        return hashlib.sha256(
            f"{self.input_pattern}:{self.output_pattern}".encode()
        ).hexdigest()[:16]
    
    def confidence_score(self) -> float:
        """
        Calculate confidence score based on usage and success.
        
        Returns:
            Confidence score (0-1)
        """
        if self.usage_count == 0:
            return 0.0
        
        # Base on success rate
        success_factor = self.success_rate() / 100.0
        
        # Weight by usage count (more usage = more confidence)
        usage_factor = min(1.0, self.usage_count / 20)  # Max out at 20 uses
        
        # Recency factor (more recent = more relevant)
        days_since_use = (datetime.now(timezone.utc) - self.last_used).days
        recency_factor = max(0.1, 1.0 - (days_since_use / 90))  # Decay over 90 days
        
        return (success_factor * 0.5 + usage_factor * 0.3 + recency_factor * 0.2)


class ProceduralMemory:
    """
    Procedural memory for learned patterns and strategies.
    
    Features:
    - Learn from successful interactions
    - Store reusable patterns with embeddings
    - Embedding-based semantic matching
    - Track performance metrics
    - Cost and latency optimization
    - Pattern matching and retrieval
    - Batch operations
    
    Example:
        ```python
        procedural = ProceduralMemory(
            storage=storage,
            agent_id="agent-123",
            embedding_function=embed_fn  # For semantic matching
        )
        await procedural.initialize()
        
        # Learn a pattern
        pattern_id = await procedural.learn(
            input_pattern="user asks for help with code",
            output_pattern="provide clear step-by-step code examples",
            success=True,
            cost=0.002,
            latency_ms=150
        )
        
        # Find matching pattern using embeddings
        pattern = await procedural.find_pattern("I need help debugging my Python code")
        
        # Update pattern performance
        await procedural.record_usage(pattern_id, success=True, cost=0.0018, latency_ms=120)
        
        # Get best patterns
        top_patterns = await procedural.get_top_patterns(limit=10)
        ```
    """
    
    def __init__(
        self,
        storage: StorageBackend,
        agent_id: str,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        min_success_rate: float = 50.0,
        ttl: Optional[int] = None
    ):
        """
        Initialize procedural memory.
        
        Args:
            storage: Storage backend to use
            agent_id: ID of the agent using this memory
            embedding_function: Function to generate embeddings for semantic matching
            min_success_rate: Minimum success rate to keep patterns (%)
            ttl: Time-to-live for patterns in seconds (None = no expiration)
        """
        self.storage = storage
        self.agent_id = agent_id
        self.embedding_function = embedding_function
        self.min_success_rate = min_success_rate
        self.ttl = ttl
        self._key_prefix = f"procedural:{agent_id}"
        
        # Async-safe cache for pattern embeddings
        self._embedding_cache = AsyncLRUCache(max_size=500, default_ttl=3600)
        
        # Statistics cache
        self._stats_cache: Optional[tuple[float, Dict[str, Any]]] = None
        self.max_limit = 10000
    
    async def initialize(self) -> None:
        """Initialize procedural memory."""
        if not self.storage._initialized:
            await self.storage.initialize()
        
        logger.info(
            f"ProceduralMemory initialized for agent: {self.agent_id} "
            f"(embedding_enabled={self.embedding_function is not None})"
        )
    
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
    
    def _word_overlap_similarity(self, text1: str, text2: str) -> float:
        """Calculate word overlap similarity (fallback when no embeddings)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    async def learn(
        self,
        input_pattern: str,
        output_pattern: str,
        success: bool = True,
        cost: Optional[float] = None,
        latency_ms: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Learn a new pattern or update existing one.
        
        Args:
            input_pattern: Pattern description for inputs
            output_pattern: Successful output strategy
            success: Whether this use was successful
            cost: Cost of the interaction
            latency_ms: Latency in milliseconds
            context: Context requirements
            category: Pattern category
            tags: Pattern tags
        
        Returns:
            Pattern ID
        """
        # Check if pattern already exists
        pattern_hash = hashlib.sha256(
            f"{input_pattern}:{output_pattern}".encode()
        ).hexdigest()[:16]
        
        hash_key = f"{self._key_prefix}:hash:{pattern_hash}"
        existing_id = await self.storage.get(hash_key)
        
        if existing_id:
            # Update existing pattern
            pattern = await self.get(existing_id)
            if pattern:
                await self.record_usage(
                    existing_id,
                    success=success,
                    cost=cost,
                    latency_ms=latency_ms
                )
                return existing_id
        
        # Generate embeddings if function is available
        input_embedding = None
        output_embedding = None
        
        if self.embedding_function:
            try:
                input_embedding = await asyncio.wait_for(
                    asyncio.to_thread(self.embedding_function, input_pattern),
                    timeout=5.0
                )
                output_embedding = await asyncio.wait_for(
                    asyncio.to_thread(self.embedding_function, output_pattern),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning(f"Pattern embedding generation timed out")
            except Exception as e:
                logger.warning(f"Failed to generate pattern embeddings: {e}")
        
        # Create new pattern
        pattern = Pattern(
            input_pattern=input_pattern,
            output_pattern=output_pattern,
            input_embedding=input_embedding,
            output_embedding=output_embedding,
            usage_count=1,
            success_count=1 if success else 0,
            failure_count=0 if success else 1,
            total_cost=cost or 0.0,
            total_latency_ms=latency_ms or 0,
            context_requirements=context or {},
            category=category,
            tags=tags or [],
        )
        
        # Cache embeddings
        if input_embedding and output_embedding:
            await self._embedding_cache.set(
                pattern.id,
                (input_embedding, output_embedding),
                ttl=3600
            )
        
        # Store pattern
        key = f"{self._key_prefix}:pattern:{pattern.id}"
        await self.storage.set(
            key,
            pattern.dict(),
            ttl=self.ttl,
            metadata={
                "type": "pattern",
                "agent_id": self.agent_id,
                "category": category,
                "success_rate": pattern.success_rate(),
            }
        )
        
        # Store hash for deduplication
        await self.storage.set(hash_key, pattern.id, ttl=self.ttl)
        
        # Index by category
        if category:
            cat_key = f"{self._key_prefix}:category:{category}:{pattern.id}"
            await self.storage.set(cat_key, pattern.id, ttl=self.ttl)
        
        # Index by tags
        for tag in (tags or []):
            tag_key = f"{self._key_prefix}:tag:{tag}:{pattern.id}"
            await self.storage.set(tag_key, pattern.id, ttl=self.ttl)
        
        logger.debug(f"Learned new pattern: {pattern.id}")
        return pattern.id
    
    async def get(self, pattern_id: str) -> Optional[Pattern]:
        """
        Get a pattern by ID.
        
        Args:
            pattern_id: Pattern ID
        
        Returns:
            Pattern if found
        """
        key = f"{self._key_prefix}:pattern:{pattern_id}"
        data = await self.storage.get(key)
        
        if data is None:
            return None
        
        return Pattern(**data)
    
    async def get_batch(self, pattern_ids: List[str]) -> List[Pattern]:
        """
        Get multiple patterns by ID in batch.
        
        Args:
            pattern_ids: List of pattern IDs
        
        Returns:
            List of patterns (missing IDs omitted)
        """
        if not pattern_ids:
            return []
        
        keys = [f"{self._key_prefix}:pattern:{pid}" for pid in pattern_ids]
        data_map = await self.storage.get_many(keys)
        
        patterns = []
        for key, data in data_map.items():
            if data:
                patterns.append(Pattern(**data))
        
        return patterns
    
    async def record_usage(
        self,
        pattern_id: str,
        success: bool,
        cost: Optional[float] = None,
        latency_ms: Optional[int] = None
    ) -> bool:
        """
        Record usage of a pattern.
        
        Args:
            pattern_id: Pattern ID
            success: Whether this use was successful
            cost: Cost of the interaction
            latency_ms: Latency in milliseconds
        
        Returns:
            True if recorded successfully
        """
        pattern = await self.get(pattern_id)
        if not pattern:
            return False
        
        # Update metrics
        pattern.usage_count += 1
        if success:
            pattern.success_count += 1
        else:
            pattern.failure_count += 1
        
        if cost is not None:
            pattern.total_cost += cost
        if latency_ms is not None:
            pattern.total_latency_ms += latency_ms
        
        pattern.last_used = datetime.now(timezone.utc)
        pattern.last_updated = datetime.now(timezone.utc)
        
        # Check if pattern should be kept
        if pattern.usage_count >= 5:  # Only prune after some use
            if pattern.success_rate() < self.min_success_rate:
                # Pattern performing poorly, delete it
                logger.info(
                    f"Deleting poorly performing pattern {pattern_id} "
                    f"(success_rate={pattern.success_rate():.1f}%)"
                )
                await self.delete(pattern_id)
                return True
        
        # Update pattern
        key = f"{self._key_prefix}:pattern:{pattern_id}"
        await self.storage.set(key, pattern.dict(), ttl=self.ttl)
        
        logger.debug(f"Recorded usage for pattern {pattern_id}")
        return True
    
    async def find_pattern(
        self,
        input_text: str,
        category: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.3
    ) -> Optional[Pattern]:
        """
        Find the best matching pattern for given input.
        
        Uses embedding similarity if available, falls back to word overlap.
        
        Args:
            input_text: Input text to match
            category: Filter by category
            context: Context to match
            min_similarity: Minimum similarity threshold
        
        Returns:
            Best matching pattern or None
        """
        # Get candidate patterns
        pattern_ids = await self._get_candidate_patterns(category)
        
        if not pattern_ids:
            return None
        
        # Batch fetch patterns
        patterns = await self.get_batch(pattern_ids)
        
        if not patterns:
            return None
        
        # Generate input embedding if function available
        input_embedding = None
        if self.embedding_function:
            try:
                input_embedding = await asyncio.wait_for(
                    asyncio.to_thread(self.embedding_function, input_text),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning(f"Input embedding generation timed out")
            except Exception as e:
                logger.warning(f"Failed to generate input embedding: {e}")
        
        # Score patterns
        best_pattern = None
        best_score = 0.0
        
        for pattern in patterns:
            # Calculate similarity score
            similarity = 0.0
            
            if input_embedding and pattern.input_embedding:
                # Use embedding similarity (semantic matching)
                similarity = self._cosine_similarity(input_embedding, pattern.input_embedding)
            elif input_embedding:
                # Try to get from async cache
                cached_embeddings = await self._embedding_cache.get(pattern.id)
                if cached_embeddings:
                    cached_input, _ = cached_embeddings
                    similarity = self._cosine_similarity(input_embedding, cached_input)
                else:
                    # Fallback to word overlap
                    similarity = self._word_overlap_similarity(input_text, pattern.input_pattern)
            else:
                # No input embedding, use word overlap
                similarity = self._word_overlap_similarity(input_text, pattern.input_pattern)
            
            if similarity < min_similarity:
                continue
            
            # Weight by pattern confidence
            confidence = pattern.confidence_score()
            
            # Combined score: similarity * weight + confidence * weight
            score = similarity * 0.6 + confidence * 0.4
            
            if score > best_score:
                best_score = score
                best_pattern = pattern
        
        if best_pattern:
            logger.debug(
                f"Found matching pattern {best_pattern.id} "
                f"(score={best_score:.3f})"
            )
        
        return best_pattern
    
    async def find_similar_patterns(
        self,
        input_text: str,
        limit: int = 5,
        min_similarity: float = 0.3
    ) -> List[Tuple[Pattern, float]]:
        """
        Find multiple similar patterns.
        
        Args:
            input_text: Input text to match
            limit: Maximum number of patterns to return
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of (pattern, score) tuples sorted by score
        """
        pattern_ids = await self._get_candidate_patterns(None)
        patterns = await self.get_batch(pattern_ids)
        
        if not patterns:
            return []
        
        # Generate input embedding
        input_embedding = None
        if self.embedding_function:
            try:
                input_embedding = await asyncio.wait_for(
                    asyncio.to_thread(self.embedding_function, input_text),
                    timeout=5.0
                )
            except (asyncio.TimeoutError, Exception):
                pass
        
        # Score all patterns
        scored_patterns = []
        
        for pattern in patterns:
            if input_embedding and pattern.input_embedding:
                similarity = self._cosine_similarity(input_embedding, pattern.input_embedding)
            else:
                similarity = self._word_overlap_similarity(input_text, pattern.input_pattern)
            
            if similarity >= min_similarity:
                score = similarity * 0.6 + pattern.confidence_score() * 0.4
                scored_patterns.append((pattern, score))
        
        # Sort by score
        scored_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return scored_patterns[:limit]
    
    async def _get_candidate_patterns(self, category: Optional[str]) -> List[str]:
        """Get candidate pattern IDs."""
        if category:
            pattern = f"{self._key_prefix}:category:{category}:*"
            cat_keys = await self.storage.list_keys(pattern)
            pattern_ids = []
            for key in cat_keys:
                pid = await self.storage.get(key)
                if pid:
                    pattern_ids.append(pid)
        else:
            pattern = f"{self._key_prefix}:pattern:*"
            pattern_keys = await self.storage.list_keys(pattern)
            pattern_ids = [key.split(":")[-1] for key in pattern_keys]
        
        return pattern_ids
    
    async def get_top_patterns(
        self,
        limit: int = 10,
        category: Optional[str] = None,
        sort_by: str = "confidence"
    ) -> List[Pattern]:
        """
        Get top performing patterns with bounds checking.
        
        Args:
            limit: Maximum number of patterns (validated and capped)
            category: Filter by category
            sort_by: Sort criteria ("confidence", "success_rate", "usage_count", "avg_cost", "avg_latency")
        
        Returns:
            List of top patterns
            
        Raises:
            ValueError: If limit is invalid
        """
        # Validate and cap limit
        limit = validate_limit(limit, max_limit=self.max_limit)
        pattern_ids = await self._get_candidate_patterns(category)
        patterns = await self.get_batch(pattern_ids)
        
        if not patterns:
            return []
        
        # Sort patterns
        if sort_by == "confidence":
            patterns.sort(key=lambda p: p.confidence_score(), reverse=True)
        elif sort_by == "success_rate":
            patterns.sort(key=lambda p: p.success_rate(), reverse=True)
        elif sort_by == "usage_count":
            patterns.sort(key=lambda p: p.usage_count, reverse=True)
        elif sort_by == "avg_cost":
            patterns.sort(key=lambda p: p.avg_cost())
        elif sort_by == "avg_latency":
            patterns.sort(key=lambda p: p.avg_latency_ms())
        
        return patterns[:limit]
    
    async def delete(self, pattern_id: str) -> bool:
        """
        Delete a pattern.
        
        Args:
            pattern_id: Pattern ID
        
        Returns:
            True if deleted
        """
        # Get pattern first
        pattern = await self.get(pattern_id)
        if not pattern:
            return False
        
        # Delete main pattern
        key = f"{self._key_prefix}:pattern:{pattern_id}"
        await self.storage.delete(key)
        
        # Delete from embedding cache
        await self._embedding_cache.delete(pattern_id)
        
        # Delete hash
        pattern_hash = pattern.pattern_hash()
        hash_key = f"{self._key_prefix}:hash:{pattern_hash}"
        await self.storage.delete(hash_key)
        
        # Delete category index
        if pattern.category:
            cat_key = f"{self._key_prefix}:category:{pattern.category}:{pattern_id}"
            await self.storage.delete(cat_key)
        
        # Delete tag indexes
        for tag in pattern.tags:
            tag_key = f"{self._key_prefix}:tag:{tag}:{pattern_id}"
            await self.storage.delete(tag_key)
        
        logger.debug(f"Deleted pattern: {pattern_id}")
        return True
    
    async def clear(self) -> int:
        """
        Clear all patterns for this agent.
        
        Returns:
            Number of patterns deleted
        """
        pattern = f"{self._key_prefix}:*"
        count = await self.storage.clear(pattern)
        await self._embedding_cache.clear()
        self._stats_cache = None
        
        logger.info(f"Cleared all patterns for agent: {self.agent_id}")
        return count
    
    async def get_statistics(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get statistics about stored patterns with caching.
        
        Args:
            use_cache: Use cached statistics if available (60 second TTL)
        
        Returns:
            Dictionary with statistics
        """
        # Check cache
        if use_cache and self._stats_cache:
            cache_time, cached_stats = self._stats_cache
            if (time.time() - cache_time) < 60:
                return cached_stats
        
        pattern_pattern = f"{self._key_prefix}:pattern:*"
        pattern_keys = await self.storage.list_keys(pattern_pattern, limit=10000)
        total_patterns = len(pattern_keys)
        
        if total_patterns == 0:
            return {
                "total_patterns": 0,
                "avg_success_rate": 0,
                "avg_confidence": 0,
                "avg_usage_count": 0,
                "total_usage": 0,
                "categories": [],
                "embedding_enabled": self.embedding_function is not None,
            }
        
        # Get sample of patterns (batch fetch)
        sample_size = min(100, total_patterns)
        sample_ids = [key.split(":")[-1] for key in pattern_keys[:sample_size]]
        patterns = await self.get_batch(sample_ids)
        
        if not patterns:
            return {
                "total_patterns": total_patterns,
                "avg_success_rate": 0,
                "avg_confidence": 0,
                "avg_usage_count": 0,
                "total_usage": 0,
                "categories": [],
                "embedding_enabled": self.embedding_function is not None,
            }
        
        # Calculate statistics
        total_success_rate = sum(p.success_rate() for p in patterns)
        total_confidence = sum(p.confidence_score() for p in patterns)
        total_usage = sum(p.usage_count for p in patterns)
        categories = set(p.category for p in patterns if p.category)
        
        # Get best pattern info
        best_pattern = max(patterns, key=lambda p: p.confidence_score())
        best_pattern_info = {
            "id": best_pattern.id,
            "input_pattern": best_pattern.input_pattern[:100] + "..." if len(best_pattern.input_pattern) > 100 else best_pattern.input_pattern,
            "success_rate": best_pattern.success_rate(),
            "confidence": round(best_pattern.confidence_score(), 3),
            "usage_count": best_pattern.usage_count,
            "category": best_pattern.category
        }
        
        stats = {
            "total_patterns": total_patterns,
            "avg_success_rate": round(total_success_rate / len(patterns), 2),
            "avg_confidence": round(total_confidence / len(patterns), 3),
            "avg_usage_count": round(total_usage / len(patterns), 2),
            "total_usage": total_usage,
            "categories": list(categories),
            "best_pattern": best_pattern_info,
            "embedding_enabled": self.embedding_function is not None,
            "cached_embeddings": await self._embedding_cache.size(),
        }
        
        # Cache results
        self._stats_cache = (time.time(), stats)
        
        return stats
