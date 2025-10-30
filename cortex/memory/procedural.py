"""
Procedural Memory - Learned Patterns and Successful Strategies.

Procedural memory stores patterns of successful interactions and strategies.
It enables agents to learn from experience and reuse successful approaches.
"""

import uuid
import hashlib
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from teleon.cortex.storage import StorageBackend


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
    first_used: datetime = Field(default_factory=datetime.utcnow)
    last_used: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    # Categorization
    category: Optional[str] = Field(None, description="Pattern category")
    tags: List[str] = Field(default_factory=list, description="Pattern tags")
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
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


class ProceduralMemory:
    """
    Procedural memory for learned patterns and strategies.
    
    Features:
    - Learn from successful interactions
    - Store reusable patterns
    - Track performance metrics
    - Cost and latency optimization
    - Pattern matching and retrieval
    
    Example:
        ```python
        procedural = ProceduralMemory(storage, agent_id="agent-123")
        await procedural.initialize()
        
        # Learn a pattern
        pattern_id = await procedural.learn(
            input_pattern="user asks for help",
            output_pattern="provide clear step-by-step instructions",
            success=True,
            cost=0.002,
            latency_ms=150
        )
        
        # Find matching pattern
        pattern = await procedural.find_pattern("user needs help")
        
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
        min_success_rate: float = 50.0,
        ttl: Optional[int] = None
    ):
        """
        Initialize procedural memory.
        
        Args:
            storage: Storage backend to use
            agent_id: ID of the agent using this memory
            min_success_rate: Minimum success rate to keep patterns (%)
            ttl: Time-to-live for patterns in seconds (None = no expiration)
        """
        self.storage = storage
        self.agent_id = agent_id
        self.min_success_rate = min_success_rate
        self.ttl = ttl
        self._key_prefix = f"procedural:{agent_id}"
    
    async def initialize(self) -> None:
        """Initialize procedural memory."""
        if not self.storage._initialized:
            await self.storage.initialize()
    
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
        
        # Create new pattern
        pattern = Pattern(
            input_pattern=input_pattern,
            output_pattern=output_pattern,
            usage_count=1,
            success_count=1 if success else 0,
            failure_count=0 if success else 1,
            total_cost=cost or 0.0,
            total_latency_ms=latency_ms or 0,
            context_requirements=context or {},
            category=category,
            tags=tags or [],
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
        
        # Index by success rate (for quick retrieval of top patterns)
        success_key = f"{self._key_prefix}:success:{int(pattern.success_rate())}:{pattern.id}"
        await self.storage.set(success_key, pattern.id, ttl=self.ttl)
        
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
        
        pattern.last_used = datetime.utcnow()
        pattern.last_updated = datetime.utcnow()
        
        # Check if pattern should be kept
        if pattern.usage_count >= 5:  # Only prune after some use
            if pattern.success_rate() < self.min_success_rate:
                # Pattern performing poorly, delete it
                await self.delete(pattern_id)
                return True
        
        # Update pattern
        key = f"{self._key_prefix}:pattern:{pattern_id}"
        await self.storage.set(key, pattern.dict(), ttl=self.ttl)
        
        # Update success rate index
        old_success_key = f"{self._key_prefix}:success:*:{pattern_id}"
        await self.storage.delete(old_success_key)
        
        new_success_key = f"{self._key_prefix}:success:{int(pattern.success_rate())}:{pattern_id}"
        await self.storage.set(new_success_key, pattern_id, ttl=self.ttl)
        
        return True
    
    async def find_pattern(
        self,
        input_text: str,
        category: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Pattern]:
        """
        Find the best matching pattern for given input.
        
        Args:
            input_text: Input text to match
            category: Filter by category
            context: Context to match
        
        Returns:
            Best matching pattern or None
        """
        # Get candidate patterns
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
        
        # Score patterns by match quality and performance
        best_pattern = None
        best_score = 0.0
        
        input_lower = input_text.lower()
        
        for pattern_id in pattern_ids:
            pattern = await self.get(pattern_id)
            if not pattern:
                continue
            
            # Simple text matching (in production, use better matching)
            pattern_lower = pattern.input_pattern.lower()
            
            # Calculate match score
            common_words = set(input_lower.split()) & set(pattern_lower.split())
            if not common_words:
                continue
            
            match_score = len(common_words) / max(
                len(input_lower.split()),
                len(pattern_lower.split())
            )
            
            # Weight by success rate
            success_weight = pattern.success_rate() / 100.0
            
            # Weight by usage (more used = more trusted)
            usage_weight = min(pattern.usage_count / 10.0, 1.0)
            
            # Combined score
            score = match_score * 0.5 + success_weight * 0.3 + usage_weight * 0.2
            
            if score > best_score:
                best_score = score
                best_pattern = pattern
        
        return best_pattern
    
    async def get_top_patterns(
        self,
        limit: int = 10,
        category: Optional[str] = None,
        sort_by: str = "success_rate"
    ) -> List[Pattern]:
        """
        Get top performing patterns.
        
        Args:
            limit: Maximum number of patterns
            category: Filter by category
            sort_by: Sort criteria ("success_rate", "usage_count", "avg_cost", "avg_latency")
        
        Returns:
            List of top patterns
        """
        # Get patterns
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
        
        # Fetch patterns
        patterns = []
        for pid in pattern_ids:
            p = await self.get(pid)
            if p:
                patterns.append(p)
        
        # Sort patterns
        if sort_by == "success_rate":
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
        
        # Delete success rate index
        success_key = f"{self._key_prefix}:success:{int(pattern.success_rate())}:{pattern_id}"
        await self.storage.delete(success_key)
        
        return True
    
    async def clear(self) -> int:
        """
        Clear all patterns for this agent.
        
        Returns:
            Number of patterns deleted
        """
        pattern = f"{self._key_prefix}:*"
        return await self.storage.clear(pattern)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored patterns.
        
        Returns:
            Dictionary with statistics
        """
        # Count patterns
        pattern_pattern = f"{self._key_prefix}:pattern:*"
        pattern_keys = await self.storage.list_keys(pattern_pattern)
        total_patterns = len(pattern_keys)
        
        if total_patterns == 0:
            return {
                "total_patterns": 0,
                "avg_success_rate": 0,
                "avg_usage_count": 0,
                "total_usage": 0,
                "categories": [],
            }
        
        # Get sample of patterns (limit to avoid performance issues)
        sample_size = min(100, total_patterns)
        sample_keys = pattern_keys[:sample_size]
        
        patterns = []
        for key in sample_keys:
            data = await self.storage.get(key)
            if data:
                patterns.append(Pattern(**data))
        
        # Calculate statistics
        total_success_rate = sum(p.success_rate() for p in patterns)
        total_usage = sum(p.usage_count for p in patterns)
        categories = set(p.category for p in patterns if p.category)
        
        # Get best pattern info
        best_pattern_info = None
        if patterns:
            best_pattern = max(patterns, key=lambda p: p.success_rate())
            best_pattern_info = {
                "id": best_pattern.id,
                "input_pattern": best_pattern.input_pattern[:100] + "..." if len(best_pattern.input_pattern) > 100 else best_pattern.input_pattern,
                "success_rate": best_pattern.success_rate(),
                "usage_count": best_pattern.usage_count,
                "category": best_pattern.category
            }
        
        return {
            "total_patterns": total_patterns,
            "avg_success_rate": round(total_success_rate / len(patterns), 2) if patterns else 0,
            "avg_usage_count": round(total_usage / len(patterns), 2) if patterns else 0,
            "total_usage": total_usage,
            "categories": list(categories),
            "best_pattern": best_pattern_info,
        }

