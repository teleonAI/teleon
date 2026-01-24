"""
Advanced Caching for LLM Responses.

This module provides sophisticated caching strategies including
exact match, semantic similarity, and prefix caching.
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, ConfigDict, field_serializer
from datetime import datetime, timedelta, timezone
from enum import Enum
import asyncio
import hashlib
import json

from teleon.core import (
    StructuredLogger,
    LogLevel,
)


class CacheStrategy(str, Enum):
    """Caching strategies."""
    EXACT = "exact"              # Exact prompt match
    SEMANTIC = "semantic"        # Semantic similarity
    PREFIX = "prefix"            # Common prefix caching
    HYBRID = "hybrid"            # Combination of strategies


class CacheEvictionPolicy(str, Enum):
    """Cache eviction policies."""
    LRU = "lru"                 # Least Recently Used
    LFU = "lfu"                 # Least Frequently Used
    COST_BASED = "cost_based"   # Evict lowest cost items first
    TTL = "ttl"                 # Time To Live


class CacheEntry(BaseModel):
    """A cached LLM response."""
    
    cache_key: str = Field(..., description="Cache key")
    
    # Request
    messages: List[Dict[str, str]] = Field(..., description="Input messages")
    model: str = Field(..., description="Model used")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Request parameters")
    
    # Response
    response: Any = Field(..., description="Cached response")
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = Field(0, description="Number of times accessed")
    
    # Token tracking (cost tracking removed)
    estimated_tokens_saved: int = Field(0, description="Tokens saved by this cache")
    total_tokens_saved: int = Field(0, description="Total tokens saved from hits")
    
    # Token info
    input_tokens: int = Field(0, description="Input token count")
    output_tokens: int = Field(0, description="Output token count")
    
    # TTL
    ttl_seconds: Optional[int] = Field(None, description="Time to live")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer('created_at', 'last_accessed')
    def serialize_datetime(self, value: datetime) -> str:
        return value.isoformat() if value else None

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl_seconds is None:
            return False

        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def record_hit(self):
        """Record a cache hit."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1
        total_tokens = self.input_tokens + self.output_tokens
        self.total_tokens_saved += total_tokens


class ResponseCache:
    """
    Semantic response cache for LLM operations.
    
    Features:
    - Multiple caching strategies
    - Configurable eviction policies
    - Cost-aware caching
    - Cache hit tracking
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        strategy: CacheStrategy = CacheStrategy.EXACT,
        eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU,
        default_ttl: Optional[int] = None
    ):
        """
        Initialize response cache.
        
        Args:
            max_size: Maximum cache entries
            strategy: Caching strategy
            eviction_policy: Eviction policy
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.strategy = strategy
        self.eviction_policy = eviction_policy
        self.default_ttl = default_ttl
        
        # Cache storage
        self.cache: Dict[str, CacheEntry] = {}
        
        # Statistics
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_cost_saved = 0  # Actually tracks tokens saved, kept name for compatibility
        
        self.lock = asyncio.Lock()
        self.logger = StructuredLogger("response_cache", LogLevel.INFO)
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    def _generate_cache_key(
        self,
        messages: List[Dict[str, str]],
        model: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate cache key for request.
        
        Args:
            messages: Chat messages
            model: Model name
            parameters: Request parameters
        
        Returns:
            Cache key
        """
        # Normalize parameters (exclude non-deterministic ones)
        norm_params = {}
        if parameters:
            for key, value in parameters.items():
                if key not in ['temperature', 'top_p', 'presence_penalty', 'frequency_penalty']:
                    norm_params[key] = value
        
        # Create canonical representation
        cache_obj = {
            "messages": messages,
            "model": model,
            "parameters": norm_params
        }
        
        # Generate hash
        cache_str = json.dumps(cache_obj, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    async def get(
        self,
        messages: List[Dict[str, str]],
        model: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Get cached response.
        
        Args:
            messages: Chat messages
            model: Model name
            parameters: Request parameters
        
        Returns:
            Cached response or None
        """
        async with self.lock:
            self.total_requests += 1
            
            # Generate key based on strategy
            if self.strategy == CacheStrategy.EXACT:
                cache_key = self._generate_cache_key(messages, model, parameters)
                entry = self.cache.get(cache_key)
                
                if entry and not entry.is_expired():
                    entry.record_hit()
                    self.cache_hits += 1
                    tokens_saved = entry.input_tokens + entry.output_tokens
                    self.total_cost_saved += tokens_saved  # Track as tokens
                    
                    self.logger.debug(
                        f"Cache hit for key {cache_key[:8]}... "
                        f"(saves {tokens_saved} tokens)"
                    )
                    
                    return entry.response
            
            elif self.strategy == CacheStrategy.SEMANTIC:
                # For semantic matching, need embeddings (simplified here)
                # In production, use actual embeddings and similarity search
                best_match = await self._find_semantic_match(messages, model)
                
                if best_match:
                    best_match.record_hit()
                    self.cache_hits += 1
                    tokens_saved = best_match.input_tokens + best_match.output_tokens
                    self.total_cost_saved += tokens_saved
                    
                    self.logger.debug(
                        f"Semantic cache hit "
                        f"(saves {tokens_saved} tokens)"
                    )
                    
                    return best_match.response
            
            elif self.strategy == CacheStrategy.PREFIX:
                # Check for common prefix
                prefix_match = await self._find_prefix_match(messages, model)
                
                if prefix_match:
                    prefix_match.record_hit()
                    self.cache_hits += 1
                    tokens_saved = (prefix_match.input_tokens + prefix_match.output_tokens) // 2  # Partial savings
                    self.total_cost_saved += tokens_saved
                    
                    self.logger.debug(
                        f"Prefix cache hit "
                        f"(saves {tokens_saved} tokens)"
                    )
                    
                    return prefix_match.response
            
            self.cache_misses += 1
            return None
    
    async def set(
        self,
        messages: List[Dict[str, str]],
        model: str,
        response: Any,
        parameters: Optional[Dict[str, Any]] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        ttl: Optional[int] = None
    ):
        """
        Cache a response.
        
        Args:
            messages: Chat messages
            model: Model name
            response: Response to cache
            parameters: Request parameters
            cost: Response cost
            input_tokens: Input token count
            output_tokens: Output token count
            ttl: Time to live (seconds)
        """
        async with self.lock:
            # Check size limit
            if len(self.cache) >= self.max_size:
                await self._evict_entry()
            
            # Generate key
            cache_key = self._generate_cache_key(messages, model, parameters)
            
            # Create entry
            entry = CacheEntry(
                cache_key=cache_key,
                messages=messages,
                model=model,
                parameters=parameters or {},
                response=response,
                estimated_cost=0.0,  # Deprecated - kept for compatibility
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                ttl_seconds=ttl or self.default_ttl
            )
            
            self.cache[cache_key] = entry
            
            self.logger.debug(
                f"Cached response {cache_key[:8]}... "
                f"(tokens={input_tokens}+{output_tokens})"
            )
    
    async def _find_semantic_match(
        self,
        messages: List[Dict[str, str]],
        model: str,
        similarity_threshold: float = 0.95
    ) -> Optional[CacheEntry]:
        """
        Find semantically similar cached entry.
        
        In production, this would use embeddings and vector similarity.
        This is a simplified implementation.
        
        Args:
            messages: Chat messages
            model: Model name
            similarity_threshold: Minimum similarity score
        
        Returns:
            Matching entry or None
        """
        # Simplified: check for exact match on last message
        if not messages:
            return None
        
        last_message = messages[-1].get("content", "")
        
        for entry in self.cache.values():
            if entry.is_expired():
                continue
            
            if entry.model != model:
                continue
            
            if entry.messages:
                entry_last = entry.messages[-1].get("content", "")
                
                # Simple similarity check (in production, use embeddings)
                if last_message.lower() == entry_last.lower():
                    return entry
        
        return None
    
    async def _find_prefix_match(
        self,
        messages: List[Dict[str, str]],
        model: str,
        min_prefix_length: int = 2
    ) -> Optional[CacheEntry]:
        """
        Find entry with common prefix.
        
        Useful for conversations where early messages are shared.
        
        Args:
            messages: Chat messages
            model: Model name
            min_prefix_length: Minimum prefix length
        
        Returns:
            Matching entry or None
        """
        if len(messages) < min_prefix_length:
            return None
        
        prefix_messages = messages[:min_prefix_length]
        
        for entry in self.cache.values():
            if entry.is_expired():
                continue
            
            if entry.model != model:
                continue
            
            if len(entry.messages) >= min_prefix_length:
                entry_prefix = entry.messages[:min_prefix_length]
                
                # Check if prefixes match
                if self._messages_equal(prefix_messages, entry_prefix):
                    return entry
        
        return None
    
    def _messages_equal(
        self,
        msgs1: List[Dict[str, str]],
        msgs2: List[Dict[str, str]]
    ) -> bool:
        """Check if message lists are equal."""
        if len(msgs1) != len(msgs2):
            return False
        
        for m1, m2 in zip(msgs1, msgs2):
            if m1.get("role") != m2.get("role"):
                return False
            if m1.get("content") != m2.get("content"):
                return False
        
        return True
    
    async def _evict_entry(self):
        """Evict an entry based on eviction policy."""
        if not self.cache:
            return
        
        if self.eviction_policy == CacheEvictionPolicy.LRU:
            # Evict least recently used
            oldest_entry = min(
                self.cache.values(),
                key=lambda e: e.last_accessed
            )
            del self.cache[oldest_entry.cache_key]
            
            self.logger.debug(f"Evicted LRU entry {oldest_entry.cache_key[:8]}...")
        
        elif self.eviction_policy == CacheEvictionPolicy.LFU:
            # Evict least frequently used
            least_used = min(
                self.cache.values(),
                key=lambda e: e.access_count
            )
            del self.cache[least_used.cache_key]
            
            self.logger.debug(f"Evicted LFU entry {least_used.cache_key[:8]}...")
        
        elif self.eviction_policy == CacheEvictionPolicy.COST_BASED:
            # Evict lowest cost (keep expensive responses)
            cheapest = min(
                self.cache.values(),
                key=lambda e: e.estimated_cost
            )
            del self.cache[cheapest.cache_key]
            
            self.logger.debug(f"Evicted low-cost entry {cheapest.cache_key[:8]}...")
        
        elif self.eviction_policy == CacheEvictionPolicy.TTL:
            # Evict expired or oldest
            expired = [e for e in self.cache.values() if e.is_expired()]
            
            if expired:
                for entry in expired:
                    del self.cache[entry.cache_key]
                
                self.logger.debug(f"Evicted {len(expired)} expired entries")
            else:
                # Evict oldest if no expired
                oldest = min(
                    self.cache.values(),
                    key=lambda e: e.created_at
                )
                del self.cache[oldest.cache_key]
                
                self.logger.debug(f"Evicted oldest entry {oldest.cache_key[:8]}...")
    
    async def clear(self):
        """Clear all cache entries."""
        async with self.lock:
            self.cache.clear()
            self.logger.info("Cache cleared")
    
    async def clear_expired(self):
        """Clear expired entries."""
        async with self.lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            if expired_keys:
                self.logger.info(f"Cleared {len(expired_keys)} expired entries")
    
    async def start_cleanup_task(self, interval: int = 300):
        """
        Start background task to periodically clean expired entries.
        
        Args:
            interval: Cleanup interval in seconds (default: 5 minutes)
        """
        if self._running:
            return
        
        self._running = True
        
        async def cleanup_loop():
            while self._running:
                try:
                    await asyncio.sleep(interval)
                    await self.clear_expired()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in cache cleanup loop: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
        self.logger.info(f"Started cache cleanup task (interval: {interval}s)")
    
    async def stop_cleanup_task(self):
        """Stop background cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped cache cleanup task")
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        
        return self.cache_hits / self.total_requests
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.get_hit_rate()
        
        # Calculate average access count
        if self.cache:
            avg_access = sum(e.access_count for e in self.cache.values()) / len(self.cache)
            total_entries_savings = sum(e.total_savings for e in self.cache.values())
        else:
            avg_access = 0
            total_entries_savings = 0
        
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "utilization": len(self.cache) / max(self.max_size, 1),
            "total_tokens_saved": self.total_cost_saved,  # Actually tokens, not cost
            "avg_access_count": avg_access,
            "strategy": self.strategy.value,
            "eviction_policy": self.eviction_policy.value,
        }
    
    async def get_top_entries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top cache entries by access count."""
        async with self.lock:
            sorted_entries = sorted(
                self.cache.values(),
                key=lambda e: e.access_count,
                reverse=True
            )
            
            return [
                {
                    "cache_key": e.cache_key[:8] + "...",
                    "model": e.model,
                    "access_count": e.access_count,
                    "total_tokens_saved": e.total_tokens_saved,
                    "age_seconds": (datetime.now(timezone.utc) - e.created_at).total_seconds(),
                }
                for e in sorted_entries[:limit]
            ]


class CacheWarmer:
    """
    Proactively warm cache with common queries.
    
    Analyzes query patterns and pre-caches likely requests.
    """
    
    def __init__(self, cache: ResponseCache):
        """
        Initialize cache warmer.
        
        Args:
            cache: Response cache to warm
        """
        self.cache = cache
        self.logger = StructuredLogger("cache_warmer", LogLevel.INFO)
    
    async def warm_cache(
        self,
        common_queries: List[Tuple[List[Dict[str, str]], str]],
        llm_caller: Any
    ):
        """
        Warm cache with common queries.
        
        Args:
            common_queries: List of (messages, model) tuples
            llm_caller: Function to call LLM
        """
        self.logger.info(f"Warming cache with {len(common_queries)} queries")
        
        for messages, model in common_queries:
            # Check if already cached
            cached = await self.cache.get(messages, model)
            
            if cached is None:
                # Not cached, fetch and cache
                try:
                    response = await llm_caller(messages, model)
                    await self.cache.set(messages, model, response)
                except Exception as e:
                    self.logger.error(f"Cache warming error: {e}")
        
        self.logger.info("Cache warming complete")


# Global cache
_global_cache: Optional[ResponseCache] = None


def get_response_cache() -> ResponseCache:
    """Get global response cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ResponseCache()
    return _global_cache


def create_cache(
    max_size: int = 1000,
    strategy: CacheStrategy = CacheStrategy.EXACT,
    eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU,
    default_ttl: Optional[int] = None
) -> ResponseCache:
    """
    Create a configured response cache.
    
    Args:
        max_size: Maximum cache entries
        strategy: Caching strategy
        eviction_policy: Eviction policy
        default_ttl: Default TTL in seconds
    
    Returns:
        Configured response cache
    """
    return ResponseCache(
        max_size=max_size,
        strategy=strategy,
        eviction_policy=eviction_policy,
        default_ttl=default_ttl
    )

