"""
Episodic Memory - Conversation History and Interaction Storage.

Episodic memory stores individual episodes (interactions, conversations, events)
in chronological order. It enables agents to remember past interactions and
learn from them.

Optimized with:
- Batch operations for efficiency
- Proper async handling
- Embedding-based similarity search (optional)
"""

import uuid
import asyncio
import time
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field, ConfigDict, field_serializer
import logging

from teleon.cortex.storage.base import StorageBackend
from teleon.cortex.utils import AsyncLRUCache, validate_episode, validate_limit

logger = logging.getLogger("teleon.cortex.episodic")


class Episode(BaseModel):
    """
    Represents a single episode/interaction.
    
    An episode captures a complete interaction including input, output,
    context, and metrics.
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = Field(..., description="ID of the agent that handled this episode")
    
    # Interaction data
    input: Dict[str, Any] = Field(..., description="Input data for the episode")
    output: Dict[str, Any] = Field(..., description="Output/response data")
    
    # Context and metadata
    context: Dict[str, Any] = Field(default_factory=dict, description="Contextual information")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Metrics
    duration_ms: Optional[int] = Field(None, description="Execution duration in milliseconds")
    cost: Optional[float] = Field(None, description="Cost of the interaction")
    success: bool = Field(True, description="Whether the interaction was successful")
    
    # Temporal information
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: Optional[str] = Field(None, description="Session this episode belongs to")

    # Relationships
    parent_episode_id: Optional[str] = Field(None, description="ID of parent episode if part of conversation")
    conversation_id: Optional[str] = Field(None, description="ID of conversation thread")

    # Optional embedding for similarity search
    embedding: Optional[List[float]] = Field(None, description="Embedding of the episode content")

    # Importance for memory consolidation
    importance_score: float = Field(0.5, description="Importance score (0-1)")

    model_config = ConfigDict()

    @field_serializer('timestamp')
    def serialize_datetime(self, value: datetime) -> str:
        return value.isoformat() if value else None
    
    def get_content_for_embedding(self) -> str:
        """Get text content suitable for embedding."""
        parts = []
        
        if isinstance(self.input, dict):
            parts.append(str(self.input.get("query", self.input)))
        else:
            parts.append(str(self.input))
        
        if isinstance(self.output, dict):
            parts.append(str(self.output.get("response", self.output)))
        else:
            parts.append(str(self.output))
        
        return " ".join(parts)


class EpisodicMemory:
    """
    Episodic memory stores and retrieves past interactions.
    
    Features:
    - Store complete interaction episodes
    - Retrieve by time range, agent, session
    - Conversation threading
    - Batch operations for efficiency
    - Optional embedding-based similarity search
    - Pattern analysis
    
    Example:
        ```python
        episodic = EpisodicMemory(storage, agent_id="agent-123")
        await episodic.initialize()
        
        # Store an episode
        episode = Episode(
            agent_id="agent-123",
            input={"query": "What is AI?"},
            output={"response": "AI is..."},
            duration_ms=150,
            cost=0.002
        )
        await episodic.store(episode)
        
        # Get recent episodes (batch operation)
        recent = await episodic.get_recent(limit=10)
        
        # Search episodes with embedding similarity
        results = await episodic.search_similar("AI topics", limit=5)
        
        # Get conversation thread
        thread = await episodic.get_conversation(conversation_id)
        ```
    """
    
    def __init__(
        self,
        storage: StorageBackend,
        agent_id: str,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        ttl: Optional[int] = None,
        max_limit: int = 10000
    ):
        """
        Initialize episodic memory.
        
        Args:
            storage: Storage backend to use
            agent_id: ID of the agent using this memory
            embedding_function: Optional function for embedding-based search
            ttl: Time-to-live for episodes in seconds (None = no expiration)
            max_limit: Maximum limit for queries (default: 10000)
        """
        self.storage = storage
        self.agent_id = agent_id
        self.embedding_function = embedding_function
        self.ttl = ttl
        self.max_limit = max_limit
        self._key_prefix = f"episodic:{agent_id}"
        
        # Async-safe cache for embeddings
        self._embedding_cache = AsyncLRUCache(max_size=1000, default_ttl=3600)
        
        # Statistics cache (60 second TTL)
        self._stats_cache: Optional[tuple[float, Dict[str, Any]]] = None
    
    async def initialize(self) -> None:
        """Initialize episodic memory."""
        if not self.storage._initialized:
            await self.storage.initialize()
        
        logger.info(f"EpisodicMemory initialized for agent: {self.agent_id}")
    
    async def store(self, episode: Episode) -> str:
        """
        Store an episode with validation.
        
        Args:
            episode: Episode to store
        
        Returns:
            Episode ID
            
        Raises:
            ValueError: If episode is invalid
        """
        # Validate episode
        validate_episode(episode)
        
        # Ensure episode belongs to this agent
        if episode.agent_id != self.agent_id:
            logger.warning(
                f"Episode agent_id ({episode.agent_id}) doesn't match "
                f"memory agent_id ({self.agent_id}), updating..."
            )
            episode.agent_id = self.agent_id
        
        # Generate embedding if function is available
        if self.embedding_function and not episode.embedding:
            try:
                content = episode.get_content_for_embedding()
                if not content or len(content.strip()) == 0:
                    logger.debug("Empty content, skipping embedding")
                else:
                    # Generate embedding with timeout
                    embedding = await asyncio.wait_for(
                        asyncio.to_thread(self.embedding_function, content),
                        timeout=5.0  # 5 second timeout
                    )
                    episode.embedding = embedding
                    await self._embedding_cache.set(episode.id, embedding, ttl=3600)
            except asyncio.TimeoutError:
                logger.warning(f"Embedding generation timed out for episode {episode.id}")
                # Continue without embedding - non-blocking
            except Exception as e:
                logger.warning(
                    f"Failed to generate embedding for episode {episode.id}: {e}",
                    exc_info=True
                )
                # Continue without embedding - non-blocking
        
        # Store episode
        key = f"{self._key_prefix}:episode:{episode.id}"
        await self.storage.set(
            key,
            episode.dict(),
            ttl=self.ttl,
            metadata={
                "type": "episode",
                "agent_id": self.agent_id,
                "timestamp": episode.timestamp.isoformat(),
                "session_id": episode.session_id,
                "conversation_id": episode.conversation_id,
                "success": episode.success,
            }
        )
        
        # Index by timestamp for efficient retrieval
        timestamp_key = f"{self._key_prefix}:timeline:{episode.timestamp.isoformat()}:{episode.id}"
        await self.storage.set(timestamp_key, episode.id, ttl=self.ttl)
        
        # Index by session if provided
        if episode.session_id:
            session_key = f"{self._key_prefix}:session:{episode.session_id}:{episode.timestamp.isoformat()}:{episode.id}"
            await self.storage.set(session_key, episode.id, ttl=self.ttl)
        
        # Index by conversation if provided
        if episode.conversation_id:
            conv_key = f"{self._key_prefix}:conversation:{episode.conversation_id}:{episode.timestamp.isoformat()}:{episode.id}"
            await self.storage.set(conv_key, episode.id, ttl=self.ttl)
        
        logger.debug(f"Stored episode: {episode.id}")
        return episode.id
    
    async def store_batch(self, episodes: List[Episode]) -> List[str]:
        """
        Store multiple episodes in batch.
        
        Args:
            episodes: List of episodes to store
        
        Returns:
            List of episode IDs
        """
        if not episodes:
            return []
        
        # Prepare batch data
        items = {}
        index_items = {}
        
        for episode in episodes:
            if episode.agent_id != self.agent_id:
                episode.agent_id = self.agent_id
            
            # Validate episode
            try:
                validate_episode(episode)
            except ValueError as e:
                logger.warning(f"Skipping invalid episode in batch: {e}")
                continue
            
            # Generate embedding if needed
            if self.embedding_function and not episode.embedding:
                try:
                    content = episode.get_content_for_embedding()
                    if content and len(content.strip()) > 0:
                        embedding = await asyncio.wait_for(
                            asyncio.to_thread(self.embedding_function, content),
                            timeout=5.0
                        )
                        episode.embedding = embedding
                        await self._embedding_cache.set(episode.id, embedding, ttl=3600)
                except (asyncio.TimeoutError, Exception) as e:
                    logger.debug(f"Failed to generate embedding for episode {episode.id}: {e}")
                    # Continue without embedding
            
            # Main episode
            key = f"{self._key_prefix}:episode:{episode.id}"
            items[key] = episode.dict()
            
            # Timeline index
            timeline_key = f"{self._key_prefix}:timeline:{episode.timestamp.isoformat()}:{episode.id}"
            index_items[timeline_key] = episode.id
            
            # Session index
            if episode.session_id:
                session_key = f"{self._key_prefix}:session:{episode.session_id}:{episode.timestamp.isoformat()}:{episode.id}"
                index_items[session_key] = episode.id
            
            # Conversation index
            if episode.conversation_id:
                conv_key = f"{self._key_prefix}:conversation:{episode.conversation_id}:{episode.timestamp.isoformat()}:{episode.id}"
                index_items[conv_key] = episode.id
        
        # Store in batch
        await self.storage.set_many(items, ttl=self.ttl)
        await self.storage.set_many(index_items, ttl=self.ttl)
        
        logger.debug(f"Stored {len(episodes)} episodes in batch")
        return [ep.id for ep in episodes]
    
    async def get(self, episode_id: str) -> Optional[Episode]:
        """
        Get a specific episode by ID.
        
        Args:
            episode_id: Episode ID
        
        Returns:
            Episode if found, None otherwise
        """
        key = f"{self._key_prefix}:episode:{episode_id}"
        data = await self.storage.get(key)
        
        if data is None:
            return None
        
        return Episode(**data)
    
    async def get_batch(self, episode_ids: List[str]) -> List[Episode]:
        """
        Get multiple episodes by ID in batch.
        
        Args:
            episode_ids: List of episode IDs
        
        Returns:
            List of episodes (missing IDs omitted)
        """
        if not episode_ids:
            return []
        
        # Build keys
        keys = [f"{self._key_prefix}:episode:{eid}" for eid in episode_ids]
        
        # Batch fetch
        data_map = await self.storage.get_many(keys)
        
        episodes = []
        for key, data in data_map.items():
            if data:
                episodes.append(Episode(**data))
        
        return episodes
    
    async def get_recent(
        self,
        limit: int = 10,
        session_id: Optional[str] = None,
        include_failed: bool = True
    ) -> List[Episode]:
        """
        Get recent episodes with bounds checking.
        
        Args:
            limit: Maximum number of episodes to return (validated and capped)
            session_id: Filter by session ID
            include_failed: Include failed episodes
        
        Returns:
            List of episodes in reverse chronological order
            
        Raises:
            ValueError: If limit is invalid
        """
        # Validate and cap limit
        limit = validate_limit(limit, max_limit=self.max_limit)
        # Determine which index to use
        if session_id:
            pattern = f"{self._key_prefix}:session:{session_id}:*"
        else:
            pattern = f"{self._key_prefix}:timeline:*"
        
        # Get episode IDs from index
        index_keys = await self.storage.list_keys(pattern)
        
        # Sort by timestamp (most recent first)
        # Key format: prefix:timeline:ISO_TIMESTAMP:EPISODE_ID
        index_keys.sort(reverse=True)
        
        # Extract episode IDs and fetch in batch
        episode_ids = []
        for key in index_keys[:limit * 2]:  # Fetch extra in case of filtering
            # Extract episode ID from key
            parts = key.split(":")
            if parts:
                episode_ids.append(parts[-1])
        
        # Batch fetch episodes
        episodes = await self.get_batch(episode_ids)
        
        # Sort by timestamp and apply filters
        episodes.sort(key=lambda e: e.timestamp, reverse=True)
        
        if not include_failed:
            episodes = [e for e in episodes if e.success]
        
        return episodes[:limit]
    
    async def get_by_time_range(
        self,
        start: datetime,
        end: datetime,
        limit: Optional[int] = None
    ) -> List[Episode]:
        """
        Get episodes within a time range.
        
        Args:
            start: Start datetime
            end: End datetime
            limit: Maximum number of episodes
        
        Returns:
            List of episodes in chronological order
        """
        pattern = f"{self._key_prefix}:timeline:*"
        index_keys = await self.storage.list_keys(pattern)
        
        # Filter by time range and collect episode IDs
        episode_ids = []
        for key in index_keys:
            # Extract timestamp from key
            parts = key.split(":")
            if len(parts) >= 4:
                timestamp_str = parts[3]
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if start <= timestamp <= end:
                        episode_ids.append(parts[-1])
                except (ValueError, IndexError):
                    continue
        
        if limit:
            episode_ids = episode_ids[:limit]
        
        # Batch fetch episodes
        episodes = await self.get_batch(episode_ids)
        
        # Sort chronologically
        episodes.sort(key=lambda e: e.timestamp)
        
        return episodes
    
    async def get_conversation(
        self,
        conversation_id: str
    ) -> List[Episode]:
        """
        Get all episodes in a conversation thread.
        
        Args:
            conversation_id: Conversation ID
        
        Returns:
            List of episodes in chronological order
        """
        pattern = f"{self._key_prefix}:conversation:{conversation_id}:*"
        index_keys = await self.storage.list_keys(pattern)
        
        # Extract episode IDs
        episode_ids = [key.split(":")[-1] for key in index_keys]
        
        # Batch fetch episodes
        episodes = await self.get_batch(episode_ids)
        
        # Sort chronologically
        episodes.sort(key=lambda e: e.timestamp)
        
        return episodes
    
    async def search(
        self,
        query: str,
        limit: int = 10
    ) -> List[Episode]:
        """
        Search episodes by content (text matching).
        
        For advanced semantic search, use search_similar with embeddings.
        
        Args:
            query: Search query
            limit: Maximum number of results
        
        Returns:
            List of matching episodes
        """
        # Get recent episodes
        all_episodes = await self.get_recent(limit=limit * 10)
        
        # Search through episodes
        matching_episodes = []
        query_lower = query.lower()
        
        for episode in all_episodes:
            input_str = str(episode.input).lower()
            output_str = str(episode.output).lower()
            
            if query_lower in input_str or query_lower in output_str:
                matching_episodes.append(episode)
                
                if len(matching_episodes) >= limit:
                    break
        
        return matching_episodes
    
    async def search_similar(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.5
    ) -> List[tuple]:
        """
        Search episodes using embedding similarity.
        
        Requires embedding_function to be set.
        
        Args:
            query: Search query
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of (episode, similarity_score) tuples
        """
        if not self.embedding_function:
            logger.warning("No embedding function set, falling back to text search")
            episodes = await self.search(query, limit)
            return [(ep, 0.5) for ep in episodes]
        
        # Generate query embedding
        query_embedding = self.embedding_function(query)
        
        # Get recent episodes with embeddings
        all_episodes = await self.get_recent(limit=limit * 5)
        
        # Calculate similarities
        results = []
        for episode in all_episodes:
            if episode.embedding:
                similarity = self._cosine_similarity(query_embedding, episode.embedding)
                if similarity >= min_similarity:
                    results.append((episode, similarity))
            else:
                # Try to get from async cache
                cached_embedding = await self._embedding_cache.get(episode.id)
                if cached_embedding:
                    similarity = self._cosine_similarity(query_embedding, cached_embedding)
                    if similarity >= min_similarity:
                        results.append((episode, similarity))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:limit]
    
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
    
    async def delete(self, episode_id: str) -> bool:
        """
        Delete an episode.
        
        Args:
            episode_id: Episode ID
        
        Returns:
            True if deleted
        """
        # Get episode first to clean up indexes
        episode = await self.get(episode_id)
        if not episode:
            return False
        
        # Delete main episode
        key = f"{self._key_prefix}:episode:{episode_id}"
        await self.storage.delete(key)
        
        # Delete from embedding cache
        await self._embedding_cache.delete(episode_id)
        
        # Delete from timeline index
        timeline_key = f"{self._key_prefix}:timeline:{episode.timestamp.isoformat()}:{episode_id}"
        await self.storage.delete(timeline_key)
        
        # Delete from session index
        if episode.session_id:
            session_key = f"{self._key_prefix}:session:{episode.session_id}:{episode.timestamp.isoformat()}:{episode_id}"
            await self.storage.delete(session_key)
        
        # Delete from conversation index
        if episode.conversation_id:
            conv_key = f"{self._key_prefix}:conversation:{episode.conversation_id}:{episode.timestamp.isoformat()}:{episode_id}"
            await self.storage.delete(conv_key)
        
        logger.debug(f"Deleted episode: {episode_id}")
        return True
    
    async def delete_batch(self, episode_ids: List[str]) -> int:
        """
        Delete multiple episodes.
        
        Args:
            episode_ids: List of episode IDs
        
        Returns:
            Number of episodes deleted
        """
        if not episode_ids:
            return 0
        
        # Get all episodes first for index cleanup
        episodes = await self.get_batch(episode_ids)
        
        # Build list of all keys to delete
        keys_to_delete = []
        
        for episode in episodes:
            # Main episode key
            keys_to_delete.append(f"{self._key_prefix}:episode:{episode.id}")
            
            # Timeline index
            keys_to_delete.append(
                f"{self._key_prefix}:timeline:{episode.timestamp.isoformat()}:{episode.id}"
            )
            
            # Session index
            if episode.session_id:
                keys_to_delete.append(
                    f"{self._key_prefix}:session:{episode.session_id}:{episode.timestamp.isoformat()}:{episode.id}"
                )
            
            # Conversation index
            if episode.conversation_id:
                keys_to_delete.append(
                    f"{self._key_prefix}:conversation:{episode.conversation_id}:{episode.timestamp.isoformat()}:{episode.id}"
                )
            
            # Clear from embedding cache
            await self._embedding_cache.delete(episode.id)
        
        # Batch delete
        count = await self.storage.delete_many(keys_to_delete)
        
        logger.debug(f"Deleted {len(episodes)} episodes")
        return len(episodes)
    
    async def clear(self) -> int:
        """
        Clear all episodes for this agent.
        
        Returns:
            Number of episodes deleted
        """
        pattern = f"{self._key_prefix}:*"
        count = await self.storage.clear(pattern)
        await self._embedding_cache.clear()
        
        # Clear stats cache
        self._stats_cache = None
        
        logger.info(f"Cleared all episodes for agent: {self.agent_id}")
        return count
    
    async def get_statistics(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get statistics about stored episodes with caching.
        
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
        
        # Count episodes (limit key scanning for performance)
        episode_pattern = f"{self._key_prefix}:episode:*"
        episode_keys = await self.storage.list_keys(episode_pattern, limit=10000)
        total_episodes = len(episode_keys)
        
        if total_episodes == 0:
            return {
                "total_episodes": 0,
                "avg_cost": 0,
                "avg_duration_ms": 0,
                "success_rate": 0,
                "unique_sessions": 0,
                "unique_conversations": 0,
                "embedding_enabled": self.embedding_function is not None,
            }
        
        # Get sample of episodes for metrics (batch fetch)
        sample_size = min(100, total_episodes)
        sample_ids = [key.split(":")[-1] for key in episode_keys[:sample_size]]
        episodes = await self.get_batch(sample_ids)
        
        if not episodes:
            return {
                "total_episodes": total_episodes,
                "avg_cost": 0,
                "avg_duration_ms": 0,
                "success_rate": 0,
                "unique_sessions": 0,
                "unique_conversations": 0,
                "embedding_enabled": self.embedding_function is not None,
            }
        
        # Calculate metrics
        total_cost = sum(e.cost for e in episodes if e.cost)
        avg_cost = total_cost / len(episodes) if episodes else 0
        
        total_duration = sum(e.duration_ms for e in episodes if e.duration_ms)
        avg_duration = total_duration / len(episodes) if episodes else 0
        
        success_count = sum(1 for e in episodes if e.success)
        success_rate = (success_count / len(episodes) * 100) if episodes else 0
        
        # Get unique sessions and conversations
        sessions = set(e.session_id for e in episodes if e.session_id)
        conversations = set(e.conversation_id for e in episodes if e.conversation_id)
        
        stats = {
            "total_episodes": total_episodes,
            "avg_cost": round(avg_cost, 6),
            "avg_duration_ms": round(avg_duration, 2),
            "success_rate": round(success_rate, 2),
            "unique_sessions": len(sessions),
            "unique_conversations": len(conversations),
            "embedding_enabled": self.embedding_function is not None,
            "cached_embeddings": await self._embedding_cache.size(),
        }
        
        # Cache results
        self._stats_cache = (time.time(), stats)
        
        return stats
