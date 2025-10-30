"""
Episodic Memory - Conversation History and Interaction Storage.

Episodic memory stores individual episodes (interactions, conversations, events)
in chronological order. It enables agents to remember past interactions and
learn from them.
"""

import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from teleon.cortex.storage import StorageBackend


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
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: Optional[str] = Field(None, description="Session this episode belongs to")
    
    # Relationships
    parent_episode_id: Optional[str] = Field(None, description="ID of parent episode if part of conversation")
    conversation_id: Optional[str] = Field(None, description="ID of conversation thread")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EpisodicMemory:
    """
    Episodic memory stores and retrieves past interactions.
    
    Features:
    - Store complete interaction episodes
    - Retrieve by time range, agent, session
    - Conversation threading
    - Pattern analysis
    - Search by content
    
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
        
        # Get recent episodes
        recent = await episodic.get_recent(limit=10)
        
        # Search episodes
        results = await episodic.search("AI", limit=5)
        
        # Get conversation thread
        thread = await episodic.get_conversation(conversation_id)
        ```
    """
    
    def __init__(
        self,
        storage: StorageBackend,
        agent_id: str,
        ttl: Optional[int] = None
    ):
        """
        Initialize episodic memory.
        
        Args:
            storage: Storage backend to use
            agent_id: ID of the agent using this memory
            ttl: Time-to-live for episodes in seconds (None = no expiration)
        """
        self.storage = storage
        self.agent_id = agent_id
        self.ttl = ttl
        self._key_prefix = f"episodic:{agent_id}"
    
    async def initialize(self) -> None:
        """Initialize episodic memory."""
        if not self.storage._initialized:
            await self.storage.initialize()
    
    async def store(self, episode: Episode) -> str:
        """
        Store an episode.
        
        Args:
            episode: Episode to store
        
        Returns:
            Episode ID
        """
        # Ensure episode belongs to this agent
        if episode.agent_id != self.agent_id:
            episode.agent_id = self.agent_id
        
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
            }
        )
        
        # Index by timestamp for efficient retrieval
        timestamp_key = f"{self._key_prefix}:timeline:{episode.timestamp.isoformat()}:{episode.id}"
        await self.storage.set(timestamp_key, episode.id, ttl=self.ttl)
        
        # Index by session if provided
        if episode.session_id:
            session_key = f"{self._key_prefix}:session:{episode.session_id}:{episode.id}"
            await self.storage.set(session_key, episode.id, ttl=self.ttl)
        
        # Index by conversation if provided
        if episode.conversation_id:
            conv_key = f"{self._key_prefix}:conversation:{episode.conversation_id}:{episode.id}"
            await self.storage.set(conv_key, episode.id, ttl=self.ttl)
        
        return episode.id
    
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
    
    async def get_recent(
        self,
        limit: int = 10,
        session_id: Optional[str] = None
    ) -> List[Episode]:
        """
        Get recent episodes.
        
        Args:
            limit: Maximum number of episodes to return
            session_id: Filter by session ID
        
        Returns:
            List of episodes in reverse chronological order
        """
        # Determine which index to use
        if session_id:
            pattern = f"{self._key_prefix}:session:{session_id}:*"
        else:
            pattern = f"{self._key_prefix}:timeline:*"
        
        # Get episode IDs from index
        index_keys = await self.storage.list_keys(pattern)
        
        # Sort by timestamp (most recent first)
        index_keys.sort(reverse=True)
        
        # Limit results
        index_keys = index_keys[:limit]
        
        # Get episode IDs
        episode_ids = []
        for key in index_keys:
            ep_id = await self.storage.get(key)
            if ep_id:
                episode_ids.append(ep_id)
        
        # Fetch episodes
        episodes = []
        for ep_id in episode_ids:
            episode = await self.get(ep_id)
            if episode:
                episodes.append(episode)
        
        return episodes
    
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
        
        # Filter by time range
        filtered_keys = []
        for key in index_keys:
            # Extract timestamp from key
            parts = key.split(":")
            if len(parts) >= 4:
                timestamp_str = parts[3]
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if start <= timestamp <= end:
                        filtered_keys.append(key)
                except:
                    continue
        
        # Sort by timestamp
        filtered_keys.sort()
        
        if limit:
            filtered_keys = filtered_keys[:limit]
        
        # Fetch episodes
        episodes = []
        for key in filtered_keys:
            ep_id = await self.storage.get(key)
            if ep_id:
                episode = await self.get(ep_id)
                if episode:
                    episodes.append(episode)
        
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
        
        # Sort by timestamp (chronological order)
        index_keys.sort()
        
        # Fetch episodes
        episodes = []
        for key in index_keys:
            ep_id = await self.storage.get(key)
            if ep_id:
                episode = await self.get(ep_id)
                if episode:
                    episodes.append(episode)
        
        return episodes
    
    async def search(
        self,
        query: str,
        limit: int = 10
    ) -> List[Episode]:
        """
        Search episodes by content.
        
        Simple text search in input and output fields.
        For advanced semantic search, use SemanticMemory.
        
        Args:
            query: Search query
            limit: Maximum number of results
        
        Returns:
            List of matching episodes
        """
        # Get all episodes
        pattern = f"{self._key_prefix}:episode:*"
        episode_keys = await self.storage.list_keys(pattern)
        
        # Search through episodes
        matching_episodes = []
        query_lower = query.lower()
        
        for key in episode_keys:
            episode_data = await self.storage.get(key)
            if episode_data:
                # Search in input and output
                input_str = str(episode_data.get("input", "")).lower()
                output_str = str(episode_data.get("output", "")).lower()
                
                if query_lower in input_str or query_lower in output_str:
                    episode = Episode(**episode_data)
                    matching_episodes.append(episode)
                    
                    if len(matching_episodes) >= limit:
                        break
        
        # Sort by relevance (most recent first for now)
        matching_episodes.sort(key=lambda x: x.timestamp, reverse=True)
        
        return matching_episodes[:limit]
    
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
        
        # Delete from timeline index
        timeline_key = f"{self._key_prefix}:timeline:{episode.timestamp.isoformat()}:{episode_id}"
        await self.storage.delete(timeline_key)
        
        # Delete from session index
        if episode.session_id:
            session_key = f"{self._key_prefix}:session:{episode.session_id}:{episode_id}"
            await self.storage.delete(session_key)
        
        # Delete from conversation index
        if episode.conversation_id:
            conv_key = f"{self._key_prefix}:conversation:{episode.conversation_id}:{episode_id}"
            await self.storage.delete(conv_key)
        
        return True
    
    async def clear(self) -> int:
        """
        Clear all episodes for this agent.
        
        Returns:
            Number of episodes deleted
        """
        pattern = f"{self._key_prefix}:*"
        return await self.storage.clear(pattern)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored episodes.
        
        Returns:
            Dictionary with statistics
        """
        # Count episodes
        episode_pattern = f"{self._key_prefix}:episode:*"
        episode_keys = await self.storage.list_keys(episode_pattern)
        total_episodes = len(episode_keys)
        
        # Get sample of episodes for metrics
        sample_size = min(100, total_episodes)
        if sample_size > 0:
            sample_keys = episode_keys[:sample_size]
            episodes = []
            for key in sample_keys:
                data = await self.storage.get(key)
                if data:
                    episodes.append(Episode(**data))
            
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
            
            return {
                "total_episodes": total_episodes,
                "avg_cost": round(avg_cost, 6),
                "avg_duration_ms": round(avg_duration, 2),
                "success_rate": round(success_rate, 2),
                "unique_sessions": len(sessions),
                "unique_conversations": len(conversations),
            }
        
        return {
            "total_episodes": 0,
            "avg_cost": 0,
            "avg_duration_ms": 0,
            "success_rate": 0,
            "unique_sessions": 0,
            "unique_conversations": 0,
        }

