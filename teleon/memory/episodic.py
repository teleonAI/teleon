"""Episodic memory - stores and retrieves events/episodes."""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field
import json


class Episode(BaseModel):
    """A single episode/event."""
    
    id: str = Field(..., description="Episode ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When it happened")
    event_type: str = Field(..., description="Type of event")
    content: str = Field(..., description="Event content/description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tags: List[str] = Field(default_factory=list, description="Tags for retrieval")
    importance: float = Field(0.5, description="Importance score (0-1)")
    
    # Context
    session_id: Optional[str] = Field(None, description="Session ID")
    user_id: Optional[str] = Field(None, description="User ID")
    
    # Outcomes
    success: Optional[bool] = Field(None, description="Whether event was successful")
    outcome: Optional[str] = Field(None, description="Event outcome")


class EpisodicMemory:
    """
    Episodic memory system.
    
    Features:
    - Store events/episodes
    - Retrieve by time, type, tags
    - Search by content
    - Analyze patterns
    - Track success/failure
    """
    
    def __init__(
        self,
        max_episodes: int = 10000,
        retention_days: Optional[int] = 30
    ):
        """
        Initialize episodic memory.
        
        Args:
            max_episodes: Maximum episodes to store
            retention_days: Days to retain episodes (None = forever)
        """
        self.episodes: List[Episode] = []
        self.max_episodes = max_episodes
        self.retention_days = retention_days
        
        # Indices for fast lookup
        self._by_type: Dict[str, List[str]] = {}
        self._by_tag: Dict[str, List[str]] = {}
        self._by_session: Dict[str, List[str]] = {}
    
    async def store(
        self,
        event_type: str,
        content: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        success: Optional[bool] = None,
        outcome: Optional[str] = None
    ) -> Episode:
        """
        Store an episode.
        
        Args:
            event_type: Type of event
            content: Event content
            session_id: Session ID
            user_id: User ID
            tags: Tags for retrieval
            metadata: Additional metadata
            importance: Importance score (0-1)
            success: Whether successful
            outcome: Event outcome
        
        Returns:
            Created episode
        """
        import uuid
        
        episode = Episode(
            id=str(uuid.uuid4()),
            event_type=event_type,
            content=content,
            session_id=session_id,
            user_id=user_id,
            tags=tags or [],
            metadata=metadata or {},
            importance=importance,
            success=success,
            outcome=outcome
        )
        
        # Add to storage
        self.episodes.append(episode)
        
        # Update indices
        if event_type not in self._by_type:
            self._by_type[event_type] = []
        self._by_type[event_type].append(episode.id)
        
        for tag in episode.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = []
            self._by_tag[tag].append(episode.id)
        
        if session_id:
            if session_id not in self._by_session:
                self._by_session[session_id] = []
            self._by_session[session_id].append(episode.id)
        
        # Cleanup if needed
        await self._cleanup()
        
        return episode
    
    async def retrieve_recent(
        self,
        limit: int = 10,
        event_type: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[Episode]:
        """
        Retrieve recent episodes.
        
        Args:
            limit: Maximum number to retrieve
            event_type: Filter by event type
            session_id: Filter by session
        
        Returns:
            List of episodes
        """
        # Filter episodes
        filtered = self.episodes
        
        if event_type:
            episode_ids = set(self._by_type.get(event_type, []))
            filtered = [e for e in filtered if e.id in episode_ids]
        
        if session_id:
            episode_ids = set(self._by_session.get(session_id, []))
            filtered = [e for e in filtered if e.id in episode_ids]
        
        # Sort by timestamp (most recent first)
        sorted_episodes = sorted(
            filtered,
            key=lambda e: e.timestamp,
            reverse=True
        )
        
        return sorted_episodes[:limit]
    
    async def retrieve_by_time(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[str] = None
    ) -> List[Episode]:
        """
        Retrieve episodes within time range.
        
        Args:
            start_time: Start of range
            end_time: End of range
            event_type: Filter by type
        
        Returns:
            List of episodes
        """
        filtered = self.episodes
        
        if event_type:
            episode_ids = set(self._by_type.get(event_type, []))
            filtered = [e for e in filtered if e.id in episode_ids]
        
        if start_time:
            filtered = [e for e in filtered if e.timestamp >= start_time]
        
        if end_time:
            filtered = [e for e in filtered if e.timestamp <= end_time]
        
        return sorted(filtered, key=lambda e: e.timestamp, reverse=True)
    
    async def retrieve_by_tags(
        self,
        tags: List[str],
        match_all: bool = False,
        limit: int = 10
    ) -> List[Episode]:
        """
        Retrieve episodes by tags.
        
        Args:
            tags: Tags to search for
            match_all: If True, episode must have all tags
            limit: Maximum to retrieve
        
        Returns:
            List of episodes
        """
        if match_all:
            # Find episodes with all tags
            episode_sets = [
                set(self._by_tag.get(tag, []))
                for tag in tags
            ]
            if not episode_sets:
                return []
            
            episode_ids = episode_sets[0].intersection(*episode_sets[1:])
        else:
            # Find episodes with any tag
            episode_ids = set()
            for tag in tags:
                episode_ids.update(self._by_tag.get(tag, []))
        
        # Get episodes
        episodes = [e for e in self.episodes if e.id in episode_ids]
        
        # Sort by timestamp
        episodes = sorted(episodes, key=lambda e: e.timestamp, reverse=True)
        
        return episodes[:limit]
    
    async def search(
        self,
        query: str,
        limit: int = 10
    ) -> List[Episode]:
        """
        Search episodes by content.
        
        Args:
            query: Search query
            limit: Maximum to retrieve
        
        Returns:
            List of episodes
        """
        query_lower = query.lower()
        
        # Simple text search
        matching = [
            e for e in self.episodes
            if query_lower in e.content.lower() or
               query_lower in e.event_type.lower() or
               any(query_lower in tag.lower() for tag in e.tags)
        ]
        
        # Sort by timestamp
        matching = sorted(matching, key=lambda e: e.timestamp, reverse=True)
        
        return matching[:limit]
    
    async def analyze_patterns(
        self,
        event_type: Optional[str] = None,
        time_window_days: int = 7
    ) -> Dict[str, Any]:
        """
        Analyze episode patterns.
        
        Args:
            event_type: Analyze specific event type
            time_window_days: Time window in days
        
        Returns:
            Analysis results
        """
        # Get episodes in time window
        cutoff = datetime.now(timezone.utc) - timedelta(days=time_window_days)
        episodes = [e for e in self.episodes if e.timestamp >= cutoff]
        
        if event_type:
            episodes = [e for e in episodes if e.event_type == event_type]
        
        # Calculate stats
        total = len(episodes)
        successful = len([e for e in episodes if e.success is True])
        failed = len([e for e in episodes if e.success is False])
        
        # Success rate
        success_rate = successful / max(successful + failed, 1)
        
        # Most common tags
        tag_counts = {}
        for episode in episodes:
            for tag in episode.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Average importance
        avg_importance = sum(e.importance for e in episodes) / max(total, 1)
        
        return {
            "total_episodes": total,
            "successful": successful,
            "failed": failed,
            "success_rate": success_rate,
            "avg_importance": avg_importance,
            "top_tags": top_tags,
            "time_window_days": time_window_days
        }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_episodes": len(self.episodes),
            "event_types": len(self._by_type),
            "unique_tags": len(self._by_tag),
            "sessions": len(self._by_session),
            "max_episodes": self.max_episodes,
            "retention_days": self.retention_days
        }
    
    async def _cleanup(self):
        """Clean up old episodes."""
        # Remove oldest if over limit
        if len(self.episodes) > self.max_episodes:
            # Sort by importance and timestamp
            self.episodes.sort(
                key=lambda e: (e.importance, e.timestamp),
                reverse=True
            )
            self.episodes = self.episodes[:self.max_episodes]
        
        # Remove old episodes if retention policy set
        if self.retention_days:
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
            self.episodes = [e for e in self.episodes if e.timestamp >= cutoff]
        
        # Rebuild indices
        await self._rebuild_indices()
    
    async def _rebuild_indices(self):
        """Rebuild lookup indices."""
        self._by_type = {}
        self._by_tag = {}
        self._by_session = {}
        
        for episode in self.episodes:
            # By type
            if episode.event_type not in self._by_type:
                self._by_type[episode.event_type] = []
            self._by_type[episode.event_type].append(episode.id)
            
            # By tag
            for tag in episode.tags:
                if tag not in self._by_tag:
                    self._by_tag[tag] = []
                self._by_tag[tag].append(episode.id)
            
            # By session
            if episode.session_id:
                if episode.session_id not in self._by_session:
                    self._by_session[episode.session_id] = []
                self._by_session[episode.session_id].append(episode.id)
    
    async def clear(self):
        """Clear all episodes."""
        self.episodes = []
        self._by_type = {}
        self._by_tag = {}
        self._by_session = {}

