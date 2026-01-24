"""
Conversation Manager - Manage long conversations with automatic summarization.

Provides utilities for:
- Hierarchical conversation summarization
- Smart conversation compression
- Message prioritization
- Conversation threading
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging

from teleon.cortex.token_manager import TokenCounter
from teleon.cortex.memory.episodic import Episode

logger = logging.getLogger(__name__)


class MessagePrioritizer:
    """
    Prioritize which messages to keep in compressed conversations.
    
    Priorities:
    - Recent messages (last N)
    - Important messages (user-marked)
    - High-engagement messages
    - Turning points in conversation
    
    Example:
        ```python
        prioritizer = MessagePrioritizer()
        
        # Prioritize messages
        prioritized = prioritizer.prioritize(
            episodes=episodes,
            keep_count=5
        )
        ```
    """
    
    def __init__(
        self,
        recency_weight: float = 0.5,
        importance_weight: float = 0.3,
        engagement_weight: float = 0.2
    ):
        """
        Initialize prioritizer.
        
        Args:
            recency_weight: Weight for recency
            importance_weight: Weight for importance markers
            engagement_weight: Weight for engagement (length, complexity)
        """
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self.engagement_weight = engagement_weight
    
    def prioritize(
        self,
        episodes: List[Episode],
        keep_count: Optional[int] = None
    ) -> List[Episode]:
        """
        Prioritize episodes for retention.
        
        Args:
            episodes: List of episodes
            keep_count: Number of episodes to keep (None = rank all)
        
        Returns:
            Prioritized list of episodes
        """
        if not episodes:
            return []
        
        # Always keep most recent messages
        if len(episodes) <= 3:
            return episodes
        
        # Score each episode
        scored_episodes = []
        
        for i, episode in enumerate(episodes):
            score = self._calculate_priority_score(episode, i, len(episodes))
            scored_episodes.append((score, episode))
        
        # Sort by score descending
        scored_episodes.sort(key=lambda x: x[0], reverse=True)
        
        # Keep top K
        if keep_count:
            scored_episodes = scored_episodes[:keep_count]
        
        # Sort by timestamp to maintain chronological order
        result_episodes = [ep for _, ep in scored_episodes]
        result_episodes.sort(key=lambda x: x.timestamp)
        
        return result_episodes
    
    def _calculate_priority_score(
        self,
        episode: Episode,
        index: int,
        total: int
    ) -> float:
        """Calculate priority score for episode."""
        score = 0.0
        
        # Recency score (most recent = 1.0)
        recency_score = (total - index) / total
        score += recency_score * self.recency_weight
        
        # Importance score (from metadata or default)
        importance = episode.metadata.get("importance", 0.5)
        score += importance * self.importance_weight
        
        # Engagement score (based on content length)
        engagement_score = self._calculate_engagement(episode)
        score += engagement_score * self.engagement_weight
        
        return score
    
    def _calculate_engagement(self, episode: Episode) -> float:
        """
        Calculate engagement score based on content.
        
        Factors:
        - Response length
        - Question complexity
        - Back-and-forth indicators
        """
        # Simple: based on output length
        output_str = str(episode.output)
        
        # Normalize to 0-1 (assume 500 chars = high engagement)
        engagement = min(1.0, len(output_str) / 500)
        
        return engagement


class ConversationSummarizer:
    """
    Summarize conversations using hierarchical strategies.
    
    Strategies:
    - extractive: Extract key messages
    - abstractive: Generate summaries (requires LLM)
    - hierarchical: Multi-level summarization
    - progressive: Progressively summarize as conversation grows
    
    Example:
        ```python
        summarizer = ConversationSummarizer(token_counter)
        
        # Summarize conversation
        summary = await summarizer.summarize_conversation(
            episodes=episodes,
            target_tokens=500,
            method="hierarchical"
        )
        
        # Get compressed history
        compressed = await summarizer.get_compressed_history(
            conversation_id="conv-123",
            max_tokens=2000
        )
        ```
    """
    
    def __init__(self, token_counter: TokenCounter):
        """
        Initialize summarizer.
        
        Args:
            token_counter: Token counter instance
        """
        self.token_counter = token_counter
        self.prioritizer = MessagePrioritizer()
    
    async def summarize_conversation(
        self,
        episodes: List[Episode],
        target_tokens: int = 500,
        method: str = "hierarchical"
    ) -> str:
        """
        Summarize conversation to target token count.
        
        Args:
            episodes: List of episodes to summarize
            target_tokens: Target token count
            method: Summarization method
        
        Returns:
            Summary text
        """
        if not episodes:
            return ""
        
        if method == "extractive":
            return self._extractive_summary(episodes, target_tokens)
        elif method == "hierarchical":
            return await self._hierarchical_summary(episodes, target_tokens)
        elif method == "progressive":
            return self._progressive_summary(episodes, target_tokens)
        else:
            logger.warning(f"Unknown method {method}, using extractive")
            return self._extractive_summary(episodes, target_tokens)
    
    def _extractive_summary(
        self,
        episodes: List[Episode],
        target_tokens: int
    ) -> str:
        """
        Extract key episodes that fit within token budget.
        
        Strategy: Keep first, last, and most important middle episodes.
        """
        if not episodes:
            return ""
        
        # Always keep first and last
        if len(episodes) <= 2:
            return self._format_episodes(episodes)
        
        summary_parts = []
        used_tokens = 0
        
        # Add first episode
        first_text = self._format_episode(episodes[0])
        first_tokens = self.token_counter.count_tokens(first_text)
        summary_parts.append(first_text)
        used_tokens += first_tokens
        
        # Add last episode
        last_text = self._format_episode(episodes[-1])
        last_tokens = self.token_counter.count_tokens(last_text)
        summary_parts.append(last_text)
        used_tokens += last_tokens
        
        # Add middle episodes by priority
        remaining_budget = target_tokens - used_tokens
        middle_episodes = episodes[1:-1]
        
        if remaining_budget > 0 and middle_episodes:
            # Prioritize middle episodes
            prioritized = self.prioritizer.prioritize(middle_episodes)
            
            middle_parts = []
            for episode in prioritized:
                episode_text = self._format_episode(episode)
                episode_tokens = self.token_counter.count_tokens(episode_text)
                
                if used_tokens + episode_tokens <= target_tokens:
                    middle_parts.append(episode_text)
                    used_tokens += episode_tokens
                else:
                    break
            
            # Insert middle between first and last
            if middle_parts:
                summary_parts = [summary_parts[0]] + middle_parts + [summary_parts[1]]
        
        return "\n\n".join(summary_parts)
    
    async def _hierarchical_summary(
        self,
        episodes: List[Episode],
        target_tokens: int
    ) -> str:
        """
        Multi-level hierarchical summarization.
        
        Strategy:
        1. Split into chunks
        2. Summarize each chunk
        3. Combine chunk summaries
        """
        if not episodes:
            return ""
        
        # If small enough, just extract
        full_text = self._format_episodes(episodes)
        full_tokens = self.token_counter.count_tokens(full_text)
        
        if full_tokens <= target_tokens:
            return full_text
        
        # Split into chunks (groups of 3-5 episodes)
        chunk_size = 4
        chunks = [
            episodes[i:i + chunk_size]
            for i in range(0, len(episodes), chunk_size)
        ]
        
        # Summarize each chunk
        tokens_per_chunk = target_tokens // len(chunks)
        chunk_summaries = []
        
        for chunk in chunks:
            chunk_summary = self._extractive_summary(chunk, tokens_per_chunk)
            chunk_summaries.append(chunk_summary)
        
        return "\n\n---\n\n".join(chunk_summaries)
    
    def _progressive_summary(
        self,
        episodes: List[Episode],
        target_tokens: int
    ) -> str:
        """
        Progressive summarization - keep recent detailed, older compressed.
        
        Strategy:
        - Last 3 messages: Full detail
        - Messages 4-10: Moderate compression
        - Older: High compression
        """
        if not episodes:
            return ""
        
        summary_parts = []
        used_tokens = 0
        
        # Determine compression levels
        recent_count = min(3, len(episodes))
        medium_count = min(7, max(0, len(episodes) - recent_count))
        old_count = max(0, len(episodes) - recent_count - medium_count)
        
        # Budget allocation
        recent_budget = int(target_tokens * 0.6)
        medium_budget = int(target_tokens * 0.3)
        old_budget = int(target_tokens * 0.1)
        
        # Old messages (highly compressed)
        if old_count > 0:
            old_episodes = episodes[:old_count]
            old_summary = self._create_high_level_summary(
                old_episodes, old_budget
            )
            if old_summary:
                summary_parts.append(f"[Earlier conversation: {old_summary}]")
                used_tokens += self.token_counter.count_tokens(old_summary)
        
        # Medium-age messages (moderate detail)
        if medium_count > 0:
            start_idx = old_count
            end_idx = start_idx + medium_count
            medium_episodes = episodes[start_idx:end_idx]
            medium_summary = self._extractive_summary(
                medium_episodes, medium_budget
            )
            if medium_summary:
                summary_parts.append(medium_summary)
                used_tokens += self.token_counter.count_tokens(medium_summary)
        
        # Recent messages (full detail)
        recent_episodes = episodes[-recent_count:]
        recent_text = self._format_episodes(recent_episodes)
        recent_tokens = self.token_counter.count_tokens(recent_text)
        
        if used_tokens + recent_tokens <= target_tokens:
            summary_parts.append(recent_text)
        else:
            # Compress slightly if needed
            remaining = target_tokens - used_tokens
            compressed = self.token_counter.truncate_to_tokens(
                recent_text, remaining
            )
            summary_parts.append(compressed)
        
        return "\n\n".join(summary_parts)
    
    def _create_high_level_summary(
        self,
        episodes: List[Episode],
        max_tokens: int
    ) -> str:
        """Create high-level summary of old episodes."""
        if not episodes:
            return ""
        
        # Extract topics/themes
        first = episodes[0]
        last = episodes[-1]
        
        summary = f"Discussed from {first.timestamp.strftime('%Y-%m-%d')} to {last.timestamp.strftime('%Y-%m-%d')} ({len(episodes)} messages)"
        
        # Truncate if needed
        return self.token_counter.truncate_to_tokens(summary, max_tokens)
    
    def _format_episode(self, episode: Episode) -> str:
        """Format single episode."""
        input_text = str(episode.input)
        output_text = str(episode.output)
        
        # Extract actual text if dict
        if isinstance(episode.input, dict):
            input_text = episode.input.get("query", str(episode.input))
        if isinstance(episode.output, dict):
            output_text = episode.output.get("response", str(episode.output))
        
        return f"User: {input_text}\nAssistant: {output_text}"
    
    def _format_episodes(self, episodes: List[Episode]) -> str:
        """Format multiple episodes."""
        return "\n\n".join(self._format_episode(ep) for ep in episodes)
    
    async def get_compressed_history(
        self,
        episodes: List[Episode],
        max_tokens: int = 2000
    ) -> List[Dict[str, Any]]:
        """
        Get conversation with smart compression.
        
        Returns structured list suitable for LLM prompts.
        
        Args:
            episodes: List of episodes
            max_tokens: Maximum tokens for all history
        
        Returns:
            List of message dicts with role and content
        """
        if not episodes:
            return []
        
        # Check if compression needed
        full_tokens = sum(
            self.token_counter.count_tokens(self._format_episode(ep))
            for ep in episodes
        )
        
        if full_tokens <= max_tokens:
            # No compression needed
            return self._episodes_to_messages(episodes)
        
        # Use progressive compression
        summary = self._progressive_summary(episodes, max_tokens)
        
        # Parse back into messages (simplified)
        messages = []
        
        # Add system message about compression
        messages.append({
            "role": "system",
            "content": f"[Conversation history compressed to fit context. {len(episodes)} messages summarized.]"
        })
        
        # Add compressed content
        parts = summary.split("\n\n")
        for part in parts:
            if part.strip():
                messages.append({
                    "role": "user",
                    "content": part
                })
        
        return messages
    
    def _episodes_to_messages(
        self,
        episodes: List[Episode]
    ) -> List[Dict[str, str]]:
        """Convert episodes to message format."""
        messages = []
        
        for episode in episodes:
            # User message
            input_text = str(episode.input)
            if isinstance(episode.input, dict):
                input_text = episode.input.get("query", str(episode.input))
            
            messages.append({
                "role": "user",
                "content": input_text
            })
            
            # Assistant message
            output_text = str(episode.output)
            if isinstance(episode.output, dict):
                output_text = episode.output.get("response", str(episode.output))
            
            messages.append({
                "role": "assistant",
                "content": output_text
            })
        
        return messages
    
    async def create_conversation_summary(
        self,
        episodes: List[Episode],
        max_length: int = 200
    ) -> str:
        """
        Create persistent summary of entire conversation.
        
        Args:
            episodes: List of episodes
            max_length: Maximum character length
        
        Returns:
            Conversation summary
        """
        if not episodes:
            return "Empty conversation"
        
        # Basic stats
        start_time = episodes[0].timestamp
        end_time = episodes[-1].timestamp
        duration = end_time - start_time
        
        # Count successful interactions
        success_count = sum(1 for ep in episodes if ep.success)
        
        # Extract topics (simplified - just from first/last)
        first_input = str(episodes[0].input)
        if isinstance(episodes[0].input, dict):
            first_input = episodes[0].input.get("query", first_input)
        
        summary = (
            f"Conversation from {start_time.strftime('%Y-%m-%d %H:%M')} "
            f"to {end_time.strftime('%Y-%m-%d %H:%M')} "
            f"({len(episodes)} messages, {success_count} successful). "
            f"Started with: {first_input[:50]}..."
        )
        
        # Truncate to max length
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary


__all__ = [
    "ConversationSummarizer",
    "MessagePrioritizer",
]

