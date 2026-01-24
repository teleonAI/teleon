"""
LLM Context Builder - Build optimized context for LLM prompts from memory.

Provides utilities for:
- Building context from multiple memory types
- Token budget allocation
- Context compression with LLM summarization
- Priority-based memory selection
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timezone
import logging
import asyncio

from teleon.cortex.token_manager import TokenCounter, ContextWindowManager

logger = logging.getLogger(__name__)


class TokenBudgetAllocator:
    """
    Intelligently allocate tokens across different context components.
    
    Strategies:
    - balanced: Equal weight to all components
    - recent: Prioritize recent conversation
    - relevant: Prioritize semantic search results
    - patterns: Prioritize successful patterns
    
    Example:
        ```python
        allocator = TokenBudgetAllocator(total_tokens=2000, strategy="balanced")
        allocation = allocator.allocate()
        # Returns: {
        #     'system': 200,
        #     'recent': 800,
        #     'knowledge': 600,
        #     'patterns': 200,
        #     'working': 200
        # }
        ```
    """
    
    STRATEGIES = {
        "balanced": {
            "system": 0.10,
            "recent": 0.40,
            "knowledge": 0.30,
            "patterns": 0.10,
            "working": 0.10,
        },
        "recent": {
            "system": 0.10,
            "recent": 0.60,
            "knowledge": 0.15,
            "patterns": 0.10,
            "working": 0.05,
        },
        "relevant": {
            "system": 0.10,
            "recent": 0.20,
            "knowledge": 0.50,
            "patterns": 0.10,
            "working": 0.10,
        },
        "patterns": {
            "system": 0.10,
            "recent": 0.25,
            "knowledge": 0.25,
            "patterns": 0.30,
            "working": 0.10,
        },
    }
    
    def __init__(
        self,
        total_tokens: int,
        strategy: str = "balanced",
        custom_allocation: Optional[Dict[str, float]] = None
    ):
        """
        Initialize budget allocator.
        
        Args:
            total_tokens: Total tokens available
            strategy: Allocation strategy
            custom_allocation: Custom allocation percentages
        """
        self.total_tokens = total_tokens
        self.strategy = strategy
        
        if custom_allocation:
            self.allocation = custom_allocation
        elif strategy in self.STRATEGIES:
            self.allocation = self.STRATEGIES[strategy]
        else:
            logger.warning(f"Unknown strategy {strategy}, using balanced")
            self.allocation = self.STRATEGIES["balanced"]
    
    def allocate(self) -> Dict[str, int]:
        """
        Allocate tokens to components.
        
        Returns:
            Dictionary mapping component to token count
        """
        allocation = {}
        
        for component, percentage in self.allocation.items():
            allocation[component] = int(self.total_tokens * percentage)
        
        return allocation
    
    def reallocate_unused(
        self,
        used: Dict[str, int],
        priorities: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Reallocate unused tokens to other components.
        
        Args:
            used: Actual tokens used per component
            priorities: Components to prioritize for reallocation
        
        Returns:
            Updated allocation
        """
        allocation = self.allocate()
        total_unused = 0
        
        # Calculate unused tokens
        for component, allocated in allocation.items():
            actual_used = used.get(component, 0)
            if actual_used < allocated:
                total_unused += (allocated - actual_used)
        
        if total_unused == 0:
            return used
        
        # Redistribute to priority components
        if not priorities:
            priorities = ["knowledge", "recent", "patterns"]
        
        per_priority = total_unused // len(priorities)
        
        result = dict(used)
        for component in priorities:
            if component in result:
                result[component] += per_priority
        
        return result


class ContextCompressor:
    """
    Compress context when exceeding token limits.
    
    Strategies:
    - truncate: Simple truncation
    - summarize: Summarize using LLM (or extractive fallback)
    - extract: Extract key information
    - hierarchical: Multi-level compression
    
    Example:
        ```python
        compressor = ContextCompressor(token_counter)
        
        # Set LLM function for summarization
        compressor.set_llm_function(my_llm_summarize)
        
        compressed = await compressor.compress(
            text=long_text,
            target_tokens=500,
            method="summarize"
        )
        ```
    """
    
    def __init__(
        self,
        token_counter: TokenCounter,
        llm_function: Optional[Callable[[str, int], str]] = None
    ):
        """
        Initialize compressor.
        
        Args:
            token_counter: Token counter instance
            llm_function: Optional async function for LLM summarization
                         Signature: async (text: str, target_tokens: int) -> str
        """
        self.token_counter = token_counter
        self._llm_function = llm_function
    
    def set_llm_function(
        self,
        llm_function: Callable[[str, int], str]
    ) -> None:
        """
        Set the LLM function for summarization.
        
        Args:
            llm_function: Async function that takes (text, target_tokens) and returns summary
        """
        self._llm_function = llm_function
    
    async def compress(
        self,
        text: str,
        target_tokens: int,
        method: str = "truncate",
        context: Optional[str] = None
    ) -> str:
        """
        Compress text to target token count.
        
        Args:
            text: Text to compress
            target_tokens: Target token count
            method: Compression method
            context: Optional context for better summarization
        
        Returns:
            Compressed text
        """
        current_tokens = self.token_counter.count_tokens(text)
        
        if current_tokens <= target_tokens:
            return text
        
        if method == "truncate":
            return self._truncate(text, target_tokens)
        elif method == "summarize":
            return await self._summarize(text, target_tokens, context)
        elif method == "extract":
            return self._extract_key_points(text, target_tokens)
        elif method == "hierarchical":
            return await self._hierarchical_compress(text, target_tokens)
        else:
            logger.warning(f"Unknown method {method}, using truncate")
            return self._truncate(text, target_tokens)
    
    def _truncate(self, text: str, target_tokens: int) -> str:
        """Simple truncation."""
        return self.token_counter.truncate_to_tokens(text, target_tokens)
    
    async def _summarize(
        self,
        text: str,
        target_tokens: int,
        context: Optional[str] = None
    ) -> str:
        """
        Summarize text using LLM or extractive fallback.
        
        If an LLM function is set, uses it for abstractive summarization.
        Otherwise, falls back to extractive summarization.
        """
        if self._llm_function:
            try:
                # Use LLM for summarization
                result = await self._call_llm_summarize(text, target_tokens, context)
                return result
            except Exception as e:
                logger.warning(f"LLM summarization failed, falling back to extraction: {e}")
                return self._extract_key_points(text, target_tokens)
        else:
            # Use extractive summarization
            return self._extractive_summarize(text, target_tokens)
    
    async def _call_llm_summarize(
        self,
        text: str,
        target_tokens: int,
        context: Optional[str] = None
    ) -> str:
        """Call LLM function for summarization."""
        if asyncio.iscoroutinefunction(self._llm_function):
            return await self._llm_function(text, target_tokens)
        else:
            # Run sync function in thread pool
            return await asyncio.to_thread(self._llm_function, text, target_tokens)
    
    def _extractive_summarize(self, text: str, target_tokens: int) -> str:
        """
        Extractive summarization - select important sentences.
        
        Uses a simple scoring algorithm:
        1. Score sentences by position (first sentences more important)
        2. Score by length (moderate length preferred)
        3. Score by keyword density
        """
        sentences = self._split_sentences(text)
        
        if not sentences:
            return self._truncate(text, target_tokens)
        
        # Score sentences
        scored_sentences = []
        total_sentences = len(sentences)
        
        for i, sentence in enumerate(sentences):
            score = 0.0
            
            # Position score (first and last sentences more important)
            if i < 3:
                score += 0.3 * (3 - i) / 3
            elif i >= total_sentences - 2:
                score += 0.1
            
            # Length score (prefer moderate length)
            words = len(sentence.split())
            if 10 <= words <= 30:
                score += 0.3
            elif 5 <= words < 10 or 30 < words <= 50:
                score += 0.15
            
            # Keyword density (simple heuristic)
            important_words = {'key', 'important', 'main', 'primary', 'essential',
                             'critical', 'significant', 'note', 'remember', 'summary'}
            sentence_lower = sentence.lower()
            for word in important_words:
                if word in sentence_lower:
                    score += 0.1
                    break
            
            scored_sentences.append((i, sentence, score))
        
        # Sort by score (descending)
        scored_sentences.sort(key=lambda x: x[2], reverse=True)
        
        # Select sentences up to token limit, maintaining original order
        selected = []
        total_tokens_used = 0
        
        for idx, sentence, score in scored_sentences:
            sentence_tokens = self.token_counter.count_tokens(sentence)
            if total_tokens_used + sentence_tokens <= target_tokens:
                selected.append((idx, sentence))
                total_tokens_used += sentence_tokens
        
        # Sort by original position
        selected.sort(key=lambda x: x[0])
        
        # Join sentences
        result = " ".join(s for _, s in selected)
        
        # If still too long, truncate
        if self.token_counter.count_tokens(result) > target_tokens:
            result = self.token_counter.truncate_to_tokens(result, target_tokens)
        
        return result
    
    def _extract_key_points(self, text: str, target_tokens: int) -> str:
        """
        Extract key points from text.
        
        Strategy: Take first sentences from each paragraph.
        """
        paragraphs = text.split("\n\n")
        
        if len(paragraphs) <= 1:
            return self._extractive_summarize(text, target_tokens)
        
        # Take first sentence from each paragraph
        key_points = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            sentences = self._split_sentences(para)
            if sentences:
                key_points.append(sentences[0])
        
        result = " ".join(key_points)
        
        # Truncate if needed
        if self.token_counter.count_tokens(result) > target_tokens:
            result = self.token_counter.truncate_to_tokens(result, target_tokens)
        
        return result
    
    async def _hierarchical_compress(self, text: str, target_tokens: int) -> str:
        """
        Multi-level compression.
        
        1. Split into sections
        2. Compress each section proportionally
        3. Combine
        """
        # Split into paragraphs
        paragraphs = text.split("\n\n")
        
        if len(paragraphs) <= 1:
            return await self._summarize(text, target_tokens)
        
        # Allocate tokens proportionally
        total_chars = sum(len(p) for p in paragraphs)
        
        compressed_paras = []
        tokens_remaining = target_tokens
        
        for i, para in enumerate(paragraphs):
            if not para.strip():
                continue
            
            # Proportional allocation
            para_ratio = len(para) / total_chars
            para_target = int(target_tokens * para_ratio)
            
            # Ensure minimum
            para_target = max(para_target, 20)
            
            # Don't exceed remaining
            para_target = min(para_target, tokens_remaining)
            
            if para_target <= 0:
                break
            
            # Compress paragraph
            para_tokens = self.token_counter.count_tokens(para)
            
            if para_tokens > para_target:
                compressed = self._extractive_summarize(para, para_target)
            else:
                compressed = para
            
            compressed_paras.append(compressed)
            tokens_remaining -= self.token_counter.count_tokens(compressed)
        
        return "\n\n".join(compressed_paras)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class PriorityRanker:
    """
    Rank memories by relevance and importance for context inclusion.
    
    Scoring factors:
    - Recency
    - Relevance (similarity to query)
    - Success rate
    - Access frequency
    - Importance score
    
    Supports embedding-based relevance when embedding function is provided.
    
    Example:
        ```python
        ranker = PriorityRanker(embedding_function=embed_fn)
        
        # Rank episodes with semantic similarity
        ranked = ranker.rank_episodes(
            episodes=episodes,
            query="What is machine learning?",
            top_k=5
        )
        
        # Rank knowledge entries
        ranked = ranker.rank_knowledge(
            entries=entries,
            query="machine learning",
            top_k=3
        )
        ```
    """
    
    def __init__(
        self,
        recency_weight: float = 0.3,
        relevance_weight: float = 0.4,
        success_weight: float = 0.2,
        importance_weight: float = 0.1,
        embedding_function: Optional[Callable[[str], List[float]]] = None
    ):
        """
        Initialize ranker.
        
        Args:
            recency_weight: Weight for recency score
            relevance_weight: Weight for relevance score
            success_weight: Weight for success rate
            importance_weight: Weight for importance score
            embedding_function: Optional function for embedding-based relevance
        """
        self.recency_weight = recency_weight
        self.relevance_weight = relevance_weight
        self.success_weight = success_weight
        self.importance_weight = importance_weight
        self.embedding_function = embedding_function
    
    def rank_episodes(
        self,
        episodes: List[Any],
        query: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Any]:
        """
        Rank episodes by priority.
        
        Args:
            episodes: List of episodes
            query: Query for relevance scoring
            top_k: Return top K episodes
        
        Returns:
            Ranked list of episodes
        """
        if not episodes:
            return []
        
        # Generate query embedding if function available
        query_embedding = None
        if query and self.embedding_function:
            try:
                query_embedding = self.embedding_function(query)
            except Exception as e:
                logger.debug(f"Failed to generate query embedding: {e}")
        
        scored_episodes = []
        
        for episode in episodes:
            score = self._score_episode(episode, query, query_embedding)
            scored_episodes.append((score, episode))
        
        # Sort by score descending
        scored_episodes.sort(key=lambda x: x[0], reverse=True)
        
        # Return top K
        if top_k:
            scored_episodes = scored_episodes[:top_k]
        
        return [episode for _, episode in scored_episodes]
    
    def _score_episode(
        self,
        episode: Any,
        query: Optional[str] = None,
        query_embedding: Optional[List[float]] = None
    ) -> float:
        """Calculate priority score for episode."""
        score = 0.0
        
        # Recency score (0-1)
        recency_score = self._calculate_recency_score(episode.timestamp)
        score += recency_score * self.recency_weight
        
        # Relevance score (0-1)
        if query or query_embedding:
            relevance_score = self._calculate_relevance_score(
                episode, query, query_embedding
            )
            score += relevance_score * self.relevance_weight
        
        # Success score (0-1)
        success_score = 1.0 if getattr(episode, 'success', True) else 0.0
        score += success_score * self.success_weight
        
        # Importance score (0-1)
        importance_score = getattr(episode, 'importance_score', 0.5)
        score += importance_score * self.importance_weight
        
        return score
    
    def _calculate_recency_score(self, timestamp: datetime) -> float:
        """Calculate recency score (1.0 = most recent, 0.0 = oldest)."""
        from datetime import timedelta, timezone
        
        now = datetime.now(timezone.utc)
        age = now - timestamp
        
        # Exponential decay: half-life of 7 days
        half_life_days = 7
        decay_factor = 0.5 ** (age.days / half_life_days)
        
        return max(0.0, min(1.0, decay_factor))
    
    def _calculate_relevance_score(
        self,
        episode: Any,
        query: Optional[str],
        query_embedding: Optional[List[float]]
    ) -> float:
        """
        Calculate relevance score using embeddings or text matching.
        """
        # Try embedding-based similarity first
        if query_embedding and hasattr(episode, 'embedding') and episode.embedding:
            return self._cosine_similarity(query_embedding, episode.embedding)
        
        # Fall back to text matching
        if not query:
            return 0.5
        
        query_lower = query.lower()
        
        # Check input
        input_text = str(episode.input).lower()
        output_text = str(episode.output).lower()
        
        # Exact match bonus
        if query_lower in input_text:
            return 1.0
        if query_lower in output_text:
            return 0.9
        
        # Word overlap
        query_words = set(query_lower.split())
        episode_words = set(input_text.split() + output_text.split())
        
        overlap = len(query_words & episode_words)
        if overlap > 0:
            return 0.3 + (0.5 * overlap / len(query_words))
        
        return 0.0
    
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
    
    def rank_knowledge(
        self,
        entries: List[Tuple[Any, float]],
        top_k: Optional[int] = None
    ) -> List[Tuple[Any, float]]:
        """
        Rank knowledge entries (already have similarity scores).
        
        Args:
            entries: List of (entry, similarity_score) tuples
            top_k: Return top K entries
        
        Returns:
            Ranked list of (entry, combined_score) tuples
        """
        if not entries:
            return []
        
        scored_entries = []
        
        for entry, similarity in entries:
            # Combine similarity with other factors
            combined_score = self._score_knowledge_entry(entry, similarity)
            scored_entries.append((entry, combined_score))
        
        # Sort by score descending
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K
        if top_k:
            scored_entries = scored_entries[:top_k]
        
        return scored_entries
    
    def _score_knowledge_entry(self, entry: Any, similarity: float) -> float:
        """Calculate combined score for knowledge entry."""
        # Start with similarity
        score = similarity * 0.6
        
        # Add importance
        importance = getattr(entry, 'importance_score', 0.5)
        score += importance * 0.2
        
        # Add confidence
        confidence = getattr(entry, 'confidence_score', 0.5)
        score += confidence * 0.1
        
        # Add access frequency boost
        access_count = getattr(entry, 'access_count', 0)
        if access_count > 0:
            # Logarithmic boost for access count
            import math
            access_boost = min(0.1, math.log(access_count + 1) / 10)
            score += access_boost
        
        return min(1.0, score)


class LLMContextBuilder:
    """
    Build optimized context for LLM prompts from Cortex memory.
    
    Main interface for constructing LLM-ready context that:
    - Respects token limits
    - Prioritizes relevant information
    - Balances different memory types
    - Compresses when necessary
    - Uses embeddings for semantic relevance
    
    Example:
        ```python
        from teleon.cortex import create_cortex
        from teleon.cortex.llm_context import LLMContextBuilder
        
        cortex = await create_cortex(agent_id="agent-123")
        builder = LLMContextBuilder(cortex)
        
        # Set LLM function for summarization (optional)
        builder.set_llm_function(my_summarize_function)
        
        # Build context for a query
        context = await builder.build_context(
            query="What is machine learning?",
            max_tokens=2000,
            strategy="balanced"
        )
        
        # Use context in prompt
        prompt = f'''
        Context:
        {context['formatted']}
        
        User Query: {query}
        '''
        ```
    """
    
    def __init__(
        self,
        cortex,
        model: str = "gpt-4",
        embedding_function: Optional[Callable[[str], List[float]]] = None
    ):
        """
        Initialize context builder.
        
        Args:
            cortex: CortexMemory instance
            model: Model name for token counting
            embedding_function: Optional embedding function for semantic ranking
        """
        self.cortex = cortex
        self.token_counter = TokenCounter(model)
        self.compressor = ContextCompressor(self.token_counter)
        
        # Use cortex's embedding function if available
        if embedding_function:
            self.embedding_function = embedding_function
        elif hasattr(cortex, 'config') and cortex.config.embedding_function:
            self.embedding_function = cortex.config.embedding_function
        else:
            self.embedding_function = None
        
        self.ranker = PriorityRanker(embedding_function=self.embedding_function)
    
    def set_llm_function(
        self,
        llm_function: Callable[[str, int], str]
    ) -> None:
        """
        Set LLM function for summarization.
        
        Args:
            llm_function: Async function (text, target_tokens) -> summary
        """
        self.compressor.set_llm_function(llm_function)
    
    async def build_context(
        self,
        query: str,
        max_tokens: int = 2000,
        strategy: str = "balanced",
        include_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Build optimized context within token budget.
        
        Args:
            query: User query
            max_tokens: Maximum tokens for context
            strategy: Token allocation strategy
            include_types: Memory types to include (default: all)
        
        Returns:
            Dictionary with:
            - recent_conversation: Recent episodes
            - relevant_knowledge: Semantic search results
            - successful_patterns: Procedural patterns
            - working_memory: Current session state
            - formatted: Formatted string for prompt
            - token_usage: Token count breakdown
        """
        if include_types is None:
            include_types = ["episodic", "semantic", "procedural", "working"]
        
        # Allocate token budget
        allocator = TokenBudgetAllocator(max_tokens, strategy)
        allocation = allocator.allocate()
        
        context = {
            "recent_conversation": [],
            "relevant_knowledge": [],
            "successful_patterns": [],
            "working_memory": {},
            "token_usage": {}
        }
        
        # Get recent conversation (episodic)
        if "episodic" in include_types and self.cortex.episodic:
            recent = await self._get_recent_conversation(
                query,
                allocation.get("recent", 0)
            )
            context["recent_conversation"] = recent["episodes"]
            context["token_usage"]["recent"] = recent["tokens"]
        
        # Get relevant knowledge (semantic)
        if "semantic" in include_types and self.cortex.semantic:
            knowledge = await self._get_relevant_knowledge(
                query,
                allocation.get("knowledge", 0)
            )
            context["relevant_knowledge"] = knowledge["entries"]
            context["token_usage"]["knowledge"] = knowledge["tokens"]
        
        # Get successful patterns (procedural)
        if "procedural" in include_types and self.cortex.procedural:
            patterns = await self._get_successful_patterns(
                query,
                allocation.get("patterns", 0)
            )
            context["successful_patterns"] = patterns["patterns"]
            context["token_usage"]["patterns"] = patterns["tokens"]
        
        # Get working memory
        if "working" in include_types and self.cortex.working:
            working = await self._get_working_memory(
                allocation.get("working", 0)
            )
            context["working_memory"] = working["data"]
            context["token_usage"]["working"] = working["tokens"]
        
        # Format context as string
        context["formatted"] = self._format_context(context)
        context["token_usage"]["total"] = self.token_counter.count_tokens(
            context["formatted"]
        )
        
        return context
    
    async def _get_recent_conversation(
        self,
        query: str,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Get recent conversation within token budget."""
        if max_tokens <= 0:
            return {"episodes": [], "tokens": 0}
        
        # Get recent episodes
        episodes = await self.cortex.episodic.get_recent(limit=10)
        
        if not episodes:
            return {"episodes": [], "tokens": 0}
        
        # Rank by priority (uses embeddings if available)
        ranked_episodes = self.ranker.rank_episodes(episodes, query, top_k=10)
        
        # Add episodes until token budget exhausted
        selected_episodes = []
        total_tokens = 0
        
        for episode in ranked_episodes:
            # Format episode
            episode_text = self._format_episode(episode)
            episode_tokens = self.token_counter.count_tokens(episode_text)
            
            if total_tokens + episode_tokens <= max_tokens:
                selected_episodes.append(episode)
                total_tokens += episode_tokens
            else:
                # Try to fit compressed version
                target = max_tokens - total_tokens
                if target > 50:  # Only if meaningful space left
                    compressed = await self.compressor.compress(
                        episode_text, target, method="extract"
                    )
                    if compressed:
                        selected_episodes.append(episode)
                        total_tokens += self.token_counter.count_tokens(compressed)
                break
        
        return {"episodes": selected_episodes, "tokens": total_tokens}
    
    async def _get_relevant_knowledge(
        self,
        query: str,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Get relevant knowledge within token budget."""
        if max_tokens <= 0:
            return {"entries": [], "tokens": 0}
        
        # Search semantic memory
        results = await self.cortex.semantic.search(
            query, limit=10, min_similarity=0.3
        )
        
        if not results:
            return {"entries": [], "tokens": 0}
        
        # Rank results
        ranked_results = self.ranker.rank_knowledge(results, top_k=10)
        
        # Add entries until budget exhausted
        selected_entries = []
        total_tokens = 0
        
        for entry, score in ranked_results:
            content = entry.content
            content_tokens = self.token_counter.count_tokens(content)
            
            if total_tokens + content_tokens <= max_tokens:
                selected_entries.append((entry, score))
                total_tokens += content_tokens
            elif total_tokens < max_tokens:
                # Compress to fit
                target = max_tokens - total_tokens
                if target > 50:
                    compressed = await self.compressor.compress(
                        content, target, method="extract"
                    )
                    selected_entries.append((entry, score))
                    total_tokens += self.token_counter.count_tokens(compressed)
                break
        
        return {"entries": selected_entries, "tokens": total_tokens}
    
    async def _get_successful_patterns(
        self,
        query: str,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Get successful patterns within token budget."""
        if max_tokens <= 0:
            return {"patterns": [], "tokens": 0}
        
        try:
            # Find similar patterns (uses embeddings if available)
            patterns = await self.cortex.procedural.find_similar_patterns(
                query, limit=3, min_similarity=0.3
            )
            
            if not patterns:
                # Try single best pattern
                pattern = await self.cortex.procedural.find_pattern(query)
                if pattern:
                    patterns = [(pattern, 0.5)]
            
            if not patterns:
                return {"patterns": [], "tokens": 0}
            
            # Select patterns within budget
            selected_patterns = []
            total_tokens = 0
            
            for pattern, score in patterns:
                pattern_text = f"Pattern: {pattern.input_pattern} -> {pattern.output_pattern}"
                pattern_tokens = self.token_counter.count_tokens(pattern_text)
                
                if total_tokens + pattern_tokens <= max_tokens:
                    selected_patterns.append(pattern)
                    total_tokens += pattern_tokens
            
            return {"patterns": selected_patterns, "tokens": total_tokens}
        
        except Exception as e:
            logger.debug(f"Error getting patterns: {e}")
            return {"patterns": [], "tokens": 0}
    
    async def _get_working_memory(
        self,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Get working memory within token budget."""
        if max_tokens <= 0:
            return {"data": {}, "tokens": 0}
        
        try:
            # Get all working memory
            working_data = await self.cortex.working.get_all()
            
            if not working_data:
                return {"data": {}, "tokens": 0}
            
            # Format and check tokens
            import json
            working_text = json.dumps(working_data, indent=2, default=str)
            working_tokens = self.token_counter.count_tokens(working_text)
            
            if working_tokens <= max_tokens:
                return {"data": working_data, "tokens": working_tokens}
            else:
                # Compress
                compressed = await self.compressor.compress(
                    working_text, max_tokens, method="truncate"
                )
                return {"data": {"compressed": compressed}, "tokens": max_tokens}
        
        except Exception as e:
            logger.debug(f"Error getting working memory: {e}")
            return {"data": {}, "tokens": 0}
    
    def _format_episode(self, episode: Any) -> str:
        """Format episode for context."""
        input_str = str(episode.input)
        output_str = str(episode.output)
        return f"User: {input_str}\nAssistant: {output_str}"
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format complete context as string."""
        parts = []
        
        # Working memory
        if context["working_memory"]:
            import json
            parts.append(f"Current Session:\n{json.dumps(context['working_memory'], indent=2, default=str)}")
        
        # Recent conversation
        if context["recent_conversation"]:
            conv_parts = []
            for episode in context["recent_conversation"]:
                conv_parts.append(self._format_episode(episode))
            parts.append(f"Recent Conversation:\n" + "\n\n".join(conv_parts))
        
        # Relevant knowledge
        if context["relevant_knowledge"]:
            knowledge_parts = []
            for entry, score in context["relevant_knowledge"]:
                knowledge_parts.append(f"[{score:.2f}] {entry.content}")
            parts.append(f"Relevant Knowledge:\n" + "\n".join(knowledge_parts))
        
        # Patterns
        if context["successful_patterns"]:
            pattern_parts = []
            for pattern in context["successful_patterns"]:
                pattern_parts.append(f"{pattern.input_pattern} -> {pattern.output_pattern}")
            parts.append(f"Successful Patterns:\n" + "\n".join(pattern_parts))
        
        return "\n\n---\n\n".join(parts)


__all__ = [
    "LLMContextBuilder",
    "TokenBudgetAllocator",
    "ContextCompressor",
    "PriorityRanker",
]
