"""
Cortex - Memory and Learning System for Teleon Agents.

Cortex provides a comprehensive memory system enabling agents to remember,
learn, and improve over time.

Memory Types:
- Working Memory: Short-term session storage
- Episodic Memory: Conversation history and past interactions
- Semantic Memory: Long-term knowledge base with vector search
- Procedural Memory: Learned patterns and successful strategies

Example:
    ```python
    from teleon.cortex import CortexMemory, CortexConfig
    from teleon.cortex.storage import InMemoryStorage
    
    # Initialize Cortex
    storage = InMemoryStorage()
    await storage.initialize()
    
    cortex = CortexMemory(
        storage=storage,
        agent_id="agent-123",
        config=CortexConfig(
            working_enabled=True,
            episodic_enabled=True,
            semantic_enabled=True,
            procedural_enabled=True
        )
    )
    await cortex.initialize()
    
    # Store short-term data
    await cortex.working.set("user_name", "Alice")
    
    # Record an interaction
    from teleon.cortex.memory import Episode
    await cortex.episodic.store(Episode(
        agent_id="agent-123",
        input={"query": "Hello"},
        output={"response": "Hi there!"}
    ))
    
    # Store knowledge
    await cortex.semantic.store(
        content="Paris is the capital of France",
        category="geography"
    )
    
    # Learn a pattern
    await cortex.procedural.learn(
        input_pattern="user greets",
        output_pattern="respond warmly",
        success=True
    )
    
    # Query memories
    recent = await cortex.episodic.get_recent(limit=5)
    knowledge = await cortex.semantic.search("capitals")
    pattern = await cortex.procedural.find_pattern("hello")
    ```
"""

from typing import Optional, Dict, Any, Callable, List
from pydantic import BaseModel, Field, SkipValidation, ConfigDict
import time

from teleon.cortex.storage import StorageBackend, InMemoryStorage
from teleon.cortex.memory.episodic import EpisodicMemory, Episode
from teleon.cortex.memory.semantic import SemanticMemory, KnowledgeEntry
from teleon.cortex.memory.procedural import ProceduralMemory, Pattern

# Import working memory from existing location
try:
    from teleon.memory.working import WorkingMemory
    WORKING_MEMORY_AVAILABLE = True
except ImportError:
    WORKING_MEMORY_AVAILABLE = False
    WorkingMemory = None

# Import LLM-specific components (Phase 1 & 2)
from teleon.cortex.token_manager import (
    TokenCounter,
    ContextWindowManager,
    TokenBudgetTracker,
)
from teleon.cortex.llm_context import (
    LLMContextBuilder,
    TokenBudgetAllocator,
    ContextCompressor,
    PriorityRanker,
)
from teleon.cortex.conversation_manager import (
    ConversationSummarizer,
    MessagePrioritizer,
)
from teleon.cortex.rag import (
    RAGMemory,
    DocumentChunker,
    ChunkRanker,
    ContextFusion,
)

# Import advanced memory features (Phase 3 & 4)
from teleon.cortex.memory_consolidation import (
    ImportanceScorer,
    MemoryConsolidator,
    AutomaticForgetfulness,
    MemoryHealthMonitor,
)
from teleon.cortex.advanced_learning import (
    PatternLearner,
    SuccessAnalyzer,
    AdaptiveMemoryManager,
    PerformanceOptimizer,
)
from teleon.cortex.monitoring import CortexMetrics, PerformanceProfiler
from typing import List


class CortexConfig(BaseModel):
    """Configuration for Cortex memory system."""
    
    # Enable/disable memory types
    working_enabled: bool = Field(True, description="Enable working memory")
    episodic_enabled: bool = Field(True, description="Enable episodic memory")
    semantic_enabled: bool = Field(True, description="Enable semantic memory")
    procedural_enabled: bool = Field(True, description="Enable procedural memory")
    
    # TTL settings
    working_ttl: Optional[int] = Field(3600, description="Working memory TTL (seconds)")
    episodic_ttl: Optional[int] = Field(None, description="Episodic memory TTL")
    semantic_ttl: Optional[int] = Field(None, description="Semantic memory TTL")
    procedural_ttl: Optional[int] = Field(None, description="Procedural memory TTL")
    
    # Learning settings
    learning_enabled: bool = Field(True, description="Enable automatic learning")
    min_success_rate: float = Field(50.0, description="Minimum success rate for patterns (%)")
    
    # Embedding settings
    embedding_function: Optional[SkipValidation[Callable]] = Field(
        None,
        description="Custom embedding function for semantic memory"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CortexMemory:
    """
    Unified memory interface for Teleon agents.
    
    Provides access to all memory types through a single interface,
    enabling agents to remember, learn, and improve.
    
    Attributes:
        working: Working memory (short-term session storage)
        episodic: Episodic memory (conversation history)
        semantic: Semantic memory (knowledge base)
        procedural: Procedural memory (learned patterns)
    
    Example:
        ```python
        cortex = CortexMemory(storage, agent_id="agent-123")
        await cortex.initialize()
        
        # Use different memory types
        await cortex.working.set("context", {"user": "Alice"})
        await cortex.episodic.store(episode)
        await cortex.semantic.store("Important fact")
        await cortex.procedural.learn("pattern", "strategy")
        
        # Get statistics
        stats = await cortex.get_statistics()
        print(f"Total episodes: {stats['episodic']['total_episodes']}")
        ```
    """
    
    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        agent_id: str = "default",
        session_id: Optional[str] = None,
        config: Optional[CortexConfig] = None
    ):
        """
        Initialize Cortex memory system.
        
        Args:
            storage: Storage backend (defaults to InMemoryStorage)
            agent_id: ID of the agent using this memory
            session_id: Session ID for working memory
            config: Cortex configuration
        """
        self.storage = storage or InMemoryStorage()
        self.agent_id = agent_id
        self.session_id = session_id
        self.config = config or CortexConfig()
        
        # Initialize memory types
        self.working: Optional[WorkingMemory] = None
        self.episodic: Optional[EpisodicMemory] = None
        self.semantic: Optional[SemanticMemory] = None
        self.procedural: Optional[ProceduralMemory] = None
        
        # LLM features (Phase 1 & 2)
        self._llm_context_builder: Optional[LLMContextBuilder] = None
        self._rag_memory: Optional[RAGMemory] = None
        self._token_counter: Optional[TokenCounter] = None
        self._conversation_summarizer: Optional[ConversationSummarizer] = None
        
        # Advanced features (Phase 3 & 4)
        self._memory_consolidator: Optional[MemoryConsolidator] = None
        self._pattern_learner: Optional[PatternLearner] = None
        self._success_analyzer: Optional[SuccessAnalyzer] = None
        self._health_monitor: Optional[MemoryHealthMonitor] = None
        self._adaptive_manager: Optional[AdaptiveMemoryManager] = None
        
        # Monitoring and profiling
        self._metrics: Optional[CortexMetrics] = None
        self._profiler: Optional[PerformanceProfiler] = None
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """
        Initialize all enabled memory types with proper error handling and cleanup.
        
        Raises:
            Exception: If initialization fails, with proper cleanup of initialized components
        """
        if self._initialized:
            return
        
        initialized_components = []
        semantic_storage = None  # Track if we created a separate semantic storage
        
        try:
            # Initialize storage if needed
            if not self.storage._initialized:
                await self.storage.initialize()
                initialized_components.append("storage")
            
            # Initialize working memory
            if self.config.working_enabled and WORKING_MEMORY_AVAILABLE:
                self.working = WorkingMemory(
                    session_id=self.session_id or self.agent_id,
                    ttl=self.config.working_ttl
                )
                initialized_components.append("working")
            
            # Initialize episodic memory
            if self.config.episodic_enabled:
                self.episodic = EpisodicMemory(
                    storage=self.storage,
                    agent_id=self.agent_id,
                    embedding_function=self.config.embedding_function,
                    ttl=self.config.episodic_ttl
                )
                await self.episodic.initialize()
                initialized_components.append("episodic")
            
            # Initialize semantic memory
            if self.config.semantic_enabled:
                # Auto-use ChromaDB for semantic memory if available
                semantic_storage = self.storage
                
                # Check if we should use ChromaDB for semantic memory
                import os
                from teleon.cortex.storage import CHROMADB_AVAILABLE, create_chroma_storage
                
                deployment_id = os.getenv("DEPLOYMENT_ID")
                if deployment_id and CHROMADB_AVAILABLE:
                    try:
                        # Use ChromaDB for semantic memory (optimized for vector search)
                        semantic_storage = create_chroma_storage(
                            deployment_id=deployment_id
                        )
                        await semantic_storage.initialize()
                        # ChromaDB handles embeddings internally
                    except Exception as e:
                        # Fall back to regular storage
                        import logging
                        logger = logging.getLogger("teleon.cortex")
                        logger.warning(f"Failed to initialize ChromaDB, using regular storage: {e}")
                        semantic_storage = self.storage
                
                self.semantic = SemanticMemory(
                    storage=semantic_storage,
                    agent_id=self.agent_id,
                    embedding_function=self.config.embedding_function,
                    ttl=self.config.semantic_ttl
                )
                await self.semantic.initialize()
                initialized_components.append("semantic")
            
            # Initialize procedural memory
            if self.config.procedural_enabled:
                self.procedural = ProceduralMemory(
                    storage=self.storage,
                    agent_id=self.agent_id,
                    embedding_function=self.config.embedding_function,
                    min_success_rate=self.config.min_success_rate,
                    ttl=self.config.procedural_ttl
                )
                await self.procedural.initialize()
                initialized_components.append("procedural")
            
            # Initialize monitoring and profiling
            self._metrics = CortexMetrics(agent_id=self.agent_id)
            self._profiler = PerformanceProfiler(slow_threshold_ms=100.0)
            
            self._initialized = True
            
            # Register with global registry for API/CLI access
            try:
                from teleon.cortex.registry import registry
                asyncio.create_task(registry.register(self.agent_id, self))
            except Exception as e:
                import logging
                logger = logging.getLogger("teleon.cortex")
                logger.debug(f"Could not register Cortex instance: {e}")
            
        except Exception as e:
            # Cleanup initialized components on error
            import logging
            logger = logging.getLogger("teleon.cortex")
            logger.error(f"Initialization failed: {e}, cleaning up...", exc_info=True)
            await self._cleanup_components(initialized_components, semantic_storage)
            raise
    
    async def _cleanup_components(
        self,
        components: List[str],
        semantic_storage: Optional[Any] = None
    ) -> None:
        """
        Cleanup initialized components on error.
        
        Args:
            components: List of component names to cleanup
            semantic_storage: Optional separate semantic storage to cleanup
        """
        import logging
        logger = logging.getLogger("teleon.cortex")
        
        # Cleanup in reverse order
        for component in reversed(components):
            try:
                if component == "storage" and self.storage:
                    await self.storage.shutdown()
                elif component == "semantic" and semantic_storage and semantic_storage != self.storage:
                    # Cleanup separate semantic storage if it was created
                    await semantic_storage.shutdown()
                # Other components don't have explicit shutdown, but storage handles it
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up {component}: {cleanup_error}")
        
        # Reset state
        self.working = None
        self.episodic = None
        self.semantic = None
        self.procedural = None
        self._initialized = False
    
    async def record_interaction(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        success: bool = True,
        cost: Optional[float] = None,
        duration_ms: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Record a complete interaction across all memory types.
        
        This is a convenience method that stores the interaction in episodic
        memory and optionally learns from it in procedural memory.
        
        Args:
            input_data: Input data for the interaction
            output_data: Output/response data
            success: Whether the interaction was successful
            cost: Cost of the interaction
            duration_ms: Duration in milliseconds
            context: Contextual information
            session_id: Session ID
            conversation_id: Conversation ID
        
        Returns:
            Episode ID
        """
        if not self._initialized:
            await self.initialize()
        
        # Store in episodic memory with metrics
        episode_id = None
        if self.episodic:
            start_time = time.perf_counter()
            try:
                episode = Episode(
                    agent_id=self.agent_id,
                    input=input_data,
                    output=output_data,
                    context=context or {},
                    success=success,
                    cost=cost,
                    duration_ms=duration_ms,
                    session_id=session_id or self.session_id,
                    conversation_id=conversation_id
                )
                episode_id = await self.episodic.store(episode)
                
                # Record metrics
                if self._metrics:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    self._metrics.record_operation("episodic", "store", latency_ms, error=False)
                    self._profiler.record_operation("episodic.store", latency_ms)
            except Exception as e:
                if self._metrics:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    self._metrics.record_operation("episodic", "store", latency_ms, error=True)
                raise
        
        # Learn pattern if enabled
        if self.config.learning_enabled and self.procedural and success:
            # Simple pattern extraction: uses first 100 chars of input/output
            # This is a basic implementation that works but could be enhanced with
            # more sophisticated pattern extraction (e.g., semantic analysis, structure detection)
            input_str = str(input_data.get("query", ""))[:100]
            output_str = str(output_data.get("response", ""))[:100]
            
            if input_str and output_str:
                await self.procedural.learn(
                    input_pattern=input_str,
                    output_pattern=output_str,
                    success=success,
                    cost=cost,
                    latency_ms=duration_ms
                )
        
        return episode_id
    
    async def retrieve_context(
        self,
        query: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context from all memory types.
        
        Args:
            query: Optional query for semantic search
            limit: Maximum items per memory type
        
        Returns:
            Dictionary with context from all memory types
        """
        if not self._initialized:
            await self.initialize()
        
        context = {}
        
        # Get working memory
        if self.working:
            working_data = await self.working.get_all()
            context["working"] = working_data
        
        # Get recent episodes
        if self.episodic:
            recent = await self.episodic.get_recent(limit=limit)
            context["recent_episodes"] = [ep.model_dump(mode='python') for ep in recent]
        
        # Search semantic memory
        if self.semantic and query:
            knowledge = await self.semantic.search(query, limit=limit)
            context["relevant_knowledge"] = [
                {"content": entry.content, "similarity": score}
                for entry, score in knowledge
            ]
        
        # Find relevant patterns
        if self.procedural and query:
            pattern = await self.procedural.find_pattern(query)
            if pattern:
                context["suggested_pattern"] = pattern.model_dump(mode='python')
        
        return context
    
    async def clear_all(self) -> Dict[str, int]:
        """
        Clear all memory types.
        
        Returns:
            Dictionary with counts of cleared items per memory type
        """
        if not self._initialized:
            await self.initialize()
        
        counts = {}
        
        if self.working:
            await self.working.clear()
            counts["working"] = 1
        
        if self.episodic:
            counts["episodic"] = await self.episodic.clear()
        
        if self.semantic:
            counts["semantic"] = await self.semantic.clear()
        
        if self.procedural:
            counts["procedural"] = await self.procedural.clear()
        
        return counts
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from all memory types including monitoring metrics.
        
        Returns:
            Dictionary with statistics from all memory types
        """
        if not self._initialized:
            await self.initialize()
        
        stats = {}
        
        if self.working:
            stats["working"] = await self.working.get_statistics()
        
        if self.episodic:
            stats["episodic"] = await self.episodic.get_statistics()
        
        if self.semantic:
            stats["semantic"] = await self.semantic.get_statistics()
        
        if self.procedural:
            stats["procedural"] = await self.procedural.get_statistics()
        
        # Add storage statistics
        if self.storage:
            storage_stats = await self.storage.get_statistics()
            stats["storage"] = {
                "total_keys": storage_stats.get("total_keys", 0),
                "total_size_bytes": storage_stats.get("total_size_bytes", 0),
                "total_operations": storage_stats.get("total_operations", 0),
                "total_gets": storage_stats.get("get_operations", 0),
                "total_sets": storage_stats.get("set_operations", 0),
                "total_deletes": storage_stats.get("delete_operations", 0),
                "hit_rate": storage_stats.get("hit_rate", "0%"),
            }
        
        # Add monitoring metrics
        if self._metrics:
            stats["monitoring"] = self._metrics.get_metrics()
        
        return stats
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get real-time monitoring metrics.
        
        Returns:
            Dictionary with operation metrics, cache stats, etc.
        """
        if not self._metrics:
            return {}
        return self._metrics.get_metrics()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get performance profiling report.
        
        Returns:
            Dictionary with performance stats, slow operations, and recommendations
        """
        if not self._profiler:
            return {}
        
        return {
            "stats": self._profiler.get_stats(),
            "slow_operations": self._profiler.get_slow_operations(20),
            "recommendations": self._profiler.get_recommendations()
        }
    
    async def shutdown(self) -> None:
        """Shutdown Cortex and cleanup resources."""
        # Unregister from registry
        try:
            from teleon.cortex.registry import registry
            await registry.unregister(self.agent_id)
        except Exception as e:
            import logging
            logger = logging.getLogger("teleon.cortex")
            logger.debug(f"Could not unregister Cortex instance: {e}")
        
        try:
            if self.storage:
                await self.storage.shutdown()
        except Exception as e:
            import logging
            logger = logging.getLogger("teleon.cortex")
            logger.warning(f"Error shutting down storage: {e}")
        
        # Reset monitoring
        if self._metrics:
            self._metrics.reset()
        if self._profiler:
            self._profiler.reset()
        
        self._initialized = False
    
    # LLM Feature Convenience Methods (Phase 1 & 2)
    
    def get_token_counter(self, model: str = "gpt-4") -> TokenCounter:
        """
        Get token counter for model.
        
        Args:
            model: Model name
        
        Returns:
            TokenCounter instance
        """
        if self._token_counter is None:
            self._token_counter = TokenCounter(model)
        return self._token_counter
    
    def get_conversation_summarizer(self) -> ConversationSummarizer:
        """
        Get conversation summarizer.
        
        Returns:
            ConversationSummarizer instance
        """
        if self._conversation_summarizer is None:
            token_counter = self.get_token_counter()
            self._conversation_summarizer = ConversationSummarizer(token_counter)
        return self._conversation_summarizer
    
    async def build_llm_context(
        self,
        query: str,
        max_tokens: int = 2000,
        strategy: str = "balanced",
        model: str = "gpt-4"
    ) -> Dict[str, Any]:
        """
        Build optimized LLM context from memory.
        
        Convenience method for LLMContextBuilder.
        
        Args:
            query: Query for context building
            max_tokens: Maximum tokens for context
            strategy: Token allocation strategy
            model: Model name for token counting
        
        Returns:
            Context dictionary with formatted text and token usage
        
        Example:
            ```python
            cortex = await create_cortex(agent_id="agent-123")
            
            context = await cortex.build_llm_context(
                query="What is Python?",
                max_tokens=1000
            )
            
            # Use in prompt
            prompt = f"{context['formatted']}\n\nUser: {query}"
            ```
        """
        if not self._initialized:
            await self.initialize()
        
        if self._llm_context_builder is None:
            self._llm_context_builder = LLMContextBuilder(self, model)
        
        return await self._llm_context_builder.build_context(
            query=query,
            max_tokens=max_tokens,
            strategy=strategy
        )
    
    async def summarize_conversation(
        self,
        episode_ids: Optional[List[str]] = None,
        target_tokens: int = 500,
        method: str = "progressive"
    ) -> str:
        """
        Summarize conversation history.
        
        Convenience method for ConversationSummarizer.
        
        Args:
            episode_ids: Specific episode IDs (None = recent episodes)
            target_tokens: Target token count for summary
            method: Summarization method (extractive, progressive, hierarchical)
        
        Returns:
            Summary text
        
        Example:
            ```python
            cortex = await create_cortex(agent_id="agent-123")
            
            # Summarize recent conversation
            summary = await cortex.summarize_conversation(
                target_tokens=300,
                method="progressive"
            )
            ```
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.episodic:
            return ""
        
        # Get episodes
        if episode_ids:
            # Get specific episodes
            episodes = []
            for ep_id in episode_ids:
                ep = await self.episodic.get(ep_id)
                if ep:
                    episodes.append(ep)
        else:
            # Get recent episodes
            episodes = await self.episodic.get_recent(limit=50)
        
        if not episodes:
            return ""
        
        summarizer = self.get_conversation_summarizer()
        return await summarizer.summarize_conversation(
            episodes=episodes,
            target_tokens=target_tokens,
            method=method
        )
    
    def get_rag_memory(
        self,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> RAGMemory:
        """
        Get RAG memory interface.
        
        Args:
            chunk_size: Chunk size in tokens
            overlap: Overlap between chunks in tokens
        
        Returns:
            RAGMemory instance
        
        Example:
            ```python
            cortex = await create_cortex(agent_id="agent-123")
            rag = cortex.get_rag_memory()
            
            # Store document
            await rag.store_document("Long document text...")
            
            # Retrieve for query
            context = await rag.retrieve_for_query("query", num_chunks=3)
            ```
        """
        if self._rag_memory is None:
            self._rag_memory = RAGMemory(self, chunk_size, overlap)
        return self._rag_memory
    
    async def store_document(
        self,
        document: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
        chunking_strategy: str = "semantic"
    ) -> List[str]:
        """
        Store document in RAG memory.
        
        Convenience method for document storage.
        
        Args:
            document: Document text
            metadata: Document metadata
            chunk_size: Chunk size (None = use default)
            chunking_strategy: Chunking strategy
        
        Returns:
            List of chunk IDs
        
        Example:
            ```python
            cortex = await create_cortex(agent_id="agent-123")
            
            chunk_ids = await cortex.store_document(
                document=long_text,
                metadata={"source": "manual.pdf", "topic": "python"}
            )
            ```
        """
        if not self._initialized:
            await self.initialize()
        
        rag = self.get_rag_memory()
        return await rag.store_document(
            document=document,
            metadata=metadata,
            chunk_size=chunk_size,
            chunking_strategy=chunking_strategy
        )
    
    async def retrieve_rag_context(
        self,
        query: str,
        num_chunks: int = 5,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Retrieve RAG context for query.
        
        Convenience method for RAG retrieval.
        
        Args:
            query: Search query
            num_chunks: Number of chunks to retrieve
            max_tokens: Maximum tokens for result
        
        Returns:
            Fused context string
        
        Example:
            ```python
            cortex = await create_cortex(agent_id="agent-123")
            
            context = await cortex.retrieve_rag_context(
                query="Python web frameworks",
                num_chunks=3
            )
            ```
        """
        if not self._initialized:
            await self.initialize()
        
        rag = self.get_rag_memory()
        return await rag.retrieve_for_query(
            query=query,
            num_chunks=num_chunks,
            max_tokens=max_tokens
        )
    
    # Advanced Memory Methods (Phase 3 & 4)
    
    async def check_memory_health(self) -> Dict[str, Any]:
        """
        Check memory system health.
        
        Returns:
            Health report with status and recommendations
        
        Example:
            ```python
            cortex = await create_cortex(agent_id="agent-123")
            health = await cortex.check_memory_health()
            
            if health['status'] == 'warning':
                print(f"Issues: {health['issues']}")
            ```
        """
        if not self._initialized:
            await self.initialize()
        
        if self._health_monitor is None:
            self._health_monitor = MemoryHealthMonitor(
                embedding_function=self.config.embedding_function
            )
        
        return await self._health_monitor.check_health(self)
    
    async def consolidate_memories(
        self,
        similarity_threshold: float = 0.8
    ) -> Dict[str, int]:
        """
        Consolidate similar memories.
        
        Args:
            similarity_threshold: Threshold for similarity
        
        Returns:
            Consolidation statistics
        
        Example:
            ```python
            cortex = await create_cortex(agent_id="agent-123")
            stats = await cortex.consolidate_memories()
            print(f"Consolidated {stats['clusters']} memory clusters")
            ```
        """
        if not self._initialized:
            await self.initialize()
        
        if self._memory_consolidator is None:
            self._memory_consolidator = MemoryConsolidator(
                similarity_threshold=similarity_threshold,
                embedding_function=self.config.embedding_function
            )
        
        stats = {"clusters": 0, "episodes_consolidated": 0}
        
        if self.episodic:
            episodes = await self.episodic.get_recent(limit=100)
            clusters = self._memory_consolidator.cluster_episodes(episodes)
            
            stats["clusters"] = len(clusters)
            
            for cluster in clusters:
                stats["episodes_consolidated"] += len(cluster)
        
        return stats
    
    async def learn_patterns(
        self,
        min_occurrences: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Learn patterns from recent interactions.
        
        Args:
            min_occurrences: Minimum pattern occurrences
        
        Returns:
            List of learned patterns
        
        Example:
            ```python
            cortex = await create_cortex(agent_id="agent-123")
            patterns = await cortex.learn_patterns(min_occurrences=3)
            
            for pattern in patterns:
                print(f"Pattern: {pattern['input_pattern']}")
                print(f"Success rate: {pattern['success_rate']}")
            ```
        """
        if not self._initialized:
            await self.initialize()
        
        if self._pattern_learner is None:
            self._pattern_learner = PatternLearner()
        
        if not self.episodic:
            return []
        
        episodes = await self.episodic.get_recent(limit=100)
        return await self._pattern_learner.learn_from_episodes(
            episodes,
            min_occurrences=min_occurrences
        )
    
    async def analyze_performance(
        self,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Analyze recent performance.
        
        Args:
            time_window_hours: Time window for analysis
        
        Returns:
            Performance analysis
        
        Example:
            ```python
            cortex = await create_cortex(agent_id="agent-123")
            analysis = await cortex.analyze_performance(time_window_hours=24)
            
            print(f"Success rate: {analysis['success_rate']}")
            print(f"Avg duration: {analysis['avg_duration_ms']}ms")
            ```
        """
        if not self._initialized:
            await self.initialize()
        
        if self._success_analyzer is None:
            self._success_analyzer = SuccessAnalyzer()
        
        if not self.episodic:
            return {"success_rate": 0.0, "total_episodes": 0}
        
        episodes = await self.episodic.get_recent(limit=200)
        return await self._success_analyzer.analyze_performance(
            episodes,
            time_window_hours=time_window_hours
        )
    
    async def optimize_memory(
        self,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Run automatic memory optimization.
        
        Args:
            force: Force optimization regardless of schedule
        
        Returns:
            Optimization actions taken
        
        Example:
            ```python
            cortex = await create_cortex(agent_id="agent-123")
            result = await cortex.optimize_memory()
            
            print(f"Optimizations: {result['optimizations']}")
            ```
        """
        if not self._initialized:
            await self.initialize()
        
        if self._adaptive_manager is None:
            self._adaptive_manager = AdaptiveMemoryManager()
        
        return await self._adaptive_manager.optimize_memory(self, force=force)
    
    async def cleanup_old_memories(
        self,
        max_age_days: int = 90,
        min_importance: float = 0.3
    ) -> Dict[str, int]:
        """
        Clean up old or unimportant memories.
        
        Args:
            max_age_days: Maximum age for memories
            min_importance: Minimum importance to keep
        
        Returns:
            Cleanup statistics
        
        Example:
            ```python
            cortex = await create_cortex(agent_id="agent-123")
            stats = await cortex.cleanup_old_memories(max_age_days=90)
            
            print(f"Deleted {stats['episodes_deleted']} old episodes")
            ```
        """
        if not self._initialized:
            await self.initialize()
        
        forgetfulness = AutomaticForgetfulness(
            max_age_days=max_age_days,
            min_importance=min_importance
        )
        
        stats = {"episodes_deleted": 0, "knowledge_deleted": 0}
        
        # Clean up episodes
        if self.episodic:
            episodes = await self.episodic.get_recent(limit=500)
            to_forget = await forgetfulness.identify_forgettable_episodes(episodes)
            
            for episode_id in to_forget:
                await self.episodic.delete(episode_id)
                stats["episodes_deleted"] += 1
        
        # Clean up knowledge
        if self.semantic:
            # Get all entries by listing keys (more efficient than empty search)
            pattern = f"{self.semantic._key_prefix}:entry:*"
            entry_keys = await self.semantic.storage.list_keys(pattern, limit=500)
            
            # Extract entry IDs from keys
            entry_ids = [key.split(":")[-1] for key in entry_keys if ":entry:" in key]
            
            if entry_ids:
                # Batch fetch entries
                entries = await self.semantic.get_batch(entry_ids)
            
            to_forget = await forgetfulness.identify_forgettable_knowledge(entries)
            
            for entry_id in to_forget:
                await self.semantic.delete(entry_id)
                stats["knowledge_deleted"] += 1
        
        return stats


# Convenience function for quick setup
async def create_cortex(
    agent_id: str,
    session_id: Optional[str] = None,
    storage_backend: str = "memory",
    config: Optional[CortexConfig] = None,
    enable_chromadb: Optional[bool] = None
) -> CortexMemory:
    """
    Create and initialize a Cortex memory system.
    
    Args:
        agent_id: ID of the agent
        session_id: Optional session ID
        storage_backend: Storage backend type ("memory", "redis", "postgres", "chroma")
        config: Optional Cortex configuration
        enable_chromadb: Enable ChromaDB for semantic memory (auto-detected if None)
    
    Returns:
        Initialized CortexMemory instance
    
    Example:
        ```python
        # Basic usage
        cortex = await create_cortex(
            agent_id="agent-123",
            storage_backend="memory"
        )
        
        # Production with ChromaDB for semantic memory
        cortex = await create_cortex(
            agent_id="agent-123",
            storage_backend="redis",
            enable_chromadb=True
        )
        ```
    """
    import os
    from teleon.cortex.storage import create_storage, CHROMADB_AVAILABLE
    
    # Auto-detect ChromaDB usage
    if enable_chromadb is None:
        # Enable ChromaDB if:
        # 1. Running in production (has DEPLOYMENT_ID env var)
        # 2. ChromaDB is available
        deployment_id = os.getenv("DEPLOYMENT_ID")
        enable_chromadb = bool(deployment_id) and CHROMADB_AVAILABLE
    
    # Create primary storage backend
    storage = create_storage(storage_backend)
    await storage.initialize()
    
    # Set up config with ChromaDB embeddings if enabled
    if config is None:
        config = CortexConfig()
    
    # If ChromaDB is enabled, set up embedding function
    if enable_chromadb and config.embedding_function is None:
        try:
            from teleon.cortex.embeddings import create_fastembed_function
            config.embedding_function = create_fastembed_function()
        except ImportError:
            pass  # Will fall back to default embeddings
    
    cortex = CortexMemory(
        storage=storage,
        agent_id=agent_id,
        session_id=session_id,
        config=config
    )
    await cortex.initialize()
    
    return cortex


__all__ = [
    # Main interface
    "CortexMemory",
    "CortexConfig",
    "create_cortex",
    
    # Memory types
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",
    "WorkingMemory",
    
    # Models
    "Episode",
    "KnowledgeEntry",
    "Pattern",
    
    # LLM Integration (Phase 1)
    "TokenCounter",
    "ContextWindowManager",
    "TokenBudgetTracker",
    "LLMContextBuilder",
    "TokenBudgetAllocator",
    "ContextCompressor",
    "PriorityRanker",
    "ConversationSummarizer",
    "MessagePrioritizer",
    
    # RAG Capabilities (Phase 2)
    "RAGMemory",
    "DocumentChunker",
    "ChunkRanker",
    "ContextFusion",
    
    # Advanced Memory (Phase 3)
    "ImportanceScorer",
    "MemoryConsolidator",
    "AutomaticForgetfulness",
    "MemoryHealthMonitor",
    
    # Learning & Optimization (Phase 4)
    "PatternLearner",
    "SuccessAnalyzer",
    "AdaptiveMemoryManager",
    "PerformanceOptimizer",
]
