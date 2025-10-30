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

from typing import Optional, Dict, Any, Callable
from pydantic import BaseModel, Field, SkipValidation

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
    
    class Config:
        arbitrary_types_allowed = True


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
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all enabled memory types."""
        if self._initialized:
            return
        
        # Initialize storage if needed
        if not self.storage._initialized:
            await self.storage.initialize()
        
        # Initialize working memory
        if self.config.working_enabled and WORKING_MEMORY_AVAILABLE:
            self.working = WorkingMemory(
                session_id=self.session_id or self.agent_id,
                ttl=self.config.working_ttl
            )
        
        # Initialize episodic memory
        if self.config.episodic_enabled:
            self.episodic = EpisodicMemory(
                storage=self.storage,
                agent_id=self.agent_id,
                ttl=self.config.episodic_ttl
            )
            await self.episodic.initialize()
        
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
                    # ChromaDB handles embeddings internally
                except Exception as e:
                    # Fall back to regular storage
                    import logging
                    logger = logging.getLogger("teleon.cortex")
                    logger.warning(f"Failed to initialize ChromaDB, using regular storage: {e}")
            
            self.semantic = SemanticMemory(
                storage=semantic_storage,
                agent_id=self.agent_id,
                embedding_function=self.config.embedding_function,
                ttl=self.config.semantic_ttl
            )
            await self.semantic.initialize()
        
        # Initialize procedural memory
        if self.config.procedural_enabled:
            self.procedural = ProceduralMemory(
                storage=self.storage,
                agent_id=self.agent_id,
                min_success_rate=self.config.min_success_rate,
                ttl=self.config.procedural_ttl
            )
            await self.procedural.initialize()
        
        self._initialized = True
    
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
        
        # Store in episodic memory
        episode_id = None
        if self.episodic:
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
        
        # Learn pattern if enabled
        if self.config.learning_enabled and self.procedural and success:
            # Extract simple pattern (in production, use better pattern extraction)
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
            context["recent_episodes"] = [ep.dict() for ep in recent]
        
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
                context["suggested_pattern"] = pattern.dict()
        
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
        Get statistics from all memory types.
        
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
        
        return stats
    
    async def shutdown(self) -> None:
        """Shutdown Cortex and cleanup resources."""
        if self.storage:
            await self.storage.shutdown()
        
        self._initialized = False


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
]

