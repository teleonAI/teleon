"""
Semantic Memory - Knowledge Base with Vector Similarity Search.

Semantic memory stores long-term knowledge with vector embeddings for
similarity-based retrieval. It enables agents to build and query knowledge bases.
"""

import uuid
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field

from teleon.cortex.storage import StorageBackend


class KnowledgeEntry(BaseModel):
    """
    A piece of knowledge stored in semantic memory.
    
    Knowledge entries contain content, embeddings, and metadata for
    similarity-based retrieval.
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., description="The knowledge content")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of content")
    
    # Metadata
    source: Optional[str] = Field(None, description="Source of this knowledge")
    category: Optional[str] = Field(None, description="Knowledge category")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Usage tracking
    access_count: int = Field(0, description="Number of times accessed")
    last_accessed: Optional[datetime] = Field(None, description="Last access time")
    
    # Temporal information
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Importance scoring
    importance_score: float = Field(1.0, description="Importance score (0-1)")
    confidence_score: float = Field(1.0, description="Confidence in this knowledge (0-1)")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def content_hash(self) -> str:
        """Generate hash of content for deduplication."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


class SemanticMemory:
    """
    Semantic memory for long-term knowledge storage with vector search.
    
    Features:
    - Store knowledge with vector embeddings
    - Similarity-based retrieval
    - Automatic deduplication
    - Importance and confidence scoring
    - Category and tag organization
    - Usage tracking
    
    Example:
        ```python
        semantic = SemanticMemory(storage, agent_id="agent-123")
        await semantic.initialize()
        
        # Store knowledge (embedding generated automatically)
        entry_id = await semantic.store(
            content="Paris is the capital of France",
            category="geography",
            tags=["europe", "capitals"]
        )
        
        # Search for similar knowledge
        results = await semantic.search("What is the capital of France?", limit=5)
        
        # Get by category
        geography = await semantic.get_by_category("geography")
        
        # Get by tag
        europe = await semantic.get_by_tag("europe")
        ```
    
    Note:
        This implementation uses simple cosine similarity with in-memory vectors.
        For production, consider ChromaDB, Pinecone, or Weaviate for better performance.
    """
    
    def __init__(
        self,
        storage: StorageBackend,
        agent_id: str,
        embedding_function: Optional[callable] = None,
        ttl: Optional[int] = None
    ):
        """
        Initialize semantic memory.
        
        Args:
            storage: Storage backend to use
            agent_id: ID of the agent using this memory
            embedding_function: Optional function to generate embeddings
            ttl: Time-to-live for knowledge in seconds (None = no expiration)
        """
        self.storage = storage
        self.agent_id = agent_id
        self.embedding_function = embedding_function or self._default_embedding
        self.ttl = ttl
        self._key_prefix = f"semantic:{agent_id}"
    
    async def initialize(self) -> None:
        """Initialize semantic memory."""
        if not self.storage._initialized:
            await self.storage.initialize()
    
    def _default_embedding(self, text: str) -> List[float]:
        """
        Default simple embedding function (TF-IDF inspired).
        
        For production, replace with:
        - OpenAI embeddings
        - Sentence transformers
        - Custom embedding models
        
        Args:
            text: Text to embed
        
        Returns:
            Simple embedding vector
        """
        # Simple word-based embedding (128 dimensions)
        # This is a placeholder - use real embeddings in production!
        words = text.lower().split()
        vector = [0.0] * 128
        
        for i, word in enumerate(words[:128]):
            # Simple hash-based embedding
            hash_val = abs(hash(word))
            idx = hash_val % 128
            vector[idx] += 1.0 / (i + 1)  # Weight by position
        
        # Normalize
        magnitude = sum(x*x for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x / magnitude for x in vector]
        
        return vector
    
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
    
    async def store(
        self,
        content: str,
        source: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance_score: float = 1.0,
        confidence_score: float = 1.0
    ) -> str:
        """
        Store knowledge in semantic memory.
        
        Args:
            content: Knowledge content
            source: Source of knowledge
            category: Knowledge category
            tags: Tags for organization
            metadata: Additional metadata
            importance_score: Importance score (0-1)
            confidence_score: Confidence score (0-1)
        
        Returns:
            Knowledge entry ID
        """
        # Check for duplicates based on content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        duplicate_key = f"{self._key_prefix}:hash:{content_hash}"
        
        existing_id = await self.storage.get(duplicate_key)
        if existing_id:
            # Update existing entry
            existing = await self.get(existing_id)
            if existing:
                existing.access_count += 1
                existing.updated_at = datetime.utcnow()
                # Merge metadata
                if metadata:
                    existing.metadata.update(metadata)
                if tags:
                    existing.tags = list(set(existing.tags + tags))
                
                await self._update_entry(existing)
                return existing_id
        
        # Generate embedding
        embedding = self.embedding_function(content)
        
        # Create knowledge entry
        entry = KnowledgeEntry(
            content=content,
            embedding=embedding,
            source=source,
            category=category,
            tags=tags or [],
            metadata=metadata or {},
            importance_score=importance_score,
            confidence_score=confidence_score
        )
        
        # Store entry
        key = f"{self._key_prefix}:entry:{entry.id}"
        await self.storage.set(
            key,
            entry.dict(),
            ttl=self.ttl,
            metadata={
                "type": "knowledge",
                "agent_id": self.agent_id,
                "category": category,
                "importance": importance_score,
            }
        )
        
        # Store content hash for deduplication
        await self.storage.set(duplicate_key, entry.id, ttl=self.ttl)
        
        # Index by category
        if category:
            cat_key = f"{self._key_prefix}:category:{category}:{entry.id}"
            await self.storage.set(cat_key, entry.id, ttl=self.ttl)
        
        # Index by tags
        for tag in (tags or []):
            tag_key = f"{self._key_prefix}:tag:{tag}:{entry.id}"
            await self.storage.set(tag_key, entry.id, ttl=self.ttl)
        
        # Store embedding vector for search
        vector_key = f"{self._key_prefix}:vector:{entry.id}"
        await self.storage.set(vector_key, embedding, ttl=self.ttl)
        
        return entry.id
    
    async def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """
        Get a knowledge entry by ID.
        
        Args:
            entry_id: Entry ID
        
        Returns:
            Knowledge entry if found
        """
        key = f"{self._key_prefix}:entry:{entry_id}"
        data = await self.storage.get(key)
        
        if data is None:
            return None
        
        entry = KnowledgeEntry(**data)
        
        # Update access tracking
        entry.access_count += 1
        entry.last_accessed = datetime.utcnow()
        await self._update_entry(entry)
        
        return entry
    
    async def _update_entry(self, entry: KnowledgeEntry) -> None:
        """Update an existing entry."""
        key = f"{self._key_prefix}:entry:{entry.id}"
        await self.storage.set(key, entry.dict(), ttl=self.ttl)
    
    async def search(
        self,
        query: str,
        limit: int = 5,
        category: Optional[str] = None,
        min_similarity: float = 0.5
    ) -> List[Tuple[KnowledgeEntry, float]]:
        """
        Search for similar knowledge using vector similarity.
        
        Args:
            query: Search query
            limit: Maximum number of results
            category: Filter by category
            min_similarity: Minimum similarity score (0-1)
        
        Returns:
            List of (entry, similarity_score) tuples, sorted by similarity
        """
        # Generate query embedding
        query_embedding = self.embedding_function(query)
        
        # Get candidate entries
        if category:
            pattern = f"{self._key_prefix}:category:{category}:*"
            cat_keys = await self.storage.list_keys(pattern)
            entry_ids = []
            for key in cat_keys:
                entry_id = await self.storage.get(key)
                if entry_id:
                    entry_ids.append(entry_id)
        else:
            pattern = f"{self._key_prefix}:vector:*"
            vector_keys = await self.storage.list_keys(pattern)
            entry_ids = [key.split(":")[-1] for key in vector_keys]
        
        # Calculate similarities
        results = []
        for entry_id in entry_ids:
            # Get entry embedding
            vector_key = f"{self._key_prefix}:vector:{entry_id}"
            embedding = await self.storage.get(vector_key)
            
            if embedding:
                # Calculate similarity
                similarity = self._cosine_similarity(query_embedding, embedding)
                
                if similarity >= min_similarity:
                    # Get full entry
                    entry = await self.get(entry_id)
                    if entry:
                        results.append((entry, similarity))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:limit]
    
    async def get_by_category(
        self,
        category: str,
        limit: Optional[int] = None
    ) -> List[KnowledgeEntry]:
        """
        Get all knowledge entries in a category.
        
        Args:
            category: Category name
            limit: Maximum number of entries
        
        Returns:
            List of knowledge entries
        """
        pattern = f"{self._key_prefix}:category:{category}:*"
        cat_keys = await self.storage.list_keys(pattern, limit=limit)
        
        entries = []
        for key in cat_keys:
            entry_id = await self.storage.get(key)
            if entry_id:
                entry = await self.get(entry_id)
                if entry:
                    entries.append(entry)
        
        return entries
    
    async def get_by_tag(
        self,
        tag: str,
        limit: Optional[int] = None
    ) -> List[KnowledgeEntry]:
        """
        Get all knowledge entries with a specific tag.
        
        Args:
            tag: Tag name
            limit: Maximum number of entries
        
        Returns:
            List of knowledge entries
        """
        pattern = f"{self._key_prefix}:tag:{tag}:*"
        tag_keys = await self.storage.list_keys(pattern, limit=limit)
        
        entries = []
        for key in tag_keys:
            entry_id = await self.storage.get(key)
            if entry_id:
                entry = await self.get(entry_id)
                if entry:
                    entries.append(entry)
        
        return entries
    
    async def delete(self, entry_id: str) -> bool:
        """
        Delete a knowledge entry.
        
        Args:
            entry_id: Entry ID
        
        Returns:
            True if deleted
        """
        # Get entry first to clean up indexes
        entry = await self.get(entry_id)
        if not entry:
            return False
        
        # Delete main entry
        key = f"{self._key_prefix}:entry:{entry_id}"
        await self.storage.delete(key)
        
        # Delete vector
        vector_key = f"{self._key_prefix}:vector:{entry_id}"
        await self.storage.delete(vector_key)
        
        # Delete content hash
        content_hash = entry.content_hash()
        hash_key = f"{self._key_prefix}:hash:{content_hash}"
        await self.storage.delete(hash_key)
        
        # Delete from category index
        if entry.category:
            cat_key = f"{self._key_prefix}:category:{entry.category}:{entry_id}"
            await self.storage.delete(cat_key)
        
        # Delete from tag indexes
        for tag in entry.tags:
            tag_key = f"{self._key_prefix}:tag:{tag}:{entry_id}"
            await self.storage.delete(tag_key)
        
        return True
    
    async def clear(self) -> int:
        """
        Clear all knowledge for this agent.
        
        Returns:
            Number of entries deleted
        """
        pattern = f"{self._key_prefix}:*"
        return await self.storage.clear(pattern)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored knowledge.
        
        Returns:
            Dictionary with statistics
        """
        # Count entries
        entry_pattern = f"{self._key_prefix}:entry:*"
        entry_keys = await self.storage.list_keys(entry_pattern)
        total_entries = len(entry_keys)
        
        if total_entries == 0:
            return {
                "total_entries": 0,
                "categories": [],
                "tags": [],
                "avg_importance": 0,
                "avg_confidence": 0,
            }
        
        # Get sample for analysis
        sample_size = min(100, total_entries)
        sample_keys = entry_keys[:sample_size]
        
        entries = []
        for key in sample_keys:
            data = await self.storage.get(key)
            if data:
                entries.append(KnowledgeEntry(**data))
        
        # Collect statistics
        categories = set()
        tags = set()
        total_importance = 0
        total_confidence = 0
        
        for entry in entries:
            if entry.category:
                categories.add(entry.category)
            tags.update(entry.tags)
            total_importance += entry.importance_score
            total_confidence += entry.confidence_score
        
        return {
            "total_entries": total_entries,
            "categories": list(categories),
            "tags": list(tags),
            "avg_importance": round(total_importance / len(entries), 2) if entries else 0,
            "avg_confidence": round(total_confidence / len(entries), 2) if entries else 0,
            "most_accessed": max(entries, key=lambda e: e.access_count) if entries else None,
        }

