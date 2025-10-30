"""Semantic memory - vector-based knowledge storage."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import hashlib


class KnowledgeItem(BaseModel):
    """A piece of knowledge."""
    
    id: str = Field(..., description="Knowledge ID")
    content: str = Field(..., description="Content/knowledge")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
    tags: List[str] = Field(default_factory=list, description="Tags")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Creation time")
    source: Optional[str] = Field(None, description="Source of knowledge")
    confidence: float = Field(1.0, description="Confidence score (0-1)")


class SemanticMemory:
    """
    Semantic memory system with vector search.
    
    Features:
    - Store knowledge with embeddings
    - Semantic search (similarity)
    - Knowledge consolidation
    - Source tracking
    """
    
    def __init__(
        self,
        embedding_dim: int = 1536,  # OpenAI ada-002 dimension
        max_items: int = 10000
    ):
        """
        Initialize semantic memory.
        
        Args:
            embedding_dim: Embedding vector dimension
            max_items: Maximum knowledge items
        """
        self.embedding_dim = embedding_dim
        self.max_items = max_items
        self.knowledge: Dict[str, KnowledgeItem] = {}
        
        # Indices
        self._by_tag: Dict[str, List[str]] = {}
    
    async def store(
        self,
        content: str,
        embedding: Optional[List[float]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        confidence: float = 1.0,
        auto_embed: bool = False
    ) -> KnowledgeItem:
        """
        Store knowledge.
        
        Args:
            content: Knowledge content
            embedding: Vector embedding (optional)
            tags: Tags
            metadata: Additional metadata
            source: Source of knowledge
            confidence: Confidence score
            auto_embed: Auto-generate embedding if None
        
        Returns:
            Created knowledge item
        """
        # Generate ID from content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        # Auto-generate embedding if requested
        if auto_embed and embedding is None:
            embedding = await self._generate_embedding(content)
        
        item = KnowledgeItem(
            id=content_hash,
            content=content,
            embedding=embedding,
            tags=tags or [],
            metadata=metadata or {},
            source=source,
            confidence=confidence
        )
        
        # Store
        self.knowledge[item.id] = item
        
        # Update indices
        for tag in item.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = []
            if item.id not in self._by_tag[tag]:
                self._by_tag[tag].append(item.id)
        
        # Cleanup if needed
        if len(self.knowledge) > self.max_items:
            await self._cleanup()
        
        return item
    
    async def retrieve_by_id(self, knowledge_id: str) -> Optional[KnowledgeItem]:
        """Retrieve knowledge by ID."""
        return self.knowledge.get(knowledge_id)
    
    async def retrieve_by_tags(
        self,
        tags: List[str],
        match_all: bool = False,
        limit: int = 10
    ) -> List[KnowledgeItem]:
        """
        Retrieve knowledge by tags.
        
        Args:
            tags: Tags to search for
            match_all: If True, must have all tags
            limit: Maximum to retrieve
        
        Returns:
            List of knowledge items
        """
        if match_all:
            # Items with all tags
            item_sets = [
                set(self._by_tag.get(tag, []))
                for tag in tags
            ]
            if not item_sets:
                return []
            
            item_ids = item_sets[0].intersection(*item_sets[1:])
        else:
            # Items with any tag
            item_ids = set()
            for tag in tags:
                item_ids.update(self._by_tag.get(tag, []))
        
        # Get items
        items = [self.knowledge[id] for id in item_ids if id in self.knowledge]
        
        # Sort by timestamp
        items = sorted(items, key=lambda x: x.timestamp, reverse=True)
        
        return items[:limit]
    
    async def semantic_search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        limit: int = 5,
        min_similarity: float = 0.7,
        auto_embed: bool = False
    ) -> List[tuple[KnowledgeItem, float]]:
        """
        Semantic search using vector similarity.
        
        Args:
            query: Search query
            query_embedding: Query embedding (optional)
            limit: Maximum results
            min_similarity: Minimum similarity threshold
            auto_embed: Auto-generate query embedding
        
        Returns:
            List of (item, similarity_score) tuples
        """
        # Auto-generate embedding if requested
        if auto_embed and query_embedding is None:
            query_embedding = await self._generate_embedding(query)
        
        if query_embedding is None:
            # Fall back to text search
            return await self._text_search(query, limit)
        
        # Calculate similarities
        results = []
        for item in self.knowledge.values():
            if item.embedding is None:
                continue
            
            similarity = self._cosine_similarity(query_embedding, item.embedding)
            
            if similarity >= min_similarity:
                results.append((item, similarity))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:limit]
    
    async def text_search(
        self,
        query: str,
        limit: int = 10
    ) -> List[KnowledgeItem]:
        """
        Simple text search.
        
        Args:
            query: Search query
            limit: Maximum results
        
        Returns:
            List of knowledge items
        """
        query_lower = query.lower()
        
        matching = [
            item for item in self.knowledge.values()
            if query_lower in item.content.lower() or
               any(query_lower in tag.lower() for tag in item.tags)
        ]
        
        # Sort by timestamp
        matching = sorted(matching, key=lambda x: x.timestamp, reverse=True)
        
        return matching[:limit]
    
    async def consolidate_knowledge(
        self,
        similarity_threshold: float = 0.95
    ) -> int:
        """
        Consolidate similar knowledge items.
        
        Args:
            similarity_threshold: Similarity threshold for merging
        
        Returns:
            Number of items consolidated
        """
        consolidated = 0
        items_to_remove = set()
        
        items = list(self.knowledge.values())
        
        for i, item1 in enumerate(items):
            if item1.id in items_to_remove:
                continue
            
            if item1.embedding is None:
                continue
            
            for item2 in items[i+1:]:
                if item2.id in items_to_remove:
                    continue
                
                if item2.embedding is None:
                    continue
                
                # Check similarity
                similarity = self._cosine_similarity(item1.embedding, item2.embedding)
                
                if similarity >= similarity_threshold:
                    # Merge item2 into item1
                    # Keep higher confidence
                    if item2.confidence > item1.confidence:
                        item1.confidence = item2.confidence
                    
                    # Merge tags
                    item1.tags = list(set(item1.tags + item2.tags))
                    
                    # Merge metadata
                    item1.metadata.update(item2.metadata)
                    
                    # Mark for removal
                    items_to_remove.add(item2.id)
                    consolidated += 1
        
        # Remove consolidated items
        for item_id in items_to_remove:
            if item_id in self.knowledge:
                del self.knowledge[item_id]
        
        # Rebuild indices
        await self._rebuild_indices()
        
        return consolidated
    
    async def update(
        self,
        knowledge_id: str,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None
    ) -> Optional[KnowledgeItem]:
        """Update knowledge item."""
        item = self.knowledge.get(knowledge_id)
        if not item:
            return None
        
        if content is not None:
            item.content = content
            # Re-generate embedding
            item.embedding = await self._generate_embedding(content)
        
        if tags is not None:
            # Remove old tag references
            for tag in item.tags:
                if tag in self._by_tag and item.id in self._by_tag[tag]:
                    self._by_tag[tag].remove(item.id)
            
            item.tags = tags
            
            # Add new tag references
            for tag in tags:
                if tag not in self._by_tag:
                    self._by_tag[tag] = []
                if item.id not in self._by_tag[tag]:
                    self._by_tag[tag].append(item.id)
        
        if metadata is not None:
            item.metadata.update(metadata)
        
        if confidence is not None:
            item.confidence = confidence
        
        return item
    
    async def delete(self, knowledge_id: str) -> bool:
        """Delete knowledge item."""
        item = self.knowledge.get(knowledge_id)
        if not item:
            return False
        
        # Remove from indices
        for tag in item.tags:
            if tag in self._by_tag and item.id in self._by_tag[tag]:
                self._by_tag[tag].remove(item.id)
        
        # Delete
        del self.knowledge[knowledge_id]
        return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        items_with_embeddings = sum(
            1 for item in self.knowledge.values()
            if item.embedding is not None
        )
        
        return {
            "total_items": len(self.knowledge),
            "items_with_embeddings": items_with_embeddings,
            "unique_tags": len(self._by_tag),
            "max_items": self.max_items,
            "embedding_dim": self.embedding_dim
        }
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        This is a placeholder. In production, you'd use:
        - OpenAI embeddings API
        - Sentence transformers
        - Other embedding models
        """
        # Placeholder: return zero vector
        # In production, replace with actual embedding generation
        import random
        random.seed(hash(text) % (2**32))
        return [random.random() for _ in range(self.embedding_dim)]
    
    async def _text_search(
        self,
        query: str,
        limit: int
    ) -> List[tuple[KnowledgeItem, float]]:
        """Fallback text search with scores."""
        items = await self.text_search(query, limit)
        # Return with dummy scores
        return [(item, 0.8) for item in items]
    
    async def _cleanup(self):
        """Clean up old/low-confidence items."""
        # Sort by confidence and timestamp
        items = sorted(
            self.knowledge.values(),
            key=lambda x: (x.confidence, x.timestamp),
            reverse=True
        )
        
        # Keep top items
        keep_items = items[:self.max_items]
        
        # Rebuild knowledge dict
        self.knowledge = {item.id: item for item in keep_items}
        
        # Rebuild indices
        await self._rebuild_indices()
    
    async def _rebuild_indices(self):
        """Rebuild tag indices."""
        self._by_tag = {}
        
        for item in self.knowledge.values():
            for tag in item.tags:
                if tag not in self._by_tag:
                    self._by_tag[tag] = []
                if item.id not in self._by_tag[tag]:
                    self._by_tag[tag].append(item.id)
    
    async def clear(self):
        """Clear all knowledge."""
        self.knowledge = {}
        self._by_tag = {}

