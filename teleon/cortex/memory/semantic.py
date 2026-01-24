"""
Semantic Memory - Knowledge Base with Vector Similarity Search.

Semantic memory stores long-term knowledge with vector embeddings for
similarity-based retrieval. It enables agents to build and query knowledge bases.

This implementation supports both:
- Key-Value storage backends (with in-memory vector search)
- Vector storage backends (ChromaDB, etc. with native vector search)
"""

import uuid
import hashlib
import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict, field_serializer
import logging

from teleon.cortex.storage.base import StorageBackend
from teleon.cortex.storage.vector_base import VectorStorageBackend, VectorSearchResult
from teleon.cortex.utils import AsyncLRUCache, validate_content, validate_limit

logger = logging.getLogger("teleon.cortex.semantic")


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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Importance scoring
    importance_score: float = Field(1.0, description="Importance score (0-1)")
    confidence_score: float = Field(1.0, description="Confidence in this knowledge (0-1)")

    model_config = ConfigDict()

    @field_serializer('last_accessed', 'created_at', 'updated_at')
    def serialize_datetime(self, value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None
    
    def content_hash(self) -> str:
        """Generate hash of content for deduplication."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]
    
    @classmethod
    def from_vector_result(cls, result: VectorSearchResult) -> "KnowledgeEntry":
        """Create KnowledgeEntry from VectorSearchResult."""
        metadata = result.metadata or {}
        return cls(
            id=result.id,
            content=result.content,
            embedding=result.embedding,
            source=metadata.get("source"),
            category=metadata.get("category"),
            tags=metadata.get("tags", []) if isinstance(metadata.get("tags"), list) else [],
            metadata=metadata,
            access_count=metadata.get("access_count", 0),
            importance_score=metadata.get("importance_score", 1.0),
            confidence_score=metadata.get("confidence_score", 1.0),
        )


class SemanticMemory:
    """
    Semantic memory for long-term knowledge storage with vector search.
    
    Supports two storage modes:
    1. Key-Value Storage: Uses in-memory vector search (suitable for small datasets)
    2. Vector Storage: Uses native vector database (recommended for production)
    
    Features:
    - Store knowledge with vector embeddings
    - Similarity-based retrieval
    - Automatic deduplication
    - Importance and confidence scoring
    - Category and tag organization
    - Usage tracking
    - Batch operations for efficiency
    
    Example:
        ```python
        # With Vector Storage (recommended for production)
        from teleon.cortex.storage import create_vector_storage
        
        vector_storage = create_vector_storage("chroma", deployment_id="agent-123")
        await vector_storage.initialize()
        
        semantic = SemanticMemory(
            storage=vector_storage,
            agent_id="agent-123",
            embedding_function=embed_fn
        )
        await semantic.initialize()
        
        # Store knowledge
        entry_id = await semantic.store(
            content="Paris is the capital of France",
            category="geography",
            tags=["europe", "capitals"]
        )
        
        # Search for similar knowledge
        results = await semantic.search("What is the capital of France?", limit=5)
        ```
    """
    
    def __init__(
        self,
        storage: Union[StorageBackend, VectorStorageBackend],
        agent_id: str,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        ttl: Optional[int] = None,
        collection_name: str = "semantic"
    ):
        """
        Initialize semantic memory.
        
        Args:
            storage: Storage backend (key-value or vector)
            agent_id: ID of the agent using this memory
            embedding_function: Function to generate embeddings
            ttl: Time-to-live for knowledge in seconds (None = no expiration)
            collection_name: Collection name for vector storage
        """
        self.storage = storage
        self.agent_id = agent_id
        self.ttl = ttl
        self.collection_name = collection_name
        
        # Determine storage type
        self._is_vector_storage = isinstance(storage, VectorStorageBackend)
        
        # Set up embedding function
        if embedding_function:
            self.embedding_function = embedding_function
        else:
            # Try to get embedding function from vector storage
            if self._is_vector_storage and hasattr(storage, 'embedding_function'):
                self.embedding_function = storage.embedding_function
            else:
                self.embedding_function = self._default_embedding
        
        # For key-value storage, we need local vector index
        self._key_prefix = f"semantic:{agent_id}"
        self._local_vectors: Dict[str, List[float]] = {}  # Will be replaced with async cache
        
        # Async-safe caches
        self._vector_cache = AsyncLRUCache(max_size=2000, default_ttl=3600)
        self._entry_cache = AsyncLRUCache(max_size=500, default_ttl=60)
        
        # Statistics cache
        self._stats_cache: Optional[tuple[float, Dict[str, Any]]] = None
        self.max_limit = 10000
    
    async def initialize(self) -> None:
        """Initialize semantic memory."""
        if not self.storage._initialized:
            await self.storage.initialize()
        
        logger.info(
            f"SemanticMemory initialized with "
            f"{'vector' if self._is_vector_storage else 'key-value'} storage"
        )
    
    def _default_embedding(self, text: str) -> List[float]:
        """
        Default simple embedding function (TF-IDF inspired).
        
        For production, replace with:
        - OpenAI embeddings
        - FastEmbed
        - Sentence transformers
        
        Args:
            text: Text to embed
        
        Returns:
            Simple embedding vector
        """
        # Simple word-based embedding (128 dimensions)
        words = text.lower().split()
        vector = [0.0] * 128
        
        for i, word in enumerate(words[:128]):
            hash_val = abs(hash(word))
            idx = hash_val % 128
            vector[idx] += 1.0 / (i + 1)
        
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
        confidence_score: float = 1.0,
        entry_id: Optional[str] = None
    ) -> str:
        """
        Store knowledge in semantic memory with validation.
        
        Args:
            content: Knowledge content
            source: Source of knowledge
            category: Knowledge category
            tags: Tags for organization
            metadata: Additional metadata
            importance_score: Importance score (0-1)
            confidence_score: Confidence score (0-1)
            entry_id: Optional custom ID
        
        Returns:
            Knowledge entry ID
            
        Raises:
            ValueError: If content is invalid
        """
        # Validate content
        content = validate_content(content)
        
        # Validate scores
        if not 0 <= importance_score <= 1:
            raise ValueError("importance_score must be between 0 and 1")
        if not 0 <= confidence_score <= 1:
            raise ValueError("confidence_score must be between 0 and 1")
        
        # Generate embedding with timeout
        try:
            embedding = await asyncio.wait_for(
                asyncio.to_thread(self.embedding_function, content),
                timeout=10.0  # 10 second timeout for embeddings
            )
        except asyncio.TimeoutError:
            logger.warning("Embedding generation timed out, using default embedding")
            embedding = self._default_embedding(content)
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}, using default")
            embedding = self._default_embedding(content)
        
        # Check for duplicates
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        # Prepare metadata
        meta = metadata.copy() if metadata else {}
        meta.update({
            "source": source,
            "category": category,
            "tags": tags or [],
            "importance_score": importance_score,
            "confidence_score": confidence_score,
            "content_hash": content_hash,
            "agent_id": self.agent_id,
            "stored_at": datetime.now(timezone.utc).isoformat(),
            "access_count": 0,
        })
        
        doc_id = entry_id or str(uuid.uuid4())
        
        if self._is_vector_storage:
            # Use vector storage directly
            doc_id = await self.storage.store(
                content=content,
                embedding=embedding,
                metadata=meta,
                document_id=doc_id,
                collection=self.collection_name
            )
        else:
            # Use key-value storage with local vector index
            entry = KnowledgeEntry(
                id=doc_id,
                content=content,
                embedding=embedding,
                source=source,
                category=category,
                tags=tags or [],
                metadata=meta,
                importance_score=importance_score,
                confidence_score=confidence_score
            )
            
            # Store entry
            key = f"{self._key_prefix}:entry:{doc_id}"
            await self.storage.set(
                key,
                entry.dict(),
                ttl=self.ttl,
                metadata={"type": "knowledge", "category": category}
            )
            
            # Store in async vector cache
            await self._vector_cache.set(doc_id, embedding, ttl=3600)
            
            # Store content hash for deduplication
            hash_key = f"{self._key_prefix}:hash:{content_hash}"
            await self.storage.set(hash_key, doc_id, ttl=self.ttl)
            
            # Index by category
            if category:
                cat_key = f"{self._key_prefix}:category:{category}:{doc_id}"
                await self.storage.set(cat_key, doc_id, ttl=self.ttl)
            
            # Index by tags
            for tag in (tags or []):
                tag_key = f"{self._key_prefix}:tag:{tag}:{doc_id}"
                await self.storage.set(tag_key, doc_id, ttl=self.ttl)
        
        logger.debug(f"Stored knowledge entry: {doc_id}")
        return doc_id
    
    async def store_batch(
        self,
        entries: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Store multiple knowledge entries in batch.
        
        Args:
            entries: List of dicts with content, category, tags, metadata, etc.
        
        Returns:
            List of entry IDs
        """
        if not entries:
            return []
        
        if self._is_vector_storage:
            # Prepare batch for vector storage
            documents = []
            for entry in entries:
                content = validate_content(entry["content"])
                try:
                    embedding = await asyncio.wait_for(
                        asyncio.to_thread(self.embedding_function, content),
                        timeout=10.0
                    )
                except (asyncio.TimeoutError, Exception) as e:
                    logger.warning(f"Failed to generate embedding in batch: {e}")
                    embedding = self._default_embedding(content)
                
                meta = entry.get("metadata", {}).copy()
                meta.update({
                    "source": entry.get("source"),
                    "category": entry.get("category"),
                    "tags": entry.get("tags", []),
                    "importance_score": entry.get("importance_score", 1.0),
                    "confidence_score": entry.get("confidence_score", 1.0),
                    "content_hash": hashlib.sha256(content.encode()).hexdigest()[:16],
                    "agent_id": self.agent_id,
                    "stored_at": datetime.now(timezone.utc).isoformat(),
                    "access_count": 0,
                })
                
                documents.append({
                    "id": entry.get("id"),
                    "content": content,
                    "embedding": embedding,
                    "metadata": meta
                })
            
            return await self.storage.store_batch(documents, collection=self.collection_name)
        else:
            # Fall back to individual stores for key-value storage
            ids = []
            for entry in entries:
                doc_id = await self.store(**entry)
                ids.append(doc_id)
            return ids
    
    async def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """
        Get a knowledge entry by ID.
        
        Args:
            entry_id: Entry ID
        
        Returns:
            Knowledge entry if found
        """
        if self._is_vector_storage:
            result = await self.storage.get(entry_id, collection=self.collection_name)
            if result:
                entry = KnowledgeEntry.from_vector_result(result)
                # Update access tracking
                entry.access_count += 1
                entry.last_accessed = datetime.now(timezone.utc)
                return entry
            return None
        else:
            key = f"{self._key_prefix}:entry:{entry_id}"
            data = await self.storage.get(key)
            
            if data is None:
                return None
            
            entry = KnowledgeEntry(**data)
            entry.access_count += 1
            entry.last_accessed = datetime.now(timezone.utc)
            
            # Update access tracking
            await self.storage.set(key, entry.dict(), ttl=self.ttl)
            
            return entry
    
    async def get_batch(self, entry_ids: List[str]) -> List[KnowledgeEntry]:
        """
        Get multiple knowledge entries by ID.
        
        Args:
            entry_ids: List of entry IDs
        
        Returns:
            List of knowledge entries (missing IDs omitted)
        """
        if not entry_ids:
            return []
        
        if self._is_vector_storage:
            results = await self.storage.get_batch(entry_ids, collection=self.collection_name)
            return [KnowledgeEntry.from_vector_result(r) for r in results]
        else:
            # Batch fetch using get_many
            keys = [f"{self._key_prefix}:entry:{eid}" for eid in entry_ids]
            data_map = await self.storage.get_many(keys)
            
            entries = []
            for key, data in data_map.items():
                if data:
                    entries.append(KnowledgeEntry(**data))
            
            return entries
    
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
            limit: Maximum number of results (validated and capped)
            category: Filter by category
            min_similarity: Minimum similarity score (0-1)
        
        Returns:
            List of (entry, similarity_score) tuples, sorted by similarity
            
        Raises:
            ValueError: If query or limit is invalid
        """
        # Validate inputs
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        limit = validate_limit(limit, max_limit=self.max_limit)
        
        if not 0 <= min_similarity <= 1:
            raise ValueError("min_similarity must be between 0 and 1")
        
        # Generate query embedding with timeout
        try:
            query_embedding = await asyncio.wait_for(
                asyncio.to_thread(self.embedding_function, query),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("Query embedding generation timed out")
            query_embedding = self._default_embedding(query)
        except Exception as e:
            logger.warning(f"Failed to generate query embedding: {e}")
            query_embedding = self._default_embedding(query)
        
        if self._is_vector_storage:
            # Use native vector search
            filters = {"category": category} if category else None
            
            results = await self.storage.search(
                query_embedding=query_embedding,
                limit=limit,
                min_similarity=min_similarity,
                filters=filters,
                collection=self.collection_name
            )
            
            return [
                (KnowledgeEntry.from_vector_result(r), r.similarity)
                for r in results
            ]
        else:
            # Use local vector search for key-value storage
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
                entry_ids = list(self._local_vectors.keys())
            
            # Calculate similarities (use async cache)
            results = []
            for entry_id in entry_ids:
                embedding = await self._vector_cache.get(entry_id)
                if embedding:
                    similarity = self._cosine_similarity(query_embedding, embedding)
                    if similarity >= min_similarity:
                        results.append((entry_id, similarity))
                elif entry_id in self._local_vectors:
                    # Fallback to old dict for backward compatibility
                    embedding = self._local_vectors[entry_id]
                    similarity = self._cosine_similarity(query_embedding, embedding)
                    if similarity >= min_similarity:
                        results.append((entry_id, similarity))
            
            # Sort by similarity
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:limit]
            
            # Fetch entries in batch
            entry_ids_to_fetch = [r[0] for r in results]
            entries = await self.get_batch(entry_ids_to_fetch)
            
            # Map entries back with scores
            entry_map = {e.id: e for e in entries}
            return [
                (entry_map[entry_id], score)
                for entry_id, score in results
                if entry_id in entry_map
            ]
    
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
        if self._is_vector_storage:
            # Search with category filter (empty query returns all)
            # Use a zero vector to get all documents
            zero_vector = [0.0] * 128  # Adjust dimension as needed
            results = await self.storage.search(
                query_embedding=zero_vector,
                limit=limit or 1000,
                filters={"category": category},
                min_similarity=0.0,
                collection=self.collection_name
            )
            return [KnowledgeEntry.from_vector_result(r) for r in results]
        else:
            pattern = f"{self._key_prefix}:category:{category}:*"
            cat_keys = await self.storage.list_keys(pattern, limit=limit)
            
            # Batch fetch entry IDs
            entry_ids = []
            for key in cat_keys:
                entry_id = await self.storage.get(key)
                if entry_id:
                    entry_ids.append(entry_id)
            
            return await self.get_batch(entry_ids)
    
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
        if self._is_vector_storage:
            # Note: ChromaDB metadata filtering for arrays is limited
            # This is a workaround
            zero_vector = [0.0] * 128
            all_results = await self.storage.search(
                query_embedding=zero_vector,
                limit=limit or 1000,
                min_similarity=0.0,
                collection=self.collection_name
            )
            
            # Filter by tag in memory
            return [
                KnowledgeEntry.from_vector_result(r)
                for r in all_results
                if tag in (r.metadata.get("tags", []) or [])
            ]
        else:
            pattern = f"{self._key_prefix}:tag:{tag}:*"
            tag_keys = await self.storage.list_keys(pattern, limit=limit)
            
            entry_ids = []
            for key in tag_keys:
                entry_id = await self.storage.get(key)
                if entry_id:
                    entry_ids.append(entry_id)
            
            return await self.get_batch(entry_ids)
    
    async def delete(self, entry_id: str) -> bool:
        """
        Delete a knowledge entry.
        
        Args:
            entry_id: Entry ID
        
        Returns:
            True if deleted
        """
        if self._is_vector_storage:
            return await self.storage.delete(entry_id, collection=self.collection_name)
        else:
            # Get entry first to clean up indexes
            entry = await self.get(entry_id)
            if not entry:
                return False
            
            # Delete main entry
            key = f"{self._key_prefix}:entry:{entry_id}"
            await self.storage.delete(key)
            
            # Delete from vector cache
            await self._vector_cache.delete(entry_id)
            if entry_id in self._local_vectors:
                del self._local_vectors[entry_id]
            
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
    
    async def delete_batch(self, entry_ids: List[str]) -> int:
        """
        Delete multiple knowledge entries.
        
        Args:
            entry_ids: List of entry IDs
        
        Returns:
            Number of entries deleted
        """
        if not entry_ids:
            return 0
        
        if self._is_vector_storage:
            return await self.storage.delete_batch(entry_ids, collection=self.collection_name)
        else:
            # Delete one by one (maintains index cleanup)
            count = 0
            for entry_id in entry_ids:
                if await self.delete(entry_id):
                    count += 1
            return count
    
    async def clear(self) -> int:
        """
        Clear all knowledge for this agent.
        
        Returns:
            Number of entries deleted
        """
        if self._is_vector_storage:
            return await self.storage.clear_collection(self.collection_name)
        else:
            pattern = f"{self._key_prefix}:*"
            count = await self.storage.clear(pattern)
            await self._vector_cache.clear()
            await self._entry_cache.clear()
            self._local_vectors.clear()
            self._stats_cache = None
            return count
    
    async def get_statistics(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get statistics about stored knowledge with caching.
        
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
        
        if self._is_vector_storage:
            total_entries = await self.storage.collection_count(self.collection_name)
            
            return {
                "total_entries": total_entries,
                "storage_type": "vector",
                "collection": self.collection_name,
                "agent_id": self.agent_id,
            }
        else:
            # Count entries (limit for performance)
            entry_pattern = f"{self._key_prefix}:entry:*"
            entry_keys = await self.storage.list_keys(entry_pattern, limit=10000)
            total_entries = len(entry_keys)
            
            if total_entries == 0:
                return {
                    "total_entries": 0,
                    "categories": [],
                    "tags": [],
                    "avg_importance": 0,
                    "avg_confidence": 0,
                    "storage_type": "key-value",
                }
            
            # Get sample for analysis
            sample_size = min(100, total_entries)
            sample_ids = [key.split(":")[-1] for key in entry_keys[:sample_size]]
            entries = await self.get_batch(sample_ids)
            
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
            
            stats = {
                "total_entries": total_entries,
                "categories": list(categories),
                "tags": list(tags),
                "avg_importance": round(total_importance / len(entries), 2) if entries else 0,
                "avg_confidence": round(total_confidence / len(entries), 2) if entries else 0,
                "storage_type": "key-value",
            }
            
            # Cache results
            self._stats_cache = (time.time(), stats)
            
            return stats
    
    async def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts using embeddings.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score (0-1)
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if not text1 or not isinstance(text1, str):
            raise ValueError("text1 must be a non-empty string")
        if not text2 or not isinstance(text2, str):
            raise ValueError("text2 must be a non-empty string")
        
        try:
            emb1 = await asyncio.wait_for(
                asyncio.to_thread(self.embedding_function, text1),
                timeout=10.0
            )
            emb2 = await asyncio.wait_for(
                asyncio.to_thread(self.embedding_function, text2),
                timeout=10.0
            )
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Failed to compute similarity embeddings: {e}")
            # Fallback to default embeddings
            emb1 = self._default_embedding(text1)
            emb2 = self._default_embedding(text2)
        
        return self._cosine_similarity(emb1, emb2)
