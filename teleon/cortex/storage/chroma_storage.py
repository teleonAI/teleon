"""
ChromaDB Vector Storage Backend for Teleon Cortex.

Enterprise-grade implementation with:
- Proper async handling (no event loop blocking)
- Retry logic with exponential backoff
- Health checks and observability
- Batch operations for efficiency
- Per-deployment isolation
- FastEmbed integration (fast, free embeddings)

Optimized for Azure Container Apps deployment model.
"""

import asyncio
import os
import uuid
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime, timezone
from functools import wraps

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    Settings = None

from teleon.cortex.storage.vector_base import (
    VectorStorageBackend,
    VectorStorageConfig,
    VectorSearchResult,
    VectorStorageError,
    ConnectionError,
    CollectionNotFoundError,
    HealthStatus,
)

logger = logging.getLogger("teleon.chroma")


def async_retry(max_attempts: int = 3, base_delay: float = 0.1, max_delay: float = 2.0):
    """
    Decorator for async retry with exponential backoff.
    
    Args:
        max_attempts: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if retryable
                    if isinstance(e, VectorStorageError) and not e.retryable:
                        raise
                    
                    if attempt < max_attempts - 1:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__} "
                            f"after {delay:.2f}s: {e}"
                        )
                        await asyncio.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


class ChromaDBVectorStorage(VectorStorageBackend):
    """
    Production-ready ChromaDB storage with FastEmbed.
    
    Architecture:
    - Runs embedded in agent container (no separate server)
    - Data persisted to Azure Files volume mount (/data)
    - Each deployment has isolated ChromaDB instance
    - FastEmbed provides fast, free embeddings
    - Proper async handling with thread pool
    
    Features:
    - Vector similarity search
    - Metadata filtering
    - Batch operations
    - Automatic persistence
    - Health checks
    - Retry logic
    - Observability metrics
    
    Example:
        >>> from teleon.cortex.storage.chroma_storage import ChromaDBVectorStorage
        >>> from teleon.cortex.embeddings import create_fastembed_function
        >>> 
        >>> # Create storage
        >>> storage = ChromaDBVectorStorage(
        ...     deployment_id="user_123_agent_xyz",
        ...     embedding_function=create_fastembed_function()
        ... )
        >>> await storage.initialize()
        >>> 
        >>> # Store knowledge
        >>> doc_id = await storage.store(
        ...     content="Python is a programming language",
        ...     embedding=embed_fn("Python is a programming language"),
        ...     metadata={"topic": "programming"}
        ... )
        >>> 
        >>> # Search by meaning
        >>> results = await storage.search(
        ...     query_embedding=embed_fn("What is Python?"),
        ...     limit=5
        ... )
    """
    
    def __init__(
        self,
        deployment_id: str,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        data_path: Optional[str] = None,
        collection_prefix: str = "teleon",
        config: Optional[VectorStorageConfig] = None
    ):
        """
        Initialize ChromaDB storage for a deployment.
        
        Args:
            deployment_id: Unique deployment ID (from platform)
            embedding_function: Function to generate embeddings
            data_path: Path to persist data (defaults to /data/chroma)
            collection_prefix: Prefix for collection names
            config: Storage configuration
        """
        super().__init__(config)
        
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB not available. Install with: pip install chromadb\n"
                "Or add 'chromadb' to your requirements.txt"
            )
        
        self.deployment_id = deployment_id
        self.collection_prefix = collection_prefix
        self.embedding_function = embedding_function
        
        # Determine data path
        # Priority: explicit path > /data mount > /tmp fallback
        if data_path:
            self.data_path = Path(data_path)
        elif os.path.exists("/data"):
            # Azure Files mount point
            self.data_path = Path(f"/data/chroma/{deployment_id}")
        else:
            # Fallback for local development
            self.data_path = Path(f"/tmp/chroma/{deployment_id}")
        
        # Thread pool for blocking operations
        self._executor = None
        
        # ChromaDB client (initialized lazily)
        self._client = None
        
        # Collections cache
        self._collections: Dict[str, Any] = {}
        
        # Circuit breaker state
        self._circuit_open = False
        self._circuit_failures = 0
        self._circuit_threshold = 5
        self._circuit_reset_time = None
        self._circuit_reset_delay = 30  # seconds
    
    async def initialize(self) -> None:
        """Initialize ChromaDB client and create data directory."""
        if self._initialized:
            return
        
        # Ensure directory exists
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ChromaDB data path: {self.data_path}")
        
        # Initialize ChromaDB client in thread pool (blocking operation)
        def _init_client():
            return chromadb.PersistentClient(
                path=str(self.data_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False
                )
            )
        
        try:
            self._client = await asyncio.to_thread(_init_client)
        except Exception as e:
            raise ConnectionError(
                f"Failed to initialize ChromaDB: {e}",
                operation="initialize"
            )
        
        # Set up embedding function if not provided
        if not self.embedding_function:
            try:
                from teleon.cortex.embeddings import create_fastembed_function
                self.embedding_function = create_fastembed_function()
                logger.info("Using FastEmbed for embeddings")
            except ImportError:
                logger.warning(
                    "No embedding function provided and FastEmbed not available. "
                    "You must provide embeddings explicitly."
                )
        
        await super().initialize()
        
        logger.info(
            f"ChromaDB storage initialized for deployment: {self.deployment_id}"
        )
    
    async def shutdown(self) -> None:
        """Shutdown ChromaDB client and cleanup."""
        try:
            # Clear collections cache
            self._collections.clear()
            
            # ChromaDB PersistentClient doesn't need explicit closing
            # Data is automatically persisted
            
            await super().shutdown()
            logger.info("ChromaDB storage shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during ChromaDB shutdown: {e}")
    
    def _get_collection_name(self, collection: Optional[str] = None) -> str:
        """Get full collection name with prefix."""
        base = collection or self.config.default_collection
        return f"{self.collection_prefix}_{base}_{self.deployment_id}"
    
    async def _get_collection(self, collection: Optional[str] = None):
        """Get or create collection (async-safe)."""
        collection_name = self._get_collection_name(collection)
        
        if collection_name not in self._collections:
            def _get_or_create():
                return self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={
                        "deployment_id": self.deployment_id,
                        "created_at": datetime.now(timezone.utc).isoformat()
                    }
                )
            
            self._collections[collection_name] = await asyncio.to_thread(_get_or_create)
        
        return self._collections[collection_name]
    
    def _check_circuit(self) -> None:
        """Check circuit breaker state."""
        if self._circuit_open:
            if self._circuit_reset_time and time.time() > self._circuit_reset_time:
                # Reset circuit
                self._circuit_open = False
                self._circuit_failures = 0
                self._circuit_reset_time = None
                logger.info("Circuit breaker reset")
            else:
                raise ConnectionError(
                    "Circuit breaker is open - too many failures",
                    operation="circuit_check"
                )
    
    def _record_failure(self) -> None:
        """Record a failure for circuit breaker."""
        self._circuit_failures += 1
        if self._circuit_failures >= self._circuit_threshold:
            self._circuit_open = True
            self._circuit_reset_time = time.time() + self._circuit_reset_delay
            logger.error(
                f"Circuit breaker opened after {self._circuit_failures} failures. "
                f"Will reset in {self._circuit_reset_delay}s"
            )
    
    def _record_success(self) -> None:
        """Record a success for circuit breaker."""
        self._circuit_failures = 0
    
    @async_retry(max_attempts=3, base_delay=0.1)
    async def store(
        self,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        collection: Optional[str] = None
    ) -> str:
        """Store a document with its embedding."""
        self._check_circuit()
        start_time = time.time()
        
        try:
            # Generate ID if not provided
            doc_id = document_id or str(uuid.uuid4())
            
            # Prepare metadata
            meta = metadata.copy() if metadata else {}
            meta["stored_at"] = datetime.now(timezone.utc).isoformat()
            meta["deployment_id"] = self.deployment_id
            
            # Ensure metadata values are valid types for ChromaDB
            meta = self._sanitize_metadata(meta)
            
            # Get collection
            coll = await self._get_collection(collection)
            
            # Store in ChromaDB (blocking operation with timeout)
            def _add():
                coll.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[meta]
                )
            
            await asyncio.wait_for(
                asyncio.to_thread(_add),
                timeout=30.0  # 30 second timeout
            )
            
            # Update metrics
            latency = (time.time() - start_time) * 1000
            self._update_metrics("store", latency)
            self._record_success()
            
            logger.debug(f"Stored document {doc_id} in {latency:.2f}ms")
            return doc_id
            
        except Exception as e:
            self._record_failure()
            self._update_metrics("store", error=True)
            raise VectorStorageError(
                f"Failed to store document: {e}",
                operation="store",
                collection=collection,
                retryable=True
            )
    
    @async_retry(max_attempts=3, base_delay=0.1)
    async def store_batch(
        self,
        documents: List[Dict[str, Any]],
        collection: Optional[str] = None
    ) -> List[str]:
        """Store multiple documents in batch."""
        self._check_circuit()
        start_time = time.time()
        
        if not documents:
            return []
        
        try:
            # Prepare batch data
            ids = []
            embeddings = []
            contents = []
            metadatas = []
            
            for doc in documents:
                doc_id = doc.get("id") or str(uuid.uuid4())
                ids.append(doc_id)
                embeddings.append(doc["embedding"])
                contents.append(doc["content"])
                
                meta = doc.get("metadata", {}).copy()
                meta["stored_at"] = datetime.now(timezone.utc).isoformat()
                meta["deployment_id"] = self.deployment_id
                metadatas.append(self._sanitize_metadata(meta))
            
            # Get collection
            coll = await self._get_collection(collection)
            
            # Store batch (blocking operation with timeout)
            def _add_batch():
                coll.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=contents,
                    metadatas=metadatas
                )
            
            await asyncio.wait_for(
                asyncio.to_thread(_add_batch),
                timeout=60.0  # 60 second timeout for batch
            )
            
            # Update metrics
            latency = (time.time() - start_time) * 1000
            self._update_metrics("store", latency)
            self._record_success()
            
            logger.debug(f"Stored {len(ids)} documents in {latency:.2f}ms")
            return ids
            
        except Exception as e:
            self._record_failure()
            self._update_metrics("store", error=True)
            raise VectorStorageError(
                f"Failed to store batch: {e}",
                operation="store_batch",
                collection=collection,
                retryable=True
            )
    
    @async_retry(max_attempts=3, base_delay=0.1)
    async def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        min_similarity: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None
    ) -> List[VectorSearchResult]:
        """Search for similar documents."""
        self._check_circuit()
        start_time = time.time()
        
        try:
            min_sim = min_similarity if min_similarity is not None else self.config.min_similarity
            
            # Get collection
            coll = await self._get_collection(collection)
            
            # Search (blocking operation with timeout)
            def _query():
                return coll.query(
                    query_embeddings=[query_embedding],
                    n_results=min(limit, self.config.max_results),
                    where=filters
                )
            
            results = await asyncio.wait_for(
                asyncio.to_thread(_query),
                timeout=30.0  # 30 second timeout
            )
            
            # Process results
            search_results = []
            
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    # Convert distance to similarity
                    # ChromaDB returns L2 distance by default
                    distance = results['distances'][0][i] if 'distances' in results else 0
                    similarity = 1.0 / (1.0 + distance)
                    
                    # Apply similarity threshold
                    if similarity < min_sim:
                        continue
                    
                    search_results.append(VectorSearchResult(
                        id=results['ids'][0][i],
                        content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i] if results.get('metadatas') else {},
                        similarity=similarity,
                        embedding=results['embeddings'][0][i] if results.get('embeddings') else None
                    ))
            
            # Update metrics
            latency = (time.time() - start_time) * 1000
            self._update_metrics("search", latency)
            self._record_success()
            
            logger.debug(f"Search returned {len(search_results)} results in {latency:.2f}ms")
            return search_results
            
        except Exception as e:
            self._record_failure()
            self._update_metrics("search", error=True)
            raise VectorStorageError(
                f"Failed to search: {e}",
                operation="search",
                collection=collection,
                retryable=True
            )
    
    @async_retry(max_attempts=3, base_delay=0.1)
    async def get(
        self,
        document_id: str,
        collection: Optional[str] = None
    ) -> Optional[VectorSearchResult]:
        """Get a document by ID."""
        self._check_circuit()
        
        try:
            coll = await self._get_collection(collection)
            
            def _get():
                return coll.get(ids=[document_id], include=["documents", "metadatas", "embeddings"])
            
            results = await asyncio.wait_for(
                asyncio.to_thread(_get),
                timeout=10.0  # 10 second timeout
            )
            
            if not results['ids']:
                return None
            
            self._record_success()
            
            return VectorSearchResult(
                id=results['ids'][0],
                content=results['documents'][0] if results.get('documents') else "",
                metadata=results['metadatas'][0] if results.get('metadatas') else {},
                similarity=1.0,  # Exact match
                embedding=results['embeddings'][0] if results.get('embeddings') else None
            )
            
        except Exception as e:
            self._record_failure()
            raise VectorStorageError(
                f"Failed to get document: {e}",
                operation="get",
                collection=collection,
                retryable=True
            )
    
    @async_retry(max_attempts=3, base_delay=0.1)
    async def get_batch(
        self,
        document_ids: List[str],
        collection: Optional[str] = None
    ) -> List[VectorSearchResult]:
        """Get multiple documents by ID."""
        self._check_circuit()
        
        if not document_ids:
            return []
        
        try:
            coll = await self._get_collection(collection)
            
            def _get_batch():
                return coll.get(ids=document_ids, include=["documents", "metadatas", "embeddings"])
            
            results = await asyncio.wait_for(
                asyncio.to_thread(_get_batch),
                timeout=30.0  # 30 second timeout for batch
            )
            
            self._record_success()
            
            search_results = []
            for i in range(len(results['ids'])):
                search_results.append(VectorSearchResult(
                    id=results['ids'][i],
                    content=results['documents'][i] if results.get('documents') else "",
                    metadata=results['metadatas'][i] if results.get('metadatas') else {},
                    similarity=1.0,
                    embedding=results['embeddings'][i] if results.get('embeddings') else None
                ))
            
            return search_results
            
        except Exception as e:
            self._record_failure()
            raise VectorStorageError(
                f"Failed to get batch: {e}",
                operation="get_batch",
                collection=collection,
                retryable=True
            )
    
    @async_retry(max_attempts=3, base_delay=0.1)
    async def delete(
        self,
        document_id: str,
        collection: Optional[str] = None
    ) -> bool:
        """Delete a document."""
        self._check_circuit()
        
        try:
            coll = await self._get_collection(collection)
            
            def _delete():
                coll.delete(ids=[document_id])
            
            await asyncio.wait_for(
                asyncio.to_thread(_delete),
                timeout=10.0  # 10 second timeout
            )
            
            self._update_metrics("delete")
            self._record_success()
            
            logger.debug(f"Deleted document {document_id}")
            return True
            
        except Exception as e:
            self._record_failure()
            self._update_metrics("delete", error=True)
            # ChromaDB doesn't raise if ID not found, so we return True
            logger.warning(f"Delete operation for {document_id}: {e}")
            return True
    
    @async_retry(max_attempts=3, base_delay=0.1)
    async def delete_batch(
        self,
        document_ids: List[str],
        collection: Optional[str] = None
    ) -> int:
        """Delete multiple documents."""
        self._check_circuit()
        
        if not document_ids:
            return 0
        
        try:
            coll = await self._get_collection(collection)
            
            def _delete_batch():
                coll.delete(ids=document_ids)
            
            await asyncio.wait_for(
                asyncio.to_thread(_delete_batch),
                timeout=30.0  # 30 second timeout for batch
            )
            
            self._update_metrics("delete")
            self._record_success()
            
            logger.debug(f"Deleted {len(document_ids)} documents")
            return len(document_ids)
            
        except Exception as e:
            self._record_failure()
            self._update_metrics("delete", error=True)
            raise VectorStorageError(
                f"Failed to delete batch: {e}",
                operation="delete_batch",
                collection=collection,
                retryable=True
            )
    
    async def clear_collection(
        self,
        collection: Optional[str] = None
    ) -> int:
        """Clear all documents in a collection."""
        try:
            collection_name = self._get_collection_name(collection)
            
            # Get count before delete
            coll = await self._get_collection(collection)
            
            def _count():
                return coll.count()
            
            count = await asyncio.wait_for(
                asyncio.to_thread(_count),
                timeout=10.0  # 10 second timeout
            )
            
            # Delete and recreate collection
            def _delete_collection():
                self._client.delete_collection(collection_name)
            
            await asyncio.wait_for(
                asyncio.to_thread(_delete_collection),
                timeout=30.0  # 30 second timeout
            )
            
            # Remove from cache
            if collection_name in self._collections:
                del self._collections[collection_name]
            
            logger.info(f"Cleared collection {collection_name} ({count} documents)")
            return count
            
        except Exception as e:
            raise VectorStorageError(
                f"Failed to clear collection: {e}",
                operation="clear_collection",
                collection=collection,
                retryable=False
            )
    
    async def list_collections(self) -> List[str]:
        """List all collections for this deployment."""
        try:
            def _list():
                return self._client.list_collections()
            
            collections = await asyncio.wait_for(
                asyncio.to_thread(_list),
                timeout=10.0  # 10 second timeout
            )
            
            # Filter to only this deployment's collections
            prefix = f"{self.collection_prefix}_"
            suffix = f"_{self.deployment_id}"
            
            return [
                c.name for c in collections
                if c.name.startswith(prefix) and c.name.endswith(suffix)
            ]
            
        except Exception as e:
            raise VectorStorageError(
                f"Failed to list collections: {e}",
                operation="list_collections",
                retryable=True
            )
    
    async def collection_count(
        self,
        collection: Optional[str] = None
    ) -> int:
        """Get document count in collection."""
        try:
            coll = await self._get_collection(collection)
            
            def _count():
                return coll.count()
            
            return await asyncio.wait_for(
                asyncio.to_thread(_count),
                timeout=10.0  # 10 second timeout
            )
            
        except Exception as e:
            raise VectorStorageError(
                f"Failed to get collection count: {e}",
                operation="collection_count",
                collection=collection,
                retryable=True
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check ChromaDB health."""
        start = time.time()
        
        try:
            # Check if we can list collections
            def _heartbeat():
                return self._client.heartbeat()
            
            heartbeat = await asyncio.wait_for(
                asyncio.to_thread(_heartbeat),
                timeout=5.0  # 5 second timeout for health check
            )
            latency_ms = (time.time() - start) * 1000
            
            # Determine status
            status = HealthStatus.HEALTHY
            if latency_ms > 1000:
                status = HealthStatus.DEGRADED
            if self._circuit_open:
                status = HealthStatus.UNHEALTHY
            
            return {
                "status": status.value,
                "latency_ms": round(latency_ms, 2),
                "heartbeat": heartbeat,
                "circuit_breaker": {
                    "open": self._circuit_open,
                    "failures": self._circuit_failures,
                    "threshold": self._circuit_threshold
                },
                "data_path": str(self.data_path),
                "deployment_id": self.deployment_id,
                "initialized": self._initialized,
                "embedding_enabled": self.embedding_function is not None
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "latency_ms": (time.time() - start) * 1000,
                "error": str(e),
                "circuit_breaker": {
                    "open": self._circuit_open,
                    "failures": self._circuit_failures,
                    "threshold": self._circuit_threshold
                },
                "initialized": self._initialized
            }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = await super().get_statistics()
        
        try:
            # Add ChromaDB-specific stats
            collections = await self.list_collections()
            total_docs = 0
            
            for coll_name in collections:
                # Extract the collection type from the name
                # Format: prefix_type_deployment_id
                parts = coll_name.split("_")
                if len(parts) >= 2:
                    coll_type = parts[1]
                    count = await self.collection_count(coll_type)
                    total_docs += count
            
            stats.update({
                "backend": "ChromaDB (embedded)",
                "data_path": str(self.data_path),
                "deployment_id": self.deployment_id,
                "collections": len(collections),
                "total_documents": total_docs,
                "embedding_enabled": self.embedding_function is not None
            })
            
        except Exception as e:
            stats["error"] = str(e)
        
        return stats
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata for ChromaDB.
        
        ChromaDB only supports string, int, float, and bool values.
        """
        sanitized = {}
        
        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, datetime):
                sanitized[key] = value.isoformat()
            elif isinstance(value, (list, dict)):
                import json
                sanitized[key] = json.dumps(value)
            else:
                sanitized[key] = str(value)
        
        return sanitized


# Convenience function for backward compatibility
def create_chroma_storage(
    deployment_id: Optional[str] = None,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    data_path: Optional[str] = None
) -> ChromaDBVectorStorage:
    """
    Create a ChromaDB storage instance with FastEmbed.
    
    Args:
        deployment_id: Deployment ID (auto-generated if not provided)
        embedding_model: FastEmbed model to use
        data_path: Optional custom data path
    
    Returns:
        Configured ChromaDB storage
    
    Example:
        >>> storage = create_chroma_storage(
        ...     deployment_id="user_123_agent_xyz",
        ...     embedding_model="BAAI/bge-small-en-v1.5"
        ... )
        >>> await storage.initialize()
    """
    # Auto-generate deployment ID if not provided
    if not deployment_id:
        deployment_id = os.getenv("DEPLOYMENT_ID") or str(uuid.uuid4())
    
    # Create embedding function
    try:
        from teleon.cortex.embeddings import create_fastembed_function
        embed_fn = create_fastembed_function(model_name=embedding_model)
    except ImportError:
        logger.warning("FastEmbed not available, embeddings disabled")
        embed_fn = None
    
    return ChromaDBVectorStorage(
        deployment_id=deployment_id,
        embedding_function=embed_fn,
        data_path=data_path
    )


# Backward compatibility alias
ChromaDBStorage = ChromaDBVectorStorage
