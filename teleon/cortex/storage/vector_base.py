"""
Vector Storage Backend Interface for Cortex Memory System.

This module provides the abstract base class for vector-enabled storage backends.
Vector backends handle semantic search with embeddings, separate from key-value storage.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Tuple
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from enum import Enum
import logging

logger = logging.getLogger("teleon.vector_storage")


class HealthStatus(str, Enum):
    """Health status for storage backend."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class VectorSearchResult(BaseModel):
    """Result from vector similarity search."""
    
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    similarity: float = Field(ge=0.0, le=1.0)
    embedding: Optional[List[float]] = None


class VectorStorageMetrics(BaseModel):
    """Metrics for vector storage backend."""
    
    total_documents: int = 0
    total_collections: int = 0
    store_operations: int = 0
    search_operations: int = 0
    delete_operations: int = 0
    avg_search_latency_ms: float = 0.0
    last_operation_time: Optional[datetime] = None
    errors: int = 0
    
    def record_operation(self, operation: str, latency_ms: float = 0.0):
        """Record an operation."""
        self.last_operation_time = datetime.now(timezone.utc)
        
        if operation == "store":
            self.store_operations += 1
        elif operation == "search":
            self.search_operations += 1
            # Running average
            total_ops = self.search_operations
            self.avg_search_latency_ms = (
                (self.avg_search_latency_ms * (total_ops - 1) + latency_ms) / total_ops
            )
        elif operation == "delete":
            self.delete_operations += 1


class VectorStorageConfig(BaseModel):
    """Configuration for vector storage backends."""
    
    embedding_dimension: int = Field(384, description="Dimension of embedding vectors")
    default_collection: str = Field("default", description="Default collection name")
    max_results: int = Field(100, description="Maximum search results")
    min_similarity: float = Field(0.0, description="Minimum similarity threshold")
    enable_metrics: bool = Field(True, description="Enable metrics collection")
    retry_attempts: int = Field(3, description="Number of retry attempts")
    retry_delay_ms: int = Field(100, description="Delay between retries in ms")
    connection_timeout_ms: int = Field(5000, description="Connection timeout in ms")
    operation_timeout_ms: int = Field(30000, description="Operation timeout in ms")


class VectorStorageBackend(ABC):
    """
    Abstract base class for vector-enabled storage backends.
    
    Vector storage backends provide semantic similarity search capabilities
    using embeddings. They are optimized for:
    - Storing documents with vector embeddings
    - Similarity-based retrieval
    - Metadata filtering
    - Collection management
    
    Implementations include:
    - ChromaDBStorage: Embedded vector database
    - PineconeStorage: Managed cloud vector database
    - WeaviateStorage: Self-hosted vector database
    - QdrantStorage: High-performance vector database
    
    Example:
        ```python
        storage = ChromaDBVectorStorage(deployment_id="agent-123")
        await storage.initialize()
        
        # Store document with embedding
        doc_id = await storage.store(
            content="Paris is the capital of France",
            embedding=embed_fn("Paris is the capital of France"),
            metadata={"topic": "geography"}
        )
        
        # Search by similarity
        results = await storage.search(
            query_embedding=embed_fn("What is the capital of France?"),
            limit=5
        )
        ```
    """
    
    def __init__(self, config: Optional[VectorStorageConfig] = None):
        """
        Initialize vector storage backend.
        
        Args:
            config: Storage configuration
        """
        self.config = config or VectorStorageConfig()
        self.metrics = VectorStorageMetrics() if self.config.enable_metrics else None
        self._initialized = False
        self._healthy = True
    
    async def initialize(self) -> None:
        """
        Initialize the storage backend.
        
        Called once before first use. Subclasses should override
        to set up connections, create collections, etc.
        """
        self._initialized = True
        logger.info(f"{self.__class__.__name__} initialized")
    
    async def shutdown(self) -> None:
        """
        Shutdown the storage backend gracefully.
        
        Called when shutting down. Subclasses should override
        to close connections, flush buffers, etc.
        """
        self._initialized = False
        logger.info(f"{self.__class__.__name__} shutdown")
    
    @abstractmethod
    async def store(
        self,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        collection: Optional[str] = None
    ) -> str:
        """
        Store a document with its embedding.
        
        Args:
            content: Document content
            embedding: Vector embedding of the content
            metadata: Optional metadata
            document_id: Optional custom ID (generated if not provided)
            collection: Collection name (uses default if not provided)
        
        Returns:
            Document ID
        
        Raises:
            VectorStorageError: If storage operation fails
        """
        pass
    
    @abstractmethod
    async def store_batch(
        self,
        documents: List[Dict[str, Any]],
        collection: Optional[str] = None
    ) -> List[str]:
        """
        Store multiple documents in batch.
        
        Args:
            documents: List of dicts with 'content', 'embedding', 'metadata', 'id'
            collection: Collection name
        
        Returns:
            List of document IDs
        
        Raises:
            VectorStorageError: If storage operation fails
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        min_similarity: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None
    ) -> List[VectorSearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            limit: Maximum results to return
            min_similarity: Minimum similarity threshold (0-1)
            filters: Metadata filters
            collection: Collection name
        
        Returns:
            List of search results with similarity scores
        
        Raises:
            VectorStorageError: If search operation fails
        """
        pass
    
    @abstractmethod
    async def get(
        self,
        document_id: str,
        collection: Optional[str] = None
    ) -> Optional[VectorSearchResult]:
        """
        Get a document by ID.
        
        Args:
            document_id: Document ID
            collection: Collection name
        
        Returns:
            Document or None if not found
        
        Raises:
            VectorStorageError: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_batch(
        self,
        document_ids: List[str],
        collection: Optional[str] = None
    ) -> List[VectorSearchResult]:
        """
        Get multiple documents by ID.
        
        Args:
            document_ids: List of document IDs
            collection: Collection name
        
        Returns:
            List of documents (missing IDs omitted)
        
        Raises:
            VectorStorageError: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def delete(
        self,
        document_id: str,
        collection: Optional[str] = None
    ) -> bool:
        """
        Delete a document.
        
        Args:
            document_id: Document ID
            collection: Collection name
        
        Returns:
            True if deleted, False if not found
        
        Raises:
            VectorStorageError: If deletion fails
        """
        pass
    
    @abstractmethod
    async def delete_batch(
        self,
        document_ids: List[str],
        collection: Optional[str] = None
    ) -> int:
        """
        Delete multiple documents.
        
        Args:
            document_ids: List of document IDs
            collection: Collection name
        
        Returns:
            Number of documents deleted
        
        Raises:
            VectorStorageError: If deletion fails
        """
        pass
    
    @abstractmethod
    async def clear_collection(
        self,
        collection: Optional[str] = None
    ) -> int:
        """
        Clear all documents in a collection.
        
        Args:
            collection: Collection name
        
        Returns:
            Number of documents deleted
        
        Raises:
            VectorStorageError: If clear fails
        """
        pass
    
    @abstractmethod
    async def list_collections(self) -> List[str]:
        """
        List all collections.
        
        Returns:
            List of collection names
        """
        pass
    
    @abstractmethod
    async def collection_count(
        self,
        collection: Optional[str] = None
    ) -> int:
        """
        Get document count in collection.
        
        Args:
            collection: Collection name
        
        Returns:
            Number of documents
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check storage health.
        
        Returns:
            Health status dict with 'status', 'latency_ms', 'details'
        """
        import time
        start = time.time()
        
        try:
            # Try a simple operation
            collections = await self.list_collections()
            latency_ms = (time.time() - start) * 1000
            
            status = HealthStatus.HEALTHY
            if latency_ms > 1000:
                status = HealthStatus.DEGRADED
            
            return {
                "status": status.value,
                "latency_ms": round(latency_ms, 2),
                "collections": len(collections),
                "initialized": self._initialized,
                "details": {}
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "latency_ms": (time.time() - start) * 1000,
                "error": str(e),
                "initialized": self._initialized,
                "details": {}
            }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "initialized": self._initialized,
            "healthy": self._healthy,
        }
        
        if self.metrics:
            stats.update({
                "total_documents": self.metrics.total_documents,
                "total_collections": self.metrics.total_collections,
                "store_operations": self.metrics.store_operations,
                "search_operations": self.metrics.search_operations,
                "delete_operations": self.metrics.delete_operations,
                "avg_search_latency_ms": round(self.metrics.avg_search_latency_ms, 2),
                "errors": self.metrics.errors,
            })
        
        return stats
    
    def _update_metrics(
        self,
        operation: str,
        latency_ms: float = 0.0,
        error: bool = False
    ) -> None:
        """Update metrics for an operation."""
        if not self.metrics:
            return
        
        self.metrics.record_operation(operation, latency_ms)
        if error:
            self.metrics.errors += 1


class VectorStorageError(Exception):
    """Base exception for vector storage errors."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        collection: Optional[str] = None,
        retryable: bool = False
    ):
        self.message = message
        self.operation = operation
        self.collection = collection
        self.retryable = retryable
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        parts = [self.message]
        if self.operation:
            parts.append(f"operation={self.operation}")
        if self.collection:
            parts.append(f"collection={self.collection}")
        return " | ".join(parts)


class ConnectionError(VectorStorageError):
    """Raised when storage backend connection fails."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, retryable=True, **kwargs)


class CollectionNotFoundError(VectorStorageError):
    """Raised when collection doesn't exist."""
    pass


class DocumentNotFoundError(VectorStorageError):
    """Raised when document doesn't exist."""
    pass


class EmbeddingDimensionError(VectorStorageError):
    """Raised when embedding dimension doesn't match."""
    pass

