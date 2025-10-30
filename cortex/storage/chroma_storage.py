"""
ChromaDB Storage Backend for Teleon Cortex.

Optimized for Teleon's deployment architecture:
- Embedded mode (no separate server needed)
- Persistent storage (survives container restarts)
- Per-deployment isolation (each agent has own ChromaDB)
- FastEmbed integration (fast, free embeddings)

Perfect for Azure Container Apps deployment model.
"""

import os
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime
import logging

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    Settings = None

from teleon.cortex.storage.base import StorageBackend

logger = logging.getLogger("teleon.chroma")


class ChromaDBStorage(StorageBackend):
    """
    Production-ready ChromaDB storage with FastEmbed.
    
    Architecture:
    - Runs embedded in agent container (no separate server)
    - Data persisted to Azure Files volume mount (/data)
    - Each deployment has isolated ChromaDB instance
    - FastEmbed provides fast, free embeddings
    
    Features:
    - Vector similarity search
    - Metadata filtering
    - Automatic persistence
    - Low resource usage
    
    Example:
        >>> from teleon.cortex.storage.chroma_storage import ChromaDBStorage
        >>> from teleon.cortex.embeddings import create_fastembed_function
        >>> 
        >>> # Create storage
        >>> storage = ChromaDBStorage(
        ...     deployment_id="user_123_agent_xyz",
        ...     embedding_function=create_fastembed_function()
        ... )
        >>> 
        >>> # Store knowledge
        >>> await storage.store(
        ...     "semantic",
        ...     "Python is a programming language",
        ...     metadata={"topic": "programming"}
        ... )
        >>> 
        >>> # Search by meaning
        >>> results = await storage.search(
        ...     "semantic",
        ...     "What is Python?",
        ...     limit=5
        ... )
    """
    
    def __init__(
        self,
        deployment_id: str,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        data_path: Optional[str] = None,
        collection_prefix: str = "teleon"
    ):
        """
        Initialize ChromaDB storage for a deployment.
        
        Args:
            deployment_id: Unique deployment ID (from platform)
            embedding_function: Function to generate embeddings
            data_path: Path to persist data (defaults to /data/chroma)
            collection_prefix: Prefix for collection names
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB not available. Install with: pip install chromadb\n"
                "Or add 'chromadb' to your requirements.txt"
            )
        
        self.deployment_id = deployment_id
        self.collection_prefix = collection_prefix
        
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
        
        # Ensure directory exists
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ChromaDB data path: {self.data_path}")
        
        # Initialize ChromaDB client (embedded mode)
        self.client = chromadb.PersistentClient(
            path=str(self.data_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False
            )
        )
        
        # Set up embedding function
        self.embedding_function = embedding_function
        if not self.embedding_function:
            # Try to use FastEmbed if available
            try:
                from teleon.cortex.embeddings import create_fastembed_function
                self.embedding_function = create_fastembed_function()
                logger.info("Using FastEmbed for embeddings")
            except ImportError:
                logger.warning(
                    "No embedding function provided and FastEmbed not available. "
                    "Semantic search will not work properly."
                )
        
        # Collections cache
        self._collections: Dict[str, Any] = {}
        
        logger.info(
            f"ChromaDB storage initialized for deployment: {deployment_id}"
        )
    
    def _get_collection(self, memory_type: str):
        """Get or create collection for a memory type."""
        collection_name = f"{self.collection_prefix}_{memory_type}_{self.deployment_id}"
        
        if collection_name not in self._collections:
            self._collections[collection_name] = (
                self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={
                        "deployment_id": self.deployment_id,
                        "memory_type": memory_type
                    }
                )
            )
        
        return self._collections[collection_name]
    
    async def store(
        self,
        memory_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None
    ) -> str:
        """
        Store content with automatic embedding.
        
        Args:
            memory_type: Type of memory (semantic, episodic, etc.)
            content: Content to store
            metadata: Optional metadata
            memory_id: Optional custom ID
        
        Returns:
            Memory ID
        """
        # Generate ID if not provided
        if not memory_id:
            memory_id = str(uuid.uuid4())
        
        # Add timestamp to metadata
        metadata = metadata or {}
        metadata["stored_at"] = datetime.utcnow().isoformat()
        metadata["memory_type"] = memory_type
        
        # Generate embedding if function is available
        embedding = None
        if self.embedding_function and memory_type == "semantic":
            try:
                embedding = self.embedding_function(content)
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
        
        # Get collection
        collection = self._get_collection(memory_type)
        
        # Store in ChromaDB
        try:
            if embedding:
                collection.add(
                    ids=[memory_id],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[metadata]
                )
            else:
                # Store without embedding (for non-semantic memory)
                collection.add(
                    ids=[memory_id],
                    documents=[content],
                    metadatas=[metadata]
                )
            
            logger.debug(f"Stored {memory_type} memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing in ChromaDB: {e}")
            raise
    
    async def retrieve(
        self,
        memory_type: str,
        memory_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific memory by ID.
        
        Args:
            memory_type: Type of memory
            memory_id: Memory ID
        
        Returns:
            Memory data or None if not found
        """
        collection = self._get_collection(memory_type)
        
        try:
            results = collection.get(ids=[memory_id])
            
            if results['ids']:
                return {
                    "id": results['ids'][0],
                    "content": results['documents'][0],
                    "metadata": results['metadatas'][0]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving from ChromaDB: {e}")
            return None
    
    async def search(
        self,
        memory_type: str,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search memories using semantic similarity.
        
        Args:
            memory_type: Type of memory
            query: Search query
            limit: Maximum results to return
            filters: Optional metadata filters
        
        Returns:
            List of matching memories with similarity scores
        """
        collection = self._get_collection(memory_type)
        
        # Generate query embedding if function is available
        query_embedding = None
        if self.embedding_function and memory_type == "semantic":
            try:
                query_embedding = self.embedding_function(query)
            except Exception as e:
                logger.error(f"Error generating query embedding: {e}")
        
        try:
            # Search with or without embeddings
            if query_embedding:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit,
                    where=filters
                )
            else:
                # Fallback: text search (less effective)
                results = collection.query(
                    query_texts=[query],
                    n_results=limit,
                    where=filters
                )
            
            # Format results
            items = []
            for i in range(len(results['ids'][0])):
                # Convert distance to similarity (ChromaDB returns L2 distance)
                # similarity = 1 / (1 + distance)
                distance = results['distances'][0][i] if 'distances' in results else 0
                similarity = 1.0 / (1.0 + distance)
                
                items.append({
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "similarity": similarity
                })
            
            return items
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []
    
    async def delete(
        self,
        memory_type: str,
        memory_id: str
    ) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_type: Type of memory
            memory_id: Memory ID
        
        Returns:
            True if deleted, False otherwise
        """
        collection = self._get_collection(memory_type)
        
        try:
            collection.delete(ids=[memory_id])
            logger.debug(f"Deleted {memory_type} memory: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting from ChromaDB: {e}")
            return False
    
    async def list_all(
        self,
        memory_type: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List all memories of a type.
        
        Args:
            memory_type: Type of memory
            limit: Optional limit on results
        
        Returns:
            List of memories
        """
        collection = self._get_collection(memory_type)
        
        try:
            # Get all items (ChromaDB doesn't have a direct "get all" with limit)
            results = collection.get(
                limit=limit
            )
            
            items = []
            for i in range(len(results['ids'])):
                items.append({
                    "id": results['ids'][i],
                    "content": results['documents'][i],
                    "metadata": results['metadatas'][i]
                })
            
            return items
            
        except Exception as e:
            logger.error(f"Error listing from ChromaDB: {e}")
            return []
    
    async def clear(self, memory_type: str) -> bool:
        """
        Clear all memories of a type.
        
        Args:
            memory_type: Type of memory to clear
        
        Returns:
            True if successful
        """
        try:
            collection_name = f"{self.collection_prefix}_{memory_type}_{self.deployment_id}"
            
            # Delete the collection
            self.client.delete_collection(collection_name)
            
            # Remove from cache
            if collection_name in self._collections:
                del self._collections[collection_name]
            
            logger.info(f"Cleared {memory_type} memory collection")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing ChromaDB collection: {e}")
            return False
    
    async def get_statistics(self, memory_type: str) -> Dict[str, Any]:
        """
        Get statistics for a memory type.
        
        Args:
            memory_type: Type of memory
        
        Returns:
            Statistics dictionary
        """
        collection = self._get_collection(memory_type)
        
        try:
            count = collection.count()
            
            return {
                "memory_type": memory_type,
                "deployment_id": self.deployment_id,
                "total_items": count,
                "data_path": str(self.data_path),
                "embedding_enabled": self.embedding_function is not None,
                "backend": "ChromaDB (embedded)"
            }
            
        except Exception as e:
            logger.error(f"Error getting ChromaDB statistics: {e}")
            return {
                "memory_type": memory_type,
                "error": str(e)
            }
    
    async def close(self):
        """Close ChromaDB client and cleanup."""
        try:
            # ChromaDB doesn't require explicit closing in embedded mode
            # Data is automatically persisted
            self._collections.clear()
            logger.info("ChromaDB storage closed")
        except Exception as e:
            logger.error(f"Error closing ChromaDB: {e}")


def create_chroma_storage(
    deployment_id: Optional[str] = None,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    data_path: Optional[str] = None
) -> ChromaDBStorage:
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
    
    return ChromaDBStorage(
        deployment_id=deployment_id,
        embedding_function=embed_fn,
        data_path=data_path
    )

