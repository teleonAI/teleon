"""
FastEmbed Engine - Fast, local embeddings for production.

FastEmbed provides:
- 2-3x faster inference than sentence-transformers
- Optimized for production workloads
- Free, open-source, local execution
- No API costs
"""

from typing import List, Optional
import logging

try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    TextEmbedding = None

logger = logging.getLogger("teleon.fastembed")


class FastEmbedEngine:
    """
    FastEmbed embedding engine for Teleon.
    
    Features:
    - Ultra-fast inference (optimized ONNX models)
    - Lazy loading (only loads when first used)
    - Auto-caching in container
    - Production-ready performance
    
    Models:
    - BAAI/bge-small-en-v1.5: 384 dim, fast, recommended
    - BAAI/bge-base-en-v1.5: 768 dim, balanced
    - BAAI/bge-large-en-v1.5: 1024 dim, best quality
    - sentence-transformers/all-MiniLM-L6-v2: 384 dim, fast
    
    Example:
        >>> engine = FastEmbedEngine()
        >>> embedding = engine.embed("Hello world")
        >>> len(embedding)
        384
        
        >>> # Batch processing
        >>> embeddings = engine.embed_batch(["text1", "text2", "text3"])
        >>> len(embeddings)
        3
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize FastEmbed engine.
        
        Args:
            model_name: Model to use (defaults to fast, small model)
            cache_dir: Directory to cache models (defaults to ~/.cache/fastembed)
        """
        if not FASTEMBED_AVAILABLE:
            raise ImportError(
                "FastEmbed not available. Install with: pip install fastembed\n"
                "Or add 'fastembed' to your requirements.txt"
            )
        
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model: Optional[TextEmbedding] = None
        
        logger.info(f"FastEmbed engine initialized with model: {model_name}")
    
    def _load_model(self) -> TextEmbedding:
        """
        Lazy load model (only when first embedding is needed).
        
        Model downloads to ~/.cache/fastembed (or custom cache_dir)
        and persists across container restarts if volume is mounted.
        """
        if self._model is None:
            logger.info(f"Loading FastEmbed model: {self.model_name}")
            
            kwargs = {}
            if self.cache_dir:
                kwargs['cache_dir'] = self.cache_dir
            
            self._model = TextEmbedding(
                model_name=self.model_name,
                **kwargs
            )
            
            logger.info("FastEmbed model loaded successfully")
        
        return self._model
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector as list of floats
        
        Example:
            >>> engine = FastEmbedEngine()
            >>> embedding = engine.embed("Machine learning")
            >>> type(embedding)
            <class 'list'>
            >>> len(embedding)
            384
        """
        model = self._load_model()
        
        # FastEmbed returns generator, convert to list
        embeddings = list(model.embed([text]))
        
        # Convert numpy array to list
        return embeddings[0].tolist()
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Uses batching for optimal performance.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (default: 32)
        
        Returns:
            List of embedding vectors
        
        Example:
            >>> engine = FastEmbedEngine()
            >>> texts = ["text 1", "text 2", "text 3"]
            >>> embeddings = engine.embed_batch(texts)
            >>> len(embeddings)
            3
            >>> len(embeddings[0])
            384
        """
        if not texts:
            return []
        
        model = self._load_model()
        
        # FastEmbed handles batching internally
        embeddings = list(model.embed(texts))
        
        # Convert numpy arrays to lists
        return [emb.tolist() for emb in embeddings]
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            "model_name": self.model_name,
            "cache_dir": self.cache_dir,
            "loaded": self._model is not None,
            "backend": "FastEmbed (ONNX)",
            "cost": "Free (local execution)"
        }


def create_fastembed_function(
    model_name: str = "BAAI/bge-small-en-v1.5",
    cache_dir: Optional[str] = None
) -> callable:
    """
    Create an embedding function using FastEmbed.
    
    This is a convenience function that returns a simple callable
    for use with Cortex memory.
    
    Args:
        model_name: FastEmbed model to use
        cache_dir: Optional cache directory
    
    Returns:
        Callable that generates embeddings
    
    Example:
        >>> from teleon.cortex import CortexMemory, CortexConfig
        >>> from teleon.cortex.embeddings import create_fastembed_function
        >>> 
        >>> # Create embedding function
        >>> embed_fn = create_fastembed_function()
        >>> 
        >>> # Use with Cortex
        >>> cortex = CortexMemory(
        ...     storage=storage,
        ...     config=CortexConfig(embedding_function=embed_fn)
        ... )
    """
    engine = FastEmbedEngine(model_name=model_name, cache_dir=cache_dir)
    return engine.embed

