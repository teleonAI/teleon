"""
Embedding engines for Cortex semantic memory.

Provides multiple embedding backends:
- FastEmbed: Fast, local embeddings (recommended for production)
- OpenAI: Cloud-based embeddings (requires API key)
- Custom: Bring your own embedding function
"""

from typing import Optional, Callable, List

try:
    from teleon.cortex.embeddings.fastembed_engine import (
        FastEmbedEngine,
        create_fastembed_function
    )
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    FastEmbedEngine = None
    create_fastembed_function = None


__all__ = [
    "FastEmbedEngine",
    "create_fastembed_function",
    "create_embedding_function",
    "FASTEMBED_AVAILABLE",
]


def create_embedding_function(
    backend: str = "fastembed",
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> Callable[[str], List[float]]:
    """
    Create an embedding function.
    
    Args:
        backend: Embedding backend ("fastembed", "openai", "custom")
        model: Model name (backend-specific)
        api_key: API key (for cloud backends)
    
    Returns:
        Callable that generates embeddings
    
    Examples:
        >>> # Use FastEmbed (recommended, free)
        >>> embed_fn = create_embedding_function("fastembed")
        
        >>> # Use OpenAI
        >>> embed_fn = create_embedding_function(
        ...     "openai",
        ...     model="text-embedding-3-small",
        ...     api_key="sk-..."
        ... )
    """
    if backend == "fastembed":
        if not FASTEMBED_AVAILABLE:
            raise ImportError(
                "FastEmbed not available. Install with: pip install fastembed"
            )
        model = model or "BAAI/bge-small-en-v1.5"
        return create_fastembed_function(model)
    
    elif backend == "openai":
        if not api_key:
            raise ValueError("OpenAI backend requires api_key parameter")
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI not available. Install with: pip install openai"
            )
        
        client = OpenAI(api_key=api_key)
        model = model or "text-embedding-3-small"
        
        def openai_embed(text: str) -> List[float]:
            response = client.embeddings.create(
                model=model,
                input=text
            )
            return response.data[0].embedding
        
        return openai_embed
    
    else:
        raise ValueError(
            f"Unknown embedding backend: {backend}. "
            f"Supported: fastembed, openai"
        )

