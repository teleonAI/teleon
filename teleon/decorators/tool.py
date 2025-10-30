"""Tool decorator for defining Teleon tools."""

from typing import Callable, Optional, Any, TypeVar, cast
from functools import wraps
import inspect
import asyncio
import time
import hashlib
import json

F = TypeVar('F', bound=Callable[..., Any])

# Simple in-memory cache
_tool_cache: dict = {}


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: Optional[str] = None,
    cache_ttl: Optional[int] = None,
    timeout: Optional[float] = None,
) -> Callable[[F], F]:
    """
    Decorator to define a Teleon tool.
    
    Tools are reusable functions that agents can execute. They can be:
    - Built-in tools (web search, data processing, etc.)
    - Custom tools (your business logic)
    - Third-party integrations (APIs, databases, etc.)
    
    Example:
        >>> @tool(name="calculator", category="math")
        >>> async def calculate(expression: str) -> float:
        ...     '''Calculate a mathematical expression.'''
        ...     return eval(expression)
        
        >>> @tool(
        ...     name="web-search",
        ...     description="Search the web for information",
        ...     category="search",
        ...     cache_ttl=3600
        ... )
        >>> async def web_search(query: str, num_results: int = 5) -> dict:
        ...     '''Search the web and return results.'''
        ...     return {'results': [...]}
    
    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        category: Tool category for organization
        cache_ttl: Cache results for this many seconds
        timeout: Maximum execution time in seconds
    
    Returns:
        Decorated function with tool capabilities
    """
    def decorator(func: F) -> F:
        # Ensure function is async
        if not asyncio.iscoroutinefunction(func):
            # Wrap sync functions to be async
            original_func = func
            
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return original_func(*args, **kwargs)
            
            func = cast(F, async_wrapper)
        
        # Get function signature
        sig = inspect.signature(func)
        
        # Get description from docstring if not provided
        tool_description = description or (func.__doc__ or "").strip()
        
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper that adds tool capabilities."""
            tool_name = name or func.__name__
            start_time = time.time()
            
            # Generate cache key if caching is enabled
            cache_key = None
            if cache_ttl:
                cache_data = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True, default=str)
                cache_key = f"{tool_name}:{hashlib.md5(cache_data.encode()).hexdigest()}"
                
                # Check cache if cache_ttl is set
                if cache_key in _tool_cache:
                    cached_result, cached_time = _tool_cache[cache_key]
                    if time.time() - cached_time < cache_ttl:
                        # Cache hit - return cached result
                        return cached_result
            
            try:
                # Setup timeout if specified
                if timeout:
                    result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                else:
                    result = await func(*args, **kwargs)
                
                # Track cost and duration
                duration = time.time() - start_time
                
                # Cache result if cache_ttl is set
                if cache_key and cache_ttl:
                    _tool_cache[cache_key] = (result, time.time())
                    # Simple cache cleanup - remove if too many entries
                    if len(_tool_cache) > 1000:
                        # Remove oldest 100 entries
                        oldest_keys = sorted(_tool_cache.items(), key=lambda x: x[1][1])[:100]
                        for key, _ in oldest_keys:
                            del _tool_cache[key]
                
                # Record successful execution metrics
                try:
                    from teleon.core import get_metrics
                    get_metrics().record_tool_execution(
                        tool_name=tool_name,
                        duration=duration,
                        status="success"
                    )
                except ImportError:
                    # Metrics not available in basic setup
                    pass
                
                return result
                
            except asyncio.TimeoutError:
                # Record timeout failure
                duration = time.time() - start_time
                try:
                    from teleon.core import get_metrics
                    get_metrics().record_tool_execution(
                        tool_name=tool_name,
                        duration=duration,
                        status="timeout"
                    )
                except ImportError:
                    pass
                raise TimeoutError(f"Tool '{tool_name}' timed out after {timeout}s")
                
            except Exception as e:
                # Record failed execution
                duration = time.time() - start_time
                try:
                    from teleon.core import get_metrics
                    get_metrics().record_tool_execution(
                        tool_name=tool_name,
                        duration=duration,
                        status="error"
                    )
                    get_metrics().record_error("tool", type(e).__name__)
                except ImportError:
                    pass
                raise
        
        # Attach metadata
        wrapper._teleon_tool = True  # type: ignore
        wrapper._teleon_tool_config = {  # type: ignore
            'name': name or func.__name__,
            'description': tool_description,
            'category': category or 'general',
            'cache_ttl': cache_ttl,
            'timeout': timeout,
            'signature': sig,
        }
        
        return cast(F, wrapper)
    
    return decorator

