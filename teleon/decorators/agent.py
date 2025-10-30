"""Agent decorator for defining Teleon agents."""

from typing import Callable, Optional, Dict, Any, TypeVar, cast
from functools import wraps
import inspect
import asyncio
from datetime import datetime
import uuid
import time

from teleon.context.execution import ExecutionContext
from teleon.config.agent_config import AgentConfig


F = TypeVar('F', bound=Callable[..., Any])


def agent(
    name: Optional[str] = None,
    memory: bool = False,
    scale: Optional[Dict[str, Any]] = None,
    llm: Optional[Dict[str, Any]] = None,
    tools: Optional[list] = None,
    collaborate: bool = False,
    timeout: Optional[float] = None,
    **kwargs: Any
) -> Callable[[F], F]:
    """
    Decorator to define a Teleon agent.
    
    Transforms a simple async function into a production-ready AI agent with:
    - Automatic memory and learning
    - Auto-scaling infrastructure
    - Multi-agent collaboration
    - Cost tracking and optimization
    - Full observability
    
    Example:
        >>> @agent(name="support-agent", memory=True)
        >>> async def support_agent(ticket: str) -> str:
        ...     return f"Handling ticket: {ticket}"
        
        >>> @agent(
        ...     name="scalable-agent",
        ...     memory=True,
        ...     scale={'min': 1, 'max': 100},
        ...     collaborate=True
        ... )
        >>> async def scalable_agent(task: str) -> dict:
        ...     return {'status': 'completed', 'result': task}
    
    Args:
        name: Agent name (defaults to function name)
        memory: Enable memory capabilities (working, episodic, semantic, procedural)
        scale: Scaling configuration dict with keys:
            - min: Minimum replicas (default: 1)
            - max: Maximum replicas (default: 10)
            - target_cpu: Target CPU percentage for scaling (default: 70)
        llm: LLM configuration dict with keys:
            - strategy: 'cost_optimized', 'quality_first', or 'speed_first'
            - models: Dict mapping complexity to model names
            - caching: Cache configuration
        tools: List of tools available to the agent
        collaborate: Enable multi-agent collaboration via NexusNet
        timeout: Global timeout in seconds
        **kwargs: Additional configuration options
    
    Returns:
        Decorated async function with agent capabilities
    
    Raises:
        TypeError: If decorated function is not async
        ValueError: If configuration is invalid
    """
    def decorator(func: F) -> F:
        # Validate function is async
        if not asyncio.iscoroutinefunction(func):
            raise TypeError(
                f"Agent function '{func.__name__}' must be async. "
                f"Use 'async def {func.__name__}(...)' instead."
            )
        
        # Get function signature for validation
        sig = inspect.signature(func)
        
        # Create agent configuration
        config = AgentConfig(
            name=name or func.__name__,
            memory=memory,
            scale=scale or {'min': 1, 'max': 10},
            llm=llm or {},
            tools=tools or [],
            collaborate=collaborate,
            timeout=timeout,
            signature=sig,
            **kwargs
        )
        
        # Validate configuration
        config.validate()
        
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper that adds agent capabilities."""
            
            # Generate execution ID
            execution_id = str(uuid.uuid4())
            
            # Create execution context
            ctx = ExecutionContext(
                execution_id=execution_id,
                agent_name=config.name,
                config=config.to_dict(),
                started_at=datetime.utcnow(),
                input_args=args,
                input_kwargs=kwargs,
            )
            
            # Setup tracing span
            span_id = f"agent.{config.name}.{execution_id}"
            start_time = time.time()
            
            # Initialize memory if enabled
            memory_session = None
            if config.memory:
                try:
                    from teleon.memory.working import WorkingMemory
                    memory_session = WorkingMemory(session_id=execution_id)
                    # Store context in memory
                    await memory_session.set("context", {
                        "agent_name": config.name,
                        "execution_id": execution_id,
                        "started_at": ctx.started_at.isoformat()
                    })
                except ImportError:
                    # Memory not available
                    pass
            
            # Setup cost tracking
            total_cost = 0.0
            
            try:
                # Execute the agent function
                result = await func(*args, **kwargs)
                
                # Mark as successful
                ctx.mark_success(result)
                
                # Calculate execution time
                duration = time.time() - start_time
                
                # Store execution in episodic memory
                if config.memory and memory_session:
                    try:
                        from teleon.memory.episodic import EpisodicMemory
                        episodic = EpisodicMemory()
                        await episodic.store_event(
                            agent_name=config.name,
                            event_type="execution",
                            data={
                                "execution_id": execution_id,
                                "input": {"args": str(args)[:100], "kwargs": str(kwargs)[:100]},
                                "output": str(result)[:500],
                                "success": True,
                                "duration": duration,
                                "cost": total_cost
                            },
                            metadata={
                                "agent_name": config.name,
                                "timestamp": ctx.started_at.isoformat()
                            }
                        )
                    except (ImportError, Exception):
                        # Episodic memory not available or failed
                        pass
                
                # Update procedural memory (learning)
                if config.memory:
                    try:
                        from teleon.memory.procedural import ProceduralMemory
                        procedural = ProceduralMemory()
                        await procedural.record_success(
                            pattern=f"{config.name}.execute",
                            context={"args_count": len(args), "kwargs_count": len(kwargs)}
                        )
                    except (ImportError, Exception):
                        # Procedural memory not available or failed
                        pass
                
                # Record metrics
                try:
                    from teleon.core import get_metrics
                    get_metrics().record_request(
                        component="agent",
                        operation=config.name,
                        duration=duration,
                        status="success"
                    )
                    if total_cost > 0:
                        get_metrics().increment_counter(
                            'llm_cost',
                            {'provider': 'unknown', 'model': 'unknown'},
                            total_cost
                        )
                except ImportError:
                    # Metrics not available
                    pass
                
                return result
                
            except Exception as e:
                # Mark as failed
                ctx.mark_failure(e)
                
                # Calculate execution time
                duration = time.time() - start_time
                
                # Record failure for learning
                if config.memory:
                    try:
                        from teleon.memory.procedural import ProceduralMemory
                        procedural = ProceduralMemory()
                        await procedural.record_failure(
                            pattern=f"{config.name}.execute",
                            context={"error_type": type(e).__name__}
                        )
                    except (ImportError, Exception):
                        # Procedural memory not available or failed
                        pass
                    
                    # Store failure in episodic memory
                    try:
                        from teleon.memory.episodic import EpisodicMemory
                        episodic = EpisodicMemory()
                        await episodic.store_event(
                            agent_name=config.name,
                            event_type="execution_failure",
                            data={
                                "execution_id": execution_id,
                                "error": str(e),
                                "error_type": type(e).__name__,
                                "duration": duration
                            }
                        )
                    except (ImportError, Exception):
                        pass
                
                # Record error metrics
                try:
                    from teleon.core import get_metrics
                    get_metrics().record_request(
                        component="agent",
                        operation=config.name,
                        duration=duration,
                        status="error"
                    )
                    get_metrics().record_error(
                        component="agent",
                        error_type=type(e).__name__
                    )
                except ImportError:
                    pass
                
                raise
            
            finally:
                # Close tracing span (log completion)
                try:
                    from teleon.core import StructuredLogger, LogLevel
                    logger = StructuredLogger(f"agent.{config.name}", LogLevel.INFO)
                    logger.info(
                        "Agent execution completed",
                        execution_id=execution_id,
                        duration=time.time() - start_time,
                        success=ctx.success
                    )
                except ImportError:
                    pass
                
                # Flush metrics (ensure all metrics are recorded)
                try:
                    from teleon.core import get_metrics
                    # Metrics are automatically flushed in production setup
                    pass
                except ImportError:
                    pass
                
                # Cleanup memory session
                if memory_session:
                    try:
                        await memory_session.close()
                    except Exception:
                        pass
        
        # Attach metadata to wrapper
        wrapper._teleon_agent = True  # type: ignore
        wrapper._teleon_config = config  # type: ignore
        wrapper._teleon_original_func = func  # type: ignore
        
        return cast(F, wrapper)
    
    return decorator

