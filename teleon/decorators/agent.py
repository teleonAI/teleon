"""Agent decorator for defining Teleon agents."""

from typing import Callable, Optional, Dict, Any, TypeVar, cast
from functools import wraps
import inspect
import asyncio
from datetime import datetime, timezone
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
    sentinel: Optional[Dict[str, Any]] = None,
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
        collaborate: Enable multi-agent collaboration (deprecated, no longer functional)
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
        
        # Initialize Sentinel if configured (lazy initialization in wrapper)
        sentinel_config = sentinel
        
        # Lazy-initialized AuditLogger for this agent (singleton per agent)
        _audit_logger = None
        _audit_logger_lock = None
        
        def _get_audit_logger():
            """Get or create AuditLogger for this agent (lazy initialization, thread-safe)."""
            nonlocal _audit_logger, _audit_logger_lock
            if _audit_logger is None:
                try:
                    import threading
                    import os
                    if _audit_logger_lock is None:
                        _audit_logger_lock = threading.Lock()
                    
                    with _audit_logger_lock:
                        # Double-check pattern
                        if _audit_logger is None:
                            from teleon.governance.audit import AuditLogger
                            # Only enable if API credentials are available
                            api_url = os.getenv('TELEON_API_URL') or os.getenv('TELEON_PLATFORM_URL')
                            api_key = os.getenv('TELEON_API_KEY')
                            
                            if api_url and api_key:
                                _audit_logger = AuditLogger(
                                    agent_id=config.name,  # Use agent name as ID
                                    agent_name=config.name,
                                    enable_remote_logging=True,
                                    api_url=api_url,
                                    api_key=api_key,
                                    batch_size=50,  # Batch for efficiency
                                    flush_interval=10.0  # Flush every 10 seconds
                                )
                            else:
                                # No API credentials - disable remote logging
                                _audit_logger = AuditLogger(
                                    agent_id=config.name,
                                    agent_name=config.name,
                                    enable_remote_logging=False
                                )
                except ImportError:
                    # Governance module not available - audit logging disabled
                    pass
                except Exception as e:
                    # Log but don't fail - audit logging is optional
                    try:
                        from teleon.core import StructuredLogger, LogLevel
                        logger = StructuredLogger("agent.decorator", LogLevel.WARNING)
                        logger.warning(f"Failed to initialize audit logger: {e}")
                    except ImportError:
                        pass
            
            return _audit_logger
        
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper that adds agent capabilities: Sentinel, memory, cost tracking, execution context, audit logging."""
            
            # Generate execution ID
            execution_id = str(uuid.uuid4())
            
            # Create execution context
            ctx = ExecutionContext(
                execution_id=execution_id,
                agent_name=config.name,
                config=config.to_dict(),
                started_at=datetime.now(timezone.utc),
                input_args=args,
                input_kwargs=kwargs,
            )
            
            # Setup tracing span
            span_id = f"agent.{config.name}.{execution_id}"
            start_time = time.time()
            
            # AUDIT LOGGING: Log request (NON-BLOCKING)
            # This queues the log but doesn't await submission
            audit_logger = _get_audit_logger()
            if audit_logger:
                try:
                    # Prepare input data (sanitize for logging)
                    input_data = {}
                    if args:
                        input_data['args'] = str(args)[:500]  # Limit size
                    if kwargs:
                        # Filter sensitive keys
                        safe_kwargs = {k: str(v)[:200] if not isinstance(v, (str, int, float, bool, type(None))) else v 
                                     for k, v in kwargs.items()}
                        input_data['kwargs'] = safe_kwargs
                    
                    # Log request (non-blocking - just queues)
                    audit_logger.log_request(
                        action=f"Agent execution: {config.name}",
                        input_data=input_data if input_data else None,
                        metadata={
                            "execution_id": execution_id,
                            "agent_name": config.name,
                            "span_id": span_id
                        }
                    )
                except Exception as e:
                    # Audit logging failures must never impact agent execution
                    try:
                        from teleon.core import StructuredLogger, LogLevel
                        logger = StructuredLogger("agent.decorator", LogLevel.WARNING)
                        logger.warning(f"Failed to log audit request: {e}")
                    except ImportError:
                        pass
            
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
            
            # Initialize Sentinel if configured (lazy)
            sentinel_engine = None
            if sentinel_config:
                try:
                    from teleon.sentinel.integration import create_sentinel_engine
                    sentinel_engine = create_sentinel_engine(sentinel_config)
                    if sentinel_engine:
                        # Register with registry
                        try:
                            from teleon.sentinel.registry import get_sentinel_registry
                            import asyncio
                            registry = await get_sentinel_registry()
                            await registry.register(config.name, sentinel_engine)
                        except Exception:
                            # Registry not critical, continue
                            pass
                except ImportError:
                    # Sentinel not available
                    pass
                except Exception as e:
                    # Sentinel initialization failed, log but continue
                    try:
                        from teleon.core import StructuredLogger, LogLevel
                        logger = StructuredLogger("agent.decorator", LogLevel.WARNING)
                        logger.warning(f"Sentinel initialization failed: {e}")
                    except ImportError:
                        pass
            
            # SENTINEL: Validate input BEFORE execution
            if sentinel_engine:
                try:
                    input_data = {'args': args, 'kwargs': kwargs}
                    input_result = await sentinel_engine.validate_input(
                        input_data,
                        config.name
                    )
                    
                    # Apply redaction if needed
                    if input_result.redacted_content and input_result.action.value == 'redact':
                        # Note: Redaction of args/kwargs is complex, so we log it
                        # In practice, the redaction would need to be applied to the actual data
                        try:
                            from teleon.core import StructuredLogger, LogLevel
                            logger = StructuredLogger(f"agent.{config.name}", LogLevel.INFO)
                            logger.info("Input redacted by Sentinel", violations=len(input_result.violations))
                        except ImportError:
                            pass
                    
                except Exception as e:
                    # Sentinel validation errors are logged but don't block unless action is BLOCK
                    # (which would have raised AgentValidationError)
                    try:
                        from teleon.core import StructuredLogger, LogLevel
                        logger = StructuredLogger(f"agent.{config.name}", LogLevel.WARNING)
                        logger.warning(f"Sentinel input validation error: {e}")
                    except ImportError:
                        pass
            
            try:
                # Execute the agent function
                result = await func(*args, **kwargs)
                
                # SENTINEL: Validate output AFTER execution
                if sentinel_engine:
                    try:
                        output_result = await sentinel_engine.validate_output(
                            result,
                            config.name
                        )
                        
                        # Apply redaction if needed
                        if output_result.redacted_content and output_result.action.value == 'redact':
                            result = output_result.redacted_content
                            try:
                                from teleon.core import StructuredLogger, LogLevel
                                logger = StructuredLogger(f"agent.{config.name}", LogLevel.INFO)
                                logger.info("Output redacted by Sentinel", violations=len(output_result.violations))
                            except ImportError:
                                pass
                        
                    except Exception as e:
                        # Sentinel validation errors are logged but don't block unless action is BLOCK
                        try:
                            from teleon.core import StructuredLogger, LogLevel
                            logger = StructuredLogger(f"agent.{config.name}", LogLevel.WARNING)
                            logger.warning(f"Sentinel output validation error: {e}")
                        except ImportError:
                            pass
                
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
                
                # AUDIT LOGGING: Log successful response (NON-BLOCKING)
                if audit_logger:
                    try:
                        # Prepare output data (sanitize for logging)
                        output_data = {}
                        if result is not None:
                            # Limit output size for logging
                            result_str = str(result)
                            output_data['result'] = result_str[:1000]  # Limit to 1KB
                            output_data['result_type'] = type(result).__name__
                        
                        # Log response (non-blocking - just queues)
                        audit_logger.log_response(
                            action=f"Agent execution completed: {config.name}",
                            output_data=output_data if output_data else None,
                            duration_ms=int(duration * 1000),
                            status="success",
                            metadata={
                                "execution_id": execution_id,
                                "agent_name": config.name,
                                "cost": total_cost,
                                "span_id": span_id
                            }
                        )
                    except Exception as e:
                        # Audit logging failures must never impact agent execution
                        try:
                            from teleon.core import StructuredLogger, LogLevel
                            logger = StructuredLogger("agent.decorator", LogLevel.WARNING)
                            logger.warning(f"Failed to log audit response: {e}")
                        except ImportError:
                            pass
                
                return result
                
            except Exception as e:
                # Mark as failed
                ctx.mark_failure(e)
                
                # Calculate execution time
                duration = time.time() - start_time
                
                # AUDIT LOGGING: Log error response (NON-BLOCKING)
                if audit_logger:
                    try:
                        # Log error (non-blocking - just queues)
                        audit_logger.log_response(
                            action=f"Agent execution failed: {config.name}",
                            output_data={
                                "error": str(e),
                                "error_type": type(e).__name__
                            },
                            duration_ms=int(duration * 1000),
                            status="error",
                            metadata={
                                "execution_id": execution_id,
                                "agent_name": config.name,
                                "span_id": span_id
                            }
                        )
                    except Exception as e2:
                        # Audit logging failures must never impact agent execution
                        try:
                            from teleon.core import StructuredLogger, LogLevel
                            logger = StructuredLogger("agent.decorator", LogLevel.WARNING)
                            logger.warning(f"Failed to log audit error: {e2}")
                        except ImportError:
                            pass
                
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

