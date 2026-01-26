"""
Teleon Client - User authentication and agent registration
"""

import os
from typing import Optional, Dict, Any, Callable, List, TypeVar, cast
import hashlib
from datetime import datetime, timezone
import httpx


class TeleonClient:
    """
    Teleon Client for authenticating users and registering agents.
    
    Usage:
        client = TeleonClient(api_key="your-teleon-api-key")
        
        @client.agent(name="my-agent")
        def my_agent(input: str) -> str:
            return "response"
    """
    
    # Global registry of all clients and their agents
    _instances: Dict[str, 'TeleonClient'] = {}
    _all_agents: Dict[str, Dict[str, Any]] = {}
    
    def __init__(
        self,
        api_key: str,
        environment: str = "production",
        base_url: Optional[str] = None,
        verify_key: bool = True
    ):
        """
        Initialize Teleon client.
        
        Args:
            api_key: Your Teleon API key from the platform
            environment: Environment (production, staging, dev)
            base_url: Custom API base URL (default: uses environment)
            verify_key: Whether to verify API key with backend (default: True)
        """
        self.api_key = api_key
        self.environment = environment
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.scopes: List[str] = []  # API key scopes/permissions
        
        # Validate API key format
        if not api_key or not isinstance(api_key, str):
            raise ValueError("Valid Teleon API key is required")
        
        # Validate API key format pattern
        if not api_key.startswith(('tlk_live_', 'tlk_test_', 'teleon_')):
            raise ValueError(
                f"Invalid API key format. Expected format: tlk_live_xxx or tlk_test_xxx\n"
                f"Get your API key from: https://dashboard.teleon.ai"
            )
        
        # Set base URL based on environment
        if base_url:
            self.base_url = base_url
        elif environment == "production":
            self.base_url = "https://api.teleon.ai"
        elif environment == "staging":
            self.base_url = "https://api.staging.teleon.ai"
        else:  # dev
            self.base_url = "http://localhost:8000"
        
        # Generate user ID from API key
        self.user_id = self._generate_user_id(api_key)
        
        # Initialize HTTP client for remote operations
        self._http_client = None
        
        # Verify API key with backend if requested
        if verify_key and environment != "dev":
            self._verify_api_key_sync()
        
        # Register this client instance
        TeleonClient._instances[self.user_id] = self
        
        # Only print if not in quiet mode (used during agent discovery)
        if not os.getenv('TELEON_QUIET'):
            print(f"✓ Teleon Client initialized")
            print(f"  User ID: {self.user_id}")
            print(f"  Environment: {environment}")
            print(f"  API URL: {self.base_url}")
    
    def _generate_user_id(self, api_key: str) -> str:
        """Generate a user ID from API key."""
        return hashlib.sha256(api_key.encode()).hexdigest()[:12]
    
    def has_scope(self, *required_scopes: str) -> bool:
        """
        Check if API key has any of the required scopes.
        
        Args:
            *required_scopes: One or more required scopes (ANY match returns True)
        
        Returns:
            True if key has at least one of the required scopes
        
        Example:
            if not client.has_scope('agents:deploy'):
                raise ValueError("API key needs 'agents:deploy' permission")
        """
        if not self.scopes:
            # If scopes weren't loaded, assume dev mode or old key
            return True
        
        return any(scope in self.scopes for scope in required_scopes)
    
    def require_scope(self, *required_scopes: str):
        """
        Raise an error if API key doesn't have required scopes.
        
        Args:
            *required_scopes: One or more required scopes (ANY match is OK)
        
        Raises:
            ValueError: If key doesn't have any of the required scopes
        
        Example:
            client.require_scope('agents:deploy')  # Will raise if missing
        """
        if not self.has_scope(*required_scopes):
            raise ValueError(
                f"Insufficient API key permissions. Required scopes: {', '.join(required_scopes)}. "
                f"Your key has: {', '.join(self.scopes) if self.scopes else 'none'}. "
                f"Please create a new API key with the required permissions at https://dashboard.teleon.ai"
            )
    
    def _verify_api_key_sync(self):
        """
        Verify API key with the backend (synchronous version for __init__).
        
        This makes a simple API call to validate the key is real and active.
        """
        import time
        
        try:
            # Use synchronous httpx client for initialization
            with httpx.Client(timeout=10.0) as client:
                # Try to validate the API key using the validate endpoint
                response = client.get(
                    f"{self.base_url}/api/v1/api-keys/validate",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )
                
                # Check if authentication succeeded
                if response.status_code == 401:
                    try:
                        error_detail = response.json().get("detail", "Invalid API key")
                    except:
                        error_detail = "Invalid API key"
                    raise ValueError(
                        f"API key verification failed: {error_detail}\n"
                        f"Please check your API key and try again.\n"
                        f"Get a valid API key from: https://dashboard.teleon.ai"
                    )
                elif response.status_code == 403:
                    raise ValueError(
                        f"API key is valid but your account is inactive.\n"
                        f"Please contact support or check your account status."
                    )
                elif response.status_code == 429:
                    raise ValueError(
                        f"Rate limit exceeded. Please wait a moment and try again."
                    )
                elif response.status_code >= 500:
                    # Server error - allow to proceed but warn
                    print(f"  ⚠️  Warning: Unable to verify API key (server error)")
                    return
                elif response.status_code >= 400:
                    raise ValueError(
                        f"API key verification failed with status {response.status_code}\n"
                        f"Response: {response.text}"
                    )
                
                # Success! Get API key info including scopes
                if response.status_code == 200:
                    try:
                        key_info = response.json()
                        self.scopes = key_info.get("key", {}).get("scopes", [])
                        if not os.getenv('TELEON_QUIET'):
                            print(f"  ✓ API key verified successfully")
                            if self.scopes:
                                print(f"  ℹ️  Scopes: {', '.join(self.scopes)}")
                            else:
                                print(f"  ⚠️  Warning: No scopes assigned to this API key")
                    except Exception:
                        # Couldn't parse scopes but key is valid
                        if not os.getenv('TELEON_QUIET'):
                            print(f"  ✓ API key verified successfully")
                else:
                    if not os.getenv('TELEON_QUIET'):
                        print(f"  ✓ API key verified")
                
        except httpx.ConnectError:
            # Can't connect to server - allow in dev mode
            if not os.getenv('TELEON_QUIET'):
                print(f"  ⚠️  Warning: Could not connect to {self.base_url}")
                print(f"  Skipping API key verification (server may not be running)")
        except httpx.TimeoutException:
            if not os.getenv('TELEON_QUIET'):
                print(f"  ⚠️  Warning: Connection timeout")
                print(f"  Skipping API key verification")
        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Other errors - warn but don't block
            if not os.getenv('TELEON_QUIET'):
                print(f"  ⚠️  Warning: Could not verify API key: {str(e)}")
    
    def agent(
        self,
        name: str,
        description: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 500,
        helix: Optional[Dict[str, Any]] = None,
        cortex: Optional[Dict[str, Any]] = None,
        sentinel: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Decorator to register an agent with this client.
        
        Now includes full runtime features:
        - Sentinel validation (input/output)
        - Memory integration (working, episodic, procedural)
        - Cost tracking
        - Execution context
        - Metrics recording
        
        Args:
            name: Agent name
            description: Agent description
            model: LLM model to use
            temperature: Temperature setting
            max_tokens: Max tokens
            helix: Helix runtime configuration (auto-scaling, health checks)
            cortex: Cortex memory configuration (learning, memory types)
            sentinel: Sentinel safety and compliance configuration
            **kwargs: Additional configuration
        
        Returns:
            Decorated function with full runtime features
        
        Example:
            ```python
            @client.agent(
                name="my-agent",
                helix={'min': 2, 'max': 10, 'target_cpu': 70},
                cortex={'learning': True, 'memory_types': ['episodic', 'semantic']},
                sentinel={'content_filtering': True, 'pii_detection': True}
            )
            async def my_agent(input: str):
                return process(input)
            ```
        """
        def decorator(func: Callable):
            # Validate function is async
            import asyncio
            if not asyncio.iscoroutinefunction(func):
                raise TypeError(
                    f"Agent function '{func.__name__}' must be async. "
                    f"Use 'async def {func.__name__}(...)' instead."
                )
            
            # Generate unique agent ID
            agent_id = self._generate_agent_id(name)
            
            # Extract function signature for OpenAPI
            import inspect
            sig = inspect.signature(func)
            params = {}
            for param_name, param in sig.parameters.items():
                params[param_name] = {
                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "string",
                    "required": param.default == inspect.Parameter.empty
                }
            
            # Map cortex config to memory config (for @agent compatibility)
            memory_enabled = False
            if cortex:
                # Enable memory if cortex has learning or memory_types
                memory_enabled = (
                    cortex.get('learning', False) or
                    bool(cortex.get('memory_types', []))
                )
            
            # Map helix config to scale config (for @agent compatibility)
            scale_config = None
            if helix:
                scale_config = {
                    'min': helix.get('min', 1),
                    'max': helix.get('max', 10),
                    'target_cpu': helix.get('target_cpu', 70)
                }
            
            # Create AgentConfig for runtime features
            from teleon.config.agent_config import AgentConfig
            agent_config = AgentConfig(
                name=name,
                memory=memory_enabled,
                scale=scale_config or {'min': 1, 'max': 10},
                llm={'model': model, 'temperature': temperature, 'max_tokens': max_tokens},
                tools=kwargs.get('tools', []),
                collaborate=False,
                timeout=kwargs.get('timeout'),
                signature=sig,
                **{k: v for k, v in kwargs.items() if k not in ['tools', 'timeout']}
            )
            agent_config.validate()
            
            # Store sentinel config for wrapper
            sentinel_config = sentinel
            
            # Store cortex config for wrapper
            cortex_config = cortex
            
            # Lazy-initialized Cortex instance for this agent
            _cortex_instance = None
            _cortex_lock = None
            
            async def _get_cortex():
                """Get or create Cortex instance (lazy initialization, thread-safe)."""
                nonlocal _cortex_instance, _cortex_lock
                if _cortex_instance is None and cortex_config:
                    try:
                        import asyncio
                        if _cortex_lock is None:
                            _cortex_lock = asyncio.Lock()
                        
                        async with _cortex_lock:
                            # Double-check pattern
                            if _cortex_instance is None:
                                from teleon.cortex import create_cortex, CortexConfig
                                from teleon.cortex.registry import registry
                                
                                # Try to get from registry first
                                _cortex_instance = await registry.get(agent_id)
                                
                                if _cortex_instance is None:
                                    # Map cortex config dict to CortexConfig
                                    # The cortex_config dict has keys like 'learning', 'memory_types', etc.
                                    # which need to be mapped to CortexConfig fields
                                    cortex_cfg = None
                                    if isinstance(cortex_config, dict):
                                        # Map memory_types to enabled flags
                                        memory_types = cortex_config.get('memory_types', [])
                                        cortex_cfg = CortexConfig(
                                            working_enabled='working' in memory_types if memory_types else True,
                                            episodic_enabled='episodic' in memory_types if memory_types else True,
                                            semantic_enabled='semantic' in memory_types if memory_types else True,
                                            procedural_enabled='procedural' in memory_types if memory_types else True,
                                            learning_enabled=cortex_config.get('learning', True),
                                            min_success_rate=cortex_config.get('procedural_config', {}).get('min_success_rate', 50.0) if isinstance(cortex_config.get('procedural_config'), dict) else 50.0
                                        )
                                    
                                    # Create new Cortex instance
                                    _cortex_instance = await create_cortex(
                                        agent_id=agent_id,
                                        storage_backend=cortex_config.get('storage', 'memory') if isinstance(cortex_config, dict) else 'memory',
                                        config=cortex_cfg
                                    )
                                    await _cortex_instance.initialize()
                                    # Register it
                                    await registry.register(agent_id, _cortex_instance)
                    except Exception as e:
                        # Cortex initialization failed, log but continue
                        try:
                            from teleon.core import StructuredLogger, LogLevel
                            logger = StructuredLogger("agent.client", LogLevel.WARNING)
                            logger.warning(f"Failed to initialize Cortex: {e}")
                        except ImportError:
                            pass
                        _cortex_instance = None
                
                return _cortex_instance
            
            # Lazy-initialized AuditLogger for this agent (singleton per agent)
            _audit_logger = None
            _audit_logger_lock = None
            
            def _get_audit_logger():
                """Get or create AuditLogger for this agent (lazy initialization, thread-safe)."""
                nonlocal _audit_logger, _audit_logger_lock
                if _audit_logger is None:
                    try:
                        import threading
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
                                        agent_id=agent_id,
                                        agent_name=name,
                                        enable_remote_logging=True,
                                        api_url=api_url,
                                        api_key=api_key,
                                        batch_size=50,  # Batch for efficiency
                                        flush_interval=10.0  # Flush every 10 seconds
                                    )
                                else:
                                    # No API credentials - disable remote logging
                                    _audit_logger = AuditLogger(
                                        agent_id=agent_id,
                                        agent_name=name,
                                        enable_remote_logging=False
                                    )
                    except ImportError:
                        # Governance module not available - audit logging disabled
                        pass
                    except Exception as e:
                        # Log but don't fail - audit logging is optional
                        try:
                            from teleon.core import StructuredLogger, LogLevel
                            logger = StructuredLogger("agent.client", LogLevel.WARNING)
                            logger.warning(f"Failed to initialize audit logger: {e}")
                        except ImportError:
                            pass
                
                return _audit_logger
            
            # Create wrapped function with runtime features
            from functools import wraps
            from datetime import datetime, timezone
            import uuid
            import time
            
            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                """Wrapper that adds agent capabilities: Sentinel, memory, cost tracking, execution context, audit logging."""
                
                # Generate execution ID
                execution_id = str(uuid.uuid4())
                
                # Create execution context
                from teleon.context.execution import ExecutionContext
                ctx = ExecutionContext(
                    execution_id=execution_id,
                    agent_name=name,
                    config=agent_config.to_dict(),
                    started_at=datetime.now(timezone.utc),
                    input_args=args,
                    input_kwargs=kwargs,
                )
                
                # Setup tracing span
                span_id = f"agent.{name}.{execution_id}"
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
                            action=f"Agent execution: {name}",
                            input_data=input_data if input_data else None,
                            metadata={
                                "execution_id": execution_id,
                                "agent_id": agent_id,
                                "span_id": span_id
                            }
                        )
                    except Exception as e:
                        # Audit logging failures must never impact agent execution
                        try:
                            from teleon.core import StructuredLogger, LogLevel
                            logger = StructuredLogger("agent.client", LogLevel.WARNING)
                            logger.warning(f"Failed to log audit request: {e}")
                        except ImportError:
                            pass
                
                # Initialize memory if enabled
                memory_session = None
                if memory_enabled:
                    try:
                        from teleon.memory.working import WorkingMemory
                        memory_session = WorkingMemory(session_id=execution_id)
                        # Store context in memory
                        await memory_session.set("context", {
                            "agent_name": name,
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
                        sentinel_engine = create_sentinel_engine(
                            sentinel_config,
                            agent_id=agent_id,
                            agent_name=name
                        )
                        if sentinel_engine:
                            # Register with registry
                            try:
                                from teleon.sentinel.registry import get_sentinel_registry
                                registry = await get_sentinel_registry()
                                await registry.register(name, sentinel_engine)
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
                            logger = StructuredLogger("agent.client", LogLevel.WARNING)
                            logger.warning(f"Sentinel initialization failed: {e}")
                        except ImportError:
                            pass
                
                # SENTINEL: Validate input BEFORE execution
                if sentinel_engine:
                    try:
                        input_data = {'args': args, 'kwargs': kwargs}
                        input_result = await sentinel_engine.validate_input(
                            input_data,
                            name
                        )
                        
                        # Handle violations based on action (BLOCK already raised by engine)
                        if not input_result.passed:
                            if input_result.action.value == 'redact' and input_result.redacted_content:
                                # Note: Redaction of args/kwargs is complex, so we log it
                                try:
                                    from teleon.core import StructuredLogger, LogLevel
                                    logger = StructuredLogger(f"agent.{name}", LogLevel.INFO)
                                    logger.info("Input redacted by Sentinel", violations=len(input_result.violations))
                                except ImportError:
                                    pass
                            
                            # Log violations (FLAG mode - violations detected but not blocked)
                            if input_result.action.value != 'block':
                                try:
                                    from teleon.core import StructuredLogger, LogLevel
                                    logger = StructuredLogger(f"agent.{name}", LogLevel.WARNING)
                                    logger.warning(
                                        "Sentinel input violations detected",
                                        violations=input_result.violations,
                                        action=input_result.action.value
                                    )
                                except ImportError:
                                    pass
                    
                    except Exception as e:
                        # Re-raise AgentValidationError (blocking violations)
                        from teleon.core.exceptions import AgentValidationError
                        if isinstance(e, AgentValidationError):
                            raise
                        # Other Sentinel errors are logged but don't block
                        try:
                            from teleon.core import StructuredLogger, LogLevel
                            logger = StructuredLogger(f"agent.{name}", LogLevel.WARNING)
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
                                name
                            )
                            
                            # Apply redaction if needed
                            if output_result.redacted_content and output_result.action.value == 'redact':
                                result = output_result.redacted_content
                                try:
                                    from teleon.core import StructuredLogger, LogLevel
                                    logger = StructuredLogger(f"agent.{name}", LogLevel.INFO)
                                    logger.info("Output redacted by Sentinel", violations=len(output_result.violations))
                                except ImportError:
                                    pass
                            
                            # Log violations (FLAG mode - violations detected but not blocked)
                            if output_result.violations and output_result.action.value != 'block':
                                try:
                                    from teleon.core import StructuredLogger, LogLevel
                                    logger = StructuredLogger(f"agent.{name}", LogLevel.WARNING)
                                    logger.warning(
                                        "Sentinel output violations detected",
                                        violations=output_result.violations,
                                        action=output_result.action.value
                                    )
                                except ImportError:
                                    pass
                        
                        except Exception as e:
                            # Re-raise AgentValidationError (blocking violations)
                            from teleon.core.exceptions import AgentValidationError
                            if isinstance(e, AgentValidationError):
                                raise
                            # Other Sentinel errors are logged but don't block
                            try:
                                from teleon.core import StructuredLogger, LogLevel
                                logger = StructuredLogger(f"agent.{name}", LogLevel.WARNING)
                                logger.warning(f"Sentinel output validation error: {e}")
                            except ImportError:
                                pass
                    
                    # Mark as successful
                    ctx.mark_success(result)
                    
                    # Calculate execution time
                    duration = time.time() - start_time
                    duration_ms = int(duration * 1000)
                    
                    # AUTOMATIC CORTEX RECORDING - This is what was missing!
                    cortex_instance = await _get_cortex()
                    if cortex_instance:
                        try:
                            # Prepare input data
                            input_data = {}
                            if args:
                                # Try to extract meaningful input from args
                                if len(args) == 1 and isinstance(args[0], (str, dict)):
                                    input_data = args[0] if isinstance(args[0], dict) else {"query": args[0]}
                                else:
                                    input_data = {"args": args}
                            
                            # Merge kwargs into input_data
                            if kwargs:
                                input_data.update(kwargs)
                            
                            # Prepare output data
                            output_data = result if isinstance(result, dict) else {"response": result}
                            
                            # Extract session_id from input if available
                            session_id = input_data.get('customer_id') or input_data.get('session_id') or input_data.get('user_id') or execution_id
                            
                            # Record interaction in Cortex
                            await cortex_instance.record_interaction(
                                input_data=input_data,
                                output_data=output_data,
                                success=True,
                                cost=total_cost if total_cost > 0 else None,
                                duration_ms=duration_ms,
                                session_id=session_id,
                                context={
                                    "execution_id": execution_id,
                                    "agent_name": name,
                                    "agent_id": agent_id
                                }
                            )
                        except Exception as e:
                            # Cortex recording failed, log but don't fail execution
                            try:
                                from teleon.core import StructuredLogger, LogLevel
                                logger = StructuredLogger("agent.client", LogLevel.WARNING)
                                logger.warning(f"Failed to record interaction in Cortex: {e}")
                            except ImportError:
                                pass
                    
                    # Store execution in episodic memory (legacy - for backward compatibility)
                    if memory_enabled and memory_session and not cortex_instance:
                        try:
                            from teleon.memory.episodic import EpisodicMemory
                            episodic = EpisodicMemory()
                            await episodic.store_event(
                                agent_name=name,
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
                                    "agent_name": name,
                                    "timestamp": ctx.started_at.isoformat()
                                }
                            )
                        except (ImportError, Exception):
                            # Episodic memory not available or failed
                            pass
                    
                    # Update procedural memory (legacy - for backward compatibility)
                    if memory_enabled and not cortex_instance:
                        try:
                            from teleon.memory.procedural import ProceduralMemory
                            procedural = ProceduralMemory()
                            await procedural.record_success(
                                pattern=f"{name}.execute",
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
                            operation=name,
                            duration=duration,
                            status="success"
                        )
                        if total_cost > 0:
                            get_metrics().increment_counter(
                                'llm_cost',
                                {'provider': 'unknown', 'model': model},
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
                                action=f"Agent execution completed: {name}",
                                output_data=output_data if output_data else None,
                                duration_ms=int(duration * 1000),
                                status="success",
                                metadata={
                                    "execution_id": execution_id,
                                    "agent_id": agent_id,
                                    "cost": total_cost,
                                    "span_id": span_id
                                }
                            )
                        except Exception as e:
                            # Audit logging failures must never impact agent execution
                            try:
                                from teleon.core import StructuredLogger, LogLevel
                                logger = StructuredLogger("agent.client", LogLevel.WARNING)
                                logger.warning(f"Failed to log audit response: {e}")
                            except ImportError:
                                pass
                    
                    return result
                    
                except Exception as e:
                    # Mark as failed
                    ctx.mark_failure(e)
                    
                    # Calculate execution time
                    duration = time.time() - start_time
                    duration_ms = int(duration * 1000)
                    
                    # AUTOMATIC CORTEX RECORDING FOR FAILURES
                    cortex_instance = await _get_cortex()
                    if cortex_instance:
                        try:
                            # Prepare input data
                            input_data = {}
                            if args:
                                if len(args) == 1 and isinstance(args[0], (str, dict)):
                                    input_data = args[0] if isinstance(args[0], dict) else {"query": args[0]}
                                else:
                                    input_data = {"args": args}
                            if kwargs:
                                input_data.update(kwargs)
                            
                            session_id = input_data.get('customer_id') or input_data.get('session_id') or input_data.get('user_id') or execution_id
                            
                            # Record failed interaction in Cortex
                            await cortex_instance.record_interaction(
                                input_data=input_data,
                                output_data={"error": str(e), "error_type": type(e).__name__},
                                success=False,
                                duration_ms=duration_ms,
                                session_id=session_id,
                                context={
                                    "execution_id": execution_id,
                                    "agent_name": name,
                                    "agent_id": agent_id
                                }
                            )
                        except Exception as cortex_error:
                            try:
                                from teleon.core import StructuredLogger, LogLevel
                                logger = StructuredLogger("agent.client", LogLevel.WARNING)
                                logger.warning(f"Failed to record failure in Cortex: {cortex_error}")
                            except ImportError:
                                pass
                    
                    # Record failure for learning (legacy - for backward compatibility)
                    if memory_enabled and not cortex_instance:
                        try:
                            from teleon.memory.procedural import ProceduralMemory
                            procedural = ProceduralMemory()
                            await procedural.record_failure(
                                pattern=f"{name}.execute",
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
                                agent_name=name,
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
                    
                    # AUDIT LOGGING: Log error response (NON-BLOCKING)
                    if audit_logger:
                        try:
                            # Log error (non-blocking - just queues)
                            audit_logger.log_response(
                                action=f"Agent execution failed: {name}",
                                output_data={
                                    "error": str(e),
                                    "error_type": type(e).__name__
                                },
                                duration_ms=int(duration * 1000),
                                status="error",
                                metadata={
                                    "execution_id": execution_id,
                                    "agent_id": agent_id,
                                    "span_id": span_id
                                }
                            )
                        except Exception as e2:
                            # Audit logging failures must never impact agent execution
                            try:
                                from teleon.core import StructuredLogger, LogLevel
                                logger = StructuredLogger("agent.client", LogLevel.WARNING)
                                logger.warning(f"Failed to log audit error: {e2}")
                            except ImportError:
                                pass
                    
                    # Record error metrics
                    try:
                        from teleon.core import get_metrics
                        get_metrics().record_request(
                            component="agent",
                            operation=name,
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
                        logger = StructuredLogger(f"agent.{name}", LogLevel.INFO)
                        logger.info(
                            "Agent execution completed",
                            execution_id=execution_id,
                            duration=time.time() - start_time,
                            success=ctx.success
                        )
                    except ImportError:
                        pass
                    
                    # Cleanup memory session
                    if memory_session:
                        try:
                            await memory_session.close()
                        except Exception:
                            pass
            
            # Register agent (store wrapped function, not raw function)
            agent_info = {
                "agent_id": agent_id,
                "name": name,
                "description": description or func.__doc__ or "No description provided",
                "function": wrapper,  # Store wrapped function, not raw function
                "user_id": self.user_id,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "parameters": params,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "helix": helix,
                "cortex": cortex,
                "sentinel": sentinel,
                "config": kwargs
            }
            
            self.agents[agent_id] = agent_info
            TeleonClient._all_agents[agent_id] = agent_info
            
            # Attach metadata to wrapper (for discovery)
            wrapper._teleon_agent = True  # type: ignore
            wrapper._teleon_config = agent_config  # type: ignore
            wrapper._teleon_original_func = func  # type: ignore
            
            if not os.getenv('TELEON_QUIET'):
                print(f"✓ Agent registered: {name}")
                print(f"  Agent ID: {agent_id}")
                print(f"  URL: /{agent_id}/")
                if sentinel:
                    print(f"  🛡️  Sentinel: Enabled")
                if memory_enabled:
                    print(f"  🧠 Memory: Enabled")
            
            return wrapper
        
        return decorator
    
    def _generate_agent_id(self, name: str) -> str:
        """Generate a unique agent ID."""
        # Combine user ID and agent name for uniqueness
        unique_str = f"{self.user_id}:{name}:{datetime.now(timezone.utc).isoformat()}"
        hash_id = hashlib.sha256(unique_str.encode()).hexdigest()[:16]
        return f"agent_{hash_id}"
    
    @classmethod
    def get_all_agents(cls) -> Dict[str, Dict[str, Any]]:
        """Get all registered agents across all clients."""
        return cls._all_agents
    
    @classmethod
    def get_agent(cls, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific agent by ID."""
        return cls._all_agents.get(agent_id)
    
    @classmethod
    def get_client(cls, user_id: str) -> Optional['TeleonClient']:
        """Get a client by user ID."""
        return cls._instances.get(user_id)
    
    def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for remote operations."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=httpx.Timeout(60.0)
            )
        return self._http_client
    
    async def get_agent(self, agent_name: str) -> 'RemoteAgent':
        """
        Get a reference to a deployed agent.
        
        This returns a RemoteAgent proxy that can be used to interact with
        the deployed agent from your application.
        
        Args:
            agent_name: Name of the deployed agent
        
        Returns:
            RemoteAgent proxy for interacting with the agent
        
        Example:
            client = TeleonClient(api_key="...")
            agent = await client.get_agent("customer-support")
            result = await agent.execute(input_data={...})
        """
        from teleon.remote_agent import RemoteAgent
        
        # Fetch agent info from API
        client = self._get_http_client()
        
        try:
            # Search for agent by name
            response = await client.get(
                f"{self.base_url}/agents",
                params={"name": agent_name, "user_id": self.user_id}
            )
            response.raise_for_status()
            data = response.json()
            
            if not data.get("agents"):
                raise ValueError(f"Agent '{agent_name}' not found. Make sure it's deployed.")
            
            agent_data = data["agents"][0]
            agent_id = agent_data["agent_id"]
            
            # Return RemoteAgent proxy
            return RemoteAgent(
                agent_name=agent_name,
                agent_id=agent_id,
                api_key=self.api_key,
                base_url=self.base_url
            )
            
        except httpx.HTTPStatusError as e:
            raise ValueError(f"Failed to fetch agent: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise ValueError(f"Failed to fetch agent: {str(e)}")
    
    async def list_deployed_agents(self) -> List[Dict[str, Any]]:
        """
        List all deployed agents for this user.
        
        Returns:
            List of deployed agents with their info
        """
        client = self._get_http_client()
        
        try:
            response = await client.get(
                f"{self.base_url}/agents",
                params={"user_id": self.user_id}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("agents", [])
            
        except Exception as e:
            raise ValueError(f"Failed to list agents: {str(e)}")
    
    async def close(self):
        """Close HTTP client connections."""
        if self._http_client:
            await self._http_client.aclose()
    
    async def initialize_agent_runtime(
        self,
        agent_id: str,
        enable_helix: bool = True,
        enable_cortex: bool = True,
        enable_sentinel: bool = True
    ) -> Dict[str, Any]:
        """
        Initialize runtime features for an agent.
        
        This sets up Helix and Cortex for the agent based on
        its configuration.
        
        Args:
            agent_id: Agent ID to initialize
            enable_helix: Enable Helix runtime
            enable_cortex: Enable Cortex memory
            enable_sentinel: Enable Sentinel safety and compliance
        
        Returns:
            Dictionary with initialized components
        
        Example:
            ```python
            components = await client.initialize_agent_runtime("agent-123")
            cortex = components['cortex']
            runtime = components['helix_runtime']
            ```
        """
        agent_info = self.agents.get(agent_id)
        if not agent_info:
            raise ValueError(f"Agent {agent_id} not found")
        
        components = {}
        
        # Initialize Helix if configured
        if enable_helix and agent_info.get('helix'):
            from teleon.helix.integration import register_agent_with_helix, parse_helix_config
            
            helix_config = parse_helix_config(agent_info['helix'])
            
            # Register with Helix (will create wrapper)
            wrapper = await register_agent_with_helix(
                agent_id=agent_id,
                agent_func=agent_info['function'],
                helix_config=helix_config
            )
            
            components['helix_wrapper'] = wrapper
            components['helix_config'] = helix_config
        
        # Initialize Cortex if configured
        if enable_cortex and agent_info.get('cortex'):
            from teleon.cortex import create_cortex, CortexConfig
            
            cortex_config_dict = agent_info['cortex']
            
            # Create Cortex instance
            cortex = await create_cortex(
                agent_id=agent_id,
                session_id=self.user_id,
                storage_backend=cortex_config_dict.get('storage', 'memory'),
                config=CortexConfig(**cortex_config_dict) if isinstance(cortex_config_dict, dict) else None
            )
            
            components['cortex'] = cortex
        
        # Initialize Sentinel if configured
        if enable_sentinel and agent_info.get('sentinel'):
            from teleon.sentinel.integration import register_agent_with_sentinel
            
            sentinel_config = agent_info['sentinel']
            sentinel_engine = await register_agent_with_sentinel(
                agent_id=agent_id,
                sentinel_config=sentinel_config,
                agent_name=agent_info['name']
            )
            
            components['sentinel_engine'] = sentinel_engine
            components['sentinel_config'] = sentinel_config
        
        return components


# Convenience function for quick setup
def init_teleon(api_key: str, environment: str = "dev") -> TeleonClient:
    """
    Initialize Teleon client.
    
    Args:
        api_key: Your Teleon API key
        environment: Environment (dev, staging, production)
    
    Returns:
        TeleonClient instance
    """
    return TeleonClient(api_key=api_key, environment=environment)
