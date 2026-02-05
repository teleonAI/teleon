"""
Teleon Client - User authentication and agent registration
"""

import os
from typing import Optional, Dict, Any, Callable, List, TypeVar, Union, cast
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
        verify_key: bool = True,
        is_paid_tier: bool = False
    ):
        """
        Initialize Teleon client.

        Args:
            api_key: Your Teleon API key from the platform
            environment: Environment (production, staging, dev)
            base_url: Custom API base URL (default: uses environment)
            verify_key: Whether to verify API key with backend (default: True)
            is_paid_tier: Whether user is on paid tier (affects embedding model)
        """
        self.api_key = api_key
        self.environment = environment
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.scopes: List[str] = []  # API key scopes/permissions
        self.is_paid_tier = is_paid_tier

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
        """
        if not self.scopes:
            return True
        return any(scope in self.scopes for scope in required_scopes)

    def require_scope(self, *required_scopes: str):
        """Raise an error if API key doesn't have required scopes."""
        if not self.has_scope(*required_scopes):
            raise ValueError(
                f"Insufficient API key permissions. Required scopes: {', '.join(required_scopes)}. "
                f"Your key has: {', '.join(self.scopes) if self.scopes else 'none'}."
            )

    def _verify_api_key_sync(self):
        """Verify API key with the backend (synchronous version for __init__)."""
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    f"{self.base_url}/api/v1/api-keys/validate",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )

                if response.status_code == 401:
                    try:
                        error_detail = response.json().get("detail", "Invalid API key")
                    except:
                        error_detail = "Invalid API key"
                    raise ValueError(f"API key verification failed: {error_detail}")
                elif response.status_code == 403:
                    raise ValueError("API key is valid but your account is inactive.")
                elif response.status_code == 429:
                    raise ValueError("Rate limit exceeded. Please wait and try again.")
                elif response.status_code >= 500:
                    if not os.getenv('TELEON_QUIET'):
                        print(f"  ⚠️  Warning: Unable to verify API key (server error)")
                    return
                elif response.status_code >= 400:
                    raise ValueError(f"API key verification failed with status {response.status_code}")

                if response.status_code == 200:
                    try:
                        key_info = response.json()
                        self.scopes = key_info.get("key", {}).get("scopes", [])
                        # Check if paid tier from API response
                        self.is_paid_tier = key_info.get("key", {}).get("is_paid", self.is_paid_tier)
                        if not os.getenv('TELEON_QUIET'):
                            print(f"  ✓ API key verified successfully")
                    except Exception:
                        if not os.getenv('TELEON_QUIET'):
                            print(f"  ✓ API key verified successfully")

        except httpx.ConnectError:
            if not os.getenv('TELEON_QUIET'):
                print(f"  ⚠️  Warning: Could not connect to {self.base_url}")
        except httpx.TimeoutException:
            if not os.getenv('TELEON_QUIET'):
                print(f"  ⚠️  Warning: Connection timeout")
        except ValueError:
            raise
        except Exception as e:
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
        cortex: Optional[Union[bool, Dict[str, Any]]] = None,
        sentinel: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Decorator to register an agent with this client.

        Args:
            name: Agent name
            description: Agent description
            model: LLM model to use
            temperature: Temperature setting
            max_tokens: Max tokens
            helix: Helix runtime configuration (auto-scaling, health checks)
            cortex: Cortex memory configuration:
                - True: Enable with defaults (auto-save enabled)
                - False/None: Disabled
                - Dict: Custom configuration
            sentinel: Sentinel safety and compliance configuration
            **kwargs: Additional configuration

        Returns:
            Decorated function with full runtime features

        Example:
            ```python
            # Simple enable
            @client.agent(name="support", cortex=True)
            async def support_agent(query: str, customer_id: str, cortex: Memory):
                # cortex.context has auto-retrieved relevant memories
                await cortex.store(content=f"Query: {query}", customer_id=customer_id)
                return response

            # With configuration
            @client.agent(
                name="support",
                cortex={
                    "auto": True,
                    "scope": ["customer_id"],
                    "fields": ["query", "type"],
                }
            )
            async def support_agent(query: str, customer_id: str, cortex: Memory):
                ...
            ```
        """
        def decorator(func: Callable):
            import asyncio
            import inspect
            from functools import wraps
            import uuid
            import time
            import logging

            # Validate function is async
            if not asyncio.iscoroutinefunction(func):
                raise TypeError(
                    f"Agent function '{func.__name__}' must be async. "
                    f"Use 'async def {func.__name__}(...)' instead."
                )

            # Generate unique agent ID
            agent_id = self._generate_agent_id(name)

            # Extract function signature
            sig = inspect.signature(func)
            params = {}
            has_cortex_param = False
            for param_name, param in sig.parameters.items():
                params[param_name] = {
                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "string",
                    "required": param.default == inspect.Parameter.empty
                }
                if param_name == "cortex":
                    has_cortex_param = True

            # Parse cortex configuration
            cortex_config = cortex
            cortex_enabled = bool(cortex)
            memory_manager = None

            if cortex_enabled:
                try:
                    from teleon.cortex import MemoryManager, parse_memory_config

                    # Parse config (True -> default config, dict -> custom config)
                    if cortex is True:
                        parsed_config = parse_memory_config(True)
                    else:
                        parsed_config = parse_memory_config(cortex)

                    # Create memory manager for this agent
                    memory_manager = MemoryManager(
                        config=parsed_config,
                        agent_name=name,
                        is_paid_tier=self.is_paid_tier
                    )
                except ImportError as e:
                    logging.getLogger("teleon.agent").warning(f"Cortex not available: {e}")
                    cortex_enabled = False
                except Exception as e:
                    logging.getLogger("teleon.agent").error(f"Failed to initialize Cortex: {e}")
                    cortex_enabled = False

            # Map helix config to scale config
            scale_config = None
            if helix:
                scale_config = {
                    'min': helix.get('min', 1),
                    'max': helix.get('max', 10),
                    'target_cpu': helix.get('target_cpu', 70)
                }

            # Create AgentConfig
            from teleon.config.agent_config import AgentConfig
            agent_config = AgentConfig(
                name=name,
                memory=cortex_enabled,
                scale=scale_config or {'min': 1, 'max': 10},
                llm={'model': model, 'temperature': temperature, 'max_tokens': max_tokens},
                tools=kwargs.get('tools', []),
                collaborate=False,
                timeout=kwargs.get('timeout'),
                signature=sig,
                **{k: v for k, v in kwargs.items() if k not in ['tools', 'timeout']}
            )
            agent_config.validate()

            # Store configs
            sentinel_config = sentinel

            # Lazy-initialized AuditLogger
            _audit_logger = None
            _audit_logger_lock = None

            def _get_audit_logger():
                nonlocal _audit_logger, _audit_logger_lock
                if _audit_logger is None:
                    try:
                        import threading
                        if _audit_logger_lock is None:
                            _audit_logger_lock = threading.Lock()

                        with _audit_logger_lock:
                            if _audit_logger is None:
                                from teleon.governance.audit import AuditLogger
                                api_url = os.getenv('TELEON_API_URL') or os.getenv('TELEON_PLATFORM_URL')
                                api_key = os.getenv('TELEON_API_KEY')

                                if api_url and api_key:
                                    _audit_logger = AuditLogger(
                                        agent_id=agent_id,
                                        agent_name=name,
                                        enable_remote_logging=True,
                                        api_url=api_url,
                                        api_key=api_key,
                                        batch_size=50,
                                        flush_interval=10.0
                                    )
                                else:
                                    _audit_logger = AuditLogger(
                                        agent_id=agent_id,
                                        agent_name=name,
                                        enable_remote_logging=False
                                    )
                    except ImportError:
                        pass
                    except Exception:
                        pass
                return _audit_logger

            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                """Wrapper that adds Cortex memory, Sentinel validation, and other features."""

                logger = logging.getLogger(f"teleon.agent.{name}")
                execution_id = str(uuid.uuid4())
                start_time = time.time()

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
                ctx.agent_id = agent_id

                # Audit logging
                audit_logger = _get_audit_logger()
                if audit_logger:
                    try:
                        input_data = {}
                        if args:
                            input_data['args'] = str(args)[:500]
                        if kwargs:
                            safe_kwargs = {k: str(v)[:200] if not isinstance(v, (str, int, float, bool, type(None))) else v
                                         for k, v in kwargs.items() if k != 'cortex'}
                            input_data['kwargs'] = safe_kwargs

                        audit_logger.log_request(
                            action=f"Agent execution: {name}",
                            input_data=input_data if input_data else None,
                            metadata={"execution_id": execution_id, "agent_id": agent_id}
                        )
                    except Exception:
                        pass

                # Initialize Cortex Memory
                memory_instance = None
                scope_values = {}
                field_values = {}
                query_text = None

                if cortex_enabled and memory_manager:
                    try:
                        # Extract scope values from function arguments
                        scope_values = memory_manager.extract_scope_values(args, kwargs, func)

                        # Extract field values for auto-save
                        field_values = memory_manager.extract_field_values(args, kwargs, func)

                        # Create Memory instance
                        memory_instance = memory_manager.create_memory(scope_values)

                        # Extract query for context retrieval
                        # Look for common query parameter names
                        full_kwargs = dict(kwargs)
                        param_names = list(sig.parameters.keys())
                        for i, arg in enumerate(args):
                            if i < len(param_names):
                                full_kwargs[param_names[i]] = arg

                        query_text = full_kwargs.get('query') or full_kwargs.get('input') or full_kwargs.get('message')
                        if query_text is None and args:
                            # Use first string argument as query
                            for arg in args:
                                if isinstance(arg, str):
                                    query_text = arg
                                    break

                        # Retrieve context
                        context = await memory_manager.retrieve_context(
                            memory=memory_instance,
                            query=query_text,
                            filter=scope_values
                        )
                        memory_instance._set_context(context)

                        logger.debug(f"Cortex initialized: {len(context.entries)} context entries")

                    except Exception as e:
                        logger.error(f"Failed to initialize Cortex memory: {e}")
                        memory_instance = None

                # Inject memory into kwargs if function expects it
                if has_cortex_param and memory_instance:
                    kwargs['cortex'] = memory_instance
                elif has_cortex_param:
                    # Create empty memory if cortex param expected but not configured
                    try:
                        from teleon.cortex import Memory
                        from teleon.cortex.storage.inmemory import InMemoryBackend
                        from teleon.cortex.embeddings.service import get_embedding_service

                        # Create minimal memory instance
                        kwargs['cortex'] = Memory(
                            backend=InMemoryBackend(),
                            embedding_service=get_embedding_service(self.is_paid_tier),
                            memory_name=name,
                            scope=[],
                            scope_values={},
                            layers=None
                        )
                    except Exception:
                        pass

                # Initialize Sentinel
                sentinel_engine = None
                if sentinel_config:
                    try:
                        from teleon.sentinel.integration import create_sentinel_engine
                        sentinel_engine = create_sentinel_engine(
                            sentinel_config,
                            agent_id=agent_id,
                            agent_name=name
                        )
                    except ImportError:
                        pass
                    except Exception as e:
                        logger.warning(f"Sentinel initialization failed: {e}")

                # Sentinel input validation
                if sentinel_engine:
                    try:
                        input_data = {'args': args, 'kwargs': {k: v for k, v in kwargs.items() if k != 'cortex'}}
                        input_result = await sentinel_engine.validate_input(input_data, name)

                        if not input_result.passed:
                            from teleon.core.exceptions import AgentValidationError
                            if input_result.action.value == 'block':
                                raise AgentValidationError(
                                    f"Input blocked by Sentinel: {input_result.violations}"
                                )
                    except Exception as e:
                        from teleon.core.exceptions import AgentValidationError
                        if isinstance(e, AgentValidationError):
                            raise
                        logger.warning(f"Sentinel input validation error: {e}")

                total_cost = 0.0

                try:
                    # Execute the agent function
                    result = await func(*args, **kwargs)

                    # Sentinel output validation
                    if sentinel_engine:
                        try:
                            output_result = await sentinel_engine.validate_output(result, name)
                            if output_result.redacted_content and output_result.action.value == 'redact':
                                result = output_result.redacted_content
                        except Exception as e:
                            from teleon.core.exceptions import AgentValidationError
                            if isinstance(e, AgentValidationError):
                                raise
                            logger.warning(f"Sentinel output validation error: {e}")

                    ctx.mark_success(result)
                    duration = time.time() - start_time
                    duration_ms = int(duration * 1000)

                    # Auto-save to Cortex
                    if cortex_enabled and memory_manager and memory_instance and query_text:
                        try:
                            response_text = result if isinstance(result, str) else str(result)
                            await memory_manager.auto_save(
                                memory=memory_instance,
                                query=query_text,
                                response=response_text,
                                fields=field_values
                            )
                            logger.debug(f"Cortex auto-saved conversation")
                        except Exception as e:
                            logger.warning(f"Cortex auto-save failed: {e}")

                    # Record metrics
                    try:
                        from teleon.core import get_metrics
                        get_metrics().record_request(
                            component="agent",
                            operation=name,
                            duration=duration,
                            status="success"
                        )
                    except ImportError:
                        pass

                    # Audit logging
                    if audit_logger:
                        try:
                            output_data = {}
                            if result is not None:
                                output_data['result'] = str(result)[:1000]
                                output_data['result_type'] = type(result).__name__

                            audit_logger.log_response(
                                action=f"Agent execution completed: {name}",
                                output_data=output_data if output_data else None,
                                duration_ms=duration_ms,
                                status="success",
                                metadata={"execution_id": execution_id, "agent_id": agent_id}
                            )
                        except Exception:
                            pass

                    return result

                except Exception as e:
                    ctx.mark_failure(e)
                    duration = time.time() - start_time

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

                    # Audit logging
                    if audit_logger:
                        try:
                            audit_logger.log_response(
                                action=f"Agent execution failed: {name}",
                                output_data={"error": str(e), "error_type": type(e).__name__},
                                duration_ms=int(duration * 1000),
                                status="error",
                                metadata={"execution_id": execution_id, "agent_id": agent_id}
                            )
                        except Exception:
                            pass

                    raise

                finally:
                    try:
                        from teleon.core import StructuredLogger, LogLevel
                        struct_logger = StructuredLogger(f"agent.{name}", LogLevel.INFO)
                        struct_logger.info(
                            "Agent execution completed",
                            execution_id=execution_id,
                            duration=time.time() - start_time,
                            success=ctx.success
                        )
                    except ImportError:
                        pass

            # Register agent
            agent_info = {
                "agent_id": agent_id,
                "name": name,
                "description": description or func.__doc__ or "No description provided",
                "function": wrapper,
                "user_id": self.user_id,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "parameters": params,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "helix": helix,
                "cortex": cortex_config,
                "sentinel": sentinel,
                "config": kwargs
            }

            self.agents[agent_id] = agent_info
            TeleonClient._all_agents[agent_id] = agent_info

            # Attach metadata to wrapper
            wrapper._teleon_agent = True
            wrapper._teleon_config = agent_config
            wrapper._teleon_original_func = func

            if not os.getenv('TELEON_QUIET'):
                print(f"✓ Agent registered: {name}")
                print(f"  Agent ID: {agent_id}")
                print(f"  URL: /{agent_id}/")
                if sentinel:
                    print(f"  🛡️  Sentinel: Enabled")
                if cortex_enabled:
                    print(f"  🧠 Cortex: Enabled")

            return wrapper

        return decorator

    def _generate_agent_id(self, name: str) -> str:
        """Generate a unique agent ID."""
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

    async def get_remote_agent(self, agent_name: str) -> 'RemoteAgent':
        """
        Get a reference to a deployed agent.

        Args:
            agent_name: Name of the deployed agent

        Returns:
            RemoteAgent proxy for interacting with the agent
        """
        from teleon.remote_agent import RemoteAgent

        client = self._get_http_client()

        try:
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
        """List all deployed agents for this user."""
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


# Convenience function for quick setup
def init_teleon(api_key: str, environment: str = "dev", is_paid_tier: bool = False) -> TeleonClient:
    """
    Initialize Teleon client.

    Args:
        api_key: Your Teleon API key
        environment: Environment (dev, staging, production)
        is_paid_tier: Whether user is on paid tier

    Returns:
        TeleonClient instance
    """
    return TeleonClient(api_key=api_key, environment=environment, is_paid_tier=is_paid_tier)
