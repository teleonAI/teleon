"""
Helix Integration Layer - Bridges TeleonClient and AgentRuntime.

This module provides the integration between user-facing agent decorators
and the Helix production runtime system.
"""

from typing import Any, Dict, Optional, Callable
import asyncio
from datetime import datetime

from teleon.helix.runtime import AgentRuntime, RuntimeConfig, ResourceConfig, get_runtime
from teleon.helix.health import HealthCheck, CheckType
from teleon.core import StructuredLogger, LogLevel


class HelixConfig(Dict):
    """
    Configuration for Helix runtime features.
    
    Used in @agent decorator to enable runtime features:
    
    Example:
        ```python
        @client.agent(
            name="my-agent",
            helix={
                'min_instances': 2,
                'max_instances': 10,
                'target_cpu': 70,
                'memory_limit_mb': 512,
                'health_check_interval': 30,
                'auto_scale': True
            }
        )
        async def my_agent(input: str):
            return process(input)
        ```
    """
    pass


class AgentWrapper:
    """
    Wrapper for agent functions that integrates with Helix runtime.
    
    Provides:
    - Lifecycle hooks
    - Health checking
    - Metrics collection
    - Error handling
    - Resource tracking
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_func: Callable,
        helix_config: Optional[Dict[str, Any]] = None,
        cortex: Optional[Any] = None,
        nexusnet: Optional[Any] = None
    ):
        """
        Initialize agent wrapper.
        
        Args:
            agent_id: Unique agent identifier
            agent_func: Agent function to wrap
            helix_config: Helix configuration
            cortex: Cortex memory instance
            nexusnet: NexusNet collaboration instance
        """
        self.agent_id = agent_id
        self.agent_func = agent_func
        self.helix_config = helix_config or {}
        self.cortex = cortex
        self.nexusnet = nexusnet
        
        self.logger = StructuredLogger(f"agent.{agent_id}", LogLevel.INFO)
        self._execution_count = 0
        self._error_count = 0
    
    async def __call__(self, *args, **kwargs) -> Any:
        """
        Execute agent with Helix integration.
        
        Provides:
        - Pre/post execution hooks
        - Error handling
        - Metrics tracking
        - Memory recording
        """
        execution_id = f"{self.agent_id}_{self._execution_count}"
        start_time = datetime.utcnow()
        
        self._execution_count += 1
        
        try:
            # Pre-execution hook
            await self._pre_execution(execution_id, args, kwargs)
            
            # Execute agent
            result = await self.agent_func(*args, **kwargs)
            
            # Post-execution hook
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self._post_execution(
                execution_id,
                args,
                kwargs,
                result,
                success=True,
                duration_ms=duration_ms
            )
            
            return result
            
        except Exception as e:
            self._error_count += 1
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            self.logger.error(
                "Agent execution failed",
                agent_id=self.agent_id,
                execution_id=execution_id,
                error=str(e),
                duration_ms=duration_ms
            )
            
            # Post-execution hook with error
            await self._post_execution(
                execution_id,
                args,
                kwargs,
                None,
                success=False,
                duration_ms=duration_ms,
                error=str(e)
            )
            
            raise
    
    async def _pre_execution(
        self,
        execution_id: str,
        args: tuple,
        kwargs: dict
    ) -> None:
        """Pre-execution hook."""
        self.logger.debug(
            "Agent execution started",
            agent_id=self.agent_id,
            execution_id=execution_id
        )
    
    async def _post_execution(
        self,
        execution_id: str,
        args: tuple,
        kwargs: dict,
        result: Any,
        success: bool,
        duration_ms: int,
        error: Optional[str] = None
    ) -> None:
        """Post-execution hook."""
        self.logger.info(
            "Agent execution completed",
            agent_id=self.agent_id,
            execution_id=execution_id,
            success=success,
            duration_ms=duration_ms,
            error=error
        )
        
        # Record in Cortex if available
        if self.cortex and hasattr(self.cortex, 'record_interaction'):
            try:
                await self.cortex.record_interaction(
                    input_data={"args": args, "kwargs": kwargs},
                    output_data={"result": result} if success else {"error": error},
                    success=success,
                    duration_ms=duration_ms
                )
            except Exception as e:
                self.logger.warning(
                    "Failed to record interaction in Cortex",
                    error=str(e)
                )
    
    def get_health(self) -> Dict[str, Any]:
        """Get agent health status."""
        error_rate = (
            (self._error_count / self._execution_count * 100)
            if self._execution_count > 0 else 0
        )
        
        return {
            "agent_id": self.agent_id,
            "status": "healthy" if error_rate < 10 else "degraded",
            "executions": self._execution_count,
            "errors": self._error_count,
            "error_rate": round(error_rate, 2)
        }


async def register_agent_with_helix(
    agent_id: str,
    agent_func: Callable,
    helix_config: Dict[str, Any],
    cortex: Optional[Any] = None,
    nexusnet: Optional[Any] = None,
    runtime: Optional[AgentRuntime] = None
) -> AgentWrapper:
    """
    Register an agent with Helix runtime.
    
    This creates the bridge between a decorated agent function
    and the Helix runtime system.
    
    Args:
        agent_id: Unique agent identifier
        agent_func: Agent function
        helix_config: Helix configuration from decorator
        cortex: Optional Cortex memory instance
        nexusnet: Optional NexusNet instance
        runtime: Optional runtime instance (creates one if None)
    
    Returns:
        AgentWrapper that integrates with Helix
    
    Example:
        ```python
        wrapper = await register_agent_with_helix(
            agent_id="agent-123",
            agent_func=my_agent_function,
            helix_config={
                'min_instances': 2,
                'max_instances': 10,
                'target_cpu': 70
            }
        )
        
        # Agent is now registered with Helix runtime
        result = await wrapper(input_data)
        ```
    """
    # Get or create runtime
    if runtime is None:
        runtime = get_runtime()
        if not runtime.running:
            await runtime.start()
    
    # Create agent wrapper
    wrapper = AgentWrapper(
        agent_id=agent_id,
        agent_func=agent_func,
        helix_config=helix_config,
        cortex=cortex,
        nexusnet=nexusnet
    )
    
    # Extract Helix configuration
    resource_config = ResourceConfig(
        cpu_limit=helix_config.get('cpu_limit'),
        memory_limit_mb=helix_config.get('memory_limit_mb'),
        min_instances=helix_config.get('min_instances', 1),
        max_instances=helix_config.get('max_instances', 1),
        startup_timeout=helix_config.get('startup_timeout', 30),
        shutdown_timeout=helix_config.get('shutdown_timeout', 30),
        health_check_enabled=helix_config.get('health_check_enabled', True),
        health_check_interval=helix_config.get('health_check_interval', 30)
    )
    
    # Create health check
    health_check = None
    if resource_config.health_check_enabled:
        health_check = HealthCheck(
            name=f"{agent_id}_health",
            check_type=CheckType.READINESS,
            interval=resource_config.health_check_interval,
            timeout=5,
            success_threshold=2,
            failure_threshold=3
        )
    
    # Register with runtime
    await runtime.register_agent(
        agent_id=agent_id,
        agent_callable=wrapper,
        resources=resource_config,
        health_check=health_check
    )
    
    # Start agent if auto_start enabled
    # NOTE: Only auto-start if the agent doesn't require arguments
    # Request-response agents should be called on-demand, not spawned as workers
    auto_start = helix_config.get('auto_start', False)  # Default to False for safety
    
    if auto_start:
        # Check if agent function requires arguments
        import inspect
        sig = inspect.signature(agent_func)
        required_params = [
            p for p in sig.parameters.values()
            if p.default == inspect.Parameter.empty and p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD
            )
        ]
        
        if required_params:
            runtime.logger.debug(
                "Agent uses on-demand execution (requires arguments)",
                agent_id=agent_id,
                required_params=[p.name for p in required_params]
            )
        else:
            # Safe to auto-start (no required args)
            await runtime.start_agent(agent_id)
    
    return wrapper


def parse_helix_config(helix_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parse and validate Helix configuration from decorator.
    
    Args:
        helix_config: Raw Helix config from @agent decorator
    
    Returns:
        Validated and normalized configuration
    
    Example:
        ```python
        # Shorthand
        config = parse_helix_config({'min': 2, 'max': 10})
        # Returns: {'min_instances': 2, 'max_instances': 10, ...}
        
        # Full form
        config = parse_helix_config({
            'min_instances': 2,
            'max_instances': 10,
            'target_cpu': 70
        })
        ```
    """
    if not helix_config:
        return {}
    
    # Normalize shorthand keys
    normalized = {}
    
    # Handle shorthand keys
    key_mapping = {
        'min': 'min_instances',
        'max': 'max_instances',
        'cpu': 'target_cpu',
        'memory': 'memory_limit_mb',
        'health_interval': 'health_check_interval'
    }
    
    for key, value in helix_config.items():
        # Map shorthand to full key
        full_key = key_mapping.get(key, key)
        normalized[full_key] = value
    
    # Set defaults
    defaults = {
        'min_instances': 1,
        'max_instances': 1,
        'auto_start': True,
        'health_check_enabled': True,
        'health_check_interval': 30,
        'startup_timeout': 30,
        'shutdown_timeout': 30
    }
    
    for key, default_value in defaults.items():
        if key not in normalized:
            normalized[key] = default_value
    
    # Validate ranges
    if normalized['min_instances'] > normalized['max_instances']:
        normalized['max_instances'] = normalized['min_instances']
    
    if normalized['min_instances'] < 1:
        normalized['min_instances'] = 1
    
    if normalized['max_instances'] > 100:
        normalized['max_instances'] = 100
    
    return normalized


async def scale_agent(
    agent_id: str,
    instances: int,
    runtime: Optional[AgentRuntime] = None
) -> bool:
    """
    Scale an agent to a specific number of instances.
    
    Args:
        agent_id: Agent to scale
        instances: Desired number of instances
        runtime: Optional runtime instance
    
    Returns:
        True if scaled successfully
    
    Example:
        ```python
        # Scale up
        await scale_agent("my-agent", 5)
        
        # Scale down
        await scale_agent("my-agent", 2)
        ```
    """
    if runtime is None:
        runtime = get_runtime()
    
    try:
        await runtime.scale_agent(agent_id, instances)
        return True
    except Exception as e:
        logger = StructuredLogger("helix.integration", LogLevel.ERROR)
        logger.error("Failed to scale agent", agent_id=agent_id, error=str(e))
        return False


async def get_agent_status(
    agent_id: str,
    runtime: Optional[AgentRuntime] = None
) -> Dict[str, Any]:
    """
    Get agent runtime status.
    
    Args:
        agent_id: Agent ID
        runtime: Optional runtime instance
    
    Returns:
        Status dictionary
    
    Example:
        ```python
        status = await get_agent_status("my-agent")
        print(f"Instances: {status['instances']}")
        print(f"Health: {status['health']}")
        ```
    """
    if runtime is None:
        runtime = get_runtime()
    
    return await runtime.get_agent_status(agent_id)


async def restart_agent(
    agent_id: str,
    runtime: Optional[AgentRuntime] = None
) -> bool:
    """
    Restart an agent.
    
    Args:
        agent_id: Agent ID
        runtime: Optional runtime instance
    
    Returns:
        True if restarted successfully
    
    Example:
        ```python
        await restart_agent("my-agent")
        ```
    """
    if runtime is None:
        runtime = get_runtime()
    
    try:
        await runtime.restart_agent(agent_id)
        return True
    except Exception as e:
        logger = StructuredLogger("helix.integration", LogLevel.ERROR)
        logger.error("Failed to restart agent", agent_id=agent_id, error=str(e))
        return False


async def stop_agent(
    agent_id: str,
    force: bool = False,
    runtime: Optional[AgentRuntime] = None
) -> bool:
    """
    Stop an agent.
    
    Args:
        agent_id: Agent ID
        force: Force kill (True) or graceful shutdown (False)
        runtime: Optional runtime instance
    
    Returns:
        True if stopped successfully
    
    Example:
        ```python
        # Graceful shutdown
        await stop_agent("my-agent")
        
        # Force kill
        await stop_agent("my-agent", force=True)
        ```
    """
    if runtime is None:
        runtime = get_runtime()
    
    try:
        await runtime.stop_agent(agent_id, force=force)
        return True
    except Exception as e:
        logger = StructuredLogger("helix.integration", LogLevel.ERROR)
        logger.error("Failed to stop agent", agent_id=agent_id, error=str(e))
        return False


def create_agent_health_check(
    agent_id: str,
    check_function: Optional[Callable] = None,
    interval: int = 30
) -> HealthCheck:
    """
    Create a custom health check for an agent.
    
    Args:
        agent_id: Agent ID
        check_function: Optional custom check function
        interval: Check interval in seconds
    
    Returns:
        HealthCheck instance
    
    Example:
        ```python
        async def my_health_check():
            # Custom health logic
            return True
        
        health = create_agent_health_check(
            "my-agent",
            check_function=my_health_check,
            interval=60
        )
        ```
    """
    return HealthCheck(
        name=f"{agent_id}_health",
        check_type=CheckType.CUSTOM if check_function else CheckType.READINESS,
        interval=interval,
        timeout=5,
        success_threshold=2,
        failure_threshold=3,
        check_fn=check_function
    )

