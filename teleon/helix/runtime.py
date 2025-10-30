"""
Agent Runtime - Production-grade runtime for managing agent processes.

Features:
- Agent lifecycle management
- Resource allocation
- Process monitoring
- Graceful shutdown
- Runtime configuration
"""

from typing import Dict, Optional, Any, List, Callable
from datetime import datetime
from pydantic import BaseModel, Field
import asyncio
import signal
import sys

from teleon.core import (
    AgentError,
    get_metrics,
    StructuredLogger,
    LogLevel,
)
from teleon.helix.process import ProcessManager, ProcessInfo, ProcessStatus
from teleon.helix.health import HealthChecker, HealthCheck
from teleon.helix.scaling import Scaler, ScalingPolicy


class ResourceConfig(BaseModel):
    """Resource configuration for agents."""
    
    cpu_limit: Optional[float] = Field(None, ge=0.1, le=32, description="CPU cores limit")
    memory_limit_mb: Optional[int] = Field(None, ge=64, description="Memory limit in MB")
    
    # Scaling
    min_instances: int = Field(1, ge=1, le=100, description="Minimum instances")
    max_instances: int = Field(1, ge=1, le=100, description="Maximum instances")
    
    # Timeouts
    startup_timeout: int = Field(30, ge=5, description="Startup timeout (seconds)")
    shutdown_timeout: int = Field(30, ge=5, description="Shutdown timeout (seconds)")
    
    # Health checks
    health_check_enabled: bool = Field(True, description="Enable health checks")
    health_check_interval: int = Field(30, ge=5, description="Health check interval")


class RuntimeConfig(BaseModel):
    """Runtime configuration."""
    
    # Environment
    environment: str = Field("development", description="Runtime environment")
    debug: bool = Field(False, description="Debug mode")
    
    # Hot reload
    hot_reload: bool = Field(True, description="Enable hot reload in development")
    watch_paths: List[str] = Field(default_factory=lambda: ["teleon", "agents"], description="Paths to watch")
    
    # Resources
    default_resources: ResourceConfig = Field(
        default_factory=ResourceConfig,
        description="Default resource configuration"
    )
    
    # Process management
    max_workers: int = Field(10, ge=1, le=100, description="Maximum worker processes")
    
    # Logging
    log_level: str = Field("INFO", description="Log level")
    
    class Config:
        validate_assignment = True


class AgentRuntime:
    """
    Production-grade agent runtime.
    
    Features:
    - Process lifecycle management
    - Resource monitoring
    - Health checking
    - Automatic scaling
    - Hot reload (development)
    """
    
    def __init__(self, config: Optional[RuntimeConfig] = None):
        """
        Initialize agent runtime.
        
        Args:
            config: Runtime configuration
        """
        self.config = config or RuntimeConfig()
        
        # Component managers
        self.process_manager = ProcessManager()
        self.health_checker = HealthChecker()
        self.scaler = Scaler()
        
        # Agent registry
        self.agents: Dict[str, Dict[str, Any]] = {}
        
        # State
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Logging
        self.logger = StructuredLogger(
            "helix_runtime",
            LogLevel[self.config.log_level]
        )
        
        # Background tasks
        self.tasks: List[asyncio.Task] = []
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown")
            # Signal handlers must use get_event_loop() not create_task()
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.shutdown())
            except RuntimeError:
                # No running loop, schedule for next iteration
                asyncio.ensure_future(self.shutdown())
        
        # Handle SIGTERM and SIGINT
        if sys.platform != 'win32':
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
    
    async def register_agent(
        self,
        agent_id: str,
        agent_callable: Callable,
        resources: Optional[ResourceConfig] = None,
        health_check: Optional[HealthCheck] = None
    ):
        """
        Register an agent with the runtime.
        
        Args:
            agent_id: Unique agent identifier
            agent_callable: Agent function/callable
            resources: Resource configuration
            health_check: Health check configuration
        """
        if agent_id in self.agents:
            raise AgentError(
                f"Agent {agent_id} already registered",
                error_code="DUPLICATE_AGENT"
            )
        
        # Use default resources if not provided
        agent_resources = resources or self.config.default_resources
        
        self.agents[agent_id] = {
            "callable": agent_callable,
            "resources": agent_resources,
            "health_check": health_check,
            "processes": [],
            "created_at": datetime.utcnow()
        }
        
        self.logger.info(
            "Agent registered",
            agent_id=agent_id,
            min_instances=agent_resources.min_instances,
            max_instances=agent_resources.max_instances
        )
    
    async def start_agent(self, agent_id: str) -> List[str]:
        """
        Start agent processes.
        
        Args:
            agent_id: Agent identifier
        
        Returns:
            List of process IDs
        """
        if agent_id not in self.agents:
            raise AgentError(f"Agent {agent_id} not registered", error_code="AGENT_NOT_FOUND")
        
        agent_info = self.agents[agent_id]
        resources = agent_info["resources"]
        
        # Start minimum number of instances
        process_ids = []
        for i in range(resources.min_instances):
            process_id = await self.process_manager.start_process(
                name=f"{agent_id}-{i}",
                target=agent_info["callable"],
                resources={
                    "cpu_limit": resources.cpu_limit,
                    "memory_limit_mb": resources.memory_limit_mb
                }
            )
            process_ids.append(process_id)
            agent_info["processes"].append(process_id)
        
        self.logger.info(
            "Agent started",
            agent_id=agent_id,
            instances=len(process_ids)
        )
        
        # Register health check if provided
        if agent_info["health_check"] and resources.health_check_enabled:
            await self.health_checker.register_check(
                agent_id,
                agent_info["health_check"]
            )
        
        # Setup scaling policy
        if resources.max_instances > resources.min_instances:
            policy = ScalingPolicy(
                min_instances=resources.min_instances,
                max_instances=resources.max_instances,
                target_cpu_percent=70.0,
                target_memory_percent=80.0
            )
            await self.scaler.register_policy(agent_id, policy)
        
        # Record metrics
        get_metrics().set_gauge(
            'memory_size',
            {'memory_type': 'running_agents'},
            len([a for a in self.agents.values() if a["processes"]])
        )
        
        return process_ids
    
    async def stop_agent(self, agent_id: str, force: bool = False):
        """
        Stop agent processes.
        
        Args:
            agent_id: Agent identifier
            force: Force kill if True, graceful shutdown if False
        """
        if agent_id not in self.agents:
            return
        
        agent_info = self.agents[agent_id]
        
        # Stop all processes
        for process_id in agent_info["processes"]:
            if force:
                await self.process_manager.kill_process(process_id)
            else:
                await self.process_manager.stop_process(
                    process_id,
                    timeout=agent_info["resources"].shutdown_timeout
                )
        
        agent_info["processes"].clear()
        
        # Unregister health check
        await self.health_checker.unregister_check(agent_id)
        
        # Remove scaling policy
        await self.scaler.unregister_policy(agent_id)
        
        self.logger.info("Agent stopped", agent_id=agent_id, force=force)
        
        # Update metrics
        get_metrics().set_gauge(
            'memory_size',
            {'memory_type': 'running_agents'},
            len([a for a in self.agents.values() if a["processes"]])
        )
    
    async def restart_agent(self, agent_id: str):
        """
        Restart agent (graceful stop + start).
        
        Args:
            agent_id: Agent identifier
        """
        self.logger.info("Restarting agent", agent_id=agent_id)
        await self.stop_agent(agent_id, force=False)
        await self.start_agent(agent_id)
    
    async def scale_agent(self, agent_id: str, instances: int):
        """
        Scale agent to specified number of instances.
        
        Args:
            agent_id: Agent identifier
            instances: Desired number of instances
        """
        if agent_id not in self.agents:
            raise AgentError(f"Agent {agent_id} not registered", error_code="AGENT_NOT_FOUND")
        
        agent_info = self.agents[agent_id]
        resources = agent_info["resources"]
        
        # Validate instance count
        if instances < resources.min_instances or instances > resources.max_instances:
            raise AgentError(
                f"Instance count {instances} outside allowed range [{resources.min_instances}, {resources.max_instances}]",
                error_code="INVALID_SCALE"
            )
        
        current_count = len(agent_info["processes"])
        
        if instances > current_count:
            # Scale up
            for i in range(instances - current_count):
                process_id = await self.process_manager.start_process(
                    name=f"{agent_id}-{current_count + i}",
                    target=agent_info["callable"],
                    resources={
                        "cpu_limit": resources.cpu_limit,
                        "memory_limit_mb": resources.memory_limit_mb
                    }
                )
                agent_info["processes"].append(process_id)
        
        elif instances < current_count:
            # Scale down
            for _ in range(current_count - instances):
                process_id = agent_info["processes"].pop()
                await self.process_manager.stop_process(process_id)
        
        self.logger.info(
            "Agent scaled",
            agent_id=agent_id,
            from_instances=current_count,
            to_instances=instances
        )
    
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """
        Get agent status.
        
        Args:
            agent_id: Agent identifier
        
        Returns:
            Status dictionary
        """
        if agent_id not in self.agents:
            return {"status": "not_found"}
        
        agent_info = self.agents[agent_id]
        
        # Get process statuses
        process_statuses = []
        for process_id in agent_info["processes"]:
            process = await self.process_manager.get_process(process_id)
            if process:
                process_statuses.append({
                    "process_id": process_id,
                    "status": process.status.value,
                    "cpu_percent": process.cpu_percent,
                    "memory_mb": process.memory_mb
                })
        
        # Get health status
        health_status = await self.health_checker.check_health(agent_id)
        
        return {
            "status": "running" if agent_info["processes"] else "stopped",
            "instances": len(agent_info["processes"]),
            "resources": agent_info["resources"].dict(),
            "processes": process_statuses,
            "health": health_status,
            "created_at": agent_info["created_at"].isoformat()
        }
    
    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents."""
        return [
            {
                "agent_id": agent_id,
                **await self.get_agent_status(agent_id)
            }
            for agent_id in self.agents.keys()
        ]
    
    async def start(self):
        """Start the runtime."""
        if self.running:
            return
        
        self.running = True
        self.logger.info(
            "Runtime starting",
            environment=self.config.environment,
            debug=self.config.debug
        )
        
        # Start background tasks
        self.tasks.append(
            asyncio.create_task(self.health_checker.start_monitoring())
        )
        
        self.tasks.append(
            asyncio.create_task(self.scaler.start_monitoring(self))
        )
        
        # Start hot reload in development
        if self.config.hot_reload and self.config.environment == "development":
            from teleon.helix.watcher import FileWatcher, WatcherConfig
            
            watcher_config = WatcherConfig(
                watch_paths=self.config.watch_paths
            )
            self.file_watcher = FileWatcher(watcher_config, self)
            self.tasks.append(
                asyncio.create_task(self.file_watcher.start())
            )
            self.logger.info("Hot reload enabled")
        
        self.logger.info("Runtime started")
    
    async def shutdown(self):
        """Gracefully shutdown the runtime."""
        if not self.running:
            return
        
        self.logger.info("Runtime shutdown initiated")
        self.running = False
        
        # Stop all agents
        for agent_id in list(self.agents.keys()):
            await self.stop_agent(agent_id, force=False)
        
        # Cancel background tasks
        for task in self.tasks:
            task.cancel()
        
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Shutdown components
        await self.process_manager.shutdown()
        await self.health_checker.shutdown()
        await self.scaler.shutdown()
        
        self.shutdown_event.set()
        
        self.logger.info("Runtime shutdown complete")
    
    async def wait_for_shutdown(self):
        """Wait for shutdown signal."""
        await self.shutdown_event.wait()


# Global runtime instance
_runtime: Optional[AgentRuntime] = None


def get_runtime(config: Optional[RuntimeConfig] = None) -> AgentRuntime:
    """Get global runtime instance."""
    global _runtime
    if _runtime is None:
        _runtime = AgentRuntime(config)
    return _runtime

