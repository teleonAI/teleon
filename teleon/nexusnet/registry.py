"""
Agent Registry - Production-grade agent discovery and registration.

Features:
- Agent registration with metadata
- Health checking
- Service discovery
- Capability-based lookup
- Load tracking
- Automatic deregistration
"""

from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import asyncio

from teleon.core import (
    AgentError,
    AgentNotFoundError,
    get_metrics,
    StructuredLogger,
    LogLevel,
)


class AgentStatus(str, Enum):
    """Agent status."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    IDLE = "idle"


class AgentCapability(str, Enum):
    """Common agent capabilities."""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    DATA_ANALYSIS = "data_analysis"
    IMAGE_GENERATION = "image_generation"
    WEB_SEARCH = "web_search"
    DATABASE_QUERY = "database_query"
    API_INTEGRATION = "api_integration"
    CUSTOM = "custom"


class AgentInfo(BaseModel):
    """Agent information."""
    
    agent_id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent name")
    description: str = Field("", description="Agent description")
    
    # Capabilities - now supports custom strings
    capabilities: List[str] = Field(
        default_factory=list,
        description="Agent capabilities (custom strings supported)"
    )
    tags: List[str] = Field(default_factory=list, description="Custom tags")
    
    # Status
    status: AgentStatus = Field(AgentStatus.IDLE, description="Current status")
    
    # Metadata
    version: str = Field("1.0.0", description="Agent version")
    endpoint: Optional[str] = Field(None, description="Agent endpoint URL")
    
    # Performance
    load: float = Field(0.0, ge=0, le=1, description="Current load (0-1)")
    max_concurrent_tasks: int = Field(10, ge=1, description="Max concurrent tasks")
    current_tasks: int = Field(0, ge=0, description="Current task count")
    
    # Timestamps
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    
    # Health
    health_check_interval: int = Field(30, ge=5, description="Health check interval (seconds)")
    
    def is_available(self) -> bool:
        """Check if agent is available for new tasks."""
        return (
            self.status in [AgentStatus.ONLINE, AgentStatus.IDLE] and
            self.current_tasks < self.max_concurrent_tasks
        )
    
    def is_healthy(self) -> bool:
        """Check if agent is healthy based on last heartbeat."""
        threshold = datetime.utcnow() - timedelta(seconds=self.health_check_interval * 3)
        return self.last_heartbeat > threshold
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentRegistry:
    """
    Production-grade agent registry.
    
    Features:
    - Thread-safe registration
    - Health monitoring
    - Capability-based discovery
    - Load balancing support
    - Automatic cleanup
    """
    
    def __init__(self):
        """Initialize agent registry."""
        self.agents: Dict[str, AgentInfo] = {}
        self.lock = asyncio.Lock()
        
        self.logger = StructuredLogger("agent_registry", LogLevel.INFO)
        
        # Health check task
        self.health_check_task: Optional[asyncio.Task] = None
    
    async def register(
        self,
        agent_id: str,
        name: str,
        capabilities: List[str],
        **kwargs
    ) -> AgentInfo:
        """
        Register an agent.
        
        Args:
            agent_id: Unique agent identifier
            name: Agent name
            capabilities: Agent capabilities (as strings)
            **kwargs: Additional agent info
        
        Returns:
            AgentInfo instance
        """
        async with self.lock:
            # Create agent info
            agent_info = AgentInfo(
                agent_id=agent_id,
                name=name,
                capabilities=capabilities,
                **kwargs
            )
            
            # Register
            self.agents[agent_id] = agent_info
            
            self.logger.info(
                "Agent registered",
                agent_id=agent_id,
                name=name,
                capabilities=capabilities  # Now already strings
            )
            
            # Record metrics
            get_metrics().set_gauge(
                'memory_size',  # Reusing existing metric
                {'memory_type': 'agent_count'},
                len(self.agents)
            )
            
            return agent_info
    
    async def deregister(self, agent_id: str):
        """
        Deregister an agent.
        
        Args:
            agent_id: Agent identifier
        """
        async with self.lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
                
                self.logger.info("Agent deregistered", agent_id=agent_id)
                
                # Record metrics
                get_metrics().set_gauge(
                    'memory_size',
                    {'memory_type': 'agent_count'},
                    len(self.agents)
                )
    
    async def heartbeat(self, agent_id: str):
        """
        Update agent heartbeat.
        
        Args:
            agent_id: Agent identifier
        
        Raises:
            AgentNotFoundError: If agent not found
        """
        async with self.lock:
            if agent_id not in self.agents:
                raise AgentNotFoundError(agent_id)
            
            self.agents[agent_id].last_heartbeat = datetime.utcnow()
    
    async def update_status(
        self,
        agent_id: str,
        status: AgentStatus,
        current_tasks: Optional[int] = None
    ):
        """
        Update agent status.
        
        Args:
            agent_id: Agent identifier
            status: New status
            current_tasks: Current task count
        
        Raises:
            AgentNotFoundError: If agent not found
        """
        async with self.lock:
            if agent_id not in self.agents:
                raise AgentNotFoundError(agent_id)
            
            agent = self.agents[agent_id]
            agent.status = status
            
            if current_tasks is not None:
                agent.current_tasks = current_tasks
                agent.load = current_tasks / agent.max_concurrent_tasks
            
            self.logger.info(
                "Agent status updated",
                agent_id=agent_id,
                status=status.value,
                load=agent.load
            )
    
    async def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """
        Get agent by ID.
        
        Args:
            agent_id: Agent identifier
        
        Returns:
            AgentInfo or None
        """
        async with self.lock:
            return self.agents.get(agent_id)
    
    async def find_agents(
        self,
        capabilities: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        status: Optional[AgentStatus] = None,
        available_only: bool = False
    ) -> List[AgentInfo]:
        """
        Find agents by criteria.
        
        Args:
            capabilities: Required capabilities (as strings)
            tags: Required tags
            status: Required status
            available_only: Only return available agents
        
        Returns:
            List of matching agents
        """
        async with self.lock:
            results = []
            
            for agent in self.agents.values():
                # Check health
                if not agent.is_healthy():
                    continue
                
                # Check availability
                if available_only and not agent.is_available():
                    continue
                
                # Check status
                if status and agent.status != status:
                    continue
                
                # Check capabilities
                if capabilities:
                    agent_caps = set(agent.capabilities)
                    required_caps = set(capabilities)
                    if not required_caps.issubset(agent_caps):
                        continue
                
                # Check tags
                if tags:
                    agent_tags = set(agent.tags)
                    required_tags = set(tags)
                    if not required_tags.issubset(agent_tags):
                        continue
                
                results.append(agent)
            
            # Sort by load (least loaded first)
            results.sort(key=lambda a: a.load)
            
            return results
    
    async def list_agents(self) -> List[AgentInfo]:
        """
        List all registered agents.
        
        Returns:
            List of all agents
        """
        async with self.lock:
            return list(self.agents.values())
    
    async def list_all(self) -> List[AgentInfo]:
        """
        Alias for list_agents().
        
        Returns:
            List of all agents
        """
        return await self.list_agents()
    
    async def discover(self, capability: str) -> List[AgentInfo]:
        """
        Discover agents by capability.
        
        Args:
            capability: Capability to search for
        
        Returns:
            List of agents with the specified capability
        """
        return await self.find_agents(capabilities=[capability])
    
    async def start_health_checks(self, interval: int = 30):
        """
        Start periodic health checks.
        
        Args:
            interval: Check interval in seconds
        """
        if self.health_check_task:
            return  # Already running
        
        async def health_check_loop():
            while True:
                await asyncio.sleep(interval)
                await self._cleanup_unhealthy()
        
        self.health_check_task = asyncio.create_task(health_check_loop())
        self.logger.info(f"Health checks started (interval: {interval}s)")
    
    async def _cleanup_unhealthy(self):
        """Remove unhealthy agents."""
        async with self.lock:
            unhealthy = []
            
            for agent_id, agent in self.agents.items():
                if not agent.is_healthy():
                    unhealthy.append(agent_id)
            
            for agent_id in unhealthy:
                del self.agents[agent_id]
                self.logger.warning("Removed unhealthy agent", agent_id=agent_id)
            
            if unhealthy:
                # Update metrics
                get_metrics().set_gauge(
                    'memory_size',
                    {'memory_type': 'agent_count'},
                    len(self.agents)
                )
    
    async def shutdown(self):
        """Gracefully shutdown registry."""
        if self.health_check_task:
            self.health_check_task.cancel()
        
        self.logger.info("Agent registry shutdown")


# Global registry instance
_registry: Optional[AgentRegistry] = None


def get_registry() -> AgentRegistry:
    """Get global agent registry."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry

