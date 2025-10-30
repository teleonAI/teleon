"""
NexusNet Client API - User-facing Collaboration Interface.

Provides simple, intuitive APIs for multi-agent collaboration.
"""

from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import asyncio

from teleon.nexusnet import (
    AgentRegistry,
    AgentInfo,
    AgentCapability,
    AgentStatus,
    get_registry,
    MessageBus,
    Message,
    MessageType,
    MessagePriority,
    TaskDelegator,
    Task,
    TaskStatus,
    Coordinator,
)
from teleon.core import StructuredLogger, LogLevel


class CollaborationAPI:
    """
    User-facing API for multi-agent collaboration.
    
    This provides a simple interface for agents to discover, communicate,
    and collaborate with each other.
    
    Features:
    - Agent discovery by capability
    - Task delegation
    - Message passing
    - Broadcasting
    - Workflow coordination
    
    Example:
        ```python
        # In agent decorator
        @client.agent(
            name="researcher",
            nexusnet={'capabilities': ['research'], 'collaborate': True}
        )
        async def researcher(topic: str, collaboration: CollaborationAPI):
            # Find an analyst agent
            analyst = await collaboration.find_agent('analysis')
            
            # Delegate data collection
            data = await collaboration.delegate_to(
                analyst,
                task={'action': 'collect_data', 'topic': topic}
            )
            
            # Broadcast progress
            await collaboration.broadcast("Research complete", priority="normal")
            
            return analyze(data)
        ```
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        registry: Optional[AgentRegistry] = None,
        message_bus: Optional[MessageBus] = None,
        delegator: Optional[TaskDelegator] = None
    ):
        """
        Initialize collaboration API.
        
        Args:
            agent_id: ID of the agent using this API
            agent_name: Name of the agent
            registry: Agent registry (creates one if None)
            message_bus: Message bus (creates one if None)
            delegator: Task delegator (creates one if None)
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        
        self.registry = registry or get_registry()
        self.message_bus = message_bus or MessageBus()
        self.delegator = delegator or TaskDelegator(self.registry, self.message_bus)
        
        self.logger = StructuredLogger(f"nexusnet.{agent_id}", LogLevel.INFO)
        
        # Track collaborations
        self._active_tasks: Dict[str, Task] = {}
        self._message_history: List[Message] = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about collaboration activity.
        
        Returns:
            Dictionary with collaboration statistics
        """
        return {
            "active_tasks": len(self._active_tasks),
            "messages_sent": len(self._message_history),
            "agent_id": self.agent_id,
            "agent_name": self.agent_name
        }
    
    async def shutdown(self):
        """Shutdown the collaboration API and cleanup resources."""
        self.logger.info("Shutting down CollaborationAPI", agent_id=self.agent_id)
        
        # Shutdown registry (which will cancel health check tasks)
        await self.registry.shutdown()
        
        self.logger.info("CollaborationAPI shutdown complete")
    
    async def find_agent(
        self,
        capability: str,
        filters: Optional[Dict[str, Any]] = None,
        require_healthy: bool = True
    ) -> Optional[str]:
        """
        Find an agent with a specific capability.
        
        Args:
            capability: Required capability name
            filters: Additional filters (e.g., {'name': 'my-agent'})
            require_healthy: Only return healthy agents
        
        Returns:
            Agent ID if found, None otherwise
        
        Example:
            ```python
            # Find any analyst
            analyst = await collaboration.find_agent('analysis')
            
            # Find specific agent
            my_analyst = await collaboration.find_agent(
                'analysis',
                filters={'name': 'expert-analyst'}
            )
            ```
        """
        try:
            # Find agents with the specified capability
            agents = await self.registry.find_agents(
                capabilities=[capability] if capability else None
            )
            
            if not agents:
                self.logger.warning(
                    "No agents found",
                    capability=capability,
                    agent_id=self.agent_id
                )
                return None
            
            # Apply filters
            if filters:
                filtered = []
                for agent in agents:
                    match = True
                    for key, value in filters.items():
                        if getattr(agent, key, None) != value:
                            match = False
                            break
                    if match:
                        filtered.append(agent)
                agents = filtered
            
            # Filter by health
            if require_healthy:
                agents = [a for a in agents if a.status == AgentStatus.HEALTHY]
            
            if not agents:
                return None
            
            # Return first matching agent
            selected = agents[0]
            
            self.logger.info(
                "Agent found",
                capability=capability,
                found_agent=selected.agent_id,
                requesting_agent=self.agent_id
            )
            
            return selected.agent_id
            
        except Exception as e:
            self.logger.error(
                "Agent discovery failed",
                capability=capability,
                error=str(e)
            )
            return None
    
    async def find_all_agents(
        self,
        capability: Optional[str] = None,
        require_healthy: bool = True
    ) -> List[str]:
        """
        Find all agents with optional capability filter.
        
        Args:
            capability: Optional capability filter
            require_healthy: Only return healthy agents
        
        Returns:
            List of agent IDs
        
        Example:
            ```python
            # Find all healthy agents
            all_agents = await collaboration.find_all_agents()
            
            # Find all analysts
            analysts = await collaboration.find_all_agents('analysis')
            ```
        """
        if capability:
            agents = await self.registry.find_agents(capabilities=[capability])
        else:
            agents = await self.registry.list_agents()
        
        if require_healthy:
            agents = [a for a in agents if a.is_healthy()]
        
        return [a.agent_id for a in agents]
    
    async def delegate_to(
        self,
        target_agent: str,
        task: Dict[str, Any],
        timeout: float = 30.0,
        priority: str = "normal"
    ) -> Any:
        """
        Delegate a task to another agent.
        
        Args:
            target_agent: Target agent ID
            task: Task data
            timeout: Task timeout in seconds
            priority: Task priority (low, normal, high)
        
        Returns:
            Task result
        
        Raises:
            TimeoutError: If task times out
            RuntimeError: If task fails
        
        Example:
            ```python
            result = await collaboration.delegate_to(
                "analyst-123",
                task={'action': 'analyze', 'data': [1, 2, 3]},
                timeout=60.0
            )
            ```
        """
        self.logger.info(
            "Delegating task",
            from_agent=self.agent_id,
            to_agent=target_agent,
            task=task
        )
        
        try:
            # Create task
            task_obj = Task(
                task_id=f"task_{self.agent_id}_{len(self._active_tasks)}",
                requester=self.agent_id,
                assignee=target_agent,
                task_data=task,
                priority=priority
            )
            
            self._active_tasks[task_obj.task_id] = task_obj
            
            # Delegate through delegator
            result = await self.delegator.delegate(task_obj, timeout=timeout)
            
            # Clean up
            del self._active_tasks[task_obj.task_id]
            
            self.logger.info(
                "Task completed",
                task_id=task_obj.task_id,
                from_agent=self.agent_id,
                to_agent=target_agent
            )
            
            return result
            
        except asyncio.TimeoutError:
            self.logger.error(
                "Task timeout",
                from_agent=self.agent_id,
                to_agent=target_agent,
                timeout=timeout
            )
            raise TimeoutError(f"Task to {target_agent} timed out after {timeout}s")
        
        except Exception as e:
            self.logger.error(
                "Task failed",
                from_agent=self.agent_id,
                to_agent=target_agent,
                error=str(e)
            )
            raise RuntimeError(f"Task failed: {str(e)}")
    
    async def send_message(
        self,
        to_agent: str,
        content: str,
        message_type: str = "info",
        priority: str = "normal"
    ) -> str:
        """
        Send a message to another agent.
        
        Args:
            to_agent: Target agent ID
            content: Message content
            message_type: Message type (info, request, response, error)
            priority: Message priority (low, normal, high)
        
        Returns:
            Message ID
        
        Example:
            ```python
            msg_id = await collaboration.send_message(
                "analyst-123",
                "Please start analysis",
                message_type="request"
            )
            ```
        """
        message = Message(
            from_agent=self.agent_id,
            to_agent=to_agent,
            content=content,
            message_type=MessageType[message_type.upper()],
            priority=MessagePriority[priority.upper()]
        )
        
        await self.message_bus.publish(message)
        self._message_history.append(message)
        
        self.logger.debug(
            "Message sent",
            from_agent=self.agent_id,
            to_agent=to_agent,
            message_id=message.message_id
        )
        
        return message.message_id
    
    async def broadcast(
        self,
        content: str,
        capability: Optional[str] = None,
        message_type: str = "info",
        priority: str = "normal"
    ) -> List[str]:
        """
        Broadcast a message to multiple agents.
        
        Args:
            content: Message content
            capability: Optional capability filter
            message_type: Message type
            priority: Message priority
        
        Returns:
            List of message IDs
        
        Example:
            ```python
            # Broadcast to all analysts
            msg_ids = await collaboration.broadcast(
                "Starting data collection",
                capability="analysis"
            )
            
            # Broadcast to all agents
            msg_ids = await collaboration.broadcast(
                "System maintenance in 5 minutes"
            )
            ```
        """
        # Find target agents
        target_agents = await self.find_all_agents(capability=capability)
        
        # Remove self
        target_agents = [a for a in target_agents if a != self.agent_id]
        
        # Send to all
        message_ids = []
        for agent in target_agents:
            msg_id = await self.send_message(
                agent,
                content,
                message_type=message_type,
                priority=priority
            )
            message_ids.append(msg_id)
        
        self.logger.info(
            "Broadcast sent",
            from_agent=self.agent_id,
            target_count=len(target_agents),
            capability=capability
        )
        
        return message_ids
    
    async def receive_messages(
        self,
        limit: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> List[Message]:
        """
        Receive messages sent to this agent.
        
        Args:
            limit: Maximum number of messages to retrieve
            since: Only get messages after this time
        
        Returns:
            List of messages
        
        Example:
            ```python
            # Get latest 10 messages
            messages = await collaboration.receive_messages(limit=10)
            
            for msg in messages:
                print(f"From {msg.from_agent}: {msg.content}")
            ```
        """
        messages = await self.message_bus.get_messages(
            self.agent_id,
            limit=limit
        )
        
        # Filter by time if specified
        if since:
            messages = [m for m in messages if m.timestamp >= since]
        
        return messages
    
    async def get_active_tasks(self) -> List[Task]:
        """
        Get currently active tasks for this agent.
        
        Returns:
            List of active tasks
        """
        return list(self._active_tasks.values())
    
    async def get_message_history(
        self,
        limit: Optional[int] = None
    ) -> List[Message]:
        """
        Get message history.
        
        Args:
            limit: Maximum number of messages
        
        Returns:
            List of messages
        """
        history = self._message_history.copy()
        if limit:
            history = history[-limit:]
        return history
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get collaboration statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "agent_id": self.agent_id,
            "active_tasks": len(self._active_tasks),
            "messages_sent": len(self._message_history),
            "message_history_size": len(self._message_history)
        }


async def create_collaboration_api(
    agent_id: str,
    agent_name: str,
    capabilities: Optional[List[str]] = None,
    auto_register: bool = True
) -> CollaborationAPI:
    """
    Create and initialize a collaboration API.
    
    Args:
        agent_id: Agent ID
        agent_name: Agent name
        capabilities: Agent capabilities
        auto_register: Automatically register with registry
    
    Returns:
        CollaborationAPI instance
    
    Example:
        ```python
        collab = await create_collaboration_api(
            agent_id="researcher-123",
            agent_name="Research Agent",
            capabilities=["research", "analysis"]
        )
        
        # Use collaboration API
        analyst = await collab.find_agent("analysis")
        ```
    """
    api = CollaborationAPI(
        agent_id=agent_id,
        agent_name=agent_name
    )
    
    # Register agent if requested
    if auto_register and capabilities:
        # Register agent with NexusNet
        await api.registry.register(
            agent_id=agent_id,
            name=agent_name,
            capabilities=capabilities,  # List of strings
            description=f"{agent_name} with capabilities: {', '.join(capabilities)}"
        )
    
    return api


__all__ = [
    "CollaborationAPI",
    "create_collaboration_api",
]

