"""
Task Delegation - Production-grade task distribution system.

Features:
- Intelligent task routing
- Load balancing
- Capability matching
- Task tracking
- Failure handling
- Result aggregation
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import uuid

from teleon.core import (
    get_metrics,
    StructuredLogger,
    LogLevel,
)
from teleon.nexusnet.registry import AgentRegistry, AgentCapability, get_registry
from teleon.nexusnet.messaging import MessageBus, MessageType, MessagePriority, get_message_bus


class TaskStatus(str, Enum):
    """Task status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(BaseModel):
    """Task model."""
    
    task_id: str = Field(..., description="Unique task ID")
    name: str = Field(..., description="Task name")
    description: str = Field("", description="Task description")
    
    # Requirements
    required_capabilities: List[AgentCapability] = Field(
        default_factory=list,
        description="Required capabilities"
    )
    priority: MessagePriority = Field(MessagePriority.NORMAL, description="Task priority")
    
    # Payload
    input_data: Dict[str, Any] = Field(..., description="Task input")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Task output")
    
    # Assignment
    assigned_to: Optional[str] = Field(None, description="Assigned agent ID")
    assigned_at: Optional[datetime] = Field(None, description="Assignment time")
    
    # Status
    status: TaskStatus = Field(TaskStatus.PENDING, description="Task status")
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(None, description="Start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    timeout: Optional[int] = Field(None, ge=1, description="Timeout in seconds")
    
    # Error handling
    error: Optional[str] = Field(None, description="Error message")
    retry_count: int = Field(0, ge=0, description="Retry count")
    max_retries: int = Field(3, ge=0, description="Max retries")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")
    
    def is_timed_out(self) -> bool:
        """Check if task has timed out."""
        if self.timeout and self.started_at:
            elapsed = (datetime.utcnow() - self.started_at).total_seconds()
            return elapsed > self.timeout
        return False
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class TaskDelegator:
    """
    Production-grade task delegator.
    
    Features:
    - Intelligent agent selection
    - Load balancing
    - Automatic retry
    - Task monitoring
    - Result collection
    """
    
    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        message_bus: Optional[MessageBus] = None
    ):
        """
        Initialize task delegator.
        
        Args:
            registry: Agent registry
            message_bus: Message bus
        """
        self.registry = registry or get_registry()
        self.message_bus = message_bus or get_message_bus()
        
        # Task tracking
        self.tasks: Dict[str, Task] = {}
        self.lock = asyncio.Lock()
        
        self.logger = StructuredLogger("task_delegator", LogLevel.INFO)
        
        # Monitoring task
        self.monitor_task: Optional[asyncio.Task] = None
    
    async def create_task(
        self,
        name: str,
        input_data: Dict[str, Any],
        required_capabilities: Optional[List[AgentCapability]] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Task:
        """
        Create a new task.
        
        Args:
            name: Task name
            input_data: Task input
            required_capabilities: Required capabilities
            priority: Task priority
            timeout: Timeout in seconds
            **kwargs: Additional task parameters
        
        Returns:
            Created task
        """
        task_id = str(uuid.uuid4())
        
        task = Task(
            task_id=task_id,
            name=name,
            input_data=input_data,
            required_capabilities=required_capabilities or [],
            priority=priority,
            timeout=timeout,
            **kwargs
        )
        
        async with self.lock:
            self.tasks[task_id] = task
        
        self.logger.info(
            "Task created",
            task_id=task_id,
            name=name,
            capabilities=[c.value for c in (required_capabilities or [])]
        )
        
        return task
    
    async def delegate(self, task: Task) -> Optional[str]:
        """
        Delegate task to suitable agent.
        
        Args:
            task: Task to delegate
        
        Returns:
            Assigned agent ID or None
        """
        # Find suitable agents
        agents = await self.registry.find_agents(
            capabilities=task.required_capabilities if task.required_capabilities else None,
            available_only=True
        )
        
        if not agents:
            self.logger.warning(
                "No suitable agents found",
                task_id=task.task_id,
                capabilities=[c.value for c in task.required_capabilities]
            )
            return None
        
        # Select agent (least loaded)
        agent = agents[0]
        
        # Assign task
        async with self.lock:
            task.assigned_to = agent.agent_id
            task.assigned_at = datetime.utcnow()
            task.status = TaskStatus.ASSIGNED
        
        # Send task message
        await self.message_bus.send(
            from_agent="delegator",
            to_agent=agent.agent_id,
            payload={
                "task_id": task.task_id,
                "name": task.name,
                "input": task.input_data,
                "timeout": task.timeout
            },
            message_type=MessageType.REQUEST,
            priority=task.priority
        )
        
        self.logger.info(
            "Task delegated",
            task_id=task.task_id,
            agent_id=agent.agent_id,
            agent_load=agent.load
        )
        
        # Update agent status
        await self.registry.update_status(
            agent.agent_id,
            agent.status,
            agent.current_tasks + 1
        )
        
        # Record metrics
        get_metrics().increment_counter(
            'memory_operations',
            {'memory_type': 'tasks', 'operation': 'delegated'},
            1
        )
        
        return agent.agent_id
    
    async def execute_task(self, task: Task) -> Task:
        """
        Execute task (create and delegate).
        
        Args:
            task: Task to execute
        
        Returns:
            Updated task
        """
        agent_id = await self.delegate(task)
        
        if not agent_id:
            async with self.lock:
                task.status = TaskStatus.FAILED
                task.error = "No suitable agents available"
            return task
        
        # Mark as in progress
        async with self.lock:
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.utcnow()
        
        return task
    
    async def complete_task(
        self,
        task_id: str,
        output_data: Dict[str, Any],
        success: bool = True,
        error: Optional[str] = None
    ) -> Task:
        """
        Mark task as complete.
        
        Args:
            task_id: Task ID
            output_data: Task output
            success: Whether task succeeded
            error: Error message if failed
        
        Returns:
            Updated task
        """
        async with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")
            
            task.output_data = output_data
            task.completed_at = datetime.utcnow()
            
            if success:
                task.status = TaskStatus.COMPLETED
            else:
                task.status = TaskStatus.FAILED
                task.error = error
            
            # Update agent status
            if task.assigned_to:
                agent = await self.registry.get_agent(task.assigned_to)
                if agent:
                    await self.registry.update_status(
                        agent.agent_id,
                        agent.status,
                        max(0, agent.current_tasks - 1)
                    )
        
        self.logger.info(
            "Task completed",
            task_id=task_id,
            success=success,
            duration=(task.completed_at - task.created_at).total_seconds() if task.completed_at else None
        )
        
        # Record metrics
        get_metrics().increment_counter(
            'memory_operations',
            {'memory_type': 'tasks', 'operation': 'completed' if success else 'failed'},
            1
        )
        
        return task
    
    async def retry_task(self, task_id: str) -> bool:
        """
        Retry a failed task.
        
        Args:
            task_id: Task ID
        
        Returns:
            True if retry initiated
        """
        async with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return False
            
            if not task.can_retry():
                self.logger.warning(
                    "Task cannot be retried",
                    task_id=task_id,
                    retry_count=task.retry_count,
                    max_retries=task.max_retries
                )
                return False
            
            # Reset task state
            task.retry_count += 1
            task.status = TaskStatus.PENDING
            task.assigned_to = None
            task.assigned_at = None
            task.started_at = None
            task.error = None
        
        # Re-delegate
        agent_id = await self.delegate(task)
        
        if agent_id:
            self.logger.info(
                "Task retried",
                task_id=task_id,
                retry_count=task.retry_count,
                agent_id=agent_id
            )
            return True
        
        return False
    
    async def cancel_task(self, task_id: str):
        """
        Cancel a task.
        
        Args:
            task_id: Task ID
        """
        async with self.lock:
            task = self.tasks.get(task_id)
            if task:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.utcnow()
                
                self.logger.info("Task cancelled", task_id=task_id)
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        async with self.lock:
            return self.tasks.get(task_id)
    
    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None
    ) -> List[Task]:
        """
        List tasks.
        
        Args:
            status: Filter by status
        
        Returns:
            List of tasks
        """
        async with self.lock:
            tasks = list(self.tasks.values())
            
            if status:
                tasks = [t for t in tasks if t.status == status]
            
            return tasks
    
    async def start_monitoring(self, interval: int = 10):
        """
        Start task monitoring.
        
        Args:
            interval: Check interval in seconds
        """
        if self.monitor_task:
            return
        
        async def monitor_loop():
            while True:
                await asyncio.sleep(interval)
                await self._check_timeouts()
        
        self.monitor_task = asyncio.create_task(monitor_loop())
        self.logger.info(f"Task monitoring started (interval: {interval}s)")
    
    async def _check_timeouts(self):
        """Check for timed out tasks and retry."""
        async with self.lock:
            for task in self.tasks.values():
                if (task.status == TaskStatus.IN_PROGRESS and 
                    task.is_timed_out()):
                    
                    self.logger.warning(
                        "Task timed out",
                        task_id=task.task_id,
                        timeout=task.timeout
                    )
                    
                    task.error = f"Task timed out after {task.timeout}s"
                    task.status = TaskStatus.FAILED
                    
                    # Attempt retry
                    if task.can_retry():
                        await self.retry_task(task.task_id)
    
    async def shutdown(self):
        """Gracefully shutdown delegator."""
        if self.monitor_task:
            self.monitor_task.cancel()
        
        self.logger.info("Task delegator shutdown")

