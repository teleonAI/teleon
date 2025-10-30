"""
Workflow Engine - Production-grade multi-agent workflow orchestration.

Features:
- DAG-based workflows
- Parallel execution
- Conditional branching
- Error handling
- State management
- Workflow templates
"""

from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
from collections import defaultdict

from teleon.core import (
    get_metrics,
    StructuredLogger,
    LogLevel,
)
from teleon.nexusnet.delegation import TaskDelegator, Task, TaskStatus, get_registry
from teleon.nexusnet.registry import AgentCapability


class NodeType(str, Enum):
    """Workflow node types."""
    TASK = "task"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    WAIT = "wait"


class NodeStatus(str, Enum):
    """Node execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowNode(BaseModel):
    """Workflow node."""
    
    node_id: str = Field(..., description="Unique node ID")
    name: str = Field(..., description="Node name")
    node_type: NodeType = Field(NodeType.TASK, description="Node type")
    
    # Task configuration
    task_name: Optional[str] = Field(None, description="Task name")
    task_input: Dict[str, Any] = Field(default_factory=dict, description="Task input")
    required_capabilities: List[AgentCapability] = Field(
        default_factory=list,
        description="Required capabilities"
    )
    
    # Dependencies
    depends_on: List[str] = Field(default_factory=list, description="Node dependencies")
    
    # Conditional execution
    condition: Optional[str] = Field(None, description="Condition expression")
    
    # Status
    status: NodeStatus = Field(NodeStatus.PENDING, description="Node status")
    output: Optional[Dict[str, Any]] = Field(None, description="Node output")
    error: Optional[str] = Field(None, description="Error message")
    
    # Timing
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class Workflow(BaseModel):
    """Workflow definition."""
    
    workflow_id: str = Field(..., description="Unique workflow ID")
    name: str = Field(..., description="Workflow name")
    description: str = Field("", description="Workflow description")
    
    # Nodes
    nodes: Dict[str, WorkflowNode] = Field(..., description="Workflow nodes")
    
    # State
    state: Dict[str, Any] = Field(default_factory=dict, description="Workflow state")
    
    # Execution
    status: str = Field("pending", description="Workflow status")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class WorkflowEngine:
    """
    Production-grade workflow engine.
    
    Features:
    - DAG execution
    - Parallel task execution
    - Error handling
    - State management
    """
    
    def __init__(self, delegator: Optional[TaskDelegator] = None):
        """
        Initialize workflow engine.
        
        Args:
            delegator: Task delegator
        """
        self.delegator = delegator or TaskDelegator()
        
        # Active workflows
        self.workflows: Dict[str, Workflow] = {}
        self.lock = asyncio.Lock()
        
        self.logger = StructuredLogger("workflow_engine", LogLevel.INFO)
    
    async def create_workflow(
        self,
        name: str,
        nodes: List[WorkflowNode],
        **kwargs
    ) -> Workflow:
        """
        Create a new workflow.
        
        Args:
            name: Workflow name
            nodes: Workflow nodes
            **kwargs: Additional workflow parameters
        
        Returns:
            Created workflow
        """
        import uuid
        
        workflow_id = str(uuid.uuid4())
        
        # Convert nodes list to dict
        nodes_dict = {node.node_id: node for node in nodes}
        
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            nodes=nodes_dict,
            **kwargs
        )
        
        # Validate DAG
        self._validate_dag(workflow)
        
        async with self.lock:
            self.workflows[workflow_id] = workflow
        
        self.logger.info(
            "Workflow created",
            workflow_id=workflow_id,
            name=name,
            node_count=len(nodes)
        )
        
        return workflow
    
    def _validate_dag(self, workflow: Workflow):
        """Validate workflow is a valid DAG."""
        # Check for cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            node = workflow.nodes[node_id]
            for dep in node.depends_on:
                if dep not in workflow.nodes:
                    raise ValueError(f"Invalid dependency: {dep}")
                
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in workflow.nodes:
            if node_id not in visited:
                if has_cycle(node_id):
                    raise ValueError("Workflow contains cycle")
    
    async def execute_workflow(self, workflow_id: str) -> Workflow:
        """
        Execute workflow.
        
        Args:
            workflow_id: Workflow ID
        
        Returns:
            Completed workflow
        """
        async with self.lock:
            workflow = self.workflows.get(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            workflow.status = "running"
            workflow.started_at = datetime.utcnow()
        
        self.logger.info("Workflow execution started", workflow_id=workflow_id)
        
        try:
            # Execute nodes in topological order
            await self._execute_nodes(workflow)
            
            async with self.lock:
                workflow.status = "completed"
                workflow.completed_at = datetime.utcnow()
            
            self.logger.info(
                "Workflow completed",
                workflow_id=workflow_id,
                duration=(workflow.completed_at - workflow.started_at).total_seconds()
            )
        
        except Exception as e:
            async with self.lock:
                workflow.status = "failed"
                workflow.completed_at = datetime.utcnow()
            
            self.logger.error(
                "Workflow failed",
                workflow_id=workflow_id,
                error=str(e)
            )
            raise
        
        return workflow
    
    async def _execute_nodes(self, workflow: Workflow):
        """Execute workflow nodes."""
        completed = set()
        
        while len(completed) < len(workflow.nodes):
            # Find ready nodes (dependencies completed)
            ready_nodes = []
            
            for node_id, node in workflow.nodes.items():
                if node_id in completed:
                    continue
                
                if node.status != NodeStatus.PENDING:
                    continue
                
                # Check dependencies
                deps_ready = all(
                    dep in completed or workflow.nodes[dep].status == NodeStatus.SKIPPED
                    for dep in node.depends_on
                )
                
                if deps_ready:
                    ready_nodes.append(node)
            
            if not ready_nodes:
                # Check for deadlock
                pending = [n for n in workflow.nodes.values() if n.status == NodeStatus.PENDING]
                if pending:
                    raise RuntimeError("Workflow deadlock detected")
                break
            
            # Execute ready nodes in parallel
            tasks = [
                self._execute_node(workflow, node)
                for node in ready_nodes
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Mark completed
            for node, result in zip(ready_nodes, results):
                if isinstance(result, Exception):
                    node.status = NodeStatus.FAILED
                    node.error = str(result)
                    raise result
                else:
                    completed.add(node.node_id)
    
    async def _execute_node(self, workflow: Workflow, node: WorkflowNode):
        """Execute a single node."""
        node.status = NodeStatus.RUNNING
        node.started_at = datetime.utcnow()
        
        try:
            # Check condition
            if node.condition:
                if not self._evaluate_condition(node.condition, workflow.state):
                    node.status = NodeStatus.SKIPPED
                    self.logger.info("Node skipped", node_id=node.node_id)
                    return
            
            # Execute based on type
            if node.node_type == NodeType.TASK:
                output = await self._execute_task_node(workflow, node)
                node.output = output
            
            elif node.node_type == NodeType.WAIT:
                await asyncio.sleep(node.task_input.get("duration", 1))
                node.output = {}
            
            node.status = NodeStatus.COMPLETED
            node.completed_at = datetime.utcnow()
            
            self.logger.info(
                "Node completed",
                node_id=node.node_id,
                node_type=node.node_type.value
            )
        
        except Exception as e:
            node.status = NodeStatus.FAILED
            node.error = str(e)
            node.completed_at = datetime.utcnow()
            
            self.logger.error(
                "Node failed",
                node_id=node.node_id,
                error=str(e)
            )
            raise
    
    async def _execute_task_node(
        self,
        workflow: Workflow,
        node: WorkflowNode
    ) -> Dict[str, Any]:
        """Execute task node."""
        # Interpolate input from workflow state
        task_input = self._interpolate_input(node.task_input, workflow.state)
        
        # Create task
        task = await self.delegator.create_task(
            name=node.task_name or node.name,
            input_data=task_input,
            required_capabilities=node.required_capabilities
        )
        
        # Execute task
        task = await self.delegator.execute_task(task)
        
        # Wait for completion
        while task.status in [TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]:
            await asyncio.sleep(1)
            task = await self.delegator.get_task(task.task_id)
        
        if task.status == TaskStatus.COMPLETED:
            # Update workflow state
            workflow.state[node.node_id] = task.output_data
            return task.output_data
        else:
            raise RuntimeError(f"Task failed: {task.error}")
    
    def _interpolate_input(
        self,
        input_data: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Interpolate input from workflow state."""
        result = {}
        
        for key, value in input_data.items():
            if isinstance(value, str) and value.startswith("$"):
                # Reference to state
                state_key = value[1:]
                result[key] = state.get(state_key, value)
            else:
                result[key] = value
        
        return result
    
    def _evaluate_condition(self, condition: str, state: Dict[str, Any]) -> bool:
        """Evaluate condition expression."""
        # Simple condition evaluation
        # In production, use a proper expression evaluator
        try:
            return eval(condition, {"state": state})
        except:
            return False
    
    async def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow by ID."""
        async with self.lock:
            return self.workflows.get(workflow_id)
    
    async def list_workflows(self) -> List[Workflow]:
        """List all workflows."""
        async with self.lock:
            return list(self.workflows.values())
    
    async def shutdown(self):
        """Gracefully shutdown workflow engine."""
        self.logger.info("Workflow engine shutdown")

