"""
NexusNet - Multi-Agent Collaboration System.

This package provides production-grade multi-agent collaboration features:
- Agent Registry: Discovery and registration
- Message Passing: Reliable communication
- Task Delegation: Intelligent task distribution
- Multi-Agent Workflows: Complex orchestration
- Coordination Primitives: Synchronization and state sharing
"""

from teleon.nexusnet.registry import (
    AgentRegistry,
    AgentInfo,
    AgentCapability,
    AgentStatus,
    get_registry
)
from teleon.nexusnet.messaging import (
    MessageBus,
    Message,
    MessageType,
    MessagePriority
)
from teleon.nexusnet.delegation import (
    TaskDelegator,
    Task,
    TaskStatus
)
from teleon.nexusnet.workflow import (
    Workflow,
    WorkflowNode,
    WorkflowEngine,
    NodeType
)
from teleon.nexusnet.coordination import (
    Coordinator,
    Lock,
    SharedState
)
from teleon.nexusnet.client import (
    CollaborationAPI,
    create_collaboration_api
)

__all__ = [
    # Registry
    "AgentRegistry",
    "AgentInfo",
    "AgentCapability",
    "AgentStatus",
    "get_registry",
    
    # Messaging
    "MessageBus",
    "Message",
    "MessageType",
    "MessagePriority",
    
    # Delegation
    "TaskDelegator",
    "Task",
    "TaskStatus",
    
    # Workflow
    "Workflow",
    "WorkflowNode",
    "WorkflowEngine",
    "NodeType",
    
    # Coordination
    "Coordinator",
    "Lock",
    "SharedState",
    
    # Client API
    "CollaborationAPI",
    "create_collaboration_api",
]

