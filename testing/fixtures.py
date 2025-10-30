"""
Test Fixtures - Helper functions for test setup.

Features:
- Agent fixtures
- Context fixtures
- Configuration fixtures
- Data fixtures
"""

from typing import Any, Dict, Optional, Callable
from datetime import datetime
import uuid

from teleon.config.agent_config import AgentConfig
from teleon.context.execution import ExecutionContext


def create_test_agent(
    name: str = "test_agent",
    description: str = "Test agent",
    **kwargs
) -> AgentConfig:
    """
    Create a test agent configuration.
    
    Args:
        name: Agent name
        description: Agent description
        **kwargs: Additional configuration
    
    Returns:
        AgentConfig instance
    """
    config_dict = {
        "name": name,
        "description": description,
        "timeout": 30,
        "retries": 3,
    }
    config_dict.update(kwargs)
    
    return AgentConfig(**config_dict)


def create_test_context(
    agent_name: str = "test_agent",
    execution_id: Optional[str] = None,
    **kwargs
) -> ExecutionContext:
    """
    Create a test execution context.
    
    Args:
        agent_name: Agent name
        execution_id: Execution ID
        **kwargs: Additional context data
    
    Returns:
        ExecutionContext instance
    """
    if execution_id is None:
        execution_id = str(uuid.uuid4())
    
    ctx = ExecutionContext(
        agent_name=agent_name,
        execution_id=execution_id,
        started_at=datetime.utcnow()
    )
    
    # Add custom attributes
    for key, value in kwargs.items():
        setattr(ctx, key, value)
    
    return ctx


def create_test_config(
    **overrides
) -> Dict[str, Any]:
    """
    Create a test configuration dictionary.
    
    Args:
        **overrides: Configuration overrides
    
    Returns:
        Configuration dictionary
    """
    config = {
        "environment": "test",
        "debug": True,
        "telemetry_enabled": False,
        "log_level": "DEBUG",
    }
    config.update(overrides)
    
    return config


def create_test_llm_request(
    model: str = "gpt-4",
    messages: Optional[list] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a test LLM request.
    
    Args:
        model: Model name
        messages: Messages
        **kwargs: Additional parameters
    
    Returns:
        LLM request dictionary
    """
    if messages is None:
        messages = [
            {"role": "user", "content": "Test message"}
        ]
    
    request = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 100,
    }
    request.update(kwargs)
    
    return request


def create_test_tool(
    name: str = "test_tool",
    description: str = "Test tool",
    execute_fn: Optional[Callable] = None
):
    """
    Create a test tool.
    
    Args:
        name: Tool name
        description: Tool description
        execute_fn: Execute function
    
    Returns:
        Tool instance
    """
    from teleon.tools.base import BaseTool
    
    class TestTool(BaseTool):
        def __init__(self):
            super().__init__(name=name, description=description)
            self._execute_fn = execute_fn
        
        async def _execute(self, *args, **kwargs):
            if self._execute_fn:
                return await self._execute_fn(*args, **kwargs)
            return {"success": True, "data": "Test result"}
    
    return TestTool()


def create_test_messages(
    count: int = 3,
    role: str = "user"
) -> list:
    """
    Create test messages.
    
    Args:
        count: Number of messages
        role: Message role
    
    Returns:
        List of messages
    """
    return [
        {
            "role": role,
            "content": f"Test message {i+1}"
        }
        for i in range(count)
    ]


def create_test_events(
    count: int = 5,
    event_type: str = "test"
) -> list:
    """
    Create test events.
    
    Args:
        count: Number of events
        event_type: Event type
    
    Returns:
        List of events
    """
    return [
        {
            "event_type": event_type,
            "timestamp": datetime.utcnow(),
            "data": {"index": i}
        }
        for i in range(count)
    ]


def create_test_memory_data(
    key_count: int = 10
) -> Dict[str, Any]:
    """
    Create test memory data.
    
    Args:
        key_count: Number of keys
    
    Returns:
        Dictionary of test data
    """
    return {
        f"key_{i}": {
            "value": f"value_{i}",
            "metadata": {"index": i}
        }
        for i in range(key_count)
    }

