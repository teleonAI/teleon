"""
Testing Framework - Core test classes and decorators.

Features:
- AgentTestCase for agent testing
- ToolTestCase for tool testing
- IntegrationTestCase for end-to-end tests
- Test decorators
- Async test support
"""

from typing import Any, Dict, Optional, Callable, List
import asyncio
import unittest
from datetime import datetime, timezone
from functools import wraps

from teleon.core import StructuredLogger, LogLevel
from teleon.testing.mocks import MockLLM, MockTool, MockMemory


class AgentTestCase(unittest.TestCase):
    """
    Base test case for agent testing.
    
    Features:
    - Automatic setup/teardown
    - Mock LLM, tools, memory
    - Assertion helpers
    - Execution tracking
    """
    
    def setUp(self):
        """Set up test case."""
        super().setUp()
        
        # Initialize mocks
        self.mock_llm = MockLLM()
        self.mock_memory = MockMemory()
        self.mock_tools: Dict[str, MockTool] = {}
        
        # Track executions
        self.executions: List[Dict[str, Any]] = []
        
        # Logger
        self.logger = StructuredLogger("test", LogLevel.DEBUG)
    
    def tearDown(self):
        """Clean up after test."""
        super().tearDown()
        
        # Reset mocks
        self.mock_llm.reset()
        self.mock_memory.reset()
        for tool in self.mock_tools.values():
            tool.reset()
    
    def add_mock_tool(self, name: str, tool: MockTool):
        """Add a mock tool."""
        self.mock_tools[name] = tool
    
    async def execute_agent(
        self,
        agent_fn: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute an agent and track execution.
        
        Args:
            agent_fn: Agent function
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Agent execution result
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Execute agent
            if asyncio.iscoroutinefunction(agent_fn):
                result = await agent_fn(*args, **kwargs)
            else:
                result = agent_fn(*args, **kwargs)
            
            # Track execution
            execution = {
                "success": True,
                "result": result,
                "args": args,
                "kwargs": kwargs,
                "start_time": start_time,
                "end_time": datetime.now(timezone.utc),
                "duration": (datetime.now(timezone.utc) - start_time).total_seconds()
            }
            self.executions.append(execution)
            
            return result
        
        except Exception as e:
            # Track failed execution
            execution = {
                "success": False,
                "error": str(e),
                "exception": e,
                "args": args,
                "kwargs": kwargs,
                "start_time": start_time,
                "end_time": datetime.now(timezone.utc),
                "duration": (datetime.now(timezone.utc) - start_time).total_seconds()
            }
            self.executions.append(execution)
            raise
    
    def assert_execution_count(self, expected: int):
        """Assert number of executions."""
        actual = len(self.executions)
        self.assertEqual(
            actual,
            expected,
            f"Expected {expected} executions, got {actual}"
        )
    
    def assert_execution_success(self, index: int = -1):
        """Assert execution succeeded."""
        execution = self.executions[index]
        self.assertTrue(
            execution["success"],
            f"Execution failed: {execution.get('error')}"
        )
    
    def assert_execution_failed(self, index: int = -1):
        """Assert execution failed."""
        execution = self.executions[index]
        self.assertFalse(
            execution["success"],
            "Expected execution to fail, but it succeeded"
        )
    
    def get_last_execution(self) -> Dict[str, Any]:
        """Get last execution."""
        if not self.executions:
            raise AssertionError("No executions recorded")
        return self.executions[-1]


class ToolTestCase(unittest.TestCase):
    """
    Base test case for tool testing.
    
    Features:
    - Tool execution helpers
    - Input/output validation
    - Error testing
    - Performance tracking
    """
    
    def setUp(self):
        """Set up test case."""
        super().setUp()
        
        # Track executions
        self.executions: List[Dict[str, Any]] = []
        
        # Logger
        self.logger = StructuredLogger("test", LogLevel.DEBUG)
    
    async def execute_tool(
        self,
        tool: Any,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a tool and track execution.
        
        Args:
            tool: Tool instance
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Tool execution result
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Execute tool
            if asyncio.iscoroutinefunction(tool.execute):
                result = await tool.execute(*args, **kwargs)
            else:
                result = tool.execute(*args, **kwargs)
            
            # Track execution
            execution = {
                "success": True,
                "result": result,
                "args": args,
                "kwargs": kwargs,
                "start_time": start_time,
                "end_time": datetime.now(timezone.utc),
                "duration": (datetime.now(timezone.utc) - start_time).total_seconds(),
                "tool_name": getattr(tool, "name", "unknown")
            }
            self.executions.append(execution)
            
            return result
        
        except Exception as e:
            # Track failed execution
            execution = {
                "success": False,
                "error": str(e),
                "exception": e,
                "args": args,
                "kwargs": kwargs,
                "start_time": start_time,
                "end_time": datetime.now(timezone.utc),
                "duration": (datetime.now(timezone.utc) - start_time).total_seconds(),
                "tool_name": getattr(tool, "name", "unknown")
            }
            self.executions.append(execution)
            raise
    
    def assert_tool_execution_time(self, max_seconds: float, index: int = -1):
        """Assert tool execution time is below threshold."""
        execution = self.executions[index]
        duration = execution["duration"]
        self.assertLess(
            duration,
            max_seconds,
            f"Tool execution took {duration}s, expected < {max_seconds}s"
        )


class IntegrationTestCase(unittest.TestCase):
    """
    Base test case for integration testing.
    
    Features:
    - End-to-end testing
    - Multi-component integration
    - Real dependency testing
    - Performance profiling
    """
    
    def setUp(self):
        """Set up test case."""
        super().setUp()
        
        # Track test metadata
        self.test_start = datetime.now(timezone.utc)
        self.logger = StructuredLogger("integration_test", LogLevel.DEBUG)
    
    def tearDown(self):
        """Clean up after test."""
        super().tearDown()
        
        # Log test duration
        duration = (datetime.now(timezone.utc) - self.test_start).total_seconds()
        self.logger.info(f"Integration test completed in {duration}s")


def test_agent(
    name: Optional[str] = None,
    timeout: Optional[float] = None
):
    """
    Decorator for agent tests.
    
    Args:
        name: Test name
        timeout: Execution timeout
    
    Returns:
        Decorated test function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            test_name = name or func.__name__
            
            try:
                if timeout:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout
                    )
                else:
                    result = await func(*args, **kwargs)
                
                return result
            
            except asyncio.TimeoutError:
                raise AssertionError(
                    f"Test '{test_name}' exceeded timeout of {timeout}s"
                )
        
        wrapper.__test_name__ = name
        wrapper.__test_timeout__ = timeout
        return wrapper
    
    return decorator


def test_tool(
    name: Optional[str] = None,
    max_duration: Optional[float] = None
):
    """
    Decorator for tool tests.
    
    Args:
        name: Test name
        max_duration: Maximum allowed duration
    
    Returns:
        Decorated test function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            test_name = name or func.__name__
            start_time = datetime.now(timezone.utc)
            
            try:
                result = await func(*args, **kwargs)
                
                # Check duration
                if max_duration:
                    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                    if duration > max_duration:
                        raise AssertionError(
                            f"Tool test '{test_name}' took {duration}s, "
                            f"expected < {max_duration}s"
                        )
                
                return result
            
            except Exception as e:
                raise
        
        wrapper.__test_name__ = name
        wrapper.__max_duration__ = max_duration
        return wrapper
    
    return decorator


class AsyncTestRunner:
    """Helper for running async tests."""
    
    @staticmethod
    def run(coro):
        """Run async test."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    
    @staticmethod
    def run_multiple(*coros):
        """Run multiple async tests."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(asyncio.gather(*coros))

