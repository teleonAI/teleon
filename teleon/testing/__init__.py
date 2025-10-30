"""
Testing Framework - Production-grade testing utilities.

This package provides comprehensive testing support:
- Test framework for agents and tools
- Mock LLMs, tools, and memory
- Test fixtures and utilities
- Assertions and matchers
- Load testing capabilities
"""

from teleon.testing.framework import (
    AgentTestCase,
    ToolTestCase,
    IntegrationTestCase,
    test_agent,
    test_tool,
)
from teleon.testing.mocks import (
    MockLLM,
    MockTool,
    MockMemory,
    MockMessageBus,
)
from teleon.testing.fixtures import (
    create_test_agent,
    create_test_context,
    create_test_config,
)
from teleon.testing.assertions import (
    assert_agent_output,
    assert_tool_called,
    assert_memory_stored,
    assert_cost_below,
)
from teleon.testing.load import LoadTester, LoadTestConfig, LoadTestResult

__all__ = [
    # Framework
    "AgentTestCase",
    "ToolTestCase",
    "IntegrationTestCase",
    "test_agent",
    "test_tool",
    
    # Mocks
    "MockLLM",
    "MockTool",
    "MockMemory",
    "MockMessageBus",
    
    # Fixtures
    "create_test_agent",
    "create_test_context",
    "create_test_config",
    
    # Assertions
    "assert_agent_output",
    "assert_tool_called",
    "assert_memory_stored",
    "assert_cost_below",
    
    # Load Testing
    "LoadTester",
    "LoadTestConfig",
    "LoadTestResult",
]

