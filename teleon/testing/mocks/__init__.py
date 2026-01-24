"""
Mock System - Production-grade mocks for testing.

Features:
- Mock LLM providers
- Mock tools
- Mock memory systems
- Mock message bus
- Call tracking and verification
"""

from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timezone
from pydantic import BaseModel

try:
    from teleon.llm.types import LLMRequest, LLMResponse
except ImportError:
    # Fallback if types not available
    LLMRequest = Dict[str, Any]
    LLMResponse = Dict[str, Any]


class MockLLM:
    """
    Mock LLM provider for testing.
    
    Features:
    - Predefined responses
    - Call tracking
    - Latency simulation
    - Error injection
    """
    
    def __init__(self):
        """Initialize mock LLM."""
        self.calls: List[Dict[str, Any]] = []
        self.responses: List[str] = []
        self.current_response_index = 0
        
        # Configuration
        self.latency_ms: float = 0
        self.error_rate: float = 0
        self.should_fail: bool = False
        self.fail_message: str = "Mock LLM error"
    
    def add_response(self, response: str):
        """Add a predefined response."""
        self.responses.append(response)
    
    def add_responses(self, responses: List[str]):
        """Add multiple responses."""
        self.responses.extend(responses)
    
    def set_latency(self, ms: float):
        """Set response latency."""
        self.latency_ms = ms
    
    def set_error_rate(self, rate: float):
        """Set error rate (0.0 to 1.0)."""
        self.error_rate = rate
    
    def fail_next(self, message: str = "Mock LLM error"):
        """Make next call fail."""
        self.should_fail = True
        self.fail_message = message
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response."""
        import asyncio
        import random
        
        # Track call
        call = {
            "timestamp": datetime.now(timezone.utc),
            "request": request,
            "model": request.model,
            "messages": request.messages
        }
        self.calls.append(call)
        
        # Simulate latency
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)
        
        # Check for forced failure
        if self.should_fail:
            self.should_fail = False
            raise Exception(self.fail_message)
        
        # Check error rate
        if self.error_rate > 0 and random.random() < self.error_rate:
            raise Exception("Random mock error")
        
        # Get response
        if self.responses:
            response_text = self.responses[self.current_response_index % len(self.responses)]
            self.current_response_index += 1
        else:
            response_text = "Mock LLM response"
        
        # Create response
        return LLMResponse(
            content=response_text,
            model=request.model,
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            },
            cost=0.001
        )
    
    def get_call_count(self) -> int:
        """Get number of calls made."""
        return len(self.calls)
    
    def get_last_call(self) -> Optional[Dict[str, Any]]:
        """Get last call."""
        return self.calls[-1] if self.calls else None
    
    def assert_called(self):
        """Assert LLM was called."""
        if not self.calls:
            raise AssertionError("Mock LLM was not called")
    
    def assert_called_times(self, expected: int):
        """Assert LLM was called specific number of times."""
        actual = len(self.calls)
        if actual != expected:
            raise AssertionError(
                f"Expected {expected} calls, got {actual}"
            )
    
    def assert_called_with_model(self, model: str):
        """Assert LLM was called with specific model."""
        if not self.calls:
            raise AssertionError("Mock LLM was not called")
        
        last_call = self.calls[-1]
        if last_call["model"] != model:
            raise AssertionError(
                f"Expected model '{model}', got '{last_call['model']}'"
            )
    
    def reset(self):
        """Reset mock state."""
        self.calls.clear()
        self.responses.clear()
        self.current_response_index = 0
        self.latency_ms = 0
        self.error_rate = 0
        self.should_fail = False


class MockTool:
    """
    Mock tool for testing.
    
    Features:
    - Predefined results
    - Call tracking
    - Error injection
    - Execution time simulation
    """
    
    def __init__(self, name: str = "mock_tool"):
        """Initialize mock tool."""
        self.name = name
        self.calls: List[Dict[str, Any]] = []
        self.results: List[Any] = []
        self.current_result_index = 0
        
        # Configuration
        self.execution_time_ms: float = 0
        self.should_fail: bool = False
        self.fail_message: str = "Mock tool error"
    
    def add_result(self, result: Any):
        """Add a predefined result."""
        self.results.append(result)
    
    def add_results(self, results: List[Any]):
        """Add multiple results."""
        self.results.extend(results)
    
    def set_execution_time(self, ms: float):
        """Set execution time."""
        self.execution_time_ms = ms
    
    def fail_next(self, message: str = "Mock tool error"):
        """Make next execution fail."""
        self.should_fail = True
        self.fail_message = message
    
    async def execute(self, *args, **kwargs) -> Any:
        """Execute tool."""
        import asyncio
        
        # Track call
        call = {
            "timestamp": datetime.now(timezone.utc),
            "args": args,
            "kwargs": kwargs
        }
        self.calls.append(call)
        
        # Simulate execution time
        if self.execution_time_ms > 0:
            await asyncio.sleep(self.execution_time_ms / 1000)
        
        # Check for failure
        if self.should_fail:
            self.should_fail = False
            raise Exception(self.fail_message)
        
        # Get result
        if self.results:
            result = self.results[self.current_result_index % len(self.results)]
            self.current_result_index += 1
            return result
        else:
            return {"success": True, "data": "Mock result"}
    
    def get_call_count(self) -> int:
        """Get number of calls."""
        return len(self.calls)
    
    def assert_called(self):
        """Assert tool was called."""
        if not self.calls:
            raise AssertionError(f"Mock tool '{self.name}' was not called")
    
    def assert_called_times(self, expected: int):
        """Assert tool was called specific number of times."""
        actual = len(self.calls)
        if actual != expected:
            raise AssertionError(
                f"Expected {expected} calls, got {actual}"
            )
    
    def reset(self):
        """Reset mock state."""
        self.calls.clear()
        self.results.clear()
        self.current_result_index = 0
        self.execution_time_ms = 0
        self.should_fail = False


class MockMemory:
    """
    Mock memory system for testing.
    
    Features:
    - In-memory storage
    - Call tracking
    - Query simulation
    """
    
    def __init__(self):
        """Initialize mock memory."""
        self.storage: Dict[str, Any] = {}
        self.calls: List[Dict[str, Any]] = []
    
    async def set(self, key: str, value: Any):
        """Store value."""
        self.calls.append({
            "operation": "set",
            "key": key,
            "value": value,
            "timestamp": datetime.now(timezone.utc)
        })
        self.storage[key] = value
    
    async def get(self, key: str) -> Optional[Any]:
        """Retrieve value."""
        self.calls.append({
            "operation": "get",
            "key": key,
            "timestamp": datetime.now(timezone.utc)
        })
        return self.storage.get(key)
    
    async def delete(self, key: str):
        """Delete value."""
        self.calls.append({
            "operation": "delete",
            "key": key,
            "timestamp": datetime.now(timezone.utc)
        })
        if key in self.storage:
            del self.storage[key]
    
    async def search(self, query: str, limit: int = 10) -> List[Any]:
        """Search memory."""
        self.calls.append({
            "operation": "search",
            "query": query,
            "limit": limit,
            "timestamp": datetime.now(timezone.utc)
        })
        # Simple search - return all values
        return list(self.storage.values())[:limit]
    
    def get_call_count(self) -> int:
        """Get number of calls."""
        return len(self.calls)
    
    def get_operations(self, operation: str) -> List[Dict[str, Any]]:
        """Get calls for specific operation."""
        return [c for c in self.calls if c["operation"] == operation]
    
    def assert_stored(self, key: str):
        """Assert key is stored."""
        if key not in self.storage:
            raise AssertionError(f"Key '{key}' not found in mock memory")
    
    def assert_not_stored(self, key: str):
        """Assert key is not stored."""
        if key in self.storage:
            raise AssertionError(f"Key '{key}' found in mock memory")
    
    def reset(self):
        """Reset mock state."""
        self.storage.clear()
        self.calls.clear()


class MockMessageBus:
    """
    Mock message bus for testing multi-agent communication.
    
    Features:
    - Message tracking
    - Subscription simulation
    - Delivery verification
    """
    
    def __init__(self):
        """Initialize mock message bus."""
        self.messages: List[Dict[str, Any]] = []
        self.subscriptions: Dict[str, List[Callable]] = {}
    
    async def publish(
        self,
        topic: str,
        message: Any,
        priority: str = "NORMAL"
    ):
        """Publish message."""
        msg = {
            "topic": topic,
            "message": message,
            "priority": priority,
            "timestamp": datetime.now(timezone.utc)
        }
        self.messages.append(msg)
        
        # Notify subscribers
        if topic in self.subscriptions:
            for handler in self.subscriptions[topic]:
                await handler(message)
    
    async def subscribe(self, topic: str, handler: Callable):
        """Subscribe to topic."""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        self.subscriptions[topic].append(handler)
    
    def get_messages(self, topic: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get messages."""
        if topic:
            return [m for m in self.messages if m["topic"] == topic]
        return self.messages
    
    def get_message_count(self, topic: Optional[str] = None) -> int:
        """Get message count."""
        return len(self.get_messages(topic))
    
    def assert_message_sent(self, topic: str):
        """Assert message was sent to topic."""
        if not any(m["topic"] == topic for m in self.messages):
            raise AssertionError(f"No messages sent to topic '{topic}'")
    
    def reset(self):
        """Reset mock state."""
        self.messages.clear()
        self.subscriptions.clear()


__all__ = [
    "MockLLM",
    "MockTool",
    "MockMemory",
    "MockMessageBus",
]

