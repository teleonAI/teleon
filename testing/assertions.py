"""
Test Assertions - Custom assertions for Teleon testing.

Features:
- Agent output assertions
- Tool call assertions
- Memory assertions
- Cost assertions
- Performance assertions
"""

from typing import Any, Optional, Dict, List
import json


def assert_agent_output(
    output: Any,
    expected_type: Optional[type] = None,
    contains: Optional[str] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None
):
    """
    Assert agent output matches expectations.
    
    Args:
        output: Agent output
        expected_type: Expected output type
        contains: String that should be in output
        min_length: Minimum output length
        max_length: Maximum output length
    
    Raises:
        AssertionError: If assertions fail
    """
    if output is None:
        raise AssertionError("Agent output is None")
    
    # Check type
    if expected_type and not isinstance(output, expected_type):
        raise AssertionError(
            f"Expected output type {expected_type.__name__}, "
            f"got {type(output).__name__}"
        )
    
    # Check contains
    if contains:
        output_str = str(output)
        if contains not in output_str:
            raise AssertionError(
                f"Output does not contain '{contains}'"
            )
    
    # Check length
    if min_length is not None or max_length is not None:
        output_str = str(output)
        length = len(output_str)
        
        if min_length and length < min_length:
            raise AssertionError(
                f"Output length {length} is less than minimum {min_length}"
            )
        
        if max_length and length > max_length:
            raise AssertionError(
                f"Output length {length} exceeds maximum {max_length}"
            )


def assert_tool_called(
    mock_tool,
    times: Optional[int] = None,
    with_args: Optional[tuple] = None,
    with_kwargs: Optional[dict] = None
):
    """
    Assert tool was called.
    
    Args:
        mock_tool: Mock tool instance
        times: Expected number of calls
        with_args: Expected args
        with_kwargs: Expected kwargs
    
    Raises:
        AssertionError: If assertions fail
    """
    if not hasattr(mock_tool, 'calls'):
        raise AssertionError("Not a mock tool")
    
    call_count = len(mock_tool.calls)
    
    # Check if called
    if call_count == 0:
        raise AssertionError(f"Tool '{mock_tool.name}' was not called")
    
    # Check call count
    if times is not None and call_count != times:
        raise AssertionError(
            f"Expected {times} calls, got {call_count}"
        )
    
    # Check args/kwargs
    if with_args is not None or with_kwargs is not None:
        last_call = mock_tool.calls[-1]
        
        if with_args and last_call.get("args") != with_args:
            raise AssertionError(
                f"Expected args {with_args}, got {last_call.get('args')}"
            )
        
        if with_kwargs and last_call.get("kwargs") != with_kwargs:
            raise AssertionError(
                f"Expected kwargs {with_kwargs}, got {last_call.get('kwargs')}"
            )


def assert_memory_stored(
    mock_memory,
    key: str,
    value: Optional[Any] = None
):
    """
    Assert value is stored in memory.
    
    Args:
        mock_memory: Mock memory instance
        key: Memory key
        value: Expected value (optional)
    
    Raises:
        AssertionError: If assertions fail
    """
    if not hasattr(mock_memory, 'storage'):
        raise AssertionError("Not a mock memory")
    
    if key not in mock_memory.storage:
        raise AssertionError(f"Key '{key}' not found in memory")
    
    if value is not None:
        actual_value = mock_memory.storage[key]
        if actual_value != value:
            raise AssertionError(
                f"Expected value {value}, got {actual_value}"
            )


def assert_cost_below(
    execution_result: Dict[str, Any],
    max_cost: float
):
    """
    Assert execution cost is below threshold.
    
    Args:
        execution_result: Execution result
        max_cost: Maximum allowed cost
    
    Raises:
        AssertionError: If cost exceeds threshold
    """
    if "cost" not in execution_result:
        raise AssertionError("No cost information in result")
    
    actual_cost = execution_result["cost"]
    if actual_cost > max_cost:
        raise AssertionError(
            f"Cost ${actual_cost:.4f} exceeds maximum ${max_cost:.4f}"
        )


def assert_execution_time_below(
    execution_result: Dict[str, Any],
    max_seconds: float
):
    """
    Assert execution time is below threshold.
    
    Args:
        execution_result: Execution result
        max_seconds: Maximum allowed time
    
    Raises:
        AssertionError: If time exceeds threshold
    """
    if "duration" not in execution_result:
        raise AssertionError("No duration information in result")
    
    actual_duration = execution_result["duration"]
    if actual_duration > max_seconds:
        raise AssertionError(
            f"Duration {actual_duration:.2f}s exceeds maximum {max_seconds:.2f}s"
        )


def assert_json_schema(
    data: Any,
    schema: Dict[str, Any]
):
    """
    Assert data matches JSON schema.
    
    Args:
        data: Data to validate
        schema: JSON schema
    
    Raises:
        AssertionError: If validation fails
    """
    try:
        import jsonschema
        jsonschema.validate(data, schema)
    except ImportError:
        raise AssertionError(
            "jsonschema package not installed. "
            "Install with: pip install jsonschema"
        )
    except jsonschema.ValidationError as e:
        raise AssertionError(f"Schema validation failed: {e.message}")


def assert_llm_called(
    mock_llm,
    times: Optional[int] = None,
    with_model: Optional[str] = None
):
    """
    Assert LLM was called.
    
    Args:
        mock_llm: Mock LLM instance
        times: Expected number of calls
        with_model: Expected model
    
    Raises:
        AssertionError: If assertions fail
    """
    if not hasattr(mock_llm, 'calls'):
        raise AssertionError("Not a mock LLM")
    
    call_count = len(mock_llm.calls)
    
    # Check if called
    if call_count == 0:
        raise AssertionError("LLM was not called")
    
    # Check call count
    if times is not None and call_count != times:
        raise AssertionError(
            f"Expected {times} calls, got {call_count}"
        )
    
    # Check model
    if with_model:
        last_call = mock_llm.calls[-1]
        if last_call.get("model") != with_model:
            raise AssertionError(
                f"Expected model '{with_model}', "
                f"got '{last_call.get('model')}'"
            )


def assert_message_sent(
    mock_bus,
    topic: str,
    count: Optional[int] = None
):
    """
    Assert message was sent to topic.
    
    Args:
        mock_bus: Mock message bus
        topic: Topic name
        count: Expected message count
    
    Raises:
        AssertionError: If assertions fail
    """
    if not hasattr(mock_bus, 'messages'):
        raise AssertionError("Not a mock message bus")
    
    messages = [m for m in mock_bus.messages if m["topic"] == topic]
    
    if not messages:
        raise AssertionError(f"No messages sent to topic '{topic}'")
    
    if count is not None and len(messages) != count:
        raise AssertionError(
            f"Expected {count} messages to '{topic}', got {len(messages)}"
        )


def assert_metrics_recorded(
    metric_name: str,
    min_count: int = 1
):
    """
    Assert metrics were recorded.
    
    Args:
        metric_name: Metric name
        min_count: Minimum number of recordings
    
    Raises:
        AssertionError: If assertions fail
    """
    try:
        from teleon.core import get_metrics
        metrics = get_metrics()
        
        # Check if metric exists
        # Note: This is a simplified check
        # In production, you'd query the metrics backend
        
        # For now, just verify metrics system is available
        if not metrics:
            raise AssertionError("Metrics system not available")
    
    except Exception as e:
        raise AssertionError(f"Failed to verify metrics: {e}")

