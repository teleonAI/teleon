"""Tool schema conversion for provider-agnostic tool calling.

Converts BaseTool instances to provider-specific formats (OpenAI, Anthropic)
and provides prompt-based fallback for providers without native tool support.
"""

import json
import re
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from teleon.llm.types import ToolCallRequest
from teleon.tools.base import BaseTool


def tools_to_openai_format(tools: list) -> List[Dict[str, Any]]:
    """Convert BaseTool instances to OpenAI tools format.

    Args:
        tools: List of BaseTool instances

    Returns:
        List of tool definitions in OpenAI function-calling format
    """
    openai_tools = []
    for tool in tools:
        if not isinstance(tool, BaseTool):
            continue
        schema = tool.get_schema()
        openai_tools.append({
            "type": "function",
            "function": {
                "name": schema.name,
                "description": schema.description,
                "parameters": schema.parameters,
            },
        })
    return openai_tools


def tools_to_anthropic_format(tools: list) -> List[Dict[str, Any]]:
    """Convert BaseTool instances to Anthropic tools format.

    Args:
        tools: List of BaseTool instances

    Returns:
        List of tool definitions in Anthropic format
    """
    anthropic_tools = []
    for tool in tools:
        if not isinstance(tool, BaseTool):
            continue
        schema = tool.get_schema()
        anthropic_tools.append({
            "name": schema.name,
            "description": schema.description,
            "input_schema": schema.parameters,
        })
    return anthropic_tools


def tools_to_prompt_description(tools: list) -> str:
    """Convert BaseTool instances to a text description for prompt injection.

    Used as a fallback for providers that don't support native tool calling.

    Args:
        tools: List of BaseTool instances

    Returns:
        A string describing available tools and the expected JSON response format
    """
    lines = ["Available tools:"]
    for tool in tools:
        if not isinstance(tool, BaseTool):
            continue
        schema = tool.get_schema()
        params = schema.parameters.get("properties", {})
        required = schema.parameters.get("required", [])
        param_parts = []
        for pname, pinfo in params.items():
            ptype = pinfo.get("type", "any")
            req = " (required)" if pname in required else ""
            param_parts.append(f"{pname}: {ptype}{req}")
        param_str = ", ".join(param_parts) if param_parts else "none"
        lines.append(f"- {schema.name}: {schema.description}. Parameters: {param_str}")

    lines.append("")
    lines.append(
        'When you want to call a tool, respond ONLY with a JSON object in this exact format '
        '(no other text before or after):'
    )
    lines.append('{"tool_calls": [{"name": "tool_name", "arguments": {"param": "value"}}]}')
    lines.append("")
    lines.append("You may call multiple tools at once by including multiple items in the array.")
    lines.append("After receiving tool results, provide your final response as plain text.")
    return "\n".join(lines)


def build_tool_map(tools: list) -> Dict[str, BaseTool]:
    """Build a name-to-tool lookup for execution.

    Args:
        tools: List of BaseTool instances

    Returns:
        Dict mapping tool name to BaseTool instance
    """
    tool_map: Dict[str, BaseTool] = {}
    for tool in tools:
        if isinstance(tool, BaseTool):
            tool_map[tool.name] = tool
    return tool_map


def parse_tool_calls_from_text(text: str) -> List[ToolCallRequest]:
    """Parse tool calls from LLM text response (prompt-based fallback).

    Handles the messy reality of LLM output: the model might wrap the JSON
    in explanation text, markdown code fences, or add commentary around it.

    Strategies (tried in order):
    1. Entire response is valid JSON.
    2. JSON inside a ```json ... ``` or ``` ... ``` code block.
    3. Brace-balanced extraction: find the outermost ``{`` that contains
       ``"tool_calls"`` and walk forward counting braces to find its matching
       ``}``.  This handles nested objects (e.g. arguments with dicts)
       regardless of surrounding prose.

    Args:
        text: Raw text from LLM response

    Returns:
        List of parsed ToolCallRequest objects, empty if none found
    """
    if not text or not text.strip():
        return []

    stripped = text.strip()

    # Strategy 1: The entire response is JSON
    parsed = _try_parse_tool_calls_json(stripped)
    if parsed:
        return parsed

    # Strategy 2: JSON inside a markdown code block
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block_match:
        parsed = _try_parse_tool_calls_json(code_block_match.group(1).strip())
        if parsed:
            return parsed

    # Strategy 3: Brace-balanced extraction around "tool_calls"
    candidate = _extract_brace_balanced_json(stripped, "tool_calls")
    if candidate:
        parsed = _try_parse_tool_calls_json(candidate)
        if parsed:
            return parsed

    return []


def _extract_brace_balanced_json(text: str, marker: str) -> Optional[str]:
    """Find the outermost brace-balanced ``{ ... }`` containing *marker*.

    Walks backward from the marker to find the opening ``{``, then walks
    forward counting brace depth to locate the matching closing ``}``.
    This correctly handles nested objects/arrays and surrounding prose.
    """
    idx = text.find(f'"{marker}"')
    if idx == -1:
        return None

    # Walk backward to find the opening brace
    start = None
    for i in range(idx - 1, -1, -1):
        if text[i] == "{":
            start = i
            break

    if start is None:
        return None

    # Walk forward from start, counting braces
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            if in_string:
                escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def _try_parse_tool_calls_json(text: str) -> Optional[List[ToolCallRequest]]:
    """Attempt to parse a JSON string into ToolCallRequest list."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict):
        return None

    raw_calls = data.get("tool_calls")
    if not isinstance(raw_calls, list):
        return None

    results = []
    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        if not name:
            continue
        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        results.append(
            ToolCallRequest(
                id=f"tc_{uuid.uuid4().hex[:12]}",
                name=name,
                arguments=arguments,
            )
        )
    return results if results else None
