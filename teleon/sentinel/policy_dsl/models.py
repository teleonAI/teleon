"""
Policy DSL Data Models.

Pydantic models for declarative policy definitions.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RuleType(str, Enum):
    TEXT_MATCH = "text_match"
    REGEX = "regex"
    LENGTH = "length"
    CONDITION = "condition"
    TOOL_ALLOWLIST = "tool_allowlist"
    TOOL_BLOCKLIST = "tool_blocklist"
    TOOL_ARGUMENT = "tool_argument"
    RATE_LIMIT = "rate_limit"


class PolicyRule(BaseModel):
    type: RuleType
    operator: Optional[str] = None
    field: Optional[str] = None
    value: Optional[Any] = None
    values: Optional[List[Any]] = None
    pattern: Optional[str] = None
    case_sensitive: bool = False
    allowed_tools: Optional[List[str]] = None
    blocked_tools: Optional[List[str]] = None
    tool: Optional[str] = None
    argument: Optional[str] = None
    max_count: Optional[int] = None
    window_seconds: Optional[int] = None
    scope: Optional[str] = "agent"


class PolicyDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    enabled: bool = True
    severity: str = Field(default="medium")  # low | medium | high | critical
    action: str = Field(default="flag")  # block | flag | redact | escalate
    targets: List[str] = Field(default_factory=lambda: ["input", "output"])
    rules: List[PolicyRule] = Field(default_factory=list)
    match: str = Field(default="any")  # any (OR) | all (AND)
    agent_ids: Optional[List[str]] = None  # None = all agents


class EvaluationContext(BaseModel):
    text: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)
    agent_name: str = ""
    validation_type: str = "input"  # input | output | tool_call
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
