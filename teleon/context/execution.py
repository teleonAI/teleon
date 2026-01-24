"""Execution context for agent runs."""

from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ExecutionContext:
    """
    Context for a single agent execution.
    
    This contains all information about a specific agent execution:
    - Execution metadata (ID, timestamps)
    - Agent configuration
    - Input/output data
    - Execution metrics (cost, duration)
    - Success/failure status
    """
    
    execution_id: str
    agent_name: str
    config: Dict[str, Any]
    started_at: datetime
    input_args: tuple = field(default_factory=tuple)
    input_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Execution results
    output: Any = None
    success: bool = False
    error: Optional[Exception] = None
    completed_at: Optional[datetime] = None
    
    # Metrics
    duration_ms: Optional[int] = None
    cost: float = 0.0
    tokens_used: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    memory_accesses: int = 0
    delegations: int = 0
    
    # Tracing
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    
    def mark_success(self, result: Any) -> None:
        """Mark execution as successful."""
        self.success = True
        self.output = result
        self.completed_at = datetime.now(timezone.utc)
        self._calculate_duration()
    
    def mark_failure(self, error: Exception) -> None:
        """Mark execution as failed."""
        self.success = False
        self.error = error
        self.completed_at = datetime.now(timezone.utc)
        self._calculate_duration()
    
    def _calculate_duration(self) -> None:
        """Calculate execution duration."""
        if self.completed_at:
            delta = self.completed_at - self.started_at
            self.duration_ms = int(delta.total_seconds() * 1000)
    
    def add_cost(self, cost: float) -> None:
        """Add to total execution cost."""
        self.cost += cost
    
    def increment_llm_calls(self) -> None:
        """Increment LLM call counter."""
        self.llm_calls += 1
    
    def increment_tool_calls(self) -> None:
        """Increment tool call counter."""
        self.tool_calls += 1
    
    def increment_memory_accesses(self) -> None:
        """Increment memory access counter."""
        self.memory_accesses += 1
    
    def increment_delegations(self) -> None:
        """Increment delegation counter."""
        self.delegations += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            'execution_id': self.execution_id,
            'agent_name': self.agent_name,
            'config': self.config,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'success': self.success,
            'error': str(self.error) if self.error else None,
            'duration_ms': self.duration_ms,
            'cost': self.cost,
            'tokens_used': self.tokens_used,
            'llm_calls': self.llm_calls,
            'tool_calls': self.tool_calls,
            'memory_accesses': self.memory_accesses,
            'delegations': self.delegations,
            'trace_id': self.trace_id,
            'span_id': self.span_id,
        }

