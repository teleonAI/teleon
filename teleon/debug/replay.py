"""
Execution Replay - Record and replay agent executions.

Features:
- Record all execution steps
- Replay executions
- Step-by-step replay
- State inspection at each step
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict, field_serializer
import json
import asyncio

from teleon.core import StructuredLogger, LogLevel


class ExecutionStep(BaseModel):
    """Single execution step."""

    step_number: int = Field(..., description="Step number")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    step_type: str = Field(..., description="Step type")

    # Data
    input_data: Optional[Dict[str, Any]] = Field(None, description="Input data")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Output data")
    state: Optional[Dict[str, Any]] = Field(None, description="State snapshot")

    # Metadata
    duration_ms: float = Field(0.0, description="Step duration")
    success: bool = Field(True, description="Step success")
    error: Optional[str] = Field(None, description="Error message")

    model_config = ConfigDict()

    @field_serializer('timestamp')
    def serialize_datetime(self, value: datetime) -> str:
        return value.isoformat() if value else None


class ExecutionRecording(BaseModel):
    """Complete execution recording."""

    execution_id: str = Field(..., description="Execution ID")
    agent_name: str = Field(..., description="Agent name")
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = Field(None)

    steps: List[ExecutionStep] = Field(default_factory=list)

    # Metadata
    total_duration_ms: float = Field(0.0, description="Total duration")
    success: bool = Field(True, description="Overall success")
    final_result: Optional[Any] = Field(None, description="Final result")

    model_config = ConfigDict()

    @field_serializer('started_at', 'completed_at')
    def serialize_datetime(self, value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None


class ExecutionRecorder:
    """
    Execution recorder.
    
    Features:
    - Record execution steps
    - Capture state snapshots
    - Save recordings to file
    """
    
    def __init__(self, execution_id: str, agent_name: str):
        """
        Initialize recorder.
        
        Args:
            execution_id: Execution ID
            agent_name: Agent name
        """
        self.recording = ExecutionRecording(
            execution_id=execution_id,
            agent_name=agent_name
        )
        self.current_step = 0
        self.logger = StructuredLogger("recorder", LogLevel.DEBUG)
    
    def record_step(
        self,
        step_type: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        state: Optional[Dict[str, Any]] = None,
        duration_ms: float = 0.0,
        success: bool = True,
        error: Optional[str] = None
    ):
        """
        Record an execution step.
        
        Args:
            step_type: Type of step
            input_data: Input data
            output_data: Output data
            state: State snapshot
            duration_ms: Step duration
            success: Success flag
            error: Error message
        """
        step = ExecutionStep(
            step_number=self.current_step,
            step_type=step_type,
            input_data=input_data,
            output_data=output_data,
            state=state,
            duration_ms=duration_ms,
            success=success,
            error=error
        )
        
        self.recording.steps.append(step)
        self.current_step += 1
        
        self.logger.debug(
            "Step recorded",
            step_number=self.current_step - 1,
            step_type=step_type,
            success=success
        )
    
    def complete(
        self,
        success: bool = True,
        final_result: Optional[Any] = None
    ):
        """
        Mark recording as complete.
        
        Args:
            success: Overall success
            final_result: Final result
        """
        self.recording.completed_at = datetime.now(timezone.utc)
        self.recording.success = success
        self.recording.final_result = final_result
        
        # Calculate total duration
        if self.recording.steps:
            self.recording.total_duration_ms = sum(
                step.duration_ms for step in self.recording.steps
            )
        
        self.logger.info(
            "Recording completed",
            execution_id=self.recording.execution_id,
            steps=len(self.recording.steps),
            success=success
        )
    
    def save(self, filepath: str):
        """
        Save recording to file.
        
        Args:
            filepath: File path
        """
        with open(filepath, 'w') as f:
            json.dump(self.recording.dict(), f, indent=2, default=str)
        
        self.logger.info(f"Recording saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> ExecutionRecording:
        """
        Load recording from file.
        
        Args:
            filepath: File path
        
        Returns:
            ExecutionRecording instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return ExecutionRecording(**data)


class ExecutionReplayer:
    """
    Execution replayer.
    
    Features:
    - Replay recorded executions
    - Step-by-step replay
    - State inspection
    - Breakpoints
    """
    
    def __init__(self, recording: ExecutionRecording):
        """
        Initialize replayer.
        
        Args:
            recording: Execution recording
        """
        self.recording = recording
        self.current_step_index = 0
        self.logger = StructuredLogger("replayer", LogLevel.INFO)
    
    def get_step(self, index: int) -> Optional[ExecutionStep]:
        """Get step by index."""
        if 0 <= index < len(self.recording.steps):
            return self.recording.steps[index]
        return None
    
    def next_step(self) -> Optional[ExecutionStep]:
        """Get next step."""
        step = self.get_step(self.current_step_index)
        if step:
            self.current_step_index += 1
        return step
    
    def previous_step(self) -> Optional[ExecutionStep]:
        """Get previous step."""
        if self.current_step_index > 0:
            self.current_step_index -= 1
            return self.get_step(self.current_step_index)
        return None
    
    def reset(self):
        """Reset to beginning."""
        self.current_step_index = 0
    
    def goto_step(self, index: int):
        """Go to specific step."""
        if 0 <= index < len(self.recording.steps):
            self.current_step_index = index
    
    async def replay_all(self, delay_ms: float = 0):
        """
        Replay all steps.
        
        Args:
            delay_ms: Delay between steps
        """
        self.reset()
        
        self.logger.info(
            "Starting replay",
            execution_id=self.recording.execution_id,
            total_steps=len(self.recording.steps)
        )
        
        while True:
            step = self.next_step()
            if not step:
                break
            
            self.logger.info(
                f"Step {step.step_number}: {step.step_type}",
                success=step.success,
                duration_ms=step.duration_ms
            )
            
            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000)
        
        self.logger.info("Replay completed")
    
    def print_summary(self):
        """Print execution summary."""
        print("\n" + "="*80)
        print("EXECUTION REPLAY SUMMARY")
        print("="*80)
        print(f"\nExecution ID:   {self.recording.execution_id}")
        print(f"Agent:          {self.recording.agent_name}")
        print(f"Started:        {self.recording.started_at}")
        print(f"Completed:      {self.recording.completed_at}")
        print(f"Success:        {'✓' if self.recording.success else '✗'}")
        print(f"Total Steps:    {len(self.recording.steps)}")
        print(f"Duration:       {self.recording.total_duration_ms:.2f}ms")
        
        print(f"\nSteps:")
        for step in self.recording.steps:
            status = "✓" if step.success else "✗"
            print(f"  {status} [{step.step_number}] {step.step_type} ({step.duration_ms:.2f}ms)")
            if step.error:
                print(f"      Error: {step.error}")
        
        print("\n" + "="*80 + "\n")

