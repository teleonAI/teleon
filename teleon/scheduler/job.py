"""
Job Definition - Scheduled job representation.

Defines:
- Job metadata and configuration
- Job execution history
- Job status tracking
"""

from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import uuid


class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobExecution:
    """Single job execution record."""
    
    execution_id: str
    """Unique execution ID"""
    
    job_id: str
    """Job ID"""
    
    status: JobStatus
    """Execution status"""
    
    started_at: datetime
    """Start time"""
    
    completed_at: Optional[datetime] = None
    """Completion time"""
    
    result: Optional[Any] = None
    """Execution result"""
    
    error: Optional[str] = None
    """Error message if failed"""
    
    duration: Optional[float] = None
    """Execution duration in seconds"""


@dataclass
class Job:
    """
    Scheduled job definition.
    
    Represents a task to be executed on a schedule.
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique job ID"""
    
    name: str = ""
    """Job name"""
    
    func: Optional[Callable] = None
    """Function to execute"""
    
    args: tuple = ()
    """Function arguments"""
    
    kwargs: Dict[str, Any] = field(default_factory=dict)
    """Function keyword arguments"""
    
    trigger: Optional[Any] = None
    """Job trigger (schedule)"""
    
    enabled: bool = True
    """Whether job is enabled"""
    
    max_instances: int = 1
    """Maximum concurrent instances"""
    
    misfire_grace_time: int = 60
    """Grace time for misfired jobs (seconds)"""
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """Job creation time"""
    
    next_run_time: Optional[datetime] = None
    """Next scheduled run time"""
    
    last_run_time: Optional[datetime] = None
    """Last execution time"""
    
    executions: list[JobExecution] = field(default_factory=list)
    """Execution history"""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""
    
    def __post_init__(self):
        """Initialize job."""
        if not self.name and self.func:
            self.name = self.func.__name__
    
    def add_execution(self, execution: JobExecution):
        """
        Add execution to history.
        
        Args:
            execution: Job execution record
        """
        self.executions.append(execution)
        
        # Keep only last 100 executions
        if len(self.executions) > 100:
            self.executions = self.executions[-100:]
    
    def get_last_execution(self) -> Optional[JobExecution]:
        """
        Get last execution record.
        
        Returns:
            Last execution or None
        """
        return self.executions[-1] if self.executions else None
    
    def get_success_rate(self) -> float:
        """
        Calculate job success rate.
        
        Returns:
            Success rate (0.0 to 1.0)
        """
        if not self.executions:
            return 0.0
        
        completed = sum(
            1 for e in self.executions
            if e.status == JobStatus.COMPLETED
        )
        
        return completed / len(self.executions)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert job to dictionary.
        
        Returns:
            Job data as dict
        """
        return {
            "id": self.id,
            "name": self.name,
            "enabled": self.enabled,
            "next_run_time": self.next_run_time.isoformat() if self.next_run_time else None,
            "last_run_time": self.last_run_time.isoformat() if self.last_run_time else None,
            "executions_count": len(self.executions),
            "success_rate": self.get_success_rate(),
            "metadata": self.metadata
        }

