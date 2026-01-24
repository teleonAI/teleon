"""
Job Triggers - Define when jobs should run.

Provides:
- Cron triggers (cron expression)
- Interval triggers (every N seconds/minutes/hours)
- Date triggers (one-time execution)
"""

from typing import Optional
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from croniter import croniter


class Trigger(ABC):
    """
    Abstract base class for job triggers.
    
    Triggers determine when a job should be executed next.
    """
    
    @abstractmethod
    def get_next_fire_time(
        self,
        previous_fire_time: Optional[datetime],
        now: Optional[datetime] = None
    ) -> Optional[datetime]:
        """
        Calculate next fire time.
        
        Args:
            previous_fire_time: Previous execution time
            now: Current time
            
        Returns:
            Next fire time or None if no more executions
        """
        pass


class CronTrigger(Trigger):
    """
    Cron-based trigger.
    
    Uses cron expressions to define schedule.
    
    Example:
        >>> # Every day at 9:00 AM
        >>> trigger = CronTrigger("0 9 * * *")
        >>> 
        >>> # Every hour
        >>> trigger = CronTrigger("0 * * * *")
        >>> 
        >>> # Every Monday at 10:30 AM
        >>> trigger = CronTrigger("30 10 * * MON")
    """
    
    def __init__(self, cron_expression: str, timezone: Optional[str] = None):
        """
        Initialize cron trigger.
        
        Args:
            cron_expression: Cron expression (e.g., "0 9 * * *")
            timezone: Timezone name (e.g., "America/New_York")
        """
        self.cron_expression = cron_expression
        self.timezone = timezone
        
        # Validate cron expression
        try:
            croniter(cron_expression)
        except Exception as e:
            raise ValueError(f"Invalid cron expression: {e}")
    
    def get_next_fire_time(
        self,
        previous_fire_time: Optional[datetime],
        now: Optional[datetime] = None
    ) -> Optional[datetime]:
        """Calculate next fire time based on cron expression."""
        if now is None:
            now = datetime.now(timezone.utc)
        
        base_time = previous_fire_time or now
        
        cron = croniter(self.cron_expression, base_time)
        next_time = cron.get_next(datetime)
        
        return next_time
    
    def __repr__(self) -> str:
        return f"CronTrigger('{self.cron_expression}')"


class IntervalTrigger(Trigger):
    """
    Interval-based trigger.
    
    Executes job at regular intervals.
    
    Example:
        >>> # Every 5 minutes
        >>> trigger = IntervalTrigger(minutes=5)
        >>> 
        >>> # Every 2 hours
        >>> trigger = IntervalTrigger(hours=2)
        >>> 
        >>> # Every 30 seconds
        >>> trigger = IntervalTrigger(seconds=30)
    """
    
    def __init__(
        self,
        weeks: int = 0,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ):
        """
        Initialize interval trigger.
        
        Args:
            weeks: Number of weeks
            days: Number of days
            hours: Number of hours
            minutes: Number of minutes
            seconds: Number of seconds
            start_date: Optional start date
            end_date: Optional end date
        """
        self.interval = timedelta(
            weeks=weeks,
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds
        )
        
        if self.interval.total_seconds() == 0:
            raise ValueError("Interval must be greater than 0")
        
        self.start_date = start_date
        self.end_date = end_date
    
    def get_next_fire_time(
        self,
        previous_fire_time: Optional[datetime],
        now: Optional[datetime] = None
    ) -> Optional[datetime]:
        """Calculate next fire time based on interval."""
        if now is None:
            now = datetime.now(timezone.utc)
        
        # Check if we're past end date
        if self.end_date and now > self.end_date:
            return None
        
        # First execution
        if previous_fire_time is None:
            next_time = self.start_date or now
        else:
            next_time = previous_fire_time + self.interval
        
        # Ensure we're past start date
        if self.start_date and next_time < self.start_date:
            next_time = self.start_date
        
        # Ensure we're not past end date
        if self.end_date and next_time > self.end_date:
            return None
        
        return next_time
    
    def __repr__(self) -> str:
        return f"IntervalTrigger(interval={self.interval})"


class DateTrigger(Trigger):
    """
    Date-based trigger.
    
    Executes job once at a specific date/time.
    
    Example:
        >>> # One-time execution at specific time
        >>> trigger = DateTrigger(datetime(2025, 12, 31, 23, 59, 59))
    """
    
    def __init__(self, run_date: datetime):
        """
        Initialize date trigger.
        
        Args:
            run_date: Date and time to run the job
        """
        self.run_date = run_date
        self._executed = False
    
    def get_next_fire_time(
        self,
        previous_fire_time: Optional[datetime],
        now: Optional[datetime] = None
    ) -> Optional[datetime]:
        """Calculate next fire time (one-time execution)."""
        if now is None:
            now = datetime.now(timezone.utc)
        
        # Already executed or run date in the past
        if self._executed or previous_fire_time is not None:
            return None
        
        # Run date is in the future
        if self.run_date > now:
            return self.run_date
        
        return None
    
    def mark_executed(self):
        """Mark this trigger as executed."""
        self._executed = True
    
    def __repr__(self) -> str:
        return f"DateTrigger(run_date={self.run_date})"

