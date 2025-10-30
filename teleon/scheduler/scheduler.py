"""
Scheduler - Main scheduling engine.

Provides:
- Job registration and management
- Automatic job execution
- Concurrent execution control
- Job history and monitoring
"""

from typing import Dict, Optional, Callable, Any, List
import asyncio
from datetime import datetime
import uuid

from teleon.scheduler.job import Job, JobExecution, JobStatus
from teleon.scheduler.triggers import Trigger
from teleon.core import StructuredLogger, LogLevel


class Scheduler:
    """
    Job scheduler with cron-like functionality.
    
    Features:
    - Async job execution
    - Concurrent execution limits
    - Automatic retry
    - Job history tracking
    
    Example:
        >>> scheduler = Scheduler()
        >>> 
        >>> # Add a cron job
        >>> scheduler.add_job(
        ...     func=my_function,
        ...     trigger=CronTrigger("0 9 * * *"),  # 9 AM daily
        ...     name="daily_report"
        ... )
        >>> 
        >>> # Start scheduler
        >>> await scheduler.start()
    """
    
    def __init__(self):
        """Initialize scheduler."""
        self.jobs: Dict[str, Job] = {}
        self.logger = StructuredLogger("scheduler", LogLevel.INFO)
        self.running = False
        self.running_jobs: Dict[str, int] = {}  # job_id -> instance_count
    
    def add_job(
        self,
        func: Callable,
        trigger: Trigger,
        name: Optional[str] = None,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        **job_kwargs
    ) -> Job:
        """
        Add a job to the scheduler.
        
        Args:
            func: Function to execute
            trigger: Job trigger (schedule)
            name: Job name
            args: Function arguments
            kwargs: Function keyword arguments
            **job_kwargs: Additional job configuration
            
        Returns:
            Created job
        """
        if kwargs is None:
            kwargs = {}
        
        job = Job(
            name=name or func.__name__,
            func=func,
            args=args,
            kwargs=kwargs,
            trigger=trigger,
            **job_kwargs
        )
        
        # Calculate next run time
        job.next_run_time = trigger.get_next_fire_time(None)
        
        self.jobs[job.id] = job
        
        self.logger.info(
            f"Added job: {job.name}",
            extra={
                "job_id": job.id,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None
            }
        )
        
        return job
    
    def remove_job(self, job_id: str):
        """
        Remove a job from the scheduler.
        
        Args:
            job_id: Job ID
        """
        if job_id in self.jobs:
            job = self.jobs.pop(job_id)
            self.logger.info(f"Removed job: {job.name}", extra={"job_id": job_id})
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get a job by ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job or None
        """
        return self.jobs.get(job_id)
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all jobs.
        
        Returns:
            List of job information
        """
        return [job.to_dict() for job in self.jobs.values()]
    
    async def _execute_job(self, job: Job):
        """
        Execute a job.
        
        Args:
            job: Job to execute
        """
        execution_id = str(uuid.uuid4())
        started_at = datetime.utcnow()
        
        execution = JobExecution(
            execution_id=execution_id,
            job_id=job.id,
            status=JobStatus.RUNNING,
            started_at=started_at
        )
        
        self.logger.info(
            f"Executing job: {job.name}",
            extra={"job_id": job.id, "execution_id": execution_id}
        )
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(job.func):
                result = await job.func(*job.args, **job.kwargs)
            else:
                result = job.func(*job.args, **job.kwargs)
            
            # Job completed successfully
            execution.status = JobStatus.COMPLETED
            execution.result = result
            execution.completed_at = datetime.utcnow()
            execution.duration = (execution.completed_at - started_at).total_seconds()
            
            self.logger.info(
                f"Job completed: {job.name}",
                extra={
                    "job_id": job.id,
                    "execution_id": execution_id,
                    "duration": execution.duration
                }
            )
        
        except Exception as e:
            # Job failed
            execution.status = JobStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.utcnow()
            execution.duration = (execution.completed_at - started_at).total_seconds()
            
            self.logger.error(
                f"Job failed: {job.name} - {e}",
                extra={
                    "job_id": job.id,
                    "execution_id": execution_id,
                    "error": str(e)
                }
            )
        
        finally:
            # Update job
            job.add_execution(execution)
            job.last_run_time = started_at
            
            # Update next run time
            job.next_run_time = job.trigger.get_next_fire_time(started_at)
            
            # Decrement running instance count
            self.running_jobs[job.id] = self.running_jobs.get(job.id, 1) - 1
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                now = datetime.utcnow()
                
                # Find jobs that should run
                for job in list(self.jobs.values()):
                    if not job.enabled:
                        continue
                    
                    if job.next_run_time is None:
                        continue
                    
                    if job.next_run_time > now:
                        continue
                    
                    # Check max instances
                    running_count = self.running_jobs.get(job.id, 0)
                    if running_count >= job.max_instances:
                        self.logger.warning(
                            f"Job max instances reached: {job.name}",
                            extra={"job_id": job.id}
                        )
                        continue
                    
                    # Execute job
                    self.running_jobs[job.id] = running_count + 1
                    asyncio.create_task(self._execute_job(job))
                
                # Sleep for 1 second
                await asyncio.sleep(1)
            
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(1)
    
    async def start(self):
        """Start the scheduler."""
        if self.running:
            self.logger.warning("Scheduler already running")
            return
        
        self.running = True
        self.logger.info("Scheduler started")
        
        await self._scheduler_loop()
    
    async def stop(self):
        """Stop the scheduler."""
        self.running = False
        self.logger.info("Scheduler stopped")


# Global scheduler
_scheduler: Optional[Scheduler] = None


def get_scheduler() -> Scheduler:
    """
    Get the global scheduler instance.
    
    Returns:
        Scheduler instance
    """
    global _scheduler
    
    if _scheduler is None:
        _scheduler = Scheduler()
    
    return _scheduler

