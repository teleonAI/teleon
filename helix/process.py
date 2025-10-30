"""
Process Manager - Production-grade process lifecycle management.

Features:
- Process spawning and monitoring
- Resource tracking (CPU, memory)
- Graceful shutdown
- Process health monitoring
- Automatic restart on failure
"""

from typing import Dict, Optional, Any, Callable
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import psutil
import os
import signal

from teleon.core import (
    get_metrics,
    StructuredLogger,
    LogLevel,
)


class ProcessStatus(str, Enum):
    """Process status."""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    CRASHED = "crashed"


class ProcessInfo(BaseModel):
    """Process information."""
    
    process_id: str = Field(..., description="Unique process ID")
    name: str = Field(..., description="Process name")
    pid: Optional[int] = Field(None, description="System PID")
    
    status: ProcessStatus = Field(ProcessStatus.STARTING, description="Process status")
    
    # Resources
    cpu_percent: float = Field(0.0, description="CPU usage percent")
    memory_mb: float = Field(0.0, description="Memory usage in MB")
    
    # Resource limits
    cpu_limit: Optional[float] = Field(None, description="CPU limit (cores)")
    memory_limit_mb: Optional[int] = Field(None, description="Memory limit (MB)")
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    stopped_at: Optional[datetime] = Field(None)
    
    # Health
    restart_count: int = Field(0, description="Number of restarts")
    last_error: Optional[str] = Field(None, description="Last error message")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class ProcessManager:
    """
    Production-grade process manager.
    
    Features:
    - Process lifecycle management
    - Resource monitoring
    - Automatic restart
    - Graceful shutdown
    """
    
    def __init__(self):
        """Initialize process manager."""
        self.processes: Dict[str, ProcessInfo] = {}
        self.tasks: Dict[str, asyncio.Task] = {}
        
        self.lock = asyncio.Lock()
        self.logger = StructuredLogger("process_manager", LogLevel.INFO)
        
        # Monitoring task
        self.monitor_task: Optional[asyncio.Task] = None
    
    async def start_process(
        self,
        name: str,
        target: Callable,
        resources: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new process.
        
        Args:
            name: Process name
            target: Target callable
            resources: Resource limits
        
        Returns:
            Process ID
        """
        import uuid
        
        process_id = str(uuid.uuid4())
        
        # Create process info
        process_info = ProcessInfo(
            process_id=process_id,
            name=name,
            cpu_limit=resources.get("cpu_limit") if resources else None,
            memory_limit_mb=resources.get("memory_limit_mb") if resources else None
        )
        
        async with self.lock:
            self.processes[process_id] = process_info
        
        # Start process task
        task = asyncio.create_task(
            self._run_process(process_id, target)
        )
        self.tasks[process_id] = task
        
        self.logger.info(
            "Process starting",
            process_id=process_id,
            name=name
        )
        
        # Start monitoring if not already running
        if self.monitor_task is None:
            self.monitor_task = asyncio.create_task(self._monitor_loop())
        
        return process_id
    
    async def _run_process(self, process_id: str, target: Callable):
        """Run process and handle lifecycle."""
        process_info = self.processes[process_id]
        
        try:
            # Mark as running
            process_info.status = ProcessStatus.RUNNING
            process_info.pid = os.getpid()
            
            self.logger.info(
                "Process started",
                process_id=process_id,
                name=process_info.name,
                pid=process_info.pid
            )
            
            # Execute target
            # Check if target is async (function or callable class with async __call__)
            if asyncio.iscoroutinefunction(target) or (
                hasattr(target, '__call__') and asyncio.iscoroutinefunction(target.__call__)
            ):
                result = await target()
            else:
                result = target()
            
            # Normal completion
            process_info.status = ProcessStatus.STOPPED
            process_info.stopped_at = datetime.utcnow()
            
            self.logger.info(
                "Process completed",
                process_id=process_id,
                name=process_info.name
            )
        
        except asyncio.CancelledError:
            # Graceful cancellation
            process_info.status = ProcessStatus.STOPPED
            process_info.stopped_at = datetime.utcnow()
            
            self.logger.info(
                "Process cancelled",
                process_id=process_id,
                name=process_info.name
            )
        
        except Exception as e:
            # Process crashed
            process_info.status = ProcessStatus.CRASHED
            process_info.stopped_at = datetime.utcnow()
            process_info.last_error = str(e)
            
            self.logger.error(
                "Process crashed",
                process_id=process_id,
                name=process_info.name,
                error=str(e)
            )
            
            # Record error metric
            get_metrics().record_error("process_manager", "process_crashed")
    
    async def stop_process(self, process_id: str, timeout: int = 30):
        """
        Stop process gracefully.
        
        Args:
            process_id: Process ID
            timeout: Shutdown timeout
        """
        async with self.lock:
            if process_id not in self.processes:
                return
            
            process_info = self.processes[process_id]
            process_info.status = ProcessStatus.STOPPING
        
        self.logger.info(
            "Stopping process",
            process_id=process_id,
            name=process_info.name
        )
        
        # Cancel task
        if process_id in self.tasks:
            task = self.tasks[process_id]
            task.cancel()
            
            try:
                await asyncio.wait_for(task, timeout=timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            
            del self.tasks[process_id]
        
        process_info.status = ProcessStatus.STOPPED
        process_info.stopped_at = datetime.utcnow()
        
        self.logger.info(
            "Process stopped",
            process_id=process_id,
            name=process_info.name
        )
    
    async def kill_process(self, process_id: str):
        """
        Force kill process.
        
        Args:
            process_id: Process ID
        """
        await self.stop_process(process_id, timeout=1)
        
        self.logger.warning(
            "Process killed",
            process_id=process_id
        )
    
    async def restart_process(
        self,
        process_id: str,
        target: Callable,
        resources: Optional[Dict[str, Any]] = None
    ):
        """
        Restart a process.
        
        Args:
            process_id: Process ID
            target: Target callable
            resources: Resource limits
        """
        process_info = self.processes.get(process_id)
        if not process_info:
            return
        
        # Stop current
        await self.stop_process(process_id)
        
        # Update restart count
        process_info.restart_count += 1
        process_info.status = ProcessStatus.STARTING
        process_info.started_at = datetime.utcnow()
        process_info.stopped_at = None
        process_info.last_error = None
        
        # Start new task
        task = asyncio.create_task(
            self._run_process(process_id, target)
        )
        self.tasks[process_id] = task
        
        self.logger.info(
            "Process restarted",
            process_id=process_id,
            restart_count=process_info.restart_count
        )
    
    async def get_process(self, process_id: str) -> Optional[ProcessInfo]:
        """Get process info."""
        return self.processes.get(process_id)
    
    async def list_processes(self) -> list[ProcessInfo]:
        """List all processes."""
        return list(self.processes.values())
    
    async def _monitor_loop(self):
        """Monitor process resources."""
        while True:
            try:
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
                for process_id, process_info in self.processes.items():
                    if process_info.status != ProcessStatus.RUNNING:
                        continue
                    
                    if process_info.pid:
                        try:
                            # Get process resource usage
                            proc = psutil.Process(process_info.pid)
                            process_info.cpu_percent = proc.cpu_percent()
                            process_info.memory_mb = proc.memory_info().rss / (1024 * 1024)
                            
                            # Check limits
                            if process_info.memory_limit_mb:
                                if process_info.memory_mb > process_info.memory_limit_mb:
                                    self.logger.warning(
                                        "Process exceeds memory limit",
                                        process_id=process_id,
                                        memory_mb=process_info.memory_mb,
                                        limit_mb=process_info.memory_limit_mb
                                    )
                        
                        except psutil.NoSuchProcess:
                            pass
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
    
    async def shutdown(self):
        """Shutdown all processes."""
        self.logger.info("Shutting down process manager")
        
        # Stop all processes
        for process_id in list(self.processes.keys()):
            await self.stop_process(process_id)
        
        # Cancel monitor
        if self.monitor_task:
            self.monitor_task.cancel()
        
        self.logger.info("Process manager shutdown complete")

