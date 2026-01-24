"""
Auto-scaler - Production-grade automatic scaling.

Features:
- CPU-based scaling
- Memory-based scaling
- Custom metric scaling
- Scaling policies
- Scale-up/down cooldowns
"""

from typing import Dict, Optional
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field
import asyncio
import psutil

from teleon.core import (
    get_metrics,
    StructuredLogger,
    LogLevel,
)
from teleon.helix.llm_metrics import LLMMetrics


class ScalingMetrics(BaseModel):
    """Scaling metrics."""
    
    cpu_percent: float = Field(0.0, description="Average CPU usage")
    memory_percent: float = Field(0.0, description="Average memory usage")
    request_rate: float = Field(0.0, description="Requests per second")
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ScalingPolicy(BaseModel):
    """Scaling policy configuration."""
    
    min_instances: int = Field(1, ge=1, description="Minimum instances")
    max_instances: int = Field(10, ge=1, description="Maximum instances")
    
    # Thresholds
    target_cpu_percent: float = Field(70.0, ge=0, le=100, description="Target CPU percent")
    target_memory_percent: float = Field(80.0, ge=0, le=100, description="Target memory percent")
    
    # Cooldown
    scale_up_cooldown: int = Field(60, ge=10, description="Scale up cooldown (seconds)")
    scale_down_cooldown: int = Field(300, ge=30, description="Scale down cooldown (seconds)")
    
    # Step size
    scale_up_step: int = Field(1, ge=1, description="Instances to add")
    scale_down_step: int = Field(1, ge=1, description="Instances to remove")


class Scaler:
    """
    Production-grade auto-scaler.
    
    Features:
    - Metric-based scaling
    - Cooldown periods
    - Configurable policies
    """
    
    def __init__(self):
        """Initialize scaler."""
        self.policies: Dict[str, ScalingPolicy] = {}
        self.last_scale_action: Dict[str, datetime] = {}
        
        # Track scaling history to prevent oscillation
        self.scaling_history: Dict[str, List[Dict[str, Any]]] = {}
        self.max_history_per_target = 10
        
        self.lock = asyncio.Lock()
        self.logger = StructuredLogger("scaler", LogLevel.INFO)
        
        self.running = False
        self.monitor_task: Optional[asyncio.Task] = None
    
    async def register_policy(
        self,
        target_id: str,
        policy: ScalingPolicy
    ):
        """
        Register scaling policy.
        
        Args:
            target_id: Target identifier
            policy: Scaling policy
        """
        async with self.lock:
            self.policies[target_id] = policy
        
        self.logger.info(
            "Scaling policy registered",
            target_id=target_id,
            min=policy.min_instances,
            max=policy.max_instances
        )
    
    async def unregister_policy(self, target_id: str):
        """Unregister scaling policy."""
        async with self.lock:
            if target_id in self.policies:
                del self.policies[target_id]
            if target_id in self.last_scale_action:
                del self.last_scale_action[target_id]
        
        self.logger.info("Scaling policy unregistered", target_id=target_id)
    
    async def evaluate_scaling(
        self,
        target_id: str,
        metrics: ScalingMetrics,
        current_instances: int
    ) -> Optional[int]:
        """
        Evaluate if scaling is needed.
        
        Args:
            target_id: Target identifier
            metrics: Current metrics
            current_instances: Current instance count
        
        Returns:
            Desired instance count or None if no scaling needed
        """
        policy = self.policies.get(target_id)
        if not policy:
            return None
        
        # Check cooldown
        last_action = self.last_scale_action.get(target_id)
        if last_action:
            time_since_last = (datetime.now(timezone.utc) - last_action).total_seconds()
            
            # Check if in cooldown period
            if metrics.cpu_percent > policy.target_cpu_percent:
                if time_since_last < policy.scale_up_cooldown:
                    return None
            else:
                if time_since_last < policy.scale_down_cooldown:
                    return None
        
        # Determine if scaling needed with hysteresis to prevent oscillation
        scale_up = False
        scale_down = False
        
        # Hysteresis: Use different thresholds for scale-up vs scale-down
        # Scale up threshold: 100% of target (aggressive)
        # Scale down threshold: 50% of target (conservative)
        cpu_scale_up_threshold = policy.target_cpu_percent
        cpu_scale_down_threshold = policy.target_cpu_percent * 0.5
        memory_scale_up_threshold = policy.target_memory_percent
        memory_scale_down_threshold = policy.target_memory_percent * 0.5
        
        # Check CPU with hysteresis
        if metrics.cpu_percent > cpu_scale_up_threshold:
            scale_up = True
        elif metrics.cpu_percent < cpu_scale_down_threshold:
            # Only scale down if we're consistently below threshold
            # Check if we've been below threshold for a while
            scale_down = True
        
        # Check memory with hysteresis
        if metrics.memory_percent > memory_scale_up_threshold:
            scale_up = True
        elif metrics.memory_percent < memory_scale_down_threshold and not scale_up:
            scale_down = True
        
        # Check for oscillation: rapid up/down cycles
        if target_id in self.scaling_history:
            recent_scales = self.scaling_history[target_id][-3:]  # Last 3 scales
            if len(recent_scales) >= 2:
                # Check if we're oscillating (up then down or vice versa)
                directions = [s.get("direction") for s in recent_scales]
                if len(directions) >= 2:
                    is_oscillating = (
                        (directions[-1] == "up" and directions[-2] == "down") or
                        (directions[-1] == "down" and directions[-2] == "up")
                    )
                    if is_oscillating:
                        # Suppress scaling to prevent oscillation
                        self.logger.warning(
                            "Oscillation detected - suppressing scaling",
                            target_id=target_id,
                            recent_directions=directions
                        )
                        return None
        
        # Calculate desired instances
        if scale_up:
            desired = min(
                current_instances + policy.scale_up_step,
                policy.max_instances
            )
            
            if desired > current_instances:
                # Record scaling decision
                async with self.lock:
                    if target_id not in self.scaling_history:
                        self.scaling_history[target_id] = []
                    self.scaling_history[target_id].append({
                        "timestamp": datetime.now(timezone.utc),
                        "direction": "up",
                        "from": current_instances,
                        "to": desired,
                        "cpu": metrics.cpu_percent,
                        "memory": metrics.memory_percent
                    })
                    # Keep only recent history
                    if len(self.scaling_history[target_id]) > self.max_history_per_target:
                        self.scaling_history[target_id] = self.scaling_history[target_id][-self.max_history_per_target:]
                
                self.logger.info(
                    "Scaling up",
                    target_id=target_id,
                    from_instances=current_instances,
                    to_instances=desired,
                    cpu=metrics.cpu_percent,
                    memory=metrics.memory_percent
                )
                return desired
        
        elif scale_down:
            desired = max(
                current_instances - policy.scale_down_step,
                policy.min_instances
            )
            
            if desired < current_instances:
                # Record scaling decision
                async with self.lock:
                    if target_id not in self.scaling_history:
                        self.scaling_history[target_id] = []
                    self.scaling_history[target_id].append({
                        "timestamp": datetime.now(timezone.utc),
                        "direction": "down",
                        "from": current_instances,
                        "to": desired,
                        "cpu": metrics.cpu_percent,
                        "memory": metrics.memory_percent
                    })
                    # Keep only recent history
                    if len(self.scaling_history[target_id]) > self.max_history_per_target:
                        self.scaling_history[target_id] = self.scaling_history[target_id][-self.max_history_per_target:]
                
                self.logger.info(
                    "Scaling down",
                    target_id=target_id,
                    from_instances=current_instances,
                    to_instances=desired,
                    cpu=metrics.cpu_percent,
                    memory=metrics.memory_percent
                )
                return desired
        
        return None
    
    async def evaluate_llm_scaling(
        self,
        target_id: str,
        llm_metrics: LLMMetrics,
        current_instances: int
    ) -> Optional[int]:
        """
        Evaluate scaling based on LLM metrics.
        
        Integrates with LLM-specific metrics like token throughput,
        queue depth, and cost constraints.
        
        Args:
            target_id: Target identifier
            llm_metrics: LLM metrics
            current_instances: Current instance count
        
        Returns:
            Desired instance count or None if no scaling needed
        """
        # Try to get LLM scaler from llm_scaling module
        try:
            from teleon.helix.llm_scaling import get_llm_scaler
            
            llm_scaler = get_llm_scaler()
            desired = await llm_scaler.evaluate_scaling(
                target_id,
                llm_metrics,
                current_instances
            )
            
            if desired:
                return desired
        except ImportError:
            self.logger.warning("LLM scaling not available, falling back to standard scaling")
        
        # Fallback: use overload detection from LLMMetrics
        if llm_metrics.is_overloaded():
            policy = self.policies.get(target_id)
            if policy:
                desired = min(
                    current_instances + policy.scale_up_step,
                    policy.max_instances
                )
                if desired > current_instances:
                    self.logger.info(
                        "LLM agent overloaded - scaling up",
                        target_id=target_id,
                        queue_depth=llm_metrics.queue_depth,
                        p95_latency=llm_metrics.p95_latency_ms
                    )
                    return desired
        
        elif llm_metrics.is_underutilized():
            policy = self.policies.get(target_id)
            if policy:
                desired = max(
                    current_instances - policy.scale_down_step,
                    policy.min_instances
                )
                if desired < current_instances:
                    self.logger.info(
                        "LLM agent underutilized - scaling down",
                        target_id=target_id,
                        utilization=llm_metrics.get_utilization()
                    )
                    return desired
        
        return None
    
    def _get_process_memory_limit(self, pid: int) -> Optional[float]:
        """
        Get actual memory limit from process.
        
        Args:
            pid: Process ID
        
        Returns:
            Memory limit in MB or None if unable to determine
        """
        try:
            process = psutil.Process(pid)
            
            # Try to get memory limit from cgroup (containerized environment)
            try:
                with open(f'/proc/{pid}/cgroup', 'r') as f:
                    cgroup_info = f.read()
                    
                # Check if running in a container with memory limits
                if 'memory' in cgroup_info or 'docker' in cgroup_info:
                    # Try to read memory limit from cgroup v1
                    try:
                        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                            limit_bytes = int(f.read().strip())
                            # Convert to MB, ignore unrealistic limits (> 1TB indicates no limit)
                            if limit_bytes < 1099511627776:  # 1TB in bytes
                                return limit_bytes / (1024 * 1024)
                    except (FileNotFoundError, ValueError, PermissionError):
                        pass
                    
                    # Try cgroup v2
                    try:
                        with open('/sys/fs/cgroup/memory.max', 'r') as f:
                            limit_str = f.read().strip()
                            if limit_str != 'max':
                                limit_bytes = int(limit_str)
                                if limit_bytes < 1099511627776:
                                    return limit_bytes / (1024 * 1024)
                    except (FileNotFoundError, ValueError, PermissionError):
                        pass
            except (FileNotFoundError, PermissionError):
                pass
            
            # Try to get resource limit (RLIMIT_AS or RLIMIT_DATA)
            try:
                # Get virtual memory limit
                rlimits = process.rlimit(psutil.RLIMIT_AS)
                soft_limit = rlimits[0]
                
                # RLIMIT_INFINITY is typically a very large number
                if soft_limit != psutil.RLIM_INFINITY and soft_limit < 1099511627776:
                    return soft_limit / (1024 * 1024)
            except (AttributeError, OSError):
                pass
            
            # Fallback: use system total memory as the theoretical limit
            system_memory = psutil.virtual_memory().total
            return system_memory / (1024 * 1024)
            
        except (psutil.NoSuchProcess, psutil.AccessDenied, Exception) as e:
            self.logger.debug(f"Unable to get memory limit for PID {pid}: {e}")
            return None
    
    async def record_scaling_action(self, target_id: str):
        """Record that a scaling action occurred."""
        async with self.lock:
            self.last_scale_action[target_id] = datetime.now(timezone.utc)
    
    async def start_monitoring(self, runtime):
        """
        Start monitoring and auto-scaling.
        
        Args:
            runtime: AgentRuntime instance
        """
        if self.running:
            return
        
        self.running = True
        self.logger.info("Auto-scaling monitoring started")
        
        async def monitor_loop():
            while self.running:
                try:
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
                    # Check all agents with policies
                    for target_id in list(self.policies.keys()):
                        try:
                            # Get agent status
                            status = await runtime.get_agent_status(target_id)
                            
                            if status.get("status") != "running":
                                continue
                            
                            # Calculate average metrics
                            processes = status.get("processes", [])
                            if not processes:
                                continue
                            
                            avg_cpu = sum(p.get("cpu_percent", 0) for p in processes) / len(processes)
                            avg_memory = sum(p.get("memory_mb", 0) for p in processes) / len(processes)
                            
                            # Get actual memory limit from process
                            memory_limit = None
                            if processes and processes[0].get("pid"):
                                memory_limit = self._get_process_memory_limit(processes[0]["pid"])
                            
                            # Fallback to configured limit if actual limit unavailable
                            if not memory_limit:
                                memory_limit = status.get("resources", {}).get("memory_limit_mb", 1024)
                            
                            memory_percent = (avg_memory / memory_limit) * 100 if memory_limit else 0
                            
                            metrics = ScalingMetrics(
                                cpu_percent=avg_cpu,
                                memory_percent=memory_percent
                            )
                            
                            # Evaluate scaling
                            desired_instances = await self.evaluate_scaling(
                                target_id,
                                metrics,
                                len(processes)
                            )
                            
                            if desired_instances and desired_instances != len(processes):
                                # Perform scaling
                                await runtime.scale_agent(target_id, desired_instances)
                                await self.record_scaling_action(target_id)
                        
                        except Exception as e:
                            self.logger.error(
                                "Scaling evaluation error",
                                target_id=target_id,
                                error=str(e)
                            )
                
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Monitor loop error: {e}")
        
        self.monitor_task = asyncio.create_task(monitor_loop())
    
    async def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Auto-scaling monitoring stopped")
    
    async def shutdown(self):
        """Shutdown scaler."""
        await self.stop_monitoring()
        self.logger.info("Scaler shutdown complete")

