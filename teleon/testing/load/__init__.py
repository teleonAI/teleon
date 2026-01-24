"""
Load Testing - Production-grade load testing framework.

Features:
- Concurrent request testing
- Performance profiling
- Latency analysis
- Throughput measurement
- Resource monitoring
"""

from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field, ConfigDict, field_serializer
import asyncio
import time
from statistics import mean, median, stdev

from teleon.core import StructuredLogger, LogLevel


class LoadTestConfig(BaseModel):
    """Load test configuration."""
    
    # Test parameters
    total_requests: int = Field(100, ge=1, description="Total number of requests")
    concurrent_users: int = Field(10, ge=1, description="Number of concurrent users")
    ramp_up_seconds: int = Field(0, ge=0, description="Ramp up time")
    
    # Timeouts
    request_timeout: float = Field(30.0, ge=1, description="Request timeout")
    
    # Reporting
    report_interval: int = Field(10, ge=1, description="Progress report interval")


class LoadTestResult(BaseModel):
    """Load test result."""
    
    # Summary
    total_requests: int = Field(..., description="Total requests")
    successful_requests: int = Field(..., description="Successful requests")
    failed_requests: int = Field(..., description="Failed requests")
    
    # Timing
    total_duration: float = Field(..., description="Total duration (seconds)")
    requests_per_second: float = Field(..., description="Throughput (req/s)")
    
    # Latency statistics
    mean_latency: float = Field(..., description="Mean latency (ms)")
    median_latency: float = Field(..., description="Median latency (ms)")
    p95_latency: float = Field(..., description="95th percentile latency (ms)")
    p99_latency: float = Field(..., description="99th percentile latency (ms)")
    min_latency: float = Field(..., description="Minimum latency (ms)")
    max_latency: float = Field(..., description="Maximum latency (ms)")
    stdev_latency: Optional[float] = Field(None, description="Standard deviation")
    
    # Errors
    errors: List[str] = Field(default_factory=list, description="Error messages")

    model_config = ConfigDict()


class LoadTester:
    """
    Production-grade load tester.
    
    Features:
    - Concurrent execution
    - Performance profiling
    - Real-time reporting
    - Detailed statistics
    """
    
    def __init__(self, config: Optional[LoadTestConfig] = None):
        """
        Initialize load tester.
        
        Args:
            config: Load test configuration
        """
        self.config = config or LoadTestConfig()
        self.logger = StructuredLogger("load_tester", LogLevel.INFO)
        
        # Results tracking
        self.latencies: List[float] = []
        self.successes = 0
        self.failures = 0
        self.errors: List[str] = []
        
        # Timing
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    async def run(
        self,
        target_fn: Callable,
        *args,
        **kwargs
    ) -> LoadTestResult:
        """
        Run load test.
        
        Args:
            target_fn: Target function to test
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Load test result
        """
        self.logger.info(
            "Starting load test",
            total_requests=self.config.total_requests,
            concurrent_users=self.config.concurrent_users
        )
        
        self.start_time = time.time()
        
        # Calculate requests per user
        requests_per_user = self.config.total_requests // self.config.concurrent_users
        remaining_requests = self.config.total_requests % self.config.concurrent_users
        
        # Create user tasks
        tasks = []
        for i in range(self.config.concurrent_users):
            user_requests = requests_per_user
            if i < remaining_requests:
                user_requests += 1
            
            # Calculate delay for ramp-up
            delay = (i * self.config.ramp_up_seconds) / self.config.concurrent_users if self.config.ramp_up_seconds > 0 else 0
            
            task = asyncio.create_task(
                self._run_user(i, user_requests, delay, target_fn, *args, **kwargs)
            )
            tasks.append(task)
        
        # Wait for all users to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.end_time = time.time()
        
        # Generate result
        result = self._generate_result()
        
        self.logger.info(
            "Load test completed",
            successful_requests=result.successful_requests,
            failed_requests=result.failed_requests,
            requests_per_second=result.requests_per_second,
            mean_latency=result.mean_latency
        )
        
        return result
    
    async def _run_user(
        self,
        user_id: int,
        request_count: int,
        delay: float,
        target_fn: Callable,
        *args,
        **kwargs
    ):
        """Run requests for a single user."""
        # Ramp-up delay
        if delay > 0:
            await asyncio.sleep(delay)
        
        for i in range(request_count):
            try:
                await self._execute_request(target_fn, *args, **kwargs)
            except Exception as e:
                self.logger.error(f"User {user_id} request {i} failed: {e}")
    
    async def _execute_request(
        self,
        target_fn: Callable,
        *args,
        **kwargs
    ):
        """Execute a single request."""
        start_time = time.time()
        
        try:
            # Execute with timeout
            if asyncio.iscoroutinefunction(target_fn):
                await asyncio.wait_for(
                    target_fn(*args, **kwargs),
                    timeout=self.config.request_timeout
                )
            else:
                target_fn(*args, **kwargs)
            
            # Record success
            latency_ms = (time.time() - start_time) * 1000
            self.latencies.append(latency_ms)
            self.successes += 1
        
        except asyncio.TimeoutError:
            self.failures += 1
            self.errors.append("Request timeout")
        
        except Exception as e:
            self.failures += 1
            error_msg = str(e)[:100]  # Limit error message length
            self.errors.append(error_msg)
    
    def _generate_result(self) -> LoadTestResult:
        """Generate test result."""
        if not self.start_time or not self.end_time:
            raise RuntimeError("Test has not been run")
        
        total_duration = self.end_time - self.start_time
        total_requests = self.successes + self.failures
        
        # Calculate latency statistics
        if self.latencies:
            sorted_latencies = sorted(self.latencies)
            p95_index = int(len(sorted_latencies) * 0.95)
            p99_index = int(len(sorted_latencies) * 0.99)
            
            mean_lat = mean(self.latencies)
            median_lat = median(self.latencies)
            p95_lat = sorted_latencies[p95_index] if p95_index < len(sorted_latencies) else sorted_latencies[-1]
            p99_lat = sorted_latencies[p99_index] if p99_index < len(sorted_latencies) else sorted_latencies[-1]
            min_lat = min(self.latencies)
            max_lat = max(self.latencies)
            stdev_lat = stdev(self.latencies) if len(self.latencies) > 1 else None
        else:
            mean_lat = median_lat = p95_lat = p99_lat = min_lat = max_lat = 0
            stdev_lat = None
        
        return LoadTestResult(
            total_requests=total_requests,
            successful_requests=self.successes,
            failed_requests=self.failures,
            total_duration=total_duration,
            requests_per_second=total_requests / total_duration if total_duration > 0 else 0,
            mean_latency=mean_lat,
            median_latency=median_lat,
            p95_latency=p95_lat,
            p99_latency=p99_lat,
            min_latency=min_lat,
            max_latency=max_lat,
            stdev_latency=stdev_lat,
            errors=list(set(self.errors))[:10]  # Unique errors, max 10
        )
    
    def print_summary(self, result: LoadTestResult):
        """Print test summary."""
        print("\n" + "="*80)
        print("LOAD TEST SUMMARY")
        print("="*80)
        print(f"\n📊 Overall Statistics:")
        print(f"  Total Requests:      {result.total_requests}")
        print(f"  Successful:          {result.successful_requests} ({result.successful_requests/result.total_requests*100:.1f}%)")
        print(f"  Failed:              {result.failed_requests} ({result.failed_requests/result.total_requests*100:.1f}%)")
        print(f"  Total Duration:      {result.total_duration:.2f}s")
        print(f"  Throughput:          {result.requests_per_second:.2f} req/s")
        
        print(f"\n⏱️  Latency Statistics (ms):")
        print(f"  Mean:                {result.mean_latency:.2f}")
        print(f"  Median:              {result.median_latency:.2f}")
        print(f"  95th Percentile:     {result.p95_latency:.2f}")
        print(f"  99th Percentile:     {result.p99_latency:.2f}")
        print(f"  Min:                 {result.min_latency:.2f}")
        print(f"  Max:                 {result.max_latency:.2f}")
        if result.stdev_latency:
            print(f"  Std Dev:             {result.stdev_latency:.2f}")
        
        if result.errors:
            print(f"\n❌ Errors (unique):")
            for error in result.errors[:5]:
                print(f"  • {error}")
        
        print("\n" + "="*80 + "\n")


async def quick_load_test(
    target_fn: Callable,
    requests: int = 100,
    concurrency: int = 10,
    *args,
    **kwargs
) -> LoadTestResult:
    """
    Run a quick load test.
    
    Args:
        target_fn: Target function
        requests: Number of requests
        concurrency: Concurrent users
        *args: Target function args
        **kwargs: Target function kwargs
    
    Returns:
        Load test result
    """
    config = LoadTestConfig(
        total_requests=requests,
        concurrent_users=concurrency
    )
    
    tester = LoadTester(config)
    result = await tester.run(target_fn, *args, **kwargs)
    tester.print_summary(result)
    
    return result


__all__ = [
    "LoadTester",
    "LoadTestConfig",
    "LoadTestResult",
    "quick_load_test",
]

