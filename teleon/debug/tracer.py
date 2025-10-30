"""
Trace Analysis - Detailed execution tracing and analysis.

Features:
- Function call tracing
- Performance profiling
- Call graph visualization
- Bottleneck detection
"""

from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from collections import defaultdict
import time
from functools import wraps

from teleon.core import StructuredLogger, LogLevel


class TraceEntry:
    """Single trace entry."""
    
    def __init__(
        self,
        function_name: str,
        args: tuple,
        kwargs: dict,
        start_time: float
    ):
        """Initialize trace entry."""
        self.function_name = function_name
        self.args = args
        self.kwargs = kwargs
        self.start_time = start_time
        self.end_time: Optional[float] = None
        self.duration_ms: float = 0.0
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.success: bool = True
    
    def complete(self, result: Optional[Any] = None, error: Optional[str] = None):
        """Mark entry as complete."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.result = result
        self.error = error
        self.success = error is None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "function": self.function_name,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error
        }


class Tracer:
    """
    Execution tracer.
    
    Features:
    - Trace function calls
    - Track performance
    - Analyze call patterns
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize tracer.
        
        Args:
            enabled: Enable tracing
        """
        self.enabled = enabled
        self.traces: List[TraceEntry] = []
        self.logger = StructuredLogger("tracer", LogLevel.DEBUG)
        
        # Statistics
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.total_times: Dict[str, float] = defaultdict(float)
    
    def trace(self, func: Callable) -> Callable:
        """
        Decorator to trace function execution.
        
        Args:
            func: Function to trace
        
        Returns:
            Wrapped function
        """
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not self.enabled:
                return await func(*args, **kwargs)
            
            # Start trace
            entry = TraceEntry(
                function_name=func.__name__,
                args=args,
                kwargs=kwargs,
                start_time=time.time()
            )
            
            try:
                result = await func(*args, **kwargs)
                entry.complete(result=result)
                return result
            
            except Exception as e:
                entry.complete(error=str(e))
                raise
            
            finally:
                self.traces.append(entry)
                self.call_counts[func.__name__] += 1
                self.total_times[func.__name__] += entry.duration_ms
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)
            
            # Start trace
            entry = TraceEntry(
                function_name=func.__name__,
                args=args,
                kwargs=kwargs,
                start_time=time.time()
            )
            
            try:
                result = func(*args, **kwargs)
                entry.complete(result=result)
                return result
            
            except Exception as e:
                entry.complete(error=str(e))
                raise
            
            finally:
                self.traces.append(entry)
                self.call_counts[func.__name__] += 1
                self.total_times[func.__name__] += entry.duration_ms
        
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    def get_traces(self) -> List[TraceEntry]:
        """Get all traces."""
        return self.traces
    
    def get_call_count(self, function_name: str) -> int:
        """Get call count for function."""
        return self.call_counts.get(function_name, 0)
    
    def get_total_time(self, function_name: str) -> float:
        """Get total time for function."""
        return self.total_times.get(function_name, 0.0)
    
    def get_average_time(self, function_name: str) -> float:
        """Get average time for function."""
        count = self.call_counts.get(function_name, 0)
        if count == 0:
            return 0.0
        return self.total_times.get(function_name, 0.0) / count
    
    def clear(self):
        """Clear all traces."""
        self.traces.clear()
        self.call_counts.clear()
        self.total_times.clear()
    
    def enable(self):
        """Enable tracing."""
        self.enabled = True
    
    def disable(self):
        """Disable tracing."""
        self.enabled = False


class TraceAnalyzer:
    """
    Trace analyzer.
    
    Features:
    - Analyze execution traces
    - Identify bottlenecks
    - Generate reports
    """
    
    def __init__(self, tracer: Tracer):
        """
        Initialize analyzer.
        
        Args:
            tracer: Tracer instance
        """
        self.tracer = tracer
        self.logger = StructuredLogger("trace_analyzer", LogLevel.INFO)
    
    def get_slowest_functions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get slowest functions by total time.
        
        Args:
            limit: Number of functions to return
        
        Returns:
            List of function stats
        """
        functions = []
        for func_name in self.tracer.call_counts.keys():
            functions.append({
                "function": func_name,
                "calls": self.tracer.get_call_count(func_name),
                "total_time_ms": self.tracer.get_total_time(func_name),
                "avg_time_ms": self.tracer.get_average_time(func_name)
            })
        
        # Sort by total time
        functions.sort(key=lambda x: x["total_time_ms"], reverse=True)
        return functions[:limit]
    
    def get_most_called_functions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most called functions.
        
        Args:
            limit: Number of functions to return
        
        Returns:
            List of function stats
        """
        functions = []
        for func_name in self.tracer.call_counts.keys():
            functions.append({
                "function": func_name,
                "calls": self.tracer.get_call_count(func_name),
                "total_time_ms": self.tracer.get_total_time(func_name),
                "avg_time_ms": self.tracer.get_average_time(func_name)
            })
        
        # Sort by call count
        functions.sort(key=lambda x: x["calls"], reverse=True)
        return functions[:limit]
    
    def get_bottlenecks(self, threshold_ms: float = 100) -> List[Dict[str, Any]]:
        """
        Get bottleneck functions.
        
        Args:
            threshold_ms: Average time threshold
        
        Returns:
            List of bottleneck functions
        """
        bottlenecks = []
        for func_name in self.tracer.call_counts.keys():
            avg_time = self.tracer.get_average_time(func_name)
            if avg_time > threshold_ms:
                bottlenecks.append({
                    "function": func_name,
                    "avg_time_ms": avg_time,
                    "calls": self.tracer.get_call_count(func_name),
                    "total_time_ms": self.tracer.get_total_time(func_name)
                })
        
        # Sort by average time
        bottlenecks.sort(key=lambda x: x["avg_time_ms"], reverse=True)
        return bottlenecks
    
    def print_report(self):
        """Print analysis report."""
        print("\n" + "="*80)
        print("TRACE ANALYSIS REPORT")
        print("="*80)
        
        total_traces = len(self.tracer.traces)
        unique_functions = len(self.tracer.call_counts)
        
        print(f"\n📊 Summary:")
        print(f"  Total Traces:        {total_traces}")
        print(f"  Unique Functions:    {unique_functions}")
        
        # Slowest functions
        print(f"\n🐌 Slowest Functions (by total time):")
        slowest = self.get_slowest_functions(5)
        for func in slowest:
            print(f"  {func['function']:30} {func['total_time_ms']:8.2f}ms "
                  f"({func['calls']} calls, avg {func['avg_time_ms']:.2f}ms)")
        
        # Most called
        print(f"\n📞 Most Called Functions:")
        most_called = self.get_most_called_functions(5)
        for func in most_called:
            print(f"  {func['function']:30} {func['calls']:5} calls "
                  f"(total {func['total_time_ms']:.2f}ms)")
        
        # Bottlenecks
        bottlenecks = self.get_bottlenecks(50)
        if bottlenecks:
            print(f"\n⚠️  Bottlenecks (avg > 50ms):")
            for func in bottlenecks[:5]:
                print(f"  {func['function']:30} avg {func['avg_time_ms']:8.2f}ms "
                      f"({func['calls']} calls)")
        
        print("\n" + "="*80 + "\n")

