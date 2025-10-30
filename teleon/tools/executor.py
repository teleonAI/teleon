"""Tool executor with sandboxing and monitoring."""

from typing import Dict, Any, Optional
import asyncio
import time

from teleon.tools.base import BaseTool, ToolResult
from teleon.tools.registry import get_registry
from teleon.logging import get_logger, LogLevel


class ToolExecutor:
    """
    Executes tools with monitoring and sandboxing.
    
    Features:
    - Timeout management
    - Resource limits
    - Execution tracking
    - Error handling
    """
    
    def __init__(
        self,
        default_timeout: float = 30.0,
        enable_logging: bool = True
    ):
        """
        Initialize tool executor.
        
        Args:
            default_timeout: Default timeout in seconds
            enable_logging: Whether to enable logging
        """
        self.default_timeout = default_timeout
        self.enable_logging = enable_logging
        
        if enable_logging:
            self.logger = get_logger("tool_executor", LogLevel.INFO)
        
        # Statistics
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.total_execution_time = 0.0
    
    async def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> ToolResult:
        """
        Execute a tool.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            timeout: Execution timeout (uses default if not provided)
        
        Returns:
            ToolResult with execution outcome
        """
        start_time = time.time()
        
        # Get tool from registry
        registry = get_registry()
        tool = registry.get(tool_name)
        
        if not tool:
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool '{tool_name}' not found",
                tool_name=tool_name,
                execution_time_ms=0
            )
        
        # Log execution start
        if self.enable_logging:
            self.logger.info(
                "Tool execution started",
                tool_name=tool_name,
                parameters=parameters
            )
        
        # Execute with timeout
        timeout_seconds = timeout or self.default_timeout
        
        try:
            result = await asyncio.wait_for(
                tool.safe_execute(**parameters),
                timeout=timeout_seconds
            )
            
            # Update statistics
            self.total_executions += 1
            if result.success:
                self.successful_executions += 1
            else:
                self.failed_executions += 1
            
            execution_time = (time.time() - start_time) * 1000
            self.total_execution_time += execution_time
            
            # Log execution complete
            if self.enable_logging:
                self.logger.info(
                    "Tool execution completed",
                    tool_name=tool_name,
                    success=result.success,
                    execution_time_ms=execution_time
                )
            
            return result
            
        except asyncio.TimeoutError:
            self.total_executions += 1
            self.failed_executions += 1
            
            error_msg = f"Tool execution timed out after {timeout_seconds}s"
            
            if self.enable_logging:
                self.logger.error(
                    "Tool execution timeout",
                    tool_name=tool_name,
                    timeout=timeout_seconds
                )
            
            return ToolResult(
                success=False,
                data=None,
                error=error_msg,
                tool_name=tool_name,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        except Exception as e:
            self.total_executions += 1
            self.failed_executions += 1
            
            if self.enable_logging:
                self.logger.error(
                    "Tool execution error",
                    tool_name=tool_name,
                    error=str(e)
                )
            
            return ToolResult(
                success=False,
                data=None,
                error=f"Execution error: {str(e)}",
                tool_name=tool_name,
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    async def batch_execute(
        self,
        executions: list[tuple[str, Dict[str, Any]]]
    ) -> list[ToolResult]:
        """
        Execute multiple tools in parallel.
        
        Args:
            executions: List of (tool_name, parameters) tuples
        
        Returns:
            List of ToolResults
        """
        tasks = [
            self.execute(tool_name, params)
            for tool_name, params in executions
        ]
        
        return await asyncio.gather(*tasks)
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get executor statistics.
        
        Returns:
            Statistics dictionary
        """
        avg_time = (
            self.total_execution_time / max(self.total_executions, 1)
        )
        
        success_rate = (
            self.successful_executions / max(self.total_executions, 1)
        ) * 100
        
        return {
            "total_executions": self.total_executions,
            "successful": self.successful_executions,
            "failed": self.failed_executions,
            "success_rate": f"{success_rate:.1f}%",
            "avg_execution_time_ms": f"{avg_time:.2f}",
            "total_execution_time_ms": f"{self.total_execution_time:.2f}"
        }
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.total_execution_time = 0.0

