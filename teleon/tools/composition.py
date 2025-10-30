"""Tool composition - chain and combine tools."""

from typing import List, Dict, Any, Optional
import asyncio

from teleon.tools.base import ToolResult
from teleon.tools.executor import ToolExecutor


class ToolChain:
    """
    Chain multiple tools together.
    
    The output of one tool feeds into the next.
    """
    
    def __init__(self, executor: Optional[ToolExecutor] = None):
        """
        Initialize tool chain.
        
        Args:
            executor: ToolExecutor instance (creates new if not provided)
        """
        self.executor = executor or ToolExecutor()
        self.steps: List[tuple] = []
    
    def add_step(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        output_mapping: Optional[Dict[str, str]] = None
    ) -> 'ToolChain':
        """
        Add a step to the chain.
        
        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters (can reference previous outputs)
            output_mapping: How to map output to next step's input
        
        Returns:
            Self for chaining
        """
        self.steps.append((tool_name, parameters, output_mapping or {}))
        return self
    
    async def execute(self, initial_input: Optional[Dict[str, Any]] = None) -> List[ToolResult]:
        """
        Execute the tool chain.
        
        Args:
            initial_input: Initial input data
        
        Returns:
            List of ToolResults from each step
        """
        results = []
        context = initial_input or {}
        
        for tool_name, parameters, output_mapping in self.steps:
            # Resolve parameters from context
            resolved_params = self._resolve_parameters(parameters, context)
            
            # Execute tool
            result = await self.executor.execute(tool_name, resolved_params)
            results.append(result)
            
            # Update context with result
            if result.success:
                if output_mapping:
                    for output_key, context_key in output_mapping.items():
                        if output_key == "data":
                            context[context_key] = result.data
                        else:
                            context[context_key] = getattr(result, output_key, None)
                else:
                    context["last_output"] = result.data
        
        return results
    
    def _resolve_parameters(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve parameter values from context."""
        resolved = {}
        
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("$"):
                # Reference to context variable
                context_key = value[1:]
                resolved[key] = context.get(context_key)
            else:
                resolved[key] = value
        
        return resolved


class ParallelToolExecution:
    """Execute multiple tools in parallel."""
    
    def __init__(self, executor: Optional[ToolExecutor] = None):
        """
        Initialize parallel execution.
        
        Args:
            executor: ToolExecutor instance
        """
        self.executor = executor or ToolExecutor()
    
    async def execute(
        self,
        tool_calls: List[tuple[str, Dict[str, Any]]]
    ) -> List[ToolResult]:
        """
        Execute tools in parallel.
        
        Args:
            tool_calls: List of (tool_name, parameters) tuples
        
        Returns:
            List of ToolResults
        """
        return await self.executor.batch_execute(tool_calls)


class ConditionalToolExecution:
    """Execute tools conditionally based on results."""
    
    def __init__(self, executor: Optional[ToolExecutor] = None):
        """
        Initialize conditional execution.
        
        Args:
            executor: ToolExecutor instance
        """
        self.executor = executor or ToolExecutor()
    
    async def execute_if(
        self,
        condition_tool: str,
        condition_params: Dict[str, Any],
        then_tool: str,
        then_params: Dict[str, Any],
        else_tool: Optional[str] = None,
        else_params: Optional[Dict[str, Any]] = None
    ) -> tuple[ToolResult, Optional[ToolResult]]:
        """
        Execute tool conditionally.
        
        Args:
            condition_tool: Tool to evaluate condition
            condition_params: Condition tool parameters
            then_tool: Tool to execute if condition is true
            then_params: Then tool parameters
            else_tool: Optional tool to execute if condition is false
            else_params: Else tool parameters
        
        Returns:
            Tuple of (condition_result, executed_tool_result)
        """
        # Execute condition
        condition_result = await self.executor.execute(
            condition_tool,
            condition_params
        )
        
        # Check condition (assume condition tool returns boolean-like data)
        condition_met = (
            condition_result.success and
            condition_result.data and
            condition_result.data.get("valid", False)
        )
        
        # Execute appropriate tool
        if condition_met:
            result = await self.executor.execute(then_tool, then_params)
        elif else_tool:
            result = await self.executor.execute(else_tool, else_params or {})
        else:
            result = None
        
        return (condition_result, result)

