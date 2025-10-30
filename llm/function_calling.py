"""Function calling / tool use for LLMs."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from teleon.tools.base import BaseTool, ToolSchema
from teleon.tools.registry import ToolRegistry


class FunctionCallRequest(BaseModel):
    """Function call request from LLM."""
    
    name: str = Field(..., description="Function name")
    arguments: Dict[str, Any] = Field(..., description="Function arguments")


class FunctionCallResult(BaseModel):
    """Result of function call."""
    
    name: str = Field(..., description="Function name")
    result: Any = Field(..., description="Function result")
    success: bool = Field(..., description="Whether call succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")


class ToolSchemaConverter:
    """
    Convert Teleon tool schemas to LLM-specific formats.
    
    Supports:
    - OpenAI function calling format
    - Anthropic tool use format
    """
    
    @staticmethod
    def to_openai_function(tool_schema: ToolSchema) -> Dict[str, Any]:
        """
        Convert tool schema to OpenAI function format.
        
        Args:
            tool_schema: Teleon tool schema
        
        Returns:
            OpenAI function definition
        """
        return {
            "name": tool_schema.name,
            "description": tool_schema.description,
            "parameters": tool_schema.parameters
        }
    
    @staticmethod
    def to_anthropic_tool(tool_schema: ToolSchema) -> Dict[str, Any]:
        """
        Convert tool schema to Anthropic tool format.
        
        Args:
            tool_schema: Teleon tool schema
        
        Returns:
            Anthropic tool definition
        """
        return {
            "name": tool_schema.name,
            "description": tool_schema.description,
            "input_schema": tool_schema.parameters
        }
    
    @staticmethod
    def tools_to_openai_functions(tools: List[BaseTool]) -> List[Dict[str, Any]]:
        """Convert list of tools to OpenAI functions."""
        return [
            ToolSchemaConverter.to_openai_function(tool.get_schema())
            for tool in tools
        ]
    
    @staticmethod
    def tools_to_anthropic_tools(tools: List[BaseTool]) -> List[Dict[str, Any]]:
        """Convert list of tools to Anthropic tools."""
        return [
            ToolSchemaConverter.to_anthropic_tool(tool.get_schema())
            for tool in tools
        ]


class FunctionCallingOrchestrator:
    """
    Orchestrate function calling between LLMs and tools.
    
    Handles:
    - Converting tools to LLM format
    - Parsing LLM function call requests
    - Executing tools
    - Formatting results for LLM
    """
    
    def __init__(self, registry: Optional[ToolRegistry] = None):
        """
        Initialize orchestrator.
        
        Args:
            registry: Tool registry (uses global if not provided)
        """
        from teleon.tools.registry import get_registry
        self.registry = registry or get_registry()
        self.converter = ToolSchemaConverter()
    
    def get_available_tools(
        self,
        tool_names: Optional[List[str]] = None
    ) -> List[BaseTool]:
        """
        Get available tools.
        
        Args:
            tool_names: Specific tools to include (None = all)
        
        Returns:
            List of tools
        """
        if tool_names:
            tools = []
            for name in tool_names:
                tool = self.registry.get(name)
                if tool:
                    tools.append(tool)
            return tools
        else:
            return self.registry.list_tools()
    
    def get_openai_functions(
        self,
        tool_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get tools in OpenAI function format.
        
        Args:
            tool_names: Specific tools to include
        
        Returns:
            OpenAI function definitions
        """
        tools = self.get_available_tools(tool_names)
        return self.converter.tools_to_openai_functions(tools)
    
    def get_anthropic_tools(
        self,
        tool_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get tools in Anthropic tool format.
        
        Args:
            tool_names: Specific tools to include
        
        Returns:
            Anthropic tool definitions
        """
        tools = self.get_available_tools(tool_names)
        return self.converter.tools_to_anthropic_tools(tools)
    
    async def execute_function_call(
        self,
        function_call: FunctionCallRequest
    ) -> FunctionCallResult:
        """
        Execute a function call from LLM.
        
        Args:
            function_call: Function call request
        
        Returns:
            Function call result
        """
        from teleon.tools.executor import ToolExecutor
        
        executor = ToolExecutor()
        
        # Execute tool
        result = await executor.execute(
            function_call.name,
            function_call.arguments
        )
        
        return FunctionCallResult(
            name=function_call.name,
            result=result.data,
            success=result.success,
            error=result.error
        )
    
    async def execute_multiple_function_calls(
        self,
        function_calls: List[FunctionCallRequest]
    ) -> List[FunctionCallResult]:
        """
        Execute multiple function calls.
        
        Args:
            function_calls: List of function call requests
        
        Returns:
            List of function call results
        """
        from teleon.tools.executor import ToolExecutor
        
        executor = ToolExecutor()
        
        # Prepare tool calls
        tool_calls = [
            (call.name, call.arguments)
            for call in function_calls
        ]
        
        # Execute in parallel
        results = await executor.batch_execute(tool_calls)
        
        # Convert to FunctionCallResult
        return [
            FunctionCallResult(
                name=function_calls[i].name,
                result=result.data,
                success=result.success,
                error=result.error
            )
            for i, result in enumerate(results)
        ]
    
    def format_function_result_for_llm(
        self,
        result: FunctionCallResult
    ) -> str:
        """
        Format function result for LLM consumption.
        
        Args:
            result: Function call result
        
        Returns:
            Formatted string
        """
        if result.success:
            return f"Function '{result.name}' returned: {result.result}"
        else:
            return f"Function '{result.name}' failed: {result.error}"

