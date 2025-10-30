"""Agent that can use tools via LLM function calling."""

from typing import List, Dict, Any, Optional
import json

from teleon.llm.gateway import LLMGateway
from teleon.llm.types import LLMMessage, LLMConfig, LLMResponse
from teleon.llm.function_calling import (
    FunctionCallingOrchestrator,
    FunctionCallRequest,
    FunctionCallResult
)
from teleon.logging import get_logger, LogLevel


class ToolUsingAgent:
    """
    Agent that can use tools via LLM function calling.
    
    Features:
    - Automatic tool calling
    - Multi-step reasoning
    - Tool result integration
    - Conversation history
    """
    
    def __init__(
        self,
        gateway: LLMGateway,
        available_tools: Optional[List[str]] = None,
        max_iterations: int = 5
    ):
        """
        Initialize tool-using agent.
        
        Args:
            gateway: LLM gateway
            available_tools: List of tool names (None = all)
            max_iterations: Max tool-calling iterations
        """
        self.gateway = gateway
        self.orchestrator = FunctionCallingOrchestrator()
        self.available_tools = available_tools
        self.max_iterations = max_iterations
        self.logger = get_logger("tool_using_agent", LogLevel.INFO)
        
        # Conversation history
        self.messages: List[LLMMessage] = []
    
    async def run(
        self,
        user_message: str,
        config: Optional[LLMConfig] = None
    ) -> str:
        """
        Run agent with tool use capability.
        
        Args:
            user_message: User's message
            config: LLM configuration
        
        Returns:
            Agent's final response
        """
        # Add user message
        self.messages.append(LLMMessage(role="user", content=user_message))
        
        # Use default config if not provided
        if config is None:
            config = LLMConfig(model="gpt-4", temperature=0.7)
        
        # Get available functions/tools
        if "gpt" in config.model.lower():
            functions = self.orchestrator.get_openai_functions(self.available_tools)
            return await self._run_openai(config, functions)
        elif "claude" in config.model.lower():
            tools = self.orchestrator.get_anthropic_tools(self.available_tools)
            return await self._run_anthropic(config, tools)
        else:
            # Fallback: no tool use
            response = await self.gateway.complete(self.messages, config)
            self.messages.append(LLMMessage(role="assistant", content=response.content))
            return response.content
    
    async def _run_openai(
        self,
        config: LLMConfig,
        functions: List[Dict[str, Any]]
    ) -> str:
        """Run with OpenAI function calling."""
        
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            
            self.logger.info(f"Iteration {iteration}", functions_count=len(functions))
            
            # Make LLM call with functions
            # Note: This is a simplified version. In production, you'd use
            # the actual OpenAI SDK with function calling support
            response = await self.gateway.complete(self.messages, config)
            
            # For now, we'll simulate function calling detection
            # In a real implementation, you'd check response.function_call
            
            # Check if response contains tool use (simplified detection)
            if self._detect_tool_use(response.content):
                # Extract function call (simplified)
                function_call = self._extract_function_call(response.content)
                
                if function_call:
                    # Execute function
                    result = await self.orchestrator.execute_function_call(function_call)
                    
                    self.logger.info(
                        "Function executed",
                        function=function_call.name,
                        success=result.success
                    )
                    
                    # Add function result to messages
                    result_text = self.orchestrator.format_function_result_for_llm(result)
                    self.messages.append(LLMMessage(
                        role="function",
                        content=result_text,
                        name=function_call.name
                    ))
                    
                    # Continue loop to get LLM's response to function result
                    continue
            
            # No more function calls, return final response
            self.messages.append(LLMMessage(role="assistant", content=response.content))
            return response.content
        
        # Max iterations reached
        self.logger.warning("Max iterations reached", max_iterations=self.max_iterations)
        return self.messages[-1].content if self.messages else "No response"
    
    async def _run_anthropic(
        self,
        config: LLMConfig,
        tools: List[Dict[str, Any]]
    ) -> str:
        """Run with Anthropic tool use."""
        
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            
            self.logger.info(f"Iteration {iteration}", tools_count=len(tools))
            
            # Make LLM call with tools
            response = await self.gateway.complete(self.messages, config)
            
            # Check if response contains tool use
            if self._detect_tool_use(response.content):
                function_call = self._extract_function_call(response.content)
                
                if function_call:
                    # Execute tool
                    result = await self.orchestrator.execute_function_call(function_call)
                    
                    self.logger.info(
                        "Tool executed",
                        tool=function_call.name,
                        success=result.success
                    )
                    
                    # Add tool result to messages
                    result_text = self.orchestrator.format_function_result_for_llm(result)
                    self.messages.append(LLMMessage(
                        role="user",
                        content=result_text
                    ))
                    
                    continue
            
            # No more tool calls
            self.messages.append(LLMMessage(role="assistant", content=response.content))
            return response.content
        
        return self.messages[-1].content if self.messages else "No response"
    
    def _detect_tool_use(self, content: str) -> bool:
        """
        Detect if LLM wants to use a tool (simplified).
        
        In production, you'd check the actual function_call field
        from OpenAI or tool_use from Anthropic.
        """
        # Simplified detection
        indicators = ["use_tool:", "call_function:", "execute:", "<<TOOL>>"]
        return any(indicator in content.lower() for indicator in indicators)
    
    def _extract_function_call(self, content: str) -> Optional[FunctionCallRequest]:
        """
        Extract function call from LLM response (simplified).
        
        In production, you'd parse the structured function_call
        response from the LLM provider.
        """
        # This is a placeholder. In real implementation,
        # you'd parse the actual function call from the response
        try:
            # Look for JSON-like structure
            if "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
                data = json.loads(json_str)
                
                if "name" in data and "arguments" in data:
                    return FunctionCallRequest(
                        name=data["name"],
                        arguments=data.get("arguments", {})
                    )
        except:
            pass
        
        return None
    
    def clear_history(self):
        """Clear conversation history."""
        self.messages = []

