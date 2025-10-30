"""Anthropic provider implementation."""

from typing import List, AsyncIterator, Dict, Any
import time

from teleon.llm.providers.base import LLMProvider
from teleon.llm.types import LLMMessage, LLMResponse, LLMConfig, LLMUsage, ProviderConfig


class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) LLM provider implementation."""
    
    # Model pricing (cost per 1K tokens)
    MODEL_PRICING = {
        "claude-3-opus-20240229": {
            "cost_per_1k_input": 0.015,
            "cost_per_1k_output": 0.075,
            "context_length": 200000
        },
        "claude-3-sonnet-20240229": {
            "cost_per_1k_input": 0.003,
            "cost_per_1k_output": 0.015,
            "context_length": 200000
        },
        "claude-3-haiku-20240307": {
            "cost_per_1k_input": 0.00025,
            "cost_per_1k_output": 0.00125,
            "context_length": 200000
        },
        "claude-2.1": {
            "cost_per_1k_input": 0.008,
            "cost_per_1k_output": 0.024,
            "context_length": 200000
        },
        "claude-2.0": {
            "cost_per_1k_input": 0.008,
            "cost_per_1k_output": 0.024,
            "context_length": 100000
        }
    }
    
    # Simplified model aliases
    MODEL_ALIASES = {
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307"
    }
    
    def __init__(self, config: ProviderConfig):
        """Initialize Anthropic provider."""
        super().__init__(config)
        self._client = None
    
    def _get_client(self):
        """Get or create Anthropic client (lazy initialization)."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(
                    api_key=self.config.api_key,
                    timeout=self.config.timeout
                )
            except ImportError:
                raise ImportError(
                    "Anthropic package not installed. Install it with: pip install anthropic"
                )
        return self._client
    
    def _resolve_model(self, model: str) -> str:
        """Resolve model alias to full model name."""
        return self.MODEL_ALIASES.get(model, model)
    
    def _convert_messages(self, messages: List[LLMMessage]) -> tuple:
        """
        Convert Teleon messages to Anthropic format.
        
        Anthropic requires system messages to be separate from the messages list.
        
        Returns:
            Tuple of (system_message, anthropic_messages)
        """
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        return system_message, anthropic_messages
    
    async def complete(
        self,
        messages: List[LLMMessage],
        config: LLMConfig
    ) -> LLMResponse:
        """Generate a completion using Anthropic."""
        client = self._get_client()
        start_time = time.time()
        
        # Resolve model name
        model = self._resolve_model(config.model)
        
        # Convert messages
        system_message, anthropic_messages = self._convert_messages(messages)
        
        # Prepare request parameters
        params = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": config.max_tokens or 4096,  # Required by Anthropic
            "temperature": config.temperature,
        }
        
        if system_message:
            params["system"] = system_message
        
        if config.top_p is not None:
            params["top_p"] = config.top_p
        
        if config.stop:
            params["stop_sequences"] = config.stop
        
        # Make the API call
        try:
            response = await client.messages.create(**params)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract content
            content = response.content[0].text if response.content else ""
            
            # Extract usage
            usage = LLMUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens
            )
            
            # Calculate cost
            cost = self.calculate_cost(
                usage.prompt_tokens,
                usage.completion_tokens,
                model
            )
            
            return LLMResponse(
                content=content,
                model=model,
                provider=self.name,
                usage=usage,
                finish_reason=response.stop_reason,
                latency_ms=latency_ms,
                cost=cost,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
            )
            
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}") from e
    
    async def stream_complete(
        self,
        messages: List[LLMMessage],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """Stream a completion using Anthropic."""
        client = self._get_client()
        
        # Resolve model name
        model = self._resolve_model(config.model)
        
        # Convert messages
        system_message, anthropic_messages = self._convert_messages(messages)
        
        # Prepare request parameters
        params = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": config.max_tokens or 4096,
            "temperature": config.temperature,
            "stream": True
        }
        
        if system_message:
            params["system"] = system_message
        
        try:
            async with client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            raise RuntimeError(f"Anthropic streaming error: {str(e)}") from e
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about an Anthropic model."""
        model = self._resolve_model(model)
        return self.MODEL_PRICING.get(model, {
            "cost_per_1k_input": 0.0,
            "cost_per_1k_output": 0.0,
            "context_length": 100000
        })

