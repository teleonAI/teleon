"""OpenAI provider implementation."""

from typing import List, AsyncIterator, Dict, Any
import asyncio
import time

from teleon.llm.providers.base import LLMProvider
from teleon.llm.types import LLMMessage, LLMResponse, LLMConfig, LLMUsage, ProviderConfig


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation."""
    
    # Model pricing (cost per 1K tokens)
    MODEL_PRICING = {
        "gpt-4": {
            "cost_per_1k_input": 0.03,
            "cost_per_1k_output": 0.06,
            "context_length": 8192
        },
        "gpt-4-turbo": {
            "cost_per_1k_input": 0.01,
            "cost_per_1k_output": 0.03,
            "context_length": 128000
        },
        "gpt-4-turbo-preview": {
            "cost_per_1k_input": 0.01,
            "cost_per_1k_output": 0.03,
            "context_length": 128000
        },
        "gpt-3.5-turbo": {
            "cost_per_1k_input": 0.0005,
            "cost_per_1k_output": 0.0015,
            "context_length": 16385
        },
        "gpt-3.5-turbo-16k": {
            "cost_per_1k_input": 0.001,
            "cost_per_1k_output": 0.002,
            "context_length": 16385
        }
    }
    
    def __init__(self, config: ProviderConfig):
        """Initialize OpenAI provider."""
        super().__init__(config)
        self._client = None
    
    def _get_client(self):
        """Get or create OpenAI client (lazy initialization)."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.api_base,
                    organization=self.config.organization,
                    timeout=self.config.timeout
                )
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install it with: pip install openai"
                )
        return self._client
    
    async def complete(
        self,
        messages: List[LLMMessage],
        config: LLMConfig
    ) -> LLMResponse:
        """Generate a completion using OpenAI."""
        client = self._get_client()
        start_time = time.time()
        
        # Convert messages to OpenAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        # Prepare request parameters
        params = {
            "model": config.model,
            "messages": openai_messages,
            "temperature": config.temperature,
        }
        
        if config.max_tokens:
            params["max_tokens"] = config.max_tokens
        if config.top_p is not None:
            params["top_p"] = config.top_p
        if config.frequency_penalty is not None:
            params["frequency_penalty"] = config.frequency_penalty
        if config.presence_penalty is not None:
            params["presence_penalty"] = config.presence_penalty
        if config.stop:
            params["stop"] = config.stop
        if config.functions:
            params["functions"] = config.functions
        if config.function_call:
            params["function_call"] = config.function_call
        
        # Make the API call
        try:
            response = await client.chat.completions.create(**params)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract response data
            choice = response.choices[0]
            content = choice.message.content or ""
            
            # Extract usage
            usage = LLMUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
            
            # Calculate cost
            cost = self.calculate_cost(
                usage.prompt_tokens,
                usage.completion_tokens,
                config.model
            )
            
            return LLMResponse(
                content=content,
                model=config.model,
                provider=self.name,
                usage=usage,
                finish_reason=choice.finish_reason,
                latency_ms=latency_ms,
                cost=cost,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
            )
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}") from e
    
    async def stream_complete(
        self,
        messages: List[LLMMessage],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """Stream a completion using OpenAI."""
        client = self._get_client()
        
        # Convert messages to OpenAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        # Prepare request parameters
        params = {
            "model": config.model,
            "messages": openai_messages,
            "temperature": config.temperature,
            "stream": True
        }
        
        if config.max_tokens:
            params["max_tokens"] = config.max_tokens
        
        try:
            stream = await client.chat.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise RuntimeError(f"OpenAI streaming error: {str(e)}") from e
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about an OpenAI model."""
        return self.MODEL_PRICING.get(model, {
            "cost_per_1k_input": 0.0,
            "cost_per_1k_output": 0.0,
            "context_length": 4096
        })

