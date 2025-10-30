"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import List, AsyncIterator, Optional
import time

from teleon.llm.types import LLMMessage, LLMResponse, LLMConfig, ProviderConfig


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All provider implementations (OpenAI, Anthropic, Google, etc.)
    must inherit from this class and implement the required methods.
    """
    
    def __init__(self, config: ProviderConfig):
        """
        Initialize the provider.
        
        Args:
            config: Provider configuration
        """
        self.config = config
        self.name = config.name
    
    @abstractmethod
    async def complete(
        self,
        messages: List[LLMMessage],
        config: LLMConfig
    ) -> LLMResponse:
        """
        Generate a completion for the given messages.
        
        Args:
            messages: List of conversation messages
            config: Request configuration
        
        Returns:
            LLM response
        """
        pass
    
    @abstractmethod
    async def stream_complete(
        self,
        messages: List[LLMMessage],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """
        Stream a completion for the given messages.
        
        Args:
            messages: List of conversation messages
            config: Request configuration
        
        Yields:
            Content chunks as they're generated
        """
        pass
    
    @abstractmethod
    def get_model_info(self, model: str) -> dict:
        """
        Get information about a specific model.
        
        Args:
            model: Model name
        
        Returns:
            Model information (context length, cost, etc.)
        """
        pass
    
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str
    ) -> float:
        """
        Calculate the cost of a request.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name
        
        Returns:
            Cost in USD
        """
        model_info = self.get_model_info(model)
        
        input_cost = (input_tokens / 1000) * model_info.get("cost_per_1k_input", 0.0)
        output_cost = (output_tokens / 1000) * model_info.get("cost_per_1k_output", 0.0)
        
        return input_cost + output_cost
    
    async def _execute_with_retry(
        self,
        func,
        *args,
        **kwargs
    ):
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Function result
        """
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    # Exponential backoff
                    wait_time = (2 ** attempt) * 0.5
                    await asyncio.sleep(wait_time)
                else:
                    raise last_error
        
        raise last_error


# Import asyncio for retry logic
import asyncio

