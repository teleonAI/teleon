"""LLM Gateway - Central interface for all LLM operations."""

from typing import List, Dict, Optional, AsyncIterator
import hashlib
import json

from teleon.llm.types import (
    LLMMessage, LLMResponse, LLMConfig, ProviderConfig, LLMRequest
)
from teleon.llm.providers.base import LLMProvider
from teleon.llm.cache import ResponseCache


class LLMGateway:
    """
    Central gateway for all LLM operations.
    
    Features:
    - Multi-provider support (OpenAI, Anthropic, Google, etc.)
    - Intelligent provider selection
    - Response caching
    - Cost tracking
    - Automatic retry and fallback
    """
    
    def __init__(self):
        """Initialize the LLM Gateway."""
        self.providers: Dict[str, LLMProvider] = {}
        self.cache: Optional[ResponseCache] = None
        self.total_cost = 0.0
        self.total_tokens = 0
        self.request_count = 0
    
    def register_provider(self, provider: LLMProvider) -> None:
        """
        Register an LLM provider.
        
        Args:
            provider: Provider instance to register
        """
        self.providers[provider.name] = provider
        print(f"✓ Registered LLM provider: {provider.name}")
    
    def enable_cache(self, cache: ResponseCache) -> None:
        """
        Enable response caching.
        
        Args:
            cache: Cache instance to use
        """
        self.cache = cache
        print("✓ Response caching enabled")
    
    async def complete(
        self,
        messages: List[LLMMessage],
        config: LLMConfig,
        provider: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate a completion.
        
        Args:
            messages: List of conversation messages
            config: Request configuration
            provider: Specific provider to use (optional)
        
        Returns:
            LLM response
        """
        # Check cache first
        if config.use_cache and self.cache:
            cache_key = self._generate_cache_key(messages, config)
            cached_response = await self.cache.get(cache_key)
            
            if cached_response:
                cached_response.cached = True
                return cached_response
        
        # Select provider
        selected_provider = self._select_provider(config.model, provider)
        
        if not selected_provider:
            raise ValueError(f"No provider available for model: {config.model}")
        
        # Generate completion
        response = await selected_provider.complete(messages, config)
        
        # Update statistics
        self.total_cost += response.cost or 0.0
        self.total_tokens += response.usage.total_tokens
        self.request_count += 1
        
        # Cache the response
        if config.use_cache and self.cache and config.cache_ttl:
            cache_key = self._generate_cache_key(messages, config)
            await self.cache.set(cache_key, response, ttl=config.cache_ttl)
        
        return response
    
    async def stream_complete(
        self,
        messages: List[LLMMessage],
        config: LLMConfig,
        provider: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Stream a completion.
        
        Args:
            messages: List of conversation messages
            config: Request configuration
            provider: Specific provider to use (optional)
        
        Yields:
            Content chunks as they're generated
        """
        # Select provider
        selected_provider = self._select_provider(config.model, provider)
        
        if not selected_provider:
            raise ValueError(f"No provider available for model: {config.model}")
        
        # Stream completion
        async for chunk in selected_provider.stream_complete(messages, config):
            yield chunk
        
        # Update statistics (approximate, as we don't have exact token counts)
        self.request_count += 1
    
    def _select_provider(
        self,
        model: str,
        preferred_provider: Optional[str] = None
    ) -> Optional[LLMProvider]:
        """
        Select the best provider for a given model.
        
        Args:
            model: Model name
            preferred_provider: Preferred provider name
        
        Returns:
            Selected provider or None
        """
        # If a specific provider is requested, use it
        if preferred_provider and preferred_provider in self.providers:
            return self.providers[preferred_provider]
        
        # Auto-detect provider based on model name
        if model.startswith("gpt-"):
            return self.providers.get("openai")
        elif model.startswith("claude-"):
            return self.providers.get("anthropic")
        elif model.startswith("gemini-"):
            return self.providers.get("google")
        
        # Fallback to first available provider
        if self.providers:
            return list(self.providers.values())[0]
        
        return None
    
    def _generate_cache_key(
        self,
        messages: List[LLMMessage],
        config: LLMConfig
    ) -> str:
        """
        Generate a cache key for a request.
        
        Args:
            messages: Conversation messages
            config: Request configuration
        
        Returns:
            Cache key (hash)
        """
        # Create a deterministic representation
        cache_data = {
            "messages": [msg.dict() for msg in messages],
            "model": config.model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get gateway statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
            "avg_cost_per_request": self.total_cost / max(self.request_count, 1),
            "providers": list(self.providers.keys()),
            "cache_enabled": self.cache is not None
        }
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self.total_cost = 0.0
        self.total_tokens = 0
        self.request_count = 0


# Global gateway instance (singleton)
_global_gateway: Optional[LLMGateway] = None


def get_gateway() -> LLMGateway:
    """
    Get the global LLM Gateway instance.
    
    Returns:
        Global gateway instance
    """
    global _global_gateway
    if _global_gateway is None:
        _global_gateway = LLMGateway()
    return _global_gateway


def configure_gateway(
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    enable_cache: bool = True,
    cache_ttl: int = 3600
) -> LLMGateway:
    """
    Configure the global LLM Gateway.
    
    Args:
        openai_api_key: OpenAI API key
        anthropic_api_key: Anthropic API key
        enable_cache: Whether to enable caching
        cache_ttl: Default cache TTL in seconds
    
    Returns:
        Configured gateway instance
    """
    gateway = get_gateway()
    
    # Register providers
    if openai_api_key:
        from teleon.llm.providers.openai import OpenAIProvider
        provider_config = ProviderConfig(
            name="openai",
            api_key=openai_api_key
        )
        gateway.register_provider(OpenAIProvider(provider_config))
    
    if anthropic_api_key:
        from teleon.llm.providers.anthropic import AnthropicProvider
        provider_config = ProviderConfig(
            name="anthropic",
            api_key=anthropic_api_key
        )
        gateway.register_provider(AnthropicProvider(provider_config))
    
    # Enable caching
    if enable_cache:
        from teleon.llm.cache import InMemoryCache
        cache = InMemoryCache(default_ttl=cache_ttl)
        gateway.enable_cache(cache)
    
    return gateway

