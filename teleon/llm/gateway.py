"""LLM Gateway - Central interface for all LLM operations."""

from typing import List, Dict, Optional, AsyncIterator
import hashlib
import json
import time
import os

from teleon.llm.types import (
    LLMMessage, LLMResponse, LLMConfig, ProviderConfig, LLMRequest
)
from teleon.llm.providers.base import LLMProvider
from teleon.llm.cache import ResponseCache

# Try to import agent reporter for metrics
try:
    from teleon.helix.agent_reporter import get_agent_reporter
    AGENT_REPORTER_AVAILABLE = True
except ImportError:
    AGENT_REPORTER_AVAILABLE = False

# Model pricing for cost calculation (per 1K tokens)
MODEL_PRICING = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "gemini-pro": {"input": 0.00025, "output": 0.0005},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
}


class LLMGateway:
    """
    Central gateway for all LLM operations.
    
    Features:
    - Multi-provider support (OpenAI, Anthropic, Google, etc.)
    - Intelligent provider selection
    - Response caching
    - Cost tracking
    - Automatic retry and fallback
    - Automatic metrics reporting to Teleon Platform
    """
    
    def __init__(self, enable_metrics_reporting: bool = True):
        """
        Initialize the LLM Gateway.
        
        Args:
            enable_metrics_reporting: Whether to report metrics to Teleon Platform
        """
        self.providers: Dict[str, LLMProvider] = {}
        self.cache: Optional[ResponseCache] = None
        self.total_cost = 0.0
        self.total_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0
        self.error_count = 0
        
        # Metrics reporting
        self._enable_metrics = enable_metrics_reporting and AGENT_REPORTER_AVAILABLE
        self._metrics_enabled_env = os.getenv("TELEON_METRICS_ENABLED", "true").lower() == "true"
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the cost of a request based on model pricing.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        
        Returns:
            Cost in USD
        """
        # Find matching pricing (try exact match, then prefix match)
        pricing = MODEL_PRICING.get(model)
        
        if not pricing:
            # Try prefix matching
            for model_name, prices in MODEL_PRICING.items():
                if model.startswith(model_name) or model_name.startswith(model.split("-")[0]):
                    pricing = prices
                    break
        
        if not pricing:
            # Default pricing (GPT-4 level)
            pricing = {"input": 0.03, "output": 0.06}
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    async def _report_request_metrics(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool,
        cost: float,
        error_type: Optional[str] = None
    ):
        """Report request metrics to Teleon Platform."""
        if not (self._enable_metrics and self._metrics_enabled_env):
            return
        
        try:
            reporter = get_agent_reporter()
            await reporter.report_request(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                model=model,
                success=success,
                cost=cost,
                error_type=error_type
            )
        except Exception:
            # Don't let metrics reporting break the main flow
            pass
    
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
        start_time = time.time()
        
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
        error_type = None
        try:
            response = await selected_provider.complete(messages, config)
            success = True
        except Exception as e:
            self.error_count += 1
            error_type = type(e).__name__
            
            # Report failed request
            latency_ms = (time.time() - start_time) * 1000
            await self._report_request_metrics(
                model=config.model,
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                success=False,
                cost=0.0,
                error_type=error_type
            )
            raise
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract token counts
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        
        # Calculate cost if not provided
        cost = response.cost if response.cost else self._calculate_cost(
            config.model, input_tokens, output_tokens
        )
        
        # Update statistics
        self.total_cost += cost
        self.total_tokens += response.usage.total_tokens if response.usage else 0
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.request_count += 1
        
        # Report metrics to Teleon Platform
        await self._report_request_metrics(
            model=config.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            success=True,
            cost=cost
        )
        
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
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "success_rate": (
                (self.request_count - self.error_count) / max(self.request_count, 1)
            ),
            "avg_cost_per_request": self.total_cost / max(self.request_count, 1),
            "avg_tokens_per_request": self.total_tokens / max(self.request_count, 1),
            "providers": list(self.providers.keys()),
            "cache_enabled": self.cache is not None,
            "metrics_reporting_enabled": self._enable_metrics and self._metrics_enabled_env
        }
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self.total_cost = 0.0
        self.total_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0
        self.error_count = 0


# Global gateway instance (singleton)
_global_gateway: Optional[LLMGateway] = None


def _configure_gateway_from_env(gateway: LLMGateway) -> None:
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if openai_key and "openai" not in gateway.providers:
        from teleon.llm.providers.openai import OpenAIProvider
        provider_config = ProviderConfig(
            name="openai",
            api_key=openai_key,
        )
        gateway.register_provider(OpenAIProvider(provider_config))

    if anthropic_key and "anthropic" not in gateway.providers:
        from teleon.llm.providers.anthropic import AnthropicProvider
        provider_config = ProviderConfig(
            name="anthropic",
            api_key=anthropic_key,
        )
        gateway.register_provider(AnthropicProvider(provider_config))


def get_gateway() -> LLMGateway:
    """
    Get the global LLM Gateway instance.
    
    Returns:
        Global gateway instance
    """
    global _global_gateway
    if _global_gateway is None:
        _global_gateway = LLMGateway()
    _configure_gateway_from_env(_global_gateway)
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

