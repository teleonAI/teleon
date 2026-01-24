"""Type definitions for LLM Gateway."""

from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class LLMMessage(BaseModel):
    """Represents a message in an LLM conversation."""
    
    role: Literal["system", "user", "assistant", "function"] = Field(
        ..., description="Role of the message sender"
    )
    content: str = Field(..., description="Content of the message")
    name: Optional[str] = Field(None, description="Name of the function (for function messages)")
    function_call: Optional[Dict[str, Any]] = Field(None, description="Function call data")


class LLMUsage(BaseModel):
    """Token usage information from an LLM call."""
    
    prompt_tokens: int = Field(0, description="Number of tokens in the prompt")
    completion_tokens: int = Field(0, description="Number of tokens in the completion")
    total_tokens: int = Field(0, description="Total number of tokens used")


class LLMResponse(BaseModel):
    """Response from an LLM call."""
    
    content: str = Field(..., description="Generated content from the LLM")
    model: str = Field(..., description="Model that generated the response")
    provider: str = Field(..., description="Provider that served the request")
    usage: LLMUsage = Field(default_factory=LLMUsage, description="Token usage information")
    finish_reason: Optional[str] = Field(None, description="Reason the generation finished")
    
    # Metadata
    latency_ms: Optional[float] = Field(None, description="Response latency in milliseconds")
    cost: Optional[float] = Field(None, description="Estimated cost of the request in USD")
    cached: bool = Field(False, description="Whether this response came from cache")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Response timestamp")
    
    # Raw response (for debugging)
    raw_response: Optional[Dict[str, Any]] = Field(None, description="Raw provider response")


class LLMConfig(BaseModel):
    """Configuration for LLM requests."""
    
    model: str = Field(..., description="Model to use (e.g., 'gpt-4', 'claude-3-opus')")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0, description="Presence penalty")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    
    # Function calling
    functions: Optional[List[Dict[str, Any]]] = Field(None, description="Available functions")
    function_call: Optional[str] = Field(None, description="Function call behavior")
    
    # Streaming
    stream: bool = Field(False, description="Whether to stream the response")
    
    # Caching
    cache_ttl: Optional[int] = Field(None, description="Cache TTL in seconds")
    use_cache: bool = Field(True, description="Whether to use cached responses")


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    
    name: str = Field(..., description="Provider name")
    api_key: str = Field(..., description="API key for the provider")
    api_base: Optional[str] = Field(None, description="Custom API base URL")
    organization: Optional[str] = Field(None, description="Organization ID")
    timeout: int = Field(30, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")
    
    # Cost per 1K tokens (can be model-specific)
    cost_per_1k_input_tokens: Optional[float] = Field(None, description="Input token cost")
    cost_per_1k_output_tokens: Optional[float] = Field(None, description="Output token cost")


class LLMRequest(BaseModel):
    """Request to the LLM Gateway."""
    
    messages: List[LLMMessage] = Field(..., description="Conversation messages")
    config: LLMConfig = Field(..., description="Request configuration")
    provider: Optional[str] = Field(None, description="Specific provider to use")
    
    # Metadata
    request_id: Optional[str] = Field(None, description="Unique request ID")
    user_id: Optional[str] = Field(None, description="User ID for tracking")
    tags: Optional[Dict[str, str]] = Field(None, description="Custom tags for tracking")

