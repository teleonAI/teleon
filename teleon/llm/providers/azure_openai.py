"""Azure OpenAI Provider - Production-grade Azure OpenAI integration."""

import os
from typing import List, Optional, AsyncIterator
import httpx
from datetime import datetime

from teleon.llm.providers.base import LLMProvider
from teleon.llm.types import LLMMessage, LLMResponse, LLMConfig, LLMUsage
from teleon.core.exceptions import LLMError


class AzureOpenAIProvider(LLMProvider):
    """
    Azure OpenAI Service provider.
    
    Features:
    - Native Azure OpenAI API support
    - Managed identity authentication
    - Regional deployment support
    - Enterprise security and compliance
    - Cost optimization for Azure workloads
    """
    
    name = "azure_openai"
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment_name: Optional[str] = None,
        use_managed_identity: bool = False
    ):
        """
        Initialize Azure OpenAI provider.
        
        Args:
            endpoint: Azure OpenAI endpoint (e.g., https://your-resource.openai.azure.com/)
            api_key: Azure OpenAI API key (if not using managed identity)
            api_version: Azure OpenAI API version
            deployment_name: Default deployment name (model name in Azure)
            use_managed_identity: Use Azure Managed Identity for authentication
        """
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("API_VERSION") or "2024-02-15-preview"
        self.deployment_name = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("MODEL_DEPLOYMENT")
        self.use_managed_identity = use_managed_identity
        
        if not self.endpoint:
            raise ValueError("Azure OpenAI endpoint is required. Set AZURE_OPENAI_ENDPOINT or AZURE_ENDPOINT")
        
        if not self.use_managed_identity and not self.api_key:
            raise ValueError("Azure OpenAI API key is required (or enable managed identity)")
        
        # Parse endpoint if it includes the full path
        if "/openai/deployments/" in self.endpoint:
            # Extract base endpoint and deployment from full URL
            parts = self.endpoint.split("/openai/deployments/")
            self.endpoint = parts[0]
            if not self.deployment_name and len(parts) > 1:
                # Extract deployment name from URL
                deployment_part = parts[1].split("/")[0]
                self.deployment_name = deployment_part
        
        # Remove trailing slash from endpoint
        self.endpoint = self.endpoint.rstrip('/')
        
        # Initialize HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0)
        )
    
    async def _get_auth_headers(self) -> dict:
        """Get authentication headers (API key or managed identity)."""
        if self.use_managed_identity:
            # Use Azure Managed Identity
            try:
                from azure.identity.aio import DefaultAzureCredential
                credential = DefaultAzureCredential()
                token = await credential.get_token("https://cognitiveservices.azure.com/.default")
                return {"Authorization": f"Bearer {token.token}"}
            except ImportError:
                raise LLMError(
                    "azure-identity package required for managed identity. Install: pip install azure-identity",
                    code="AZURE_IDENTITY_MISSING"
                )
        else:
            return {"api-key": self.api_key}
    
    def _get_deployment_name(self, model: str) -> str:
        """Get Azure deployment name for a model."""
        # If deployment name is explicitly set, use it
        if self.deployment_name:
            return self.deployment_name
        
        # Map OpenAI model names to common Azure deployment names
        deployment_map = {
            "gpt-4": "gpt-4",
            "gpt-4-turbo": "gpt-4-turbo",
            "gpt-4-32k": "gpt-4-32k",
            "gpt-35-turbo": "gpt-35-turbo",
            "gpt-3.5-turbo": "gpt-35-turbo",
            "text-embedding-ada-002": "text-embedding-ada-002",
        }
        
        return deployment_map.get(model, model)
    
    async def complete(
        self,
        messages: List[LLMMessage],
        config: LLMConfig
    ) -> LLMResponse:
        """
        Generate completion using Azure OpenAI.
        
        Args:
            messages: Conversation messages
            config: Request configuration
        
        Returns:
            LLM response
        """
        deployment = self._get_deployment_name(config.model)
        url = f"{self.endpoint}/openai/deployments/{deployment}/chat/completions?api-version={self.api_version}"
        
        headers = await self._get_auth_headers()
        headers["Content-Type"] = "application/json"
        
        # Prepare request payload
        payload = {
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
            "stream": False
        }
        
        # Add tools/functions if provided
        if hasattr(config, 'tools') and config.tools:
            payload["tools"] = config.tools
            if hasattr(config, 'tool_choice') and config.tool_choice:
                payload["tool_choice"] = config.tool_choice
        
        try:
            response = await self.client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            choice = data["choices"][0]
            message = choice["message"]
            
            # Extract usage
            usage_data = data.get("usage", {})
            usage = LLMUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            )
            
            # Calculate cost (Azure OpenAI pricing)
            cost = self._calculate_cost(config.model, usage)
            
            return LLMResponse(
                content=message.get("content", ""),
                model=config.model,
                provider="azure_openai",
                usage=usage,
                finish_reason=choice.get("finish_reason"),
                latency_ms=response.elapsed.total_seconds() * 1000,
                cost=cost,
                cached=False
            )
            
        except httpx.HTTPStatusError as e:
            raise LLMError(
                f"Azure OpenAI API error: {e.response.status_code} - {e.response.text}",
                code="AZURE_API_ERROR"
            )
        except Exception as e:
            raise LLMError(f"Azure OpenAI request failed: {str(e)}", code="REQUEST_FAILED")
    
    async def stream_complete(
        self,
        messages: List[LLMMessage],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """
        Stream completion from Azure OpenAI.
        
        Args:
            messages: Conversation messages
            config: Request configuration
        
        Yields:
            Content chunks
        """
        deployment = self._get_deployment_name(config.model)
        url = f"{self.endpoint}/openai/deployments/{deployment}/chat/completions?api-version={self.api_version}"
        
        headers = await self._get_auth_headers()
        headers["Content-Type"] = "application/json"
        
        payload = {
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "stream": True
        }
        
        try:
            async with self.client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        
                        if data == "[DONE]":
                            break
                        
                        try:
                            import json
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"]
                            
                            if "content" in delta:
                                yield delta["content"]
                        except:
                            continue
                            
        except Exception as e:
            raise LLMError(f"Azure OpenAI streaming failed: {str(e)}", code="STREAM_FAILED")
    
    def _calculate_cost(self, model: str, usage: LLMUsage) -> float:
        """
        Calculate cost based on Azure OpenAI pricing.
        
        Note: Prices may vary by region and contract.
        These are standard pay-as-you-go prices (USD).
        """
        # Azure OpenAI pricing per 1K tokens
        pricing = {
            "gpt-4": {
                "prompt": 0.03,
                "completion": 0.06
            },
            "gpt-4-turbo": {
                "prompt": 0.01,
                "completion": 0.03
            },
            "gpt-4-32k": {
                "prompt": 0.06,
                "completion": 0.12
            },
            "gpt-35-turbo": {
                "prompt": 0.0015,
                "completion": 0.002
            },
            "gpt-3.5-turbo": {
                "prompt": 0.0015,
                "completion": 0.002
            }
        }
        
        # Get pricing for model (default to GPT-3.5 pricing)
        model_pricing = pricing.get(model, pricing["gpt-35-turbo"])
        
        # Calculate cost
        prompt_cost = (usage.prompt_tokens / 1000) * model_pricing["prompt"]
        completion_cost = (usage.completion_tokens / 1000) * model_pricing["completion"]
        
        return prompt_cost + completion_cost
    
    def get_model_info(self, model: str) -> dict:
        """
        Get model information for cost calculation and capabilities.
        
        Args:
            model: Model name or deployment name
        
        Returns:
            Model information dictionary
        """
        # Azure OpenAI model information
        model_info = {
            "gpt-4": {
                "context_length": 8192,
                "cost_per_1k_input": 0.03,
                "cost_per_1k_output": 0.06,
                "supports_functions": True
            },
            "gpt-4-turbo": {
                "context_length": 128000,
                "cost_per_1k_input": 0.01,
                "cost_per_1k_output": 0.03,
                "supports_functions": True
            },
            "gpt-4-32k": {
                "context_length": 32768,
                "cost_per_1k_input": 0.06,
                "cost_per_1k_output": 0.12,
                "supports_functions": True
            },
            "gpt-35-turbo": {
                "context_length": 16385,
                "cost_per_1k_input": 0.0015,
                "cost_per_1k_output": 0.002,
                "supports_functions": True
            },
            "gpt-3.5-turbo": {
                "context_length": 16385,
                "cost_per_1k_input": 0.0015,
                "cost_per_1k_output": 0.002,
                "supports_functions": True
            }
        }
        
        # Return info for known models, or default for custom deployments (like gpt-5-cogeo)
        return model_info.get(model, {
            "context_length": 128000,
            "cost_per_1k_input": 0.01,
            "cost_per_1k_output": 0.03,
            "supports_functions": True
        })
    
    def supports_model(self, model: str) -> bool:
        """Check if provider supports a model."""
        # Azure OpenAI supports any deployment name, so always return True
        return True
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

