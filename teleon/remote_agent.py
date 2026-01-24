"""
Remote Agent - Interact with deployed Teleon agents from your applications
"""

from typing import Dict, Any, Optional, List
import httpx
import asyncio
from datetime import datetime, timedelta


class RemoteAgent:
    """
    A remote agent proxy for interacting with deployed Teleon agents.
    
    Usage:
        client = TeleonClient(api_key="...")
        agent = await client.get_agent("customer-support")
        result = await agent.execute(input_data={...})
    """
    
    def __init__(
        self,
        agent_name: str,
        agent_id: str,
        api_key: str,
        base_url: str = "https://api.teleon.ai"
    ):
        """
        Initialize remote agent proxy.
        
        Args:
            agent_name: Name of the agent
            agent_id: Unique agent ID
            api_key: Teleon API key
            base_url: Base URL for Teleon API
        """
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        
        # Initialize HTTP client
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=httpx.Timeout(60.0)
        )
        
        # Initialize sub-interfaces
        self.cortex = CortexInterface(self)
    
    async def execute(
        self,
        input_data: Any,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the deployed agent with input.
        
        Args:
            input_data: Input data for the agent (can be string, dict, list, etc.)
            temperature: Override temperature
            max_tokens: Override max tokens
            metadata: Additional metadata to pass
        
        Returns:
            Agent execution result with status, response, metrics, etc.
        """
        payload = {
            "input": input_data,
            "metadata": metadata or {}
        }
        
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        try:
            response = await self.client.post(
                f"{self.base_url}/agents/{self.agent_id}/execute",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise AgentExecutionError(
                f"Agent execution failed: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            raise AgentExecutionError(f"Agent execution failed: {str(e)}")
    
    async def execute_batch(
        self,
        inputs: List[Any],
        parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute agent with multiple inputs.
        
        Args:
            inputs: List of input data
            parallel: Execute in parallel (default: True)
        
        Returns:
            List of execution results
        """
        if parallel:
            tasks = [self.execute(input_data) for input_data in inputs]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for input_data in inputs:
                result = await self.execute(input_data)
                results.append(result)
            return results
    
    async def get_metrics(
        self,
        period: str = "24h"
    ) -> Dict[str, Any]:
        """
        Get agent metrics for a time period.
        
        Args:
            period: Time period (e.g., "1h", "24h", "7d", "30d")
        
        Returns:
            Metrics including success rate, response time, costs, etc.
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/agents/{self.agent_id}/metrics",
                params={"period": period}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise AgentMetricsError(f"Failed to fetch metrics: {str(e)}")
    
    async def get_info(self) -> Dict[str, Any]:
        """
        Get agent information and configuration.
        
        Returns:
            Agent info including model, config, status, etc.
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/agents/{self.agent_id}/info"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise AgentInfoError(f"Failed to fetch agent info: {str(e)}")
    
    async def get_logs(
        self,
        limit: int = 100,
        level: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get agent execution logs.
        
        Args:
            limit: Maximum number of logs to return
            level: Filter by log level (DEBUG, INFO, WARNING, ERROR)
        
        Returns:
            List of log entries
        """
        params = {"limit": limit}
        if level:
            params["level"] = level
        
        try:
            response = await self.client.get(
                f"{self.base_url}/agents/{self.agent_id}/logs",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise AgentLogsError(f"Failed to fetch logs: {str(e)}")
    
    async def update_config(
        self,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update agent configuration.
        
        Args:
            config: Configuration updates (temperature, max_tokens, etc.)
        
        Returns:
            Updated configuration
        """
        try:
            response = await self.client.patch(
                f"{self.base_url}/agents/{self.agent_id}/config",
                json=config
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise AgentConfigError(f"Failed to update config: {str(e)}")
    
    async def scale(
        self,
        replicas: int
    ) -> Dict[str, Any]:
        """
        Scale agent replicas.
        
        Args:
            replicas: Number of replicas to scale to
        
        Returns:
            Scaling status
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/agents/{self.agent_id}/scale",
                json={"replicas": replicas}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise AgentScalingError(f"Failed to scale agent: {str(e)}")
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    def __repr__(self):
        return f"RemoteAgent(name={self.agent_name}, id={self.agent_id})"


class CortexInterface:
    """
    Interface for interacting with agent's Cortex memory.
    """
    
    def __init__(self, agent: RemoteAgent):
        self.agent = agent
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        memory_type: str = "semantic"
    ) -> List[Dict[str, Any]]:
        """
        Search agent's memory.
        
        Args:
            query: Search query
            limit: Maximum results
            memory_type: Type of memory (semantic, episodic, procedural)
        
        Returns:
            List of memory entries
        """
        try:
            response = await self.agent.client.post(
                f"{self.agent.base_url}/agents/{self.agent.agent_id}/memory/search",
                json={
                    "query": query,
                    "limit": limit,
                    "type": memory_type
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise CortexError(f"Memory search failed: {str(e)}")
    
    async def store(
        self,
        content: str,
        memory_type: str = "semantic",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store data in agent's memory.
        
        Args:
            content: Content to store
            memory_type: Type of memory
            metadata: Additional metadata
        
        Returns:
            Storage confirmation
        """
        try:
            response = await self.agent.client.post(
                f"{self.agent.base_url}/agents/{self.agent.agent_id}/memory/store",
                json={
                    "content": content,
                    "type": memory_type,
                    "metadata": metadata or {}
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise CortexError(f"Memory store failed: {str(e)}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Memory usage statistics
        """
        try:
            response = await self.agent.client.get(
                f"{self.agent.base_url}/agents/{self.agent.agent_id}/memory/stats"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise CortexError(f"Failed to fetch memory stats: {str(e)}")


# Exceptions
class AgentExecutionError(Exception):
    """Raised when agent execution fails."""
    pass


class AgentMetricsError(Exception):
    """Raised when fetching metrics fails."""
    pass


class AgentInfoError(Exception):
    """Raised when fetching agent info fails."""
    pass


class AgentLogsError(Exception):
    """Raised when fetching logs fails."""
    pass


class AgentConfigError(Exception):
    """Raised when updating config fails."""
    pass


class AgentScalingError(Exception):
    """Raised when scaling fails."""
    pass


class CortexError(Exception):
    """Raised when Cortex operations fail."""
    pass

