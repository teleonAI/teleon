"""
Sentinel Registry - Global Sentinel Instance Management.

Manages active Sentinel instances per agent for CLI and API access.
"""

from typing import Dict, Optional
import asyncio
from teleon.sentinel.engine import SentinelEngine
from teleon.core import StructuredLogger, LogLevel


class SentinelRegistry:
    """
    Global registry for Sentinel instances.
    
    Tracks active Sentinel engines per agent for:
    - CLI access
    - API access
    - Monitoring
    """
    
    def __init__(self):
        """Initialize registry."""
        self.engines: Dict[str, SentinelEngine] = {}
        self.lock = asyncio.Lock()
        self.logger = StructuredLogger("sentinel_registry", LogLevel.INFO)
    
    async def register(self, agent_id: str, engine: SentinelEngine) -> None:
        """
        Register a Sentinel engine for an agent.
        
        Args:
            agent_id: Agent ID
            engine: SentinelEngine instance
        """
        async with self.lock:
            self.engines[agent_id] = engine
            self.logger.debug("Sentinel registered", agent_id=agent_id)
    
    async def unregister(self, agent_id: str) -> None:
        """
        Unregister a Sentinel engine.
        
        Args:
            agent_id: Agent ID
        """
        async with self.lock:
            if agent_id in self.engines:
                del self.engines[agent_id]
                self.logger.debug("Sentinel unregistered", agent_id=agent_id)
    
    async def get(self, agent_id: str) -> Optional[SentinelEngine]:
        """
        Get Sentinel engine for an agent.
        
        Args:
            agent_id: Agent ID
        
        Returns:
            SentinelEngine or None
        """
        async with self.lock:
            return self.engines.get(agent_id)
    
    async def list_all(self) -> Dict[str, SentinelEngine]:
        """
        List all registered Sentinel engines.
        
        Returns:
            Dictionary mapping agent IDs to engines
        """
        async with self.lock:
            return self.engines.copy()
    
    async def clear(self) -> None:
        """Clear all registrations."""
        async with self.lock:
            self.engines.clear()
            self.logger.debug("Registry cleared")


# Global registry instance
_registry: Optional[SentinelRegistry] = None
_registry_lock = asyncio.Lock()


async def get_sentinel_registry() -> SentinelRegistry:
    """
    Get global Sentinel registry instance.
    
    Returns:
        SentinelRegistry instance
    """
    global _registry
    
    if _registry is None:
        async with _registry_lock:
            if _registry is None:
                _registry = SentinelRegistry()
    
    return _registry

