"""
Cortex Registry - Production-ready instance management.

Tracks and manages active Cortex memory instances for monitoring and API access.
"""

from typing import Dict, Optional, Any
from datetime import datetime
import asyncio
import logging
from weakref import WeakValueDictionary

from teleon.cortex import CortexMemory

logger = logging.getLogger("teleon.cortex.registry")


class CortexRegistry:
    """
    Global registry for Cortex memory instances.
    
    Provides production-ready instance tracking and retrieval for:
    - Dashboard API endpoints
    - CLI commands
    - Monitoring and profiling
    
    Thread-safe and handles instance lifecycle.
    """
    
    _instances: Dict[str, CortexMemory] = {}
    _lock = asyncio.Lock()
    
    @classmethod
    async def register(cls, agent_id: str, cortex: CortexMemory) -> None:
        """
        Register a Cortex instance.
        
        Args:
            agent_id: Agent ID
            cortex: CortexMemory instance
        """
        async with cls._lock:
            if agent_id in cls._instances:
                logger.warning(f"Cortex instance for agent {agent_id} already registered, replacing")
            
            cls._instances[agent_id] = cortex
            logger.info(f"Registered Cortex instance for agent: {agent_id}")
    
    @classmethod
    async def unregister(cls, agent_id: str) -> None:
        """
        Unregister a Cortex instance.
        
        Args:
            agent_id: Agent ID
        """
        async with cls._lock:
            if agent_id in cls._instances:
                cortex = cls._instances.pop(agent_id)
                try:
                    await cortex.shutdown()
                except Exception as e:
                    logger.warning(f"Error shutting down Cortex for {agent_id}: {e}")
                logger.info(f"Unregistered Cortex instance for agent: {agent_id}")
    
    @classmethod
    async def get(cls, agent_id: str) -> Optional[CortexMemory]:
        """
        Get a registered Cortex instance.
        
        Args:
            agent_id: Agent ID
        
        Returns:
            CortexMemory instance or None if not found
        """
        async with cls._lock:
            return cls._instances.get(agent_id)
    
    @classmethod
    async def get_or_create(
        cls,
        agent_id: str,
        storage: Optional[Any] = None,
        config: Optional[Any] = None
    ) -> CortexMemory:
        """
        Get existing instance or create new one.
        
        Args:
            agent_id: Agent ID
            storage: Optional storage backend
            config: Optional Cortex config
        
        Returns:
            CortexMemory instance
        """
        async with cls._lock:
            if agent_id in cls._instances:
                return cls._instances[agent_id]
            
            # Create new instance
            from teleon.cortex import CortexMemory, CortexConfig
            cortex = CortexMemory(
                storage=storage,
                agent_id=agent_id,
                config=config or CortexConfig()
            )
            await cortex.initialize()
            
            cls._instances[agent_id] = cortex
            logger.info(f"Created and registered new Cortex instance for agent: {agent_id}")
            
            return cortex
    
    @classmethod
    async def list_agents(cls) -> list[str]:
        """
        List all registered agent IDs.
        
        Returns:
            List of agent IDs
        """
        async with cls._lock:
            return list(cls._instances.keys())
    
    @classmethod
    async def get_all_instances(cls) -> Dict[str, CortexMemory]:
        """
        Get all registered instances.
        
        Returns:
            Dictionary of agent_id -> CortexMemory
        """
        async with cls._lock:
            return cls._instances.copy()
    
    @classmethod
    async def clear(cls) -> None:
        """Clear all registered instances (for testing)."""
        async with cls._lock:
            for agent_id, cortex in list(cls._instances.items()):
                try:
                    await cortex.shutdown()
                except Exception as e:
                    logger.warning(f"Error shutting down Cortex for {agent_id}: {e}")
            cls._instances.clear()


# Global registry instance
registry = CortexRegistry()

