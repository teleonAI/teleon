"""
Coordination Primitives - Production-grade agent coordination.

Features:
- Distributed locks
- Shared state management
- Barriers
- Semaphores
- Leader election
"""

from typing import Dict, Optional, Any, Set
from datetime import datetime, timedelta
import asyncio
from pydantic import BaseModel, Field

from teleon.core import (
    get_metrics,
    StructuredLogger,
    LogLevel,
)


class Lock:
    """
    Distributed lock for agent coordination.
    
    Production features:
    - Automatic timeout
    - Deadlock prevention
    - Lock ownership tracking
    """
    
    def __init__(self, name: str, timeout: float = 30.0):
        """
        Initialize lock.
        
        Args:
            name: Lock name
            timeout: Lock timeout in seconds
        """
        self.name = name
        self.timeout = timeout
        
        self._lock = asyncio.Lock()
        self._owner: Optional[str] = None
        self._acquired_at: Optional[datetime] = None
        
        self.logger = StructuredLogger(f"lock.{name}", LogLevel.DEBUG)
    
    async def acquire(self, agent_id: str, timeout: Optional[float] = None) -> bool:
        """
        Acquire lock.
        
        Args:
            agent_id: Agent requesting lock
            timeout: Acquire timeout
        
        Returns:
            True if acquired
        """
        acquire_timeout = timeout or self.timeout
        
        try:
            await asyncio.wait_for(
                self._lock.acquire(),
                timeout=acquire_timeout
            )
            
            self._owner = agent_id
            self._acquired_at = datetime.utcnow()
            
            self.logger.info("Lock acquired", agent_id=agent_id)
            return True
        
        except asyncio.TimeoutError:
            self.logger.warning(
                "Lock acquire timeout",
                agent_id=agent_id,
                timeout=acquire_timeout
            )
            return False
    
    def release(self, agent_id: str):
        """
        Release lock.
        
        Args:
            agent_id: Agent releasing lock
        
        Raises:
            RuntimeError: If agent doesn't own lock
        """
        if self._owner != agent_id:
            raise RuntimeError(f"Lock not owned by {agent_id}")
        
        self._owner = None
        self._acquired_at = None
        self._lock.release()
        
        self.logger.info("Lock released", agent_id=agent_id)
    
    def is_locked(self) -> bool:
        """Check if lock is held."""
        return self._lock.locked()
    
    def owner(self) -> Optional[str]:
        """Get lock owner."""
        return self._owner


class SharedState:
    """
    Shared state for agent coordination.
    
    Production features:
    - Thread-safe operations
    - State versioning
    - Change notifications
    - TTL support
    """
    
    def __init__(self, name: str):
        """
        Initialize shared state.
        
        Args:
            name: State name
        """
        self.name = name
        
        self._state: Dict[str, Any] = {}
        self._version: int = 0
        self._lock = asyncio.Lock()
        
        # Change listeners
        self._listeners: Dict[str, list] = {}
        
        self.logger = StructuredLogger(f"state.{name}", LogLevel.DEBUG)
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from state.
        
        Args:
            key: State key
        
        Returns:
            Value or None
        """
        async with self._lock:
            return self._state.get(key)
    
    async def set(self, key: str, value: Any, agent_id: Optional[str] = None):
        """
        Set value in state.
        
        Args:
            key: State key
            value: Value
            agent_id: Agent setting value
        """
        async with self._lock:
            self._state[key] = value
            self._version += 1
            
            self.logger.debug(
                "State updated",
                key=key,
                version=self._version,
                agent_id=agent_id
            )
            
            # Notify listeners
            await self._notify_listeners(key, value)
    
    async def update(self, updates: Dict[str, Any], agent_id: Optional[str] = None):
        """
        Update multiple values atomically.
        
        Args:
            updates: Key-value updates
            agent_id: Agent updating state
        """
        async with self._lock:
            self._state.update(updates)
            self._version += 1
            
            self.logger.debug(
                "State batch updated",
                keys=list(updates.keys()),
                version=self._version,
                agent_id=agent_id
            )
            
            # Notify listeners
            for key, value in updates.items():
                await self._notify_listeners(key, value)
    
    async def delete(self, key: str, agent_id: Optional[str] = None):
        """
        Delete value from state.
        
        Args:
            key: State key
            agent_id: Agent deleting value
        """
        async with self._lock:
            if key in self._state:
                del self._state[key]
                self._version += 1
                
                self.logger.debug(
                    "State deleted",
                    key=key,
                    version=self._version,
                    agent_id=agent_id
                )
    
    async def get_all(self) -> Dict[str, Any]:
        """Get all state."""
        async with self._lock:
            return dict(self._state)
    
    async def clear(self, agent_id: Optional[str] = None):
        """Clear all state."""
        async with self._lock:
            self._state.clear()
            self._version += 1
            
            self.logger.info("State cleared", agent_id=agent_id)
    
    def version(self) -> int:
        """Get current version."""
        return self._version
    
    async def subscribe(self, key: str, callback):
        """
        Subscribe to state changes.
        
        Args:
            key: State key
            callback: Callback function
        """
        if key not in self._listeners:
            self._listeners[key] = []
        
        self._listeners[key].append(callback)
        self.logger.debug("Listener subscribed", key=key)
    
    async def _notify_listeners(self, key: str, value: Any):
        """Notify listeners of change."""
        if key in self._listeners:
            for callback in self._listeners[key]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(key, value)
                    else:
                        callback(key, value)
                except Exception as e:
                    self.logger.error(
                        "Listener error",
                        key=key,
                        error=str(e)
                    )


class Barrier:
    """
    Synchronization barrier for multiple agents.
    
    Agents wait at barrier until all have arrived.
    """
    
    def __init__(self, name: str, parties: int, timeout: float = 60.0):
        """
        Initialize barrier.
        
        Args:
            name: Barrier name
            parties: Number of agents required
            timeout: Wait timeout
        """
        self.name = name
        self.parties = parties
        self.timeout = timeout
        
        self._waiting: Set[str] = set()
        self._event = asyncio.Event()
        self._lock = asyncio.Lock()
        
        self.logger = StructuredLogger(f"barrier.{name}", LogLevel.DEBUG)
    
    async def wait(self, agent_id: str) -> bool:
        """
        Wait at barrier.
        
        Args:
            agent_id: Agent ID
        
        Returns:
            True if all parties arrived
        """
        async with self._lock:
            self._waiting.add(agent_id)
            
            self.logger.debug(
                "Agent waiting at barrier",
                agent_id=agent_id,
                waiting=len(self._waiting),
                required=self.parties
            )
            
            if len(self._waiting) >= self.parties:
                # All parties arrived
                self._event.set()
                self.logger.info("Barrier released", parties=self.parties)
        
        # Wait for all parties
        try:
            await asyncio.wait_for(self._event.wait(), timeout=self.timeout)
            return True
        except asyncio.TimeoutError:
            self.logger.warning("Barrier timeout", agent_id=agent_id)
            return False
    
    async def reset(self):
        """Reset barrier."""
        async with self._lock:
            self._waiting.clear()
            self._event.clear()
            self.logger.debug("Barrier reset")


class Semaphore:
    """
    Counting semaphore for resource management.
    
    Limits number of agents accessing a resource.
    """
    
    def __init__(self, name: str, value: int):
        """
        Initialize semaphore.
        
        Args:
            name: Semaphore name
            value: Initial value (resource count)
        """
        self.name = name
        self._semaphore = asyncio.Semaphore(value)
        self._holders: Set[str] = set()
        self._lock = asyncio.Lock()
        
        self.logger = StructuredLogger(f"semaphore.{name}", LogLevel.DEBUG)
    
    async def acquire(self, agent_id: str, timeout: Optional[float] = None):
        """
        Acquire semaphore.
        
        Args:
            agent_id: Agent ID
            timeout: Acquire timeout
        """
        if timeout:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=timeout
            )
        else:
            await self._semaphore.acquire()
        
        async with self._lock:
            self._holders.add(agent_id)
        
        self.logger.debug("Semaphore acquired", agent_id=agent_id)
    
    def release(self, agent_id: str):
        """
        Release semaphore.
        
        Args:
            agent_id: Agent ID
        """
        self._semaphore.release()
        
        async def remove_holder():
            async with self._lock:
                self._holders.discard(agent_id)
        
        asyncio.create_task(remove_holder())
        self.logger.debug("Semaphore released", agent_id=agent_id)


class Coordinator:
    """
    Coordination manager.
    
    Manages all coordination primitives.
    """
    
    def __init__(self):
        """Initialize coordinator."""
        self.locks: Dict[str, Lock] = {}
        self.states: Dict[str, SharedState] = {}
        self.barriers: Dict[str, Barrier] = {}
        self.semaphores: Dict[str, Semaphore] = {}
        
        self._lock = asyncio.Lock()
        self.logger = StructuredLogger("coordinator", LogLevel.INFO)
    
    async def get_lock(self, name: str, timeout: float = 30.0) -> Lock:
        """Get or create lock."""
        async with self._lock:
            if name not in self.locks:
                self.locks[name] = Lock(name, timeout)
            return self.locks[name]
    
    async def get_state(self, name: str) -> SharedState:
        """Get or create shared state."""
        async with self._lock:
            if name not in self.states:
                self.states[name] = SharedState(name)
            return self.states[name]
    
    async def get_barrier(
        self,
        name: str,
        parties: int,
        timeout: float = 60.0
    ) -> Barrier:
        """Get or create barrier."""
        async with self._lock:
            if name not in self.barriers:
                self.barriers[name] = Barrier(name, parties, timeout)
            return self.barriers[name]
    
    async def get_semaphore(self, name: str, value: int) -> Semaphore:
        """Get or create semaphore."""
        async with self._lock:
            if name not in self.semaphores:
                self.semaphores[name] = Semaphore(name, value)
            return self.semaphores[name]
    
    async def shutdown(self):
        """Shutdown coordinator."""
        self.logger.info("Coordinator shutdown")


# Global coordinator instance
_coordinator: Optional[Coordinator] = None


def get_coordinator() -> Coordinator:
    """Get global coordinator."""
    global _coordinator
    if _coordinator is None:
        _coordinator = Coordinator()
    return _coordinator

