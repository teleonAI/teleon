"""Session management for Working Memory."""

from typing import Dict, Optional
import uuid
import asyncio

from teleon.memory.working import WorkingMemory


class SessionManager:
    """
    Manages multiple Working Memory sessions.
    
    Features:
    - Session creation and retrieval
    - Automatic cleanup of expired sessions
    - Session statistics
    """
    
    def __init__(
        self,
        default_ttl: int = 3600,
        max_size: int = 1000,
        cleanup_interval: int = 300
    ):
        """
        Initialize session manager.
        
        Args:
            default_ttl: Default TTL for sessions in seconds
            max_size: Maximum size for each session
            cleanup_interval: Interval for cleanup task in seconds
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        self._sessions: Dict[str, WorkingMemory] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def create_session(
        self,
        session_id: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> WorkingMemory:
        """
        Create a new session.
        
        Args:
            session_id: Optional session ID (auto-generated if not provided)
            ttl: Optional TTL (uses default if not provided)
        
        Returns:
            Working memory instance for the session
        """
        async with self._lock:
            # Generate session ID if not provided
            if session_id is None:
                session_id = str(uuid.uuid4())
            
            # Check if session already exists
            if session_id in self._sessions:
                return self._sessions[session_id]
            
            # Create new session
            memory = WorkingMemory(
                session_id=session_id,
                ttl=ttl or self.default_ttl,
                max_size=self.max_size
            )
            
            self._sessions[session_id] = memory
            
            # Start cleanup task if not running
            if self._cleanup_task is None:
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            return memory
    
    async def get_session(
        self,
        session_id: str,
        create_if_missing: bool = True
    ) -> Optional[WorkingMemory]:
        """
        Get an existing session.
        
        Args:
            session_id: Session ID
            create_if_missing: Whether to create session if it doesn't exist
        
        Returns:
            Working memory instance or None
        """
        async with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]
            
            if create_if_missing:
                return await self.create_session(session_id=session_id)
            
            return None
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session ID
        
        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if session_id in self._sessions:
                await self._sessions[session_id].clear()
                del self._sessions[session_id]
                return True
            return False
    
    async def clear_all_sessions(self) -> None:
        """Clear all sessions."""
        async with self._lock:
            for memory in self._sessions.values():
                await memory.clear()
            self._sessions.clear()
    
    async def get_all_sessions(self) -> Dict[str, WorkingMemory]:
        """
        Get all active sessions.
        
        Returns:
            Dictionary of session ID to Working Memory
        """
        return self._sessions.copy()
    
    async def _cleanup_loop(self) -> None:
        """Background task to clean up expired sessions."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in session cleanup: {e}")
    
    async def _cleanup_expired_sessions(self) -> None:
        """Remove expired sessions."""
        async with self._lock:
            expired_sessions = [
                session_id
                for session_id, memory in self._sessions.items()
                if memory.is_expired()
            ]
            
            for session_id in expired_sessions:
                await self._sessions[session_id].clear()
                del self._sessions[session_id]
            
            if expired_sessions:
                print(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get session manager statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_sessions": len(self._sessions),
            "default_ttl": self.default_ttl,
            "max_size": self.max_size,
            "cleanup_interval": self.cleanup_interval
        }
    
    async def shutdown(self) -> None:
        """Shutdown the session manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        await self.clear_all_sessions()


# Global session manager instance
_global_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """
    Get the global session manager instance.
    
    Returns:
        Global session manager
    """
    global _global_session_manager
    if _global_session_manager is None:
        _global_session_manager = SessionManager()
    return _global_session_manager

