"""Working Memory implementation."""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import asyncio


class WorkingMemory:
    """
    Working Memory for agents.
    
    Stores short-term state and context for ongoing conversations.
    Features:
    - Session-based storage
    - Automatic expiration (TTL)
    - Context window management
    - Key-value storage
    """
    
    def __init__(
        self,
        session_id: str,
        ttl: int = 3600,
        max_size: int = 1000
    ):
        """
        Initialize working memory.
        
        Args:
            session_id: Unique session identifier
            ttl: Time-to-live for entries in seconds
            max_size: Maximum number of entries
        """
        self.session_id = session_id
        self.ttl = ttl
        self.max_size = max_size
        self._storage: Dict[str, tuple] = {}  # key -> (value, expiry_time)
        self._lock = asyncio.Lock()
        self.created_at = datetime.utcnow()
        self.last_accessed = datetime.utcnow()
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """
        Store a value in working memory.
        
        Args:
            key: Storage key
            value: Value to store
            ttl: Custom TTL (defaults to instance TTL)
        """
        async with self._lock:
            # Enforce max size
            if len(self._storage) >= self.max_size and key not in self._storage:
                # Remove oldest entry
                oldest_key = min(
                    self._storage.keys(),
                    key=lambda k: self._storage[k][1]
                )
                del self._storage[oldest_key]
            
            # Calculate expiry
            ttl = ttl or self.ttl
            expiry_time = datetime.utcnow() + timedelta(seconds=ttl)
            
            # Store value
            self._storage[key] = (value, expiry_time)
            self.last_accessed = datetime.utcnow()
    
    async def get(
        self,
        key: str,
        default: Optional[Any] = None
    ) -> Optional[Any]:
        """
        Retrieve a value from working memory.
        
        Args:
            key: Storage key
            default: Default value if not found
        
        Returns:
            Stored value or default
        """
        async with self._lock:
            if key not in self._storage:
                return default
            
            value, expiry_time = self._storage[key]
            
            # Check expiration
            if datetime.utcnow() > expiry_time:
                del self._storage[key]
                return default
            
            self.last_accessed = datetime.utcnow()
            return value
    
    async def delete(self, key: str) -> bool:
        """
        Delete a value from working memory.
        
        Args:
            key: Storage key
        
        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if key in self._storage:
                del self._storage[key]
                self.last_accessed = datetime.utcnow()
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all stored values."""
        async with self._lock:
            self._storage.clear()
            self.last_accessed = datetime.utcnow()
    
    async def keys(self) -> List[str]:
        """
        Get all valid (non-expired) keys.
        
        Returns:
            List of keys
        """
        async with self._lock:
            now = datetime.utcnow()
            valid_keys = []
            expired_keys = []
            
            for key, (value, expiry_time) in self._storage.items():
                if now > expiry_time:
                    expired_keys.append(key)
                else:
                    valid_keys.append(key)
            
            # Clean up expired keys
            for key in expired_keys:
                del self._storage[key]
            
            return valid_keys
    
    async def size(self) -> int:
        """
        Get the number of stored items.
        
        Returns:
            Number of items
        """
        keys = await self.keys()
        return len(keys)
    
    async def get_all(self) -> Dict[str, Any]:
        """
        Get all stored values.
        
        Returns:
            Dictionary of all stored values
        """
        async with self._lock:
            now = datetime.utcnow()
            result = {}
            expired_keys = []
            
            for key, (value, expiry_time) in self._storage.items():
                if now > expiry_time:
                    expired_keys.append(key)
                else:
                    result[key] = value
            
            # Clean up expired keys
            for key in expired_keys:
                del self._storage[key]
            
            self.last_accessed = datetime.utcnow()
            return result
    
    def is_expired(self) -> bool:
        """
        Check if the session is expired.
        
        Returns:
            True if expired
        """
        age = datetime.utcnow() - self.last_accessed
        return age.total_seconds() > self.ttl
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "session_id": self.session_id,
            "size": len(self._storage),
            "max_size": self.max_size,
            "ttl": self.ttl,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "age_seconds": (datetime.utcnow() - self.created_at).total_seconds()
        }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory statistics (async version for consistency with other memory types).
        
        Returns:
            Statistics dictionary
        """
        return self.get_stats()

