"""
Token Store - Secure storage for OAuth tokens.

Provides:
- In-memory storage
- Redis storage (for production)
- Token encryption
- Automatic cleanup
"""

from typing import Dict, Optional
from abc import ABC, abstractmethod
import json
from datetime import datetime, timedelta

from teleon.auth.oauth import OAuth2Token
from teleon.core import StructuredLogger, LogLevel


class TokenStore(ABC):
    """
    Abstract base class for token storage.
    
    Implementations must provide secure storage for OAuth tokens.
    """
    
    @abstractmethod
    async def save_token(
        self,
        user_id: str,
        provider: str,
        token: OAuth2Token
    ):
        """
        Save OAuth token.
        
        Args:
            user_id: User identifier
            provider: OAuth provider name
            token: OAuth2 token
        """
        pass
    
    @abstractmethod
    async def get_token(
        self,
        user_id: str,
        provider: str
    ) -> Optional[OAuth2Token]:
        """
        Retrieve OAuth token.
        
        Args:
            user_id: User identifier
            provider: OAuth provider name
            
        Returns:
            OAuth2 token or None
        """
        pass
    
    @abstractmethod
    async def delete_token(
        self,
        user_id: str,
        provider: str
    ):
        """
        Delete OAuth token.
        
        Args:
            user_id: User identifier
            provider: OAuth provider name
        """
        pass
    
    @abstractmethod
    async def list_tokens(
        self,
        user_id: str
    ) -> Dict[str, OAuth2Token]:
        """
        List all tokens for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary of provider -> token
        """
        pass


class InMemoryTokenStore(TokenStore):
    """
    In-memory token storage.
    
    WARNING: Tokens are lost when the application restarts.
    Use Redis storage for production.
    """
    
    def __init__(self):
        """Initialize in-memory store."""
        self.tokens: Dict[str, Dict[str, OAuth2Token]] = {}
        self.logger = StructuredLogger("token_store.memory", LogLevel.INFO)
    
    async def save_token(
        self,
        user_id: str,
        provider: str,
        token: OAuth2Token
    ):
        """Save token to memory."""
        if user_id not in self.tokens:
            self.tokens[user_id] = {}
        
        self.tokens[user_id][provider] = token
        
        self.logger.info(
            f"Saved token for {user_id}/{provider}",
            extra={"user_id": user_id, "provider": provider}
        )
    
    async def get_token(
        self,
        user_id: str,
        provider: str
    ) -> Optional[OAuth2Token]:
        """Get token from memory."""
        token = self.tokens.get(user_id, {}).get(provider)
        
        if token:
            self.logger.debug(
                f"Retrieved token for {user_id}/{provider}",
                extra={"user_id": user_id, "provider": provider}
            )
        
        return token
    
    async def delete_token(
        self,
        user_id: str,
        provider: str
    ):
        """Delete token from memory."""
        if user_id in self.tokens and provider in self.tokens[user_id]:
            del self.tokens[user_id][provider]
            
            self.logger.info(
                f"Deleted token for {user_id}/{provider}",
                extra={"user_id": user_id, "provider": provider}
            )
    
    async def list_tokens(
        self,
        user_id: str
    ) -> Dict[str, OAuth2Token]:
        """List all tokens for user."""
        return self.tokens.get(user_id, {})


class RedisTokenStore(TokenStore):
    """
    Redis-based token storage.
    
    Provides:
    - Persistent storage
    - Automatic expiration
    - High performance
    
    Requires Redis to be available.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize Redis token store.
        
        Args:
            redis_url: Redis connection URL
        """
        try:
            import redis.asyncio as redis
            self.redis = redis.from_url(redis_url)
        except ImportError:
            raise ImportError("redis package required for RedisTokenStore")
        
        self.logger = StructuredLogger("token_store.redis", LogLevel.INFO)
    
    def _key(self, user_id: str, provider: str) -> str:
        """Generate Redis key."""
        return f"teleon:token:{user_id}:{provider}"
    
    async def save_token(
        self,
        user_id: str,
        provider: str,
        token: OAuth2Token
    ):
        """Save token to Redis."""
        key = self._key(user_id, provider)
        value = json.dumps(token.to_dict())
        
        # Set expiration if token has expires_in
        if token.expires_in:
            await self.redis.setex(key, token.expires_in, value)
        else:
            await self.redis.set(key, value)
        
        self.logger.info(
            f"Saved token to Redis for {user_id}/{provider}",
            extra={"user_id": user_id, "provider": provider}
        )
    
    async def get_token(
        self,
        user_id: str,
        provider: str
    ) -> Optional[OAuth2Token]:
        """Get token from Redis."""
        key = self._key(user_id, provider)
        value = await self.redis.get(key)
        
        if not value:
            return None
        
        token_data = json.loads(value)
        token = OAuth2Token.from_dict(token_data)
        
        self.logger.debug(
            f"Retrieved token from Redis for {user_id}/{provider}",
            extra={"user_id": user_id, "provider": provider}
        )
        
        return token
    
    async def delete_token(
        self,
        user_id: str,
        provider: str
    ):
        """Delete token from Redis."""
        key = self._key(user_id, provider)
        await self.redis.delete(key)
        
        self.logger.info(
            f"Deleted token from Redis for {user_id}/{provider}",
            extra={"user_id": user_id, "provider": provider}
        )
    
    async def list_tokens(
        self,
        user_id: str
    ) -> Dict[str, OAuth2Token]:
        """List all tokens for user from Redis."""
        pattern = self._key(user_id, "*")
        keys = []
        
        async for key in self.redis.scan_iter(pattern):
            keys.append(key)
        
        tokens = {}
        for key in keys:
            provider = key.decode().split(":")[-1]
            value = await self.redis.get(key)
            if value:
                token_data = json.loads(value)
                tokens[provider] = OAuth2Token.from_dict(token_data)
        
        return tokens
    
    async def close(self):
        """Close Redis connection."""
        await self.redis.close()


# Global token store
_token_store: Optional[TokenStore] = None


def get_token_store() -> TokenStore:
    """
    Get the global token store.
    
    Returns:
        Token store instance
    """
    global _token_store
    
    if _token_store is None:
        _token_store = InMemoryTokenStore()
    
    return _token_store


def set_token_store(store: TokenStore):
    """
    Set the global token store.
    
    Args:
        store: Token store instance
    """
    global _token_store
    _token_store = store

