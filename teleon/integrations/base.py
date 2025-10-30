"""
Base Integration Framework - Foundation for all service integrations.

Provides:
- Base integration class with retry and rate limiting
- Integration configuration
- Integration registry for discovery
- Automatic error handling and logging
"""

from typing import Dict, Any, Optional, Callable, Type
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import asyncio
import time

from teleon.core import StructuredLogger, LogLevel, TeleonException


class IntegrationError(TeleonException):
    """Raised when an integration operation fails."""
    pass


class RateLimitError(IntegrationError):
    """Raised when rate limit is exceeded."""
    pass


class AuthenticationError(IntegrationError):
    """Raised when authentication fails."""
    pass


@dataclass
class IntegrationConfig:
    """Configuration for an integration."""
    
    name: str
    """Integration name"""
    
    api_key: Optional[str] = None
    """API key for authentication"""
    
    api_secret: Optional[str] = None
    """API secret for authentication"""
    
    base_url: Optional[str] = None
    """Base URL for API calls"""
    
    timeout: int = 30
    """Request timeout in seconds"""
    
    max_retries: int = 3
    """Maximum number of retries"""
    
    rate_limit_per_second: int = 10
    """Maximum requests per second"""
    
    rate_limit_per_minute: int = 100
    """Maximum requests per minute"""
    
    extra: Dict[str, Any] = field(default_factory=dict)
    """Extra configuration options"""


class RateLimiter:
    """
    Token bucket rate limiter.
    
    Implements rate limiting with per-second and per-minute limits.
    """
    
    def __init__(self, per_second: int, per_minute: int):
        """
        Initialize rate limiter.
        
        Args:
            per_second: Maximum requests per second
            per_minute: Maximum requests per minute
        """
        self.per_second = per_second
        self.per_minute = per_minute
        
        self.second_tokens = per_second
        self.minute_tokens = per_minute
        
        self.second_last_refill = time.time()
        self.minute_last_refill = time.time()
        
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """
        Acquire permission to make a request.
        
        Raises:
            RateLimitError: If rate limit exceeded
        """
        async with self.lock:
            now = time.time()
            
            # Refill second tokens
            elapsed = now - self.second_last_refill
            if elapsed >= 1.0:
                self.second_tokens = self.per_second
                self.second_last_refill = now
            
            # Refill minute tokens
            elapsed = now - self.minute_last_refill
            if elapsed >= 60.0:
                self.minute_tokens = self.per_minute
                self.minute_last_refill = now
            
            # Check if we have tokens
            if self.second_tokens <= 0:
                raise RateLimitError("Rate limit exceeded (per second)")
            
            if self.minute_tokens <= 0:
                raise RateLimitError("Rate limit exceeded (per minute)")
            
            # Consume tokens
            self.second_tokens -= 1
            self.minute_tokens -= 1


class BaseIntegration(ABC):
    """
    Base class for all service integrations.
    
    Provides common functionality:
    - Configuration management
    - Automatic retry with exponential backoff
    - Rate limiting
    - Error handling and logging
    - Authentication
    """
    
    def __init__(self, config: IntegrationConfig):
        """
        Initialize integration.
        
        Args:
            config: Integration configuration
        """
        self.config = config
        self.logger = StructuredLogger(f"integration.{config.name}", LogLevel.INFO)
        self.rate_limiter = RateLimiter(
            config.rate_limit_per_second,
            config.rate_limit_per_minute
        )
        self._authenticated = False
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """
        Authenticate with the service.
        
        Returns:
            True if authentication successful
            
        Raises:
            AuthenticationError: If authentication fails
        """
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """
        Test the connection to the service.
        
        Returns:
            True if connection successful
        """
        pass
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a function with automatic retry.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            IntegrationError: If all retries fail
        """
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                # Rate limiting
                await self.rate_limiter.acquire()
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                return result
            
            except RateLimitError as e:
                # Wait and retry for rate limit errors
                wait_time = min(2 ** attempt, 60)
                self.logger.warning(
                    f"Rate limit exceeded, waiting {wait_time}s",
                    extra={"attempt": attempt + 1}
                )
                await asyncio.sleep(wait_time)
                last_error = e
            
            except Exception as e:
                last_error = e
                
                # Don't retry on authentication errors
                if isinstance(e, AuthenticationError):
                    raise
                
                # Exponential backoff
                if attempt < self.config.max_retries - 1:
                    wait_time = min(2 ** attempt, 60)
                    self.logger.warning(
                        f"Request failed, retrying in {wait_time}s",
                        extra={
                            "attempt": attempt + 1,
                            "error": str(e)
                        }
                    )
                    await asyncio.sleep(wait_time)
        
        # All retries failed
        raise IntegrationError(
            f"All retries failed: {str(last_error)}"
        ) from last_error
    
    async def ensure_authenticated(self):
        """Ensure the integration is authenticated."""
        if not self._authenticated:
            await self.authenticate()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name})"


class IntegrationRegistry:
    """
    Registry for discovering and managing integrations.
    
    Allows registering and retrieving integrations by name.
    """
    
    def __init__(self):
        """Initialize the registry."""
        self._integrations: Dict[str, Type[BaseIntegration]] = {}
        self.logger = StructuredLogger("integration_registry", LogLevel.INFO)
    
    def register(
        self,
        name: str,
        integration_class: Type[BaseIntegration]
    ):
        """
        Register an integration.
        
        Args:
            name: Integration name
            integration_class: Integration class
        """
        self._integrations[name] = integration_class
        self.logger.info(f"Registered integration: {name}")
    
    def get(self, name: str) -> Optional[Type[BaseIntegration]]:
        """
        Get an integration class by name.
        
        Args:
            name: Integration name
            
        Returns:
            Integration class or None if not found
        """
        return self._integrations.get(name)
    
    def list_integrations(self) -> list[str]:
        """
        List all registered integrations.
        
        Returns:
            List of integration names
        """
        return list(self._integrations.keys())
    
    def create(
        self,
        name: str,
        config: IntegrationConfig
    ) -> BaseIntegration:
        """
        Create an integration instance.
        
        Args:
            name: Integration name
            config: Integration configuration
            
        Returns:
            Integration instance
            
        Raises:
            IntegrationError: If integration not found
        """
        integration_class = self.get(name)
        if not integration_class:
            raise IntegrationError(f"Integration not found: {name}")
        
        return integration_class(config)


# Global registry
_registry = IntegrationRegistry()


def get_registry() -> IntegrationRegistry:
    """Get the global integration registry."""
    return _registry

