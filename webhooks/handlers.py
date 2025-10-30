"""
Webhook Event Handlers - Process webhook events.

Provides:
- Base event handler class
- Event handler registry
- Common event processing patterns
"""

from typing import Dict, Any, Optional, Callable, Type
from abc import ABC, abstractmethod

from teleon.core import StructuredLogger, LogLevel


class EventHandler(ABC):
    """
    Base class for webhook event handlers.
    
    Subclass this to create custom event handlers.
    """
    
    def __init__(self):
        """Initialize event handler."""
        self.logger = StructuredLogger(
            f"event_handler.{self.__class__.__name__}",
            LogLevel.INFO
        )
    
    @abstractmethod
    async def handle(
        self,
        event_type: str,
        payload: Dict[str, Any],
        headers: Dict[str, str]
    ) -> Any:
        """
        Handle webhook event.
        
        Args:
            event_type: Event type
            payload: Event payload
            headers: Request headers
            
        Returns:
            Handler result
        """
        pass
    
    async def on_error(self, error: Exception, event_type: str):
        """
        Handle errors during event processing.
        
        Args:
            error: Exception that occurred
            event_type: Event type that failed
        """
        self.logger.error(
            f"Error handling event: {error}",
            extra={"event_type": event_type}
        )


class EventHandlerRegistry:
    """
    Registry for event handlers.
    
    Allows registering and retrieving event handlers by name.
    """
    
    def __init__(self):
        """Initialize registry."""
        self._handlers: Dict[str, Type[EventHandler]] = {}
        self.logger = StructuredLogger("event_registry", LogLevel.INFO)
    
    def register(
        self,
        name: str,
        handler_class: Type[EventHandler]
    ):
        """
        Register an event handler.
        
        Args:
            name: Handler name
            handler_class: Handler class
        """
        self._handlers[name] = handler_class
        self.logger.info(f"Registered event handler: {name}")
    
    def get(self, name: str) -> Optional[Type[EventHandler]]:
        """
        Get handler class by name.
        
        Args:
            name: Handler name
            
        Returns:
            Handler class or None
        """
        return self._handlers.get(name)
    
    def list_handlers(self) -> list[str]:
        """
        List all registered handlers.
        
        Returns:
            List of handler names
        """
        return list(self._handlers.keys())
    
    def create(self, name: str) -> EventHandler:
        """
        Create handler instance.
        
        Args:
            name: Handler name
            
        Returns:
            Handler instance
            
        Raises:
            ValueError: If handler not found
        """
        handler_class = self.get(name)
        if not handler_class:
            raise ValueError(f"Event handler not found: {name}")
        
        return handler_class()


# Global registry
_event_registry = EventHandlerRegistry()


def get_event_registry() -> EventHandlerRegistry:
    """Get the global event handler registry."""
    return _event_registry


def event_handler(name: str):
    """
    Decorator to register an event handler.
    
    Example:
        >>> @event_handler("github_push")
        ... class GitHubPushHandler(EventHandler):
        ...     async def handle(self, event_type, payload, headers):
        ...         # Process GitHub push event
        ...         pass
    """
    def decorator(cls: Type[EventHandler]):
        _event_registry.register(name, cls)
        return cls
    
    return decorator

