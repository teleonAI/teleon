"""
Webhook Router - Route webhook events to appropriate handlers.

Provides:
- Event routing based on event type
- Handler registration
- Pattern matching for events
"""

from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass
import re
import asyncio

from teleon.core import StructuredLogger, LogLevel


@dataclass
class WebhookHandler:
    """Webhook event handler configuration."""
    
    event_pattern: str
    """Event pattern (supports wildcards)"""
    
    handler: Callable
    """Handler function"""
    
    priority: int = 0
    """Handler priority (higher = first)"""
    
    async_handler: bool = True
    """Whether handler is async"""


class WebhookRouter:
    """
    Routes webhook events to registered handlers.
    
    Supports:
    - Pattern-based routing (wildcards)
    - Multiple handlers per event
    - Priority-based execution
    - Async and sync handlers
    """
    
    def __init__(self):
        """Initialize webhook router."""
        self.handlers: List[WebhookHandler] = []
        self.logger = StructuredLogger("webhook_router", LogLevel.INFO)
    
    def register(
        self,
        event_pattern: str,
        handler: Callable,
        priority: int = 0
    ):
        """
        Register a webhook handler.
        
        Args:
            event_pattern: Event pattern (e.g., "github.push", "stripe.*")
            handler: Handler function
            priority: Handler priority
        """
        webhook_handler = WebhookHandler(
            event_pattern=event_pattern,
            handler=handler,
            priority=priority,
            async_handler=asyncio.iscoroutinefunction(handler)
        )
        
        self.handlers.append(webhook_handler)
        
        # Sort by priority (descending)
        self.handlers.sort(key=lambda h: h.priority, reverse=True)
        
        self.logger.info(
            f"Registered webhook handler",
            extra={
                "pattern": event_pattern,
                "priority": priority
            }
        )
    
    def unregister(self, event_pattern: str, handler: Callable):
        """
        Unregister a webhook handler.
        
        Args:
            event_pattern: Event pattern
            handler: Handler function
        """
        self.handlers = [
            h for h in self.handlers
            if not (h.event_pattern == event_pattern and h.handler == handler)
        ]
        
        self.logger.info(f"Unregistered webhook handler: {event_pattern}")
    
    def _pattern_matches(self, pattern: str, event_type: str) -> bool:
        """
        Check if event type matches pattern.
        
        Args:
            pattern: Event pattern (supports * wildcard)
            event_type: Event type to match
            
        Returns:
            True if matches
        """
        # Convert wildcard pattern to regex
        regex_pattern = pattern.replace(".", "\\.").replace("*", ".*")
        regex_pattern = f"^{regex_pattern}$"
        
        return bool(re.match(regex_pattern, event_type))
    
    async def route(
        self,
        event_type: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ) -> List[Any]:
        """
        Route webhook event to registered handlers.
        
        Args:
            event_type: Event type (e.g., "github.push")
            payload: Event payload
            headers: Request headers
            
        Returns:
            List of handler results
        """
        if headers is None:
            headers = {}
        
        matching_handlers = [
            h for h in self.handlers
            if self._pattern_matches(h.event_pattern, event_type)
        ]
        
        if not matching_handlers:
            self.logger.warning(f"No handlers found for event: {event_type}")
            return []
        
        self.logger.info(
            f"Routing event to {len(matching_handlers)} handler(s)",
            extra={"event_type": event_type}
        )
        
        results = []
        
        for handler_config in matching_handlers:
            try:
                if handler_config.async_handler:
                    result = await handler_config.handler(
                        event_type, payload, headers
                    )
                else:
                    result = handler_config.handler(
                        event_type, payload, headers
                    )
                
                results.append(result)
            
            except Exception as e:
                self.logger.error(
                    f"Handler failed: {e}",
                    extra={
                        "event_type": event_type,
                        "handler": handler_config.handler.__name__
                    }
                )
        
        return results
    
    def list_handlers(self) -> List[Dict[str, Any]]:
        """
        List all registered handlers.
        
        Returns:
            List of handler information
        """
        return [
            {
                "pattern": h.event_pattern,
                "handler": h.handler.__name__,
                "priority": h.priority,
                "async": h.async_handler
            }
            for h in self.handlers
        ]

