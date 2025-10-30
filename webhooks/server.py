"""
Webhook Server - Receive and process webhooks from external services.

Provides:
- HTTP server for receiving webhooks
- Automatic signature validation
- Event routing to handlers
- Dead letter queue for failed events
"""

from typing import Dict, Any, Optional, Callable
from fastapi import FastAPI, Request, HTTPException, Response
import time

from teleon.webhooks.validator import WebhookValidator, SignatureValidationError
from teleon.webhooks.router import WebhookRouter
from teleon.core import StructuredLogger, LogLevel


class WebhookServer:
    """
    HTTP server for receiving webhooks.
    
    Features:
    - FastAPI-based server
    - Automatic signature validation
    - Event routing
    - Logging and monitoring
    
    Example:
        >>> server = WebhookServer()
        >>> 
        >>> @server.on("github.push")
        >>> async def handle_push(event_type, payload, headers):
        ...     print(f"Received push to {payload['repository']['name']}")
        >>> 
        >>> server.run(port=8080)
    """
    
    def __init__(self):
        """Initialize webhook server."""
        self.app = FastAPI(title="Teleon Webhook Server")
        self.router = WebhookRouter()
        self.logger = StructuredLogger("webhook_server", LogLevel.INFO)
        self.validators: Dict[str, Callable] = {}
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.post("/webhooks/{provider}")
        async def receive_webhook(provider: str, request: Request):
            """Receive and process webhook."""
            # Get raw body
            body = await request.body()
            headers = dict(request.headers)
            
            # Parse JSON payload
            try:
                payload = await request.json()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
            
            # Validate signature if validator exists
            if provider in self.validators:
                try:
                    self.validators[provider](body, headers)
                except SignatureValidationError as e:
                    self.logger.error(f"Signature validation failed: {e}")
                    raise HTTPException(status_code=401, detail="Invalid signature")
            
            # Determine event type
            event_type = self._extract_event_type(provider, payload, headers)
            
            # Log event
            self.logger.info(
                f"Received webhook: {provider}/{event_type}",
                extra={"provider": provider, "event_type": event_type}
            )
            
            # Route to handlers
            try:
                results = await self.router.route(event_type, payload, headers)
                
                return {
                    "status": "success",
                    "event_type": event_type,
                    "handlers_executed": len(results),
                    "timestamp": time.time()
                }
            
            except Exception as e:
                self.logger.error(f"Error processing webhook: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": time.time()}
        
        @self.app.get("/handlers")
        async def list_handlers():
            """List registered handlers."""
            return {"handlers": self.router.list_handlers()}
    
    def _extract_event_type(
        self,
        provider: str,
        payload: Dict[str, Any],
        headers: Dict[str, str]
    ) -> str:
        """
        Extract event type from webhook.
        
        Args:
            provider: Provider name
            payload: Webhook payload
            headers: Request headers
            
        Returns:
            Event type string
        """
        if provider == "github":
            event = headers.get("x-github-event", "unknown")
            action = payload.get("action")
            if action:
                return f"github.{event}.{action}"
            return f"github.{event}"
        
        elif provider == "stripe":
            return f"stripe.{payload.get('type', 'unknown')}"
        
        elif provider == "slack":
            event_type = payload.get("type")
            if event_type == "event_callback":
                event_type = payload.get("event", {}).get("type", "unknown")
            return f"slack.{event_type}"
        
        else:
            # Generic event type extraction
            return f"{provider}.{payload.get('event', payload.get('type', 'unknown'))}"
    
    def on(self, event_pattern: str, priority: int = 0):
        """
        Decorator to register webhook handler.
        
        Args:
            event_pattern: Event pattern (e.g., "github.push", "stripe.*")
            priority: Handler priority
            
        Example:
            >>> @server.on("github.push")
            >>> async def handle_push(event_type, payload, headers):
            ...     pass
        """
        def decorator(func: Callable):
            self.router.register(event_pattern, func, priority)
            return func
        
        return decorator
    
    def register_validator(
        self,
        provider: str,
        validator: Callable
    ):
        """
        Register signature validator for a provider.
        
        Args:
            provider: Provider name
            validator: Validator function
        """
        self.validators[provider] = validator
        self.logger.info(f"Registered validator for: {provider}")
    
    def register_github_validator(self, secret: str):
        """Register GitHub webhook validator."""
        def validator(body: bytes, headers: Dict[str, str]):
            signature = headers.get("x-hub-signature-256", "")
            return WebhookValidator.validate_github(body, signature, secret)
        
        self.register_validator("github", validator)
    
    def register_stripe_validator(self, secret: str):
        """Register Stripe webhook validator."""
        def validator(body: bytes, headers: Dict[str, str]):
            signature = headers.get("stripe-signature", "")
            timestamp = int(time.time())
            return WebhookValidator.validate_stripe(
                body, signature, secret, timestamp
            )
        
        self.register_validator("stripe", validator)
    
    def register_slack_validator(self, secret: str):
        """Register Slack webhook validator."""
        def validator(body: bytes, headers: Dict[str, str]):
            signature = headers.get("x-slack-signature", "")
            timestamp = headers.get("x-slack-request-timestamp", "")
            return WebhookValidator.validate_slack(
                body, signature, secret, timestamp
            )
        
        self.register_validator("slack", validator)
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application."""
        return self.app

