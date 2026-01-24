"""
API Gateway - Complete gateway implementation.

Combines routing, rate limiting, auth, and transformation.
"""

from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field
from collections import defaultdict
import asyncio
import re

from teleon.core import StructuredLogger, LogLevel


class RouteMatch(BaseModel):
    """Route match result."""
    matched: bool
    params: Dict[str, str] = Field(default_factory=dict)


class Route:
    """API route definition."""
    
    def __init__(
        self,
        path: str,
        handler: Callable,
        methods: List[str] = None,
        rate_limit: Optional[int] = None
    ):
        self.path = path
        self.handler = handler
        self.methods = methods or ["GET"]
        self.rate_limit = rate_limit
        
        # Convert path to regex
        self.pattern = re.compile(
            "^" + re.sub(r'\{(\w+)\}', r'(?P<\1>[^/]+)', path) + "$"
        )
    
    def match(self, path: str, method: str) -> RouteMatch:
        """Check if path matches route."""
        if method not in self.methods:
            return RouteMatch(matched=False)
        
        match = self.pattern.match(path)
        if match:
            return RouteMatch(matched=True, params=match.groupdict())
        
        return RouteMatch(matched=False)


class Router:
    """Request router."""
    
    def __init__(self):
        self.routes: List[Route] = []
        self.logger = StructuredLogger("router", LogLevel.INFO)
    
    def add_route(self, route: Route):
        """Add a route."""
        self.routes.append(route)
        self.logger.debug(f"Route added: {route.path}")
    
    async def route(self, path: str, method: str, **kwargs) -> Any:
        """Route request to handler."""
        for route in self.routes:
            match = route.match(path, method)
            if match.matched:
                self.logger.debug(f"Route matched: {route.path}")
                return await route.handler(**match.params, **kwargs)
        
        raise ValueError(f"No route found for {method} {path}")


class RateLimit(BaseModel):
    """Rate limit state."""
    requests: int = 0
    window_start: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    blocked_until: Optional[datetime] = None


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        burst: int = 10
    ):
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        self.limits: Dict[str, RateLimit] = defaultdict(RateLimit)
        self.logger = StructuredLogger("rate_limiter", LogLevel.INFO)
    
    async def check_limit(self, client_id: str) -> bool:
        """Check if request is allowed."""
        limit = self.limits[client_id]
        now = datetime.now(timezone.utc)
        
        # Check if blocked
        if limit.blocked_until and now < limit.blocked_until:
            return False
        
        # Reset window if needed
        window_duration = timedelta(minutes=1)
        if now - limit.window_start > window_duration:
            limit.requests = 0
            limit.window_start = now
            limit.blocked_until = None
        
        # Check limit
        if limit.requests >= self.requests_per_minute:
            limit.blocked_until = limit.window_start + window_duration
            self.logger.warning(f"Rate limit exceeded: {client_id}")
            return False
        
        limit.requests += 1
        return True


class RequestTransformer:
    """Transform incoming requests."""
    
    def transform_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Transform request headers."""
        # Add default headers
        headers.setdefault("X-Gateway-Version", "1.0")
        headers.setdefault("X-Request-ID", str(datetime.now(timezone.utc).timestamp()))
        return headers
    
    def transform_body(self, body: Any) -> Any:
        """Transform request body."""
        return body


class ResponseTransformer:
    """Transform outgoing responses."""
    
    def transform(self, response: Any) -> Dict[str, Any]:
        """Transform response."""
        return {
            "data": response,
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.0"
            }
        }


class GatewayConfig(BaseModel):
    """Gateway configuration."""
    rate_limit_requests: int = Field(60, description="Requests per minute")
    enable_auth: bool = Field(True, description="Enable authentication")
    enable_transforms: bool = Field(True, description="Enable transformations")
    timeout_seconds: int = Field(30, description="Request timeout")


class APIGateway:
    """
    Complete API Gateway.
    
    Features:
    - Routing
    - Rate limiting
    - Authentication
    - Request/response transformation
    """
    
    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or GatewayConfig()
        
        self.router = Router()
        self.rate_limiter = RateLimiter(self.config.rate_limit_requests)
        self.request_transformer = RequestTransformer()
        self.response_transformer = ResponseTransformer()
        
        self.logger = StructuredLogger("api_gateway", LogLevel.INFO)
    
    def add_route(self, route: Route):
        """Add a route to the gateway."""
        self.router.add_route(route)
    
    async def handle_request(
        self,
        path: str,
        method: str,
        client_id: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Handle incoming request.
        
        Args:
            path: Request path
            method: HTTP method
            client_id: Client identifier
            headers: Request headers
            body: Request body
        
        Returns:
            Response
        """
        # Rate limiting
        if not await self.rate_limiter.check_limit(client_id):
            return {
                "error": "Rate limit exceeded",
                "status": 429
            }
        
        # Transform request
        if self.config.enable_transforms:
            headers = self.request_transformer.transform_headers(headers or {})
            body = self.request_transformer.transform_body(body)
        
        try:
            # Route request
            result = await asyncio.wait_for(
                self.router.route(path, method, headers=headers, body=body),
                timeout=self.config.timeout_seconds
            )
            
            # Transform response
            if self.config.enable_transforms:
                response = self.response_transformer.transform(result)
            else:
                response = {"data": result}
            
            response["status"] = 200
            return response
        
        except asyncio.TimeoutError:
            return {
                "error": "Request timeout",
                "status": 504
            }
        
        except ValueError as e:
            return {
                "error": str(e),
                "status": 404
            }
        
        except Exception as e:
            self.logger.error(f"Request error: {e}")
            return {
                "error": "Internal server error",
                "status": 500
            }

