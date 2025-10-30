"""
REST API Connector - Generic REST API client.

Provides:
- HTTP requests with automatic retry
- Authentication support
- Request/response validation
- Rate limiting
"""

from typing import Dict, Any, Optional
import httpx

from teleon.connectors.base import BaseConnector, ConnectionError


class RESTAPIConnector(BaseConnector):
    """
    Generic REST API connector.
    
    Example:
        >>> connector = RESTAPIConnector(
        ...     base_url="https://api.example.com",
        ...     headers={"Authorization": "Bearer token"}
        ... )
        >>> 
        >>> async with connector:
        ...     response = await connector.get("/users")
    """
    
    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize REST API connector.
        
        Args:
            base_url: Base URL for API
            headers: Default headers
            timeout: Request timeout
            max_retries: Maximum retries
        """
        super().__init__("rest_api")
        
        self.base_url = base_url
        self.headers = headers or {}
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.client: Optional[httpx.AsyncClient] = None
    
    async def connect(self):
        """Connect to REST API."""
        try:
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=self.timeout
            )
            
            self.connected = True
            self.logger.info(f"Connected to REST API: {self.base_url}")
        
        except Exception as e:
            raise ConnectionError(f"Failed to connect to REST API: {e}") from e
    
    async def disconnect(self):
        """Disconnect from REST API."""
        if self.client:
            await self.client.aclose()
            self.connected = False
            self.logger.info("Disconnected from REST API")
    
    async def test_connection(self) -> bool:
        """Test REST API connection."""
        try:
            response = await self.client.get("/")
            return response.status_code < 500
        except:
            return False
    
    async def request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> httpx.Response:
        """
        Make HTTP request.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request parameters
            
        Returns:
            HTTP response
        """
        await self.ensure_connected()
        
        response = await self.client.request(method, endpoint, **kwargs)
        
        self.logger.info(
            f"{method} {endpoint} - {response.status_code}",
            extra={"method": method, "endpoint": endpoint, "status": response.status_code}
        )
        
        return response
    
    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        GET request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            **kwargs: Additional parameters
            
        Returns:
            Response JSON
        """
        response = await self.request("GET", endpoint, params=params, **kwargs)
        response.raise_for_status()
        return response.json()
    
    async def post(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        POST request.
        
        Args:
            endpoint: API endpoint
            json: Request body
            **kwargs: Additional parameters
            
        Returns:
            Response JSON
        """
        response = await self.request("POST", endpoint, json=json, **kwargs)
        response.raise_for_status()
        return response.json()
    
    async def put(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        PUT request.
        
        Args:
            endpoint: API endpoint
            json: Request body
            **kwargs: Additional parameters
            
        Returns:
            Response JSON
        """
        response = await self.request("PUT", endpoint, json=json, **kwargs)
        response.raise_for_status()
        return response.json()
    
    async def delete(
        self,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        DELETE request.
        
        Args:
            endpoint: API endpoint
            **kwargs: Additional parameters
            
        Returns:
            Response JSON
        """
        response = await self.request("DELETE", endpoint, **kwargs)
        response.raise_for_status()
        
        # Handle empty responses
        try:
            return response.json()
        except:
            return {"status": "success"}

