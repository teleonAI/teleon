"""Web and API tools."""

import re
from typing import Any
from urllib.parse import urlparse

from teleon.tools.base import BaseTool, ToolResult, ToolSchema, ToolCategory


class HTTPRequestTool(BaseTool):
    """Make HTTP requests."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute HTTP request."""
        url = kwargs.get("url")
        method = kwargs.get("method", "GET").upper()
        headers = kwargs.get("headers", {})
        data = kwargs.get("data")
        
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data if method in ["POST", "PUT", "PATCH"] else None
                )
                
                return ToolResult(
                    success=True,
                    data={
                        "status_code": response.status_code,
                        "headers": dict(response.headers),
                        "body": response.text,
                        "json": response.json() if response.headers.get("content-type", "").startswith("application/json") else None
                    },
                    tool_name=self.name,
                    metadata={"method": method, "url": url}
                )
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="http_request",
            description="Make HTTP requests to APIs and websites",
            category=ToolCategory.WEB,
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
                    "headers": {"type": "object"},
                    "data": {"type": "any"}
                },
                "required": ["url"]
            },
            returns={"type": "object"},
            tags=["http", "api", "request", "web"]
        )


class WebScraperTool(BaseTool):
    """Extract data from web pages."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute web scraping."""
        url = kwargs.get("url")
        extract_type = kwargs.get("type", "text")  # text, links, title
        
        try:
            import httpx
            from bs4 import BeautifulSoup
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                if extract_type == "text":
                    data = soup.get_text(strip=True)
                elif extract_type == "links":
                    data = [a.get('href') for a in soup.find_all('a') if a.get('href')]
                elif extract_type == "title":
                    data = soup.title.string if soup.title else None
                else:
                    data = str(soup)
                
                return ToolResult(success=True, data=data, tool_name=self.name)
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="web_scraper",
            description="Extract text, links, and data from web pages",
            category=ToolCategory.WEB,
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "type": {"type": "string", "enum": ["text", "links", "title", "html"]}
                },
                "required": ["url"]
            },
            returns={"type": "any"},
            tags=["scrape", "web", "extract"],
            requires=["httpx", "beautifulsoup4"]
        )


class URLValidatorTool(BaseTool):
    """Validate and parse URLs."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute URL validation."""
        url = kwargs.get("url")
        
        try:
            parsed = urlparse(url)
            is_valid = all([parsed.scheme, parsed.netloc])
            
            return ToolResult(
                success=True,
                data={
                    "valid": is_valid,
                    "scheme": parsed.scheme,
                    "domain": parsed.netloc,
                    "path": parsed.path,
                    "params": parsed.params,
                    "query": parsed.query,
                    "fragment": parsed.fragment
                },
                tool_name=self.name
            )
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="url_validator",
            description="Validate and parse URLs",
            category=ToolCategory.WEB,
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"}
                },
                "required": ["url"]
            },
            returns={"type": "object"},
            tags=["url", "validate", "parse"]
        )


class APIClientTool(BaseTool):
    """Generic API client tool."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute API request with authentication."""
        url = kwargs.get("url")
        method = kwargs.get("method", "GET")
        api_key = kwargs.get("api_key")
        auth_header = kwargs.get("auth_header", "Authorization")
        
        try:
            import httpx
            
            headers = {}
            if api_key:
                headers[auth_header] = f"Bearer {api_key}"
            
            async with httpx.AsyncClient() as client:
                response = await client.request(method=method, url=url, headers=headers)
                
                return ToolResult(
                    success=True,
                    data=response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                    tool_name=self.name
                )
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="api_client",
            description="Make authenticated API requests",
            category=ToolCategory.WEB,
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "method": {"type": "string"},
                    "api_key": {"type": "string"},
                    "auth_header": {"type": "string"}
                },
                "required": ["url"]
            },
            returns={"type": "any"},
            tags=["api", "auth", "request"]
        )


class WebhookTool(BaseTool):
    """Send data to webhooks."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute webhook POST."""
        url = kwargs.get("url")
        data = kwargs.get("data")
        
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data)
                
                return ToolResult(
                    success=True,
                    data={
                        "status_code": response.status_code,
                        "response": response.text
                    },
                    tool_name=self.name
                )
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="webhook",
            description="Send data to webhook URLs",
            category=ToolCategory.WEB,
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "data": {"type": "any"}
                },
                "required": ["url", "data"]
            },
            returns={"type": "object"},
            tags=["webhook", "notify", "send"]
        )

