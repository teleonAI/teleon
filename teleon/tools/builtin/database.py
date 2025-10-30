"""Database tools."""

from typing import Any
from teleon.tools.base import BaseTool, ToolResult, ToolSchema, ToolCategory


class SQLQueryTool(BaseTool):
    """Execute SQL queries (read-only for safety)."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute SQL query."""
        query = kwargs.get("query", "")
        connection_string = kwargs.get("connection_string")
        
        try:
            # Validate read-only
            query_upper = query.upper().strip()
            dangerous_keywords = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE"]
            
            if any(kw in query_upper for kw in dangerous_keywords):
                return ToolResult(
                    success=False,
                    error="Only SELECT queries allowed for safety",
                    tool_name=self.name
                )
            
            # NOTE: In production, you'd actually execute the query
            # For now, return a placeholder
            return ToolResult(
                success=True,
                data={"message": "Query would be executed", "query": query},
                tool_name=self.name,
                metadata={"rows": 0}
            )
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="sql_query",
            description="Execute read-only SQL SELECT queries",
            category=ToolCategory.DATABASE,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "connection_string": {"type": "string"}
                },
                "required": ["query"]
            },
            returns={"type": "object"},
            tags=["database", "sql", "query"]
        )


class RedisGetTool(BaseTool):
    """Get value from Redis."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Get from Redis."""
        key = kwargs.get("key")
        
        try:
            # Placeholder - in production, connect to Redis
            return ToolResult(
                success=True,
                data={"key": key, "value": None},
                tool_name=self.name
            )
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="redis_get",
            description="Get value from Redis cache",
            category=ToolCategory.DATABASE,
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string"}
                },
                "required": ["key"]
            },
            returns={"type": "object"},
            tags=["redis", "cache", "get"]
        )


class RedisSetTool(BaseTool):
    """Set value in Redis."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Set in Redis."""
        key = kwargs.get("key")
        value = kwargs.get("value")
        ttl = kwargs.get("ttl")  # Seconds
        
        try:
            # Placeholder
            return ToolResult(
                success=True,
                data={"key": key, "set": True},
                tool_name=self.name
            )
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="redis_set",
            description="Set value in Redis cache with optional TTL",
            category=ToolCategory.DATABASE,
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "any"},
                    "ttl": {"type": "integer"}
                },
                "required": ["key", "value"]
            },
            returns={"type": "object"},
            tags=["redis", "cache", "set"]
        )

