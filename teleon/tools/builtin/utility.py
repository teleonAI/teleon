"""Utility tools."""

from typing import Any
import datetime
import hashlib
import uuid
from teleon.tools.base import BaseTool, ToolResult, ToolSchema, ToolCategory


class GenerateUUIDTool(BaseTool):
    """Generate UUID."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Generate UUID."""
        version = kwargs.get("version", 4)
        
        try:
            if version == 4:
                generated_uuid = str(uuid.uuid4())
            else:
                generated_uuid = str(uuid.uuid4())
            
            return ToolResult(
                success=True,
                data={"uuid": generated_uuid},
                tool_name=self.name
            )
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="generate_uuid",
            description="Generate a UUID",
            category=ToolCategory.UTILITY,
            parameters={
                "type": "object",
                "properties": {
                    "version": {"type": "integer"}
                }
            },
            returns={"type": "object"},
            tags=["uuid", "generate", "utility"]
        )


class HashStringTool(BaseTool):
    """Hash a string."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Hash string."""
        text = kwargs.get("text")
        algorithm = kwargs.get("algorithm", "sha256")
        
        try:
            if algorithm == "md5":
                hash_obj = hashlib.md5(text.encode())
            elif algorithm == "sha1":
                hash_obj = hashlib.sha1(text.encode())
            elif algorithm == "sha256":
                hash_obj = hashlib.sha256(text.encode())
            else:
                return ToolResult(
                    success=False,
                    error=f"Unsupported algorithm: {algorithm}",
                    tool_name=self.name
                )
            
            return ToolResult(
                success=True,
                data={
                    "hash": hash_obj.hexdigest(),
                    "algorithm": algorithm
                },
                tool_name=self.name
            )
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="hash_string",
            description="Hash a string using MD5, SHA1, or SHA256",
            category=ToolCategory.UTILITY,
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "algorithm": {"type": "string", "enum": ["md5", "sha1", "sha256"]}
                },
                "required": ["text"]
            },
            returns={"type": "object"},
            tags=["hash", "crypto", "utility"]
        )


class GetTimestampTool(BaseTool):
    """Get current timestamp."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Get timestamp."""
        format_type = kwargs.get("format", "iso")
        
        try:
            now = datetime.datetime.now(timezone.utc)
            
            if format_type == "iso":
                timestamp = now.isoformat()
            elif format_type == "unix":
                timestamp = int(now.timestamp())
            elif format_type == "unix_ms":
                timestamp = int(now.timestamp() * 1000)
            else:
                timestamp = now.isoformat()
            
            return ToolResult(
                success=True,
                data={
                    "timestamp": timestamp,
                    "format": format_type
                },
                tool_name=self.name
            )
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="get_timestamp",
            description="Get current timestamp in various formats",
            category=ToolCategory.UTILITY,
            parameters={
                "type": "object",
                "properties": {
                    "format": {"type": "string", "enum": ["iso", "unix", "unix_ms"]}
                }
            },
            returns={"type": "object"},
            tags=["timestamp", "time", "utility"]
        )


class SleepTool(BaseTool):
    """Sleep/wait for specified duration."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Sleep."""
        import asyncio
        
        seconds = kwargs.get("seconds", 1)
        
        try:
            await asyncio.sleep(seconds)
            
            return ToolResult(
                success=True,
                data={"slept_seconds": seconds},
                tool_name=self.name
            )
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="sleep",
            description="Sleep/wait for specified duration",
            category=ToolCategory.UTILITY,
            parameters={
                "type": "object",
                "properties": {
                    "seconds": {"type": "number"}
                },
                "required": ["seconds"]
            },
            returns={"type": "object"},
            tags=["sleep", "wait", "delay", "utility"]
        )

