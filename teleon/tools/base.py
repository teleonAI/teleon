"""Base classes for tools."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from enum import Enum


class ToolCategory(str, Enum):
    """Tool categories."""
    DATA = "data"
    WEB = "web"
    FILE = "file"
    DATABASE = "database"
    COMMUNICATION = "communication"
    IMAGE = "image"
    ANALYTICS = "analytics"
    UTILITY = "utility"
    AI_ML = "ai_ml"


class ToolResult(BaseModel):
    """Result from a tool execution."""
    
    success: bool = Field(..., description="Whether execution was successful")
    data: Any = Field(None, description="Result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Execution info
    tool_name: str = Field(..., description="Name of the tool")
    execution_time_ms: Optional[float] = Field(None, description="Execution time in ms")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Execution timestamp")


class ToolSchema(BaseModel):
    """Schema for a tool."""
    
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    category: ToolCategory = Field(..., description="Tool category")
    parameters: Dict[str, Any] = Field(..., description="Parameter schema")
    returns: Dict[str, Any] = Field(..., description="Return value schema")
    examples: List[Dict[str, Any]] = Field(default_factory=list, description="Usage examples")
    
    # Metadata
    version: str = Field("1.0.0", description="Tool version")
    requires: List[str] = Field(default_factory=list, description="Required dependencies/permissions")
    tags: List[str] = Field(default_factory=list, description="Search tags")


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    All tool implementations must inherit from this class.
    """
    
    def __init__(self):
        """Initialize the tool."""
        self._schema: Optional[ToolSchema] = None
    
    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool with given parameters.
        
        Args:
            **kwargs: Tool parameters
        
        Returns:
            ToolResult with execution outcome
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """
        Get the tool's schema.
        
        Returns:
            Tool schema
        """
        pass
    
    @property
    def name(self) -> str:
        """Get tool name."""
        return self.get_schema().name
    
    @property
    def description(self) -> str:
        """Get tool description."""
        return self.get_schema().description
    
    @property
    def category(self) -> ToolCategory:
        """Get tool category."""
        return self.get_schema().category
    
    def validate_parameters(self, **kwargs: Any) -> bool:
        """
        Validate tool parameters.
        
        Args:
            **kwargs: Parameters to validate
        
        Returns:
            True if valid
        
        Raises:
            ValueError: If parameters are invalid
        """
        schema = self.get_schema()
        required_params = schema.parameters.get("required", [])
        
        # Check required parameters
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")
        
        return True
    
    async def safe_execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute tool with error handling.
        
        Args:
            **kwargs: Tool parameters
        
        Returns:
            ToolResult with execution outcome
        """
        import time
        
        start_time = time.time()
        
        try:
            # Validate parameters
            self.validate_parameters(**kwargs)
            
            # Execute tool
            result = await self.execute(**kwargs)
            
            # Add execution time
            execution_time = (time.time() - start_time) * 1000
            result.execution_time_ms = execution_time
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                tool_name=self.name,
                execution_time_ms=execution_time,
                metadata={"exception_type": type(e).__name__}
            )

