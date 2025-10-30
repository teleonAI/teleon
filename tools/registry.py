"""Tool registry for managing available tools."""

from typing import Dict, List, Optional
import asyncio

from teleon.tools.base import BaseTool, ToolCategory, ToolSchema


class ToolRegistry:
    """
    Central registry for all available tools.
    
    Manages tool registration, discovery, and retrieval.
    """
    
    def __init__(self):
        """Initialize the registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._lock = asyncio.Lock()
    
    async def register(self, tool: BaseTool) -> None:
        """
        Register a tool.
        
        Args:
            tool: Tool instance to register
        """
        async with self._lock:
            tool_name = tool.name
            
            if tool_name in self._tools:
                print(f"⚠️  Tool '{tool_name}' already registered, replacing...")
            
            self._tools[tool_name] = tool
            print(f"✓ Registered tool: {tool_name} ({tool.category.value})")
    
    async def unregister(self, tool_name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            tool_name: Name of tool to unregister
        
        Returns:
            True if unregistered, False if not found
        """
        async with self._lock:
            if tool_name in self._tools:
                del self._tools[tool_name]
                print(f"✓ Unregistered tool: {tool_name}")
                return True
            return False
    
    def get(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool
        
        Returns:
            Tool instance or None
        """
        return self._tools.get(tool_name)
    
    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        tags: Optional[List[str]] = None
    ) -> List[BaseTool]:
        """
        List available tools.
        
        Args:
            category: Filter by category
            tags: Filter by tags
        
        Returns:
            List of matching tools
        """
        tools = list(self._tools.values())
        
        # Filter by category
        if category:
            tools = [t for t in tools if t.category == category]
        
        # Filter by tags
        if tags:
            tools = [
                t for t in tools
                if any(tag in t.get_schema().tags for tag in tags)
            ]
        
        return tools
    
    def get_schemas(
        self,
        category: Optional[ToolCategory] = None
    ) -> List[ToolSchema]:
        """
        Get tool schemas.
        
        Args:
            category: Filter by category
        
        Returns:
            List of tool schemas
        """
        tools = self.list_tools(category=category)
        return [tool.get_schema() for tool in tools]
    
    def search(self, query: str) -> List[BaseTool]:
        """
        Search tools by name or description.
        
        Args:
            query: Search query
        
        Returns:
            List of matching tools
        """
        query_lower = query.lower()
        
        matching_tools = []
        for tool in self._tools.values():
            schema = tool.get_schema()
            
            # Search in name, description, and tags
            if (query_lower in schema.name.lower() or
                query_lower in schema.description.lower() or
                any(query_lower in tag.lower() for tag in schema.tags)):
                matching_tools.append(tool)
        
        return matching_tools
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get registry statistics.
        
        Returns:
            Statistics dictionary
        """
        category_counts = {}
        for tool in self._tools.values():
            category = tool.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "total_tools": len(self._tools),
            "by_category": category_counts,
            "tool_names": list(self._tools.keys())
        }


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """
    Get the global tool registry.
    
    Returns:
        Global registry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry

