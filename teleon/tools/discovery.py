"""
Tool Discovery - Search and discover built-in tools.

Features:
- Search tools by name, description, category
- List tools by category
- Get tool documentation
- Tool recommendations
"""

from typing import Any, Dict, List, Optional
from enum import Enum

from teleon.core import StructuredLogger, LogLevel
from teleon.tools.registry import get_registry


class ToolCategory(str, Enum):
    """Tool categories."""
    DATA = "data"
    WEB = "web"
    FILES = "files"
    DATABASE = "database"
    COMMUNICATION = "communication"
    ANALYTICS = "analytics"
    UTILITY = "utility"
    ALL = "all"


class ToolDiscovery:
    """
    Tool discovery and search system.
    
    Features:
    - Search tools by query
    - Filter by category
    - Get tool details
    - Usage examples
    """
    
    # Tool metadata with categories and descriptions
    TOOL_METADATA = {
        # Data tools
        "json_parser": {
            "category": ToolCategory.DATA,
            "description": "Parse and validate JSON data",
            "keywords": ["json", "parse", "validate", "data"]
        },
        "csv_parser": {
            "category": ToolCategory.DATA,
            "description": "Parse CSV files and data",
            "keywords": ["csv", "parse", "table", "data"]
        },
        "data_transform": {
            "category": ToolCategory.DATA,
            "description": "Transform and manipulate data",
            "keywords": ["transform", "data", "manipulate", "convert"]
        },
        "data_validator": {
            "category": ToolCategory.DATA,
            "description": "Validate data against schemas",
            "keywords": ["validate", "schema", "data", "check"]
        },
        "format_converter": {
            "category": ToolCategory.DATA,
            "description": "Convert between data formats (JSON, XML, CSV)",
            "keywords": ["convert", "format", "json", "xml", "csv"]
        },
        
        # Web tools
        "http_request": {
            "category": ToolCategory.WEB,
            "description": "Make HTTP requests (GET, POST, etc.)",
            "keywords": ["http", "request", "api", "web", "rest"]
        },
        "web_scraper": {
            "category": ToolCategory.WEB,
            "description": "Scrape data from web pages",
            "keywords": ["scrape", "web", "html", "extract", "crawl"]
        },
        "url_validator": {
            "category": ToolCategory.WEB,
            "description": "Validate and parse URLs",
            "keywords": ["url", "validate", "parse", "web"]
        },
        "api_client": {
            "category": ToolCategory.WEB,
            "description": "Generic API client for REST APIs",
            "keywords": ["api", "rest", "client", "http"]
        },
        "webhook": {
            "category": ToolCategory.WEB,
            "description": "Send webhook notifications",
            "keywords": ["webhook", "notify", "http", "callback"]
        },
        
        # File tools
        "read_file": {
            "category": ToolCategory.FILES,
            "description": "Read file contents",
            "keywords": ["read", "file", "load", "open"]
        },
        "write_file": {
            "category": ToolCategory.FILES,
            "description": "Write data to files",
            "keywords": ["write", "file", "save", "create"]
        },
        "list_directory": {
            "category": ToolCategory.FILES,
            "description": "List directory contents",
            "keywords": ["list", "directory", "folder", "files"]
        },
        "file_info": {
            "category": ToolCategory.FILES,
            "description": "Get file metadata and information",
            "keywords": ["info", "metadata", "file", "stats"]
        },
        "search_files": {
            "category": ToolCategory.FILES,
            "description": "Search for files by pattern",
            "keywords": ["search", "find", "files", "pattern", "glob"]
        },
        
        # Database tools
        "sql_query": {
            "category": ToolCategory.DATABASE,
            "description": "Execute SQL queries",
            "keywords": ["sql", "query", "database", "select"]
        },
        "redis_get": {
            "category": ToolCategory.DATABASE,
            "description": "Get value from Redis",
            "keywords": ["redis", "get", "cache", "key"]
        },
        "redis_set": {
            "category": ToolCategory.DATABASE,
            "description": "Set value in Redis",
            "keywords": ["redis", "set", "cache", "key", "store"]
        },
        
        # Communication tools
        "send_email": {
            "category": ToolCategory.COMMUNICATION,
            "description": "Send email messages",
            "keywords": ["email", "send", "smtp", "message"]
        },
        "send_sms": {
            "category": ToolCategory.COMMUNICATION,
            "description": "Send SMS text messages",
            "keywords": ["sms", "text", "message", "send"]
        },
        "slack_message": {
            "category": ToolCategory.COMMUNICATION,
            "description": "Send Slack messages",
            "keywords": ["slack", "message", "send", "notify"]
        },
        
        # Analytics tools
        "calculate_stats": {
            "category": ToolCategory.ANALYTICS,
            "description": "Calculate statistical metrics (mean, median, std)",
            "keywords": ["stats", "statistics", "mean", "median", "analytics"]
        },
        "calculate_percentile": {
            "category": ToolCategory.ANALYTICS,
            "description": "Calculate percentiles from data",
            "keywords": ["percentile", "quantile", "analytics", "statistics"]
        },
        "time_series_aggregation": {
            "category": ToolCategory.ANALYTICS,
            "description": "Aggregate time-series data",
            "keywords": ["time", "series", "aggregate", "analytics"]
        },
        "correlation": {
            "category": ToolCategory.ANALYTICS,
            "description": "Calculate correlation between datasets",
            "keywords": ["correlation", "relationship", "analytics", "statistics"]
        },
        
        # Utility tools
        "generate_uuid": {
            "category": ToolCategory.UTILITY,
            "description": "Generate UUID identifiers",
            "keywords": ["uuid", "generate", "id", "unique"]
        },
        "hash_string": {
            "category": ToolCategory.UTILITY,
            "description": "Hash strings using various algorithms",
            "keywords": ["hash", "md5", "sha256", "crypto"]
        },
        "get_timestamp": {
            "category": ToolCategory.UTILITY,
            "description": "Get current timestamp",
            "keywords": ["timestamp", "time", "date", "now"]
        },
        "sleep": {
            "category": ToolCategory.UTILITY,
            "description": "Pause execution for specified duration",
            "keywords": ["sleep", "wait", "delay", "pause"]
        }
    }
    
    def __init__(self):
        """Initialize tool discovery."""
        self.logger = StructuredLogger("tool_discovery", LogLevel.INFO)
    
    def search(
        self,
        query: str = "",
        category: Optional[ToolCategory] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for tools.
        
        Args:
            query: Search query (searches name, description, keywords)
            category: Filter by category
        
        Returns:
            List of matching tools with metadata
        """
        results = []
        query_lower = query.lower()
        
        for tool_name, metadata in self.TOOL_METADATA.items():
            # Filter by category
            if category and category != ToolCategory.ALL:
                if metadata["category"] != category:
                    continue
            
            # Search in name, description, and keywords
            if query:
                searchable = (
                    tool_name.lower() + " " +
                    metadata["description"].lower() + " " +
                    " ".join(metadata["keywords"])
                )
                
                if query_lower not in searchable:
                    continue
            
            results.append({
                "name": tool_name,
                "category": metadata["category"].value,
                "description": metadata["description"],
                "keywords": metadata["keywords"]
            })
        
        return results
    
    def list_by_category(self, category: ToolCategory) -> List[Dict[str, Any]]:
        """
        List all tools in a category.
        
        Args:
            category: Tool category
        
        Returns:
            List of tools in category
        """
        return self.search(category=category)
    
    def get_tool_details(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a tool.
        
        Args:
            tool_name: Tool name
        
        Returns:
            Tool details or None if not found
        """
        if tool_name not in self.TOOL_METADATA:
            return None
        
        metadata = self.TOOL_METADATA[tool_name]
        
        # Get tool from registry for schema
        registry = get_registry()
        tool = registry.get(tool_name)
        
        details = {
            "name": tool_name,
            "category": metadata["category"].value,
            "description": metadata["description"],
            "keywords": metadata["keywords"]
        }
        
        if tool:
            details["schema"] = tool.schema
            details["available"] = True
        else:
            details["available"] = False
        
        return details
    
    def get_categories(self) -> List[str]:
        """Get list of all categories."""
        return [c.value for c in ToolCategory if c != ToolCategory.ALL]
    
    def count_by_category(self) -> Dict[str, int]:
        """Count tools in each category."""
        counts = {}
        for metadata in self.TOOL_METADATA.values():
            category = metadata["category"].value
            counts[category] = counts.get(category, 0) + 1
        return counts
    
    def recommend_tools(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Recommend tools based on task description.
        
        Args:
            task_description: Description of the task
        
        Returns:
            List of recommended tools
        """
        # Simple keyword-based recommendation
        recommendations = self.search(query=task_description)
        
        # Sort by relevance (number of keyword matches)
        task_lower = task_description.lower()
        for rec in recommendations:
            matches = sum(
                1 for keyword in rec["keywords"]
                if keyword in task_lower
            )
            rec["relevance"] = matches
        
        recommendations.sort(key=lambda x: x["relevance"], reverse=True)
        
        return recommendations[:5]  # Top 5 recommendations
    
    def list_all_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        return self.search()


# Global instance
_discovery = None


def get_discovery() -> ToolDiscovery:
    """Get global tool discovery instance."""
    global _discovery
    if _discovery is None:
        _discovery = ToolDiscovery()
    return _discovery

