"""Analytics and statistics tools."""

from typing import Any, List
from teleon.tools.base import BaseTool, ToolResult, ToolSchema, ToolCategory


class CalculateStatsTool(BaseTool):
    """Calculate statistics for numerical data."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Calculate statistics."""
        data = kwargs.get("data", [])
        
        try:
            if not data:
                return ToolResult(
                    success=False,
                    error="No data provided",
                    tool_name=self.name
                )
            
            # Calculate stats
            mean = sum(data) / len(data)
            sorted_data = sorted(data)
            n = len(sorted_data)
            
            if n % 2 == 0:
                median = (sorted_data[n//2-1] + sorted_data[n//2]) / 2
            else:
                median = sorted_data[n//2]
            
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            std_dev = variance ** 0.5
            
            return ToolResult(
                success=True,
                data={
                    "count": len(data),
                    "sum": sum(data),
                    "mean": mean,
                    "median": median,
                    "min": min(data),
                    "max": max(data),
                    "std_dev": std_dev,
                    "variance": variance
                },
                tool_name=self.name
            )
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="calculate_stats",
            description="Calculate statistics (mean, median, std dev, etc.)",
            category=ToolCategory.ANALYTICS,
            parameters={
                "type": "object",
                "properties": {
                    "data": {"type": "array", "items": {"type": "number"}}
                },
                "required": ["data"]
            },
            returns={"type": "object"},
            tags=["statistics", "analytics", "math"]
        )


class CalculatePercentileTool(BaseTool):
    """Calculate percentile."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Calculate percentile."""
        data = kwargs.get("data", [])
        percentile = kwargs.get("percentile", 50)
        
        try:
            if not data:
                return ToolResult(
                    success=False,
                    error="No data provided",
                    tool_name=self.name
                )
            
            sorted_data = sorted(data)
            n = len(sorted_data)
            k = (n - 1) * (percentile / 100)
            f = int(k)
            c = k - f
            
            if f + 1 < n:
                result = sorted_data[f] + c * (sorted_data[f + 1] - sorted_data[f])
            else:
                result = sorted_data[f]
            
            return ToolResult(
                success=True,
                data={
                    "percentile": percentile,
                    "value": result
                },
                tool_name=self.name
            )
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="calculate_percentile",
            description="Calculate percentile for dataset",
            category=ToolCategory.ANALYTICS,
            parameters={
                "type": "object",
                "properties": {
                    "data": {"type": "array", "items": {"type": "number"}},
                    "percentile": {"type": "number"}
                },
                "required": ["data"]
            },
            returns={"type": "object"},
            tags=["percentile", "analytics", "statistics"]
        )


class TimeSeriesAggregationTool(BaseTool):
    """Aggregate time series data."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Aggregate time series."""
        data = kwargs.get("data", [])  # List of {timestamp, value}
        interval = kwargs.get("interval", "hour")  # hour, day, week
        aggregation = kwargs.get("aggregation", "sum")  # sum, avg, min, max
        
        try:
            # Placeholder - simple aggregation
            return ToolResult(
                success=True,
                data={
                    "aggregated": data,
                    "interval": interval,
                    "type": aggregation
                },
                tool_name=self.name
            )
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="timeseries_aggregate",
            description="Aggregate time series data by interval",
            category=ToolCategory.ANALYTICS,
            parameters={
                "type": "object",
                "properties": {
                    "data": {"type": "array"},
                    "interval": {"type": "string"},
                    "aggregation": {"type": "string"}
                },
                "required": ["data"]
            },
            returns={"type": "object"},
            tags=["timeseries", "analytics", "aggregation"]
        )


class CorrelationTool(BaseTool):
    """Calculate correlation between datasets."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Calculate correlation."""
        data_x = kwargs.get("data_x", [])
        data_y = kwargs.get("data_y", [])
        
        try:
            if len(data_x) != len(data_y):
                return ToolResult(
                    success=False,
                    error="Data arrays must have same length",
                    tool_name=self.name
                )
            
            if not data_x:
                return ToolResult(
                    success=False,
                    error="No data provided",
                    tool_name=self.name
                )
            
            # Calculate Pearson correlation
            mean_x = sum(data_x) / len(data_x)
            mean_y = sum(data_y) / len(data_y)
            
            numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(data_x, data_y))
            denominator_x = sum((x - mean_x) ** 2 for x in data_x) ** 0.5
            denominator_y = sum((y - mean_y) ** 2 for y in data_y) ** 0.5
            
            if denominator_x == 0 or denominator_y == 0:
                correlation = 0
            else:
                correlation = numerator / (denominator_x * denominator_y)
            
            return ToolResult(
                success=True,
                data={"correlation": correlation},
                tool_name=self.name
            )
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="correlation",
            description="Calculate Pearson correlation between two datasets",
            category=ToolCategory.ANALYTICS,
            parameters={
                "type": "object",
                "properties": {
                    "data_x": {"type": "array", "items": {"type": "number"}},
                    "data_y": {"type": "array", "items": {"type": "number"}}
                },
                "required": ["data_x", "data_y"]
            },
            returns={"type": "object"},
            tags=["correlation", "analytics", "statistics"]
        )

