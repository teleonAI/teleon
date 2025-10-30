"""Built-in tools for Teleon."""

from teleon.tools.builtin.data import (
    JSONParserTool,
    CSVParserTool,
    DataTransformTool,
    DataValidatorTool,
    FormatConverterTool
)
from teleon.tools.builtin.web import (
    HTTPRequestTool,
    WebScraperTool,
    URLValidatorTool,
    APIClientTool,
    WebhookTool
)
from teleon.tools.builtin.files import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
    FileInfoTool,
    SearchFilesTool
)
from teleon.tools.builtin.database import (
    SQLQueryTool,
    RedisGetTool,
    RedisSetTool
)
from teleon.tools.builtin.communication import (
    SendEmailTool,
    SendSMSTool,
    SlackMessageTool
)
from teleon.tools.builtin.analytics import (
    CalculateStatsTool,
    CalculatePercentileTool,
    TimeSeriesAggregationTool,
    CorrelationTool
)
from teleon.tools.builtin.utility import (
    GenerateUUIDTool,
    HashStringTool,
    GetTimestampTool,
    SleepTool
)

__all__ = [
    # Data tools (5)
    "JSONParserTool",
    "CSVParserTool",
    "DataTransformTool",
    "DataValidatorTool",
    "FormatConverterTool",
    # Web tools (5)
    "HTTPRequestTool",
    "WebScraperTool",
    "URLValidatorTool",
    "APIClientTool",
    "WebhookTool",
    # File tools (5)
    "ReadFileTool",
    "WriteFileTool",
    "ListDirectoryTool",
    "FileInfoTool",
    "SearchFilesTool",
    # Database tools (3)
    "SQLQueryTool",
    "RedisGetTool",
    "RedisSetTool",
    # Communication tools (3)
    "SendEmailTool",
    "SendSMSTool",
    "SlackMessageTool",
    # Analytics tools (4)
    "CalculateStatsTool",
    "CalculatePercentileTool",
    "TimeSeriesAggregationTool",
    "CorrelationTool",
    # Utility tools (4)
    "GenerateUUIDTool",
    "HashStringTool",
    "GetTimestampTool",
    "SleepTool",
]


async def register_all_builtin_tools():
    """Register all built-in tools with the global registry."""
    from teleon.tools.registry import get_registry
    
    registry = get_registry()
    
    # Data tools (5)
    await registry.register(JSONParserTool())
    await registry.register(CSVParserTool())
    await registry.register(DataTransformTool())
    await registry.register(DataValidatorTool())
    await registry.register(FormatConverterTool())
    
    # Web tools (5)
    await registry.register(HTTPRequestTool())
    await registry.register(WebScraperTool())
    await registry.register(URLValidatorTool())
    await registry.register(APIClientTool())
    await registry.register(WebhookTool())
    
    # File tools (5)
    await registry.register(ReadFileTool())
    await registry.register(WriteFileTool())
    await registry.register(ListDirectoryTool())
    await registry.register(FileInfoTool())
    await registry.register(SearchFilesTool())
    
    # Database tools (3)
    await registry.register(SQLQueryTool())
    await registry.register(RedisGetTool())
    await registry.register(RedisSetTool())
    
    # Communication tools (3)
    await registry.register(SendEmailTool())
    await registry.register(SendSMSTool())
    await registry.register(SlackMessageTool())
    
    # Analytics tools (4)
    await registry.register(CalculateStatsTool())
    await registry.register(CalculatePercentileTool())
    await registry.register(TimeSeriesAggregationTool())
    await registry.register(CorrelationTool())
    
    # Utility tools (4)
    await registry.register(GenerateUUIDTool())
    await registry.register(HashStringTool())
    await registry.register(GetTimestampTool())
    await registry.register(SleepTool())
    
    print(f"✓ Registered {len(registry._tools)} built-in tools")

