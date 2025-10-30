"""Data processing tools."""

import json
import csv
from io import StringIO
from typing import Any, Dict, List

from teleon.tools.base import BaseTool, ToolResult, ToolSchema, ToolCategory


class JSONParserTool(BaseTool):
    """Parse and manipulate JSON data."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute JSON parsing."""
        json_string = kwargs.get("json_string")
        operation = kwargs.get("operation", "parse")  # parse, stringify, validate
        
        try:
            if operation == "parse":
                data = json.loads(json_string)
                return ToolResult(
                    success=True,
                    data=data,
                    tool_name=self.name,
                    metadata={"operation": "parse"}
                )
            
            elif operation == "stringify":
                json_data = kwargs.get("data")
                indent = kwargs.get("indent", 2)
                result = json.dumps(json_data, indent=indent)
                return ToolResult(
                    success=True,
                    data=result,
                    tool_name=self.name,
                    metadata={"operation": "stringify"}
                )
            
            elif operation == "validate":
                try:
                    json.loads(json_string)
                    return ToolResult(
                        success=True,
                        data={"valid": True},
                        tool_name=self.name
                    )
                except json.JSONDecodeError as e:
                    return ToolResult(
                        success=True,
                        data={"valid": False, "error": str(e)},
                        tool_name=self.name
                    )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                tool_name=self.name
            )
    
    def get_schema(self) -> ToolSchema:
        """Get tool schema."""
        return ToolSchema(
            name="json_parser",
            description="Parse, stringify, and validate JSON data",
            category=ToolCategory.DATA,
            parameters={
                "type": "object",
                "properties": {
                    "json_string": {"type": "string", "description": "JSON string to parse"},
                    "data": {"type": "any", "description": "Data to stringify"},
                    "operation": {"type": "string", "enum": ["parse", "stringify", "validate"]},
                    "indent": {"type": "integer", "description": "Indentation for stringify"}
                },
                "required": []
            },
            returns={
                "type": "object",
                "properties": {
                    "data": {"type": "any"}
                }
            },
            tags=["json", "parse", "data", "format"]
        )


class CSVParserTool(BaseTool):
    """Parse and manipulate CSV data."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute CSV parsing."""
        csv_string = kwargs.get("csv_string")
        operation = kwargs.get("operation", "parse")  # parse, create
        delimiter = kwargs.get("delimiter", ",")
        
        try:
            if operation == "parse":
                csv_file = StringIO(csv_string)
                reader = csv.DictReader(csv_file, delimiter=delimiter)
                data = list(reader)
                
                return ToolResult(
                    success=True,
                    data=data,
                    tool_name=self.name,
                    metadata={"rows": len(data), "operation": "parse"}
                )
            
            elif operation == "create":
                data = kwargs.get("data", [])
                if not data:
                    raise ValueError("No data provided")
                
                output = StringIO()
                if isinstance(data[0], dict):
                    writer = csv.DictWriter(output, fieldnames=data[0].keys(), delimiter=delimiter)
                    writer.writeheader()
                    writer.writerows(data)
                else:
                    writer = csv.writer(output, delimiter=delimiter)
                    writer.writerows(data)
                
                return ToolResult(
                    success=True,
                    data=output.getvalue(),
                    tool_name=self.name,
                    metadata={"operation": "create"}
                )
        
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                tool_name=self.name
            )
    
    def get_schema(self) -> ToolSchema:
        """Get tool schema."""
        return ToolSchema(
            name="csv_parser",
            description="Parse and create CSV data",
            category=ToolCategory.DATA,
            parameters={
                "type": "object",
                "properties": {
                    "csv_string": {"type": "string"},
                    "data": {"type": "array"},
                    "operation": {"type": "string", "enum": ["parse", "create"]},
                    "delimiter": {"type": "string"}
                }
            },
            returns={"type": "any"},
            tags=["csv", "parse", "data"]
        )


class DataTransformTool(BaseTool):
    """Transform data structures."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute data transformation."""
        data = kwargs.get("data")
        operation = kwargs.get("operation")  # filter, map, reduce, sort
        
        try:
            if operation == "filter":
                # Simple key-value filtering
                filter_key = kwargs.get("key")
                filter_value = kwargs.get("value")
                
                if isinstance(data, list):
                    result = [
                        item for item in data
                        if isinstance(item, dict) and item.get(filter_key) == filter_value
                    ]
                else:
                    result = data
                
                return ToolResult(
                    success=True,
                    data=result,
                    tool_name=self.name,
                    metadata={"operation": "filter", "result_count": len(result)}
                )
            
            elif operation == "sort":
                sort_key = kwargs.get("key")
                reverse = kwargs.get("reverse", False)
                
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    result = sorted(data, key=lambda x: x.get(sort_key, ""), reverse=reverse)
                else:
                    result = sorted(data, reverse=reverse)
                
                return ToolResult(
                    success=True,
                    data=result,
                    tool_name=self.name,
                    metadata={"operation": "sort"}
                )
            
            elif operation == "group_by":
                group_key = kwargs.get("key")
                
                if not isinstance(data, list):
                    raise ValueError("Data must be a list for group_by")
                
                groups = {}
                for item in data:
                    if isinstance(item, dict):
                        key_value = item.get(group_key)
                        if key_value not in groups:
                            groups[key_value] = []
                        groups[key_value].append(item)
                
                return ToolResult(
                    success=True,
                    data=groups,
                    tool_name=self.name,
                    metadata={"operation": "group_by", "groups": len(groups)}
                )
        
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                tool_name=self.name
            )
    
    def get_schema(self) -> ToolSchema:
        """Get tool schema."""
        return ToolSchema(
            name="data_transform",
            description="Transform data structures (filter, sort, group)",
            category=ToolCategory.DATA,
            parameters={
                "type": "object",
                "properties": {
                    "data": {"type": "any"},
                    "operation": {"type": "string", "enum": ["filter", "sort", "group_by"]},
                    "key": {"type": "string"},
                    "value": {"type": "any"},
                    "reverse": {"type": "boolean"}
                },
                "required": ["data", "operation"]
            },
            returns={"type": "any"},
            tags=["transform", "filter", "sort", "data"]
        )


class DataValidatorTool(BaseTool):
    """Validate data against schemas."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute data validation."""
        data = kwargs.get("data")
        validation_type = kwargs.get("type", "type_check")
        
        try:
            if validation_type == "type_check":
                expected_type = kwargs.get("expected_type")
                type_map = {
                    "string": str,
                    "number": (int, float),
                    "integer": int,
                    "boolean": bool,
                    "array": list,
                    "object": dict
                }
                
                python_type = type_map.get(expected_type)
                if python_type and isinstance(data, python_type):
                    return ToolResult(
                        success=True,
                        data={"valid": True, "type": expected_type},
                        tool_name=self.name
                    )
                else:
                    return ToolResult(
                        success=True,
                        data={
                            "valid": False,
                            "expected": expected_type,
                            "actual": type(data).__name__
                        },
                        tool_name=self.name
                    )
            
            elif validation_type == "required_fields":
                required_fields = kwargs.get("required_fields", [])
                
                if not isinstance(data, dict):
                    return ToolResult(
                        success=True,
                        data={"valid": False, "error": "Data must be an object"},
                        tool_name=self.name
                    )
                
                missing_fields = [f for f in required_fields if f not in data]
                
                return ToolResult(
                    success=True,
                    data={
                        "valid": len(missing_fields) == 0,
                        "missing_fields": missing_fields
                    },
                    tool_name=self.name
                )
        
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                tool_name=self.name
            )
    
    def get_schema(self) -> ToolSchema:
        """Get tool schema."""
        return ToolSchema(
            name="data_validator",
            description="Validate data types and required fields",
            category=ToolCategory.DATA,
            parameters={
                "type": "object",
                "properties": {
                    "data": {"type": "any"},
                    "type": {"type": "string", "enum": ["type_check", "required_fields"]},
                    "expected_type": {"type": "string"},
                    "required_fields": {"type": "array"}
                },
                "required": ["data"]
            },
            returns={"type": "object"},
            tags=["validate", "schema", "data"]
        )


class FormatConverterTool(BaseTool):
    """Convert between different data formats."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute format conversion."""
        data = kwargs.get("data")
        from_format = kwargs.get("from_format")
        to_format = kwargs.get("to_format")
        
        try:
            # Parse input format
            if from_format == "json":
                parsed_data = json.loads(data) if isinstance(data, str) else data
            elif from_format == "csv":
                csv_file = StringIO(data)
                reader = csv.DictReader(csv_file)
                parsed_data = list(reader)
            else:
                parsed_data = data
            
            # Convert to output format
            if to_format == "json":
                result = json.dumps(parsed_data, indent=2)
            elif to_format == "csv":
                if not isinstance(parsed_data, list):
                    raise ValueError("Data must be a list for CSV conversion")
                
                output = StringIO()
                if parsed_data and isinstance(parsed_data[0], dict):
                    writer = csv.DictWriter(output, fieldnames=parsed_data[0].keys())
                    writer.writeheader()
                    writer.writerows(parsed_data)
                else:
                    writer = csv.writer(output)
                    writer.writerows(parsed_data)
                result = output.getvalue()
            else:
                result = str(parsed_data)
            
            return ToolResult(
                success=True,
                data=result,
                tool_name=self.name,
                metadata={"from": from_format, "to": to_format}
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                tool_name=self.name
            )
    
    def get_schema(self) -> ToolSchema:
        """Get tool schema."""
        return ToolSchema(
            name="format_converter",
            description="Convert between JSON, CSV, and other formats",
            category=ToolCategory.DATA,
            parameters={
                "type": "object",
                "properties": {
                    "data": {"type": "any"},
                    "from_format": {"type": "string", "enum": ["json", "csv", "auto"]},
                    "to_format": {"type": "string", "enum": ["json", "csv", "string"]}
                },
                "required": ["data", "from_format", "to_format"]
            },
            returns={"type": "string"},
            tags=["convert", "format", "json", "csv"]
        )

