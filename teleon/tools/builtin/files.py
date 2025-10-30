"""File operation tools."""

import os
from pathlib import Path
from typing import Any
import glob

from teleon.tools.base import BaseTool, ToolResult, ToolSchema, ToolCategory


class ReadFileTool(BaseTool):
    """Read file contents."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute file read."""
        file_path = kwargs.get("path")
        encoding = kwargs.get("encoding", "utf-8")
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            return ToolResult(
                success=True,
                data=content,
                tool_name=self.name,
                metadata={"size": len(content), "path": file_path}
            )
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="read_file",
            description="Read contents of a file",
            category=ToolCategory.FILE,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "encoding": {"type": "string"}
                },
                "required": ["path"]
            },
            returns={"type": "string"},
            tags=["file", "read", "io"]
        )


class WriteFileTool(BaseTool):
    """Write content to a file."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute file write."""
        file_path = kwargs.get("path")
        content = kwargs.get("content")
        mode = kwargs.get("mode", "w")  # w, a
        encoding = kwargs.get("encoding", "utf-8")
        
        try:
            # Create directory if it doesn't exist
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, mode, encoding=encoding) as f:
                f.write(content)
            
            return ToolResult(
                success=True,
                data={"path": file_path, "bytes_written": len(content)},
                tool_name=self.name
            )
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="write_file",
            description="Write content to a file",
            category=ToolCategory.FILE,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "mode": {"type": "string", "enum": ["w", "a"]},
                    "encoding": {"type": "string"}
                },
                "required": ["path", "content"]
            },
            returns={"type": "object"},
            tags=["file", "write", "io"]
        )


class ListDirectoryTool(BaseTool):
    """List directory contents."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute directory listing."""
        dir_path = kwargs.get("path", ".")
        include_hidden = kwargs.get("include_hidden", False)
        
        try:
            entries = []
            for entry in os.listdir(dir_path):
                if not include_hidden and entry.startswith('.'):
                    continue
                
                full_path = os.path.join(dir_path, entry)
                entries.append({
                    "name": entry,
                    "path": full_path,
                    "is_dir": os.path.isdir(full_path),
                    "size": os.path.getsize(full_path) if os.path.isfile(full_path) else None
                })
            
            return ToolResult(
                success=True,
                data=entries,
                tool_name=self.name,
                metadata={"count": len(entries), "path": dir_path}
            )
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="list_directory",
            description="List contents of a directory",
            category=ToolCategory.FILE,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "include_hidden": {"type": "boolean"}
                }
            },
            returns={"type": "array"},
            tags=["directory", "list", "files"]
        )


class FileInfoTool(BaseTool):
    """Get file information."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute file info retrieval."""
        file_path = kwargs.get("path")
        
        try:
            path = Path(file_path)
            stat = path.stat()
            
            return ToolResult(
                success=True,
                data={
                    "name": path.name,
                    "path": str(path.absolute()),
                    "size": stat.st_size,
                    "is_file": path.is_file(),
                    "is_dir": path.is_dir(),
                    "extension": path.suffix,
                    "created": stat.st_ctime,
                    "modified": stat.st_mtime,
                    "accessed": stat.st_atime
                },
                tool_name=self.name
            )
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="file_info",
            description="Get detailed information about a file",
            category=ToolCategory.FILE,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            },
            returns={"type": "object"},
            tags=["file", "info", "metadata"]
        )


class SearchFilesTool(BaseTool):
    """Search for files matching patterns."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute file search."""
        directory = kwargs.get("directory", ".")
        pattern = kwargs.get("pattern", "*")
        recursive = kwargs.get("recursive", False)
        
        try:
            if recursive:
                search_pattern = os.path.join(directory, "**", pattern)
                files = glob.glob(search_pattern, recursive=True)
            else:
                search_pattern = os.path.join(directory, pattern)
                files = glob.glob(search_pattern)
            
            results = [
                {
                    "path": f,
                    "name": os.path.basename(f),
                    "size": os.path.getsize(f) if os.path.isfile(f) else None
                }
                for f in files
            ]
            
            return ToolResult(
                success=True,
                data=results,
                tool_name=self.name,
                metadata={"count": len(results), "pattern": pattern}
            )
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="search_files",
            description="Search for files matching a pattern",
            category=ToolCategory.FILE,
            parameters={
                "type": "object",
                "properties": {
                    "directory": {"type": "string"},
                    "pattern": {"type": "string"},
                    "recursive": {"type": "boolean"}
                }
            },
            returns={"type": "array"},
            tags=["search", "find", "files", "glob"]
        )

