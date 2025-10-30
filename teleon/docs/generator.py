"""
Documentation generators.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import inspect
import json

from teleon.core import StructuredLogger, LogLevel


class OpenAPIGenerator:
    """Generate OpenAPI 3.0 specifications."""
    
    def __init__(self, title: str, version: str):
        self.title = title
        self.version = version
        self.paths: Dict[str, Dict] = {}
        self.logger = StructuredLogger("openapi_generator", LogLevel.INFO)
    
    def add_endpoint(
        self,
        path: str,
        method: str,
        description: str,
        parameters: Optional[List[Dict]] = None,
        responses: Optional[Dict] = None
    ):
        """Add API endpoint."""
        if path not in self.paths:
            self.paths[path] = {}
        
        self.paths[path][method.lower()] = {
            "description": description,
            "parameters": parameters or [],
            "responses": responses or {
                "200": {"description": "Success"}
            }
        }
    
    def generate(self) -> Dict[str, Any]:
        """Generate OpenAPI spec."""
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": self.title,
                "version": self.version
            },
            "paths": self.paths
        }
        
        self.logger.info("OpenAPI spec generated")
        return spec
    
    def to_json(self) -> str:
        """Convert to JSON."""
        return json.dumps(self.generate(), indent=2)


class MarkdownGenerator:
    """Generate Markdown documentation."""
    
    def __init__(self, title: str):
        self.title = title
        self.sections: List[str] = []
        self.logger = StructuredLogger("markdown_generator", LogLevel.INFO)
    
    def add_section(self, heading: str, content: str, level: int = 2):
        """Add a section."""
        heading_prefix = "#" * level
        self.sections.append(f"{heading_prefix} {heading}\n\n{content}\n")
    
    def add_code_block(self, code: str, language: str = "python"):
        """Add code block."""
        self.sections.append(f"```{language}\n{code}\n```\n")
    
    def generate(self) -> str:
        """Generate Markdown document."""
        doc = f"# {self.title}\n\n"
        doc += "\n".join(self.sections)
        
        self.logger.info("Markdown doc generated")
        return doc


class DocGenerator:
    """
    Complete documentation generator.
    
    Features:
    - Extract from code
    - Generate API docs
    - Create examples
    """
    
    def __init__(self, project_name: str, version: str):
        self.project_name = project_name
        self.version = version
        self.logger = StructuredLogger("doc_generator", LogLevel.INFO)
    
    def generate_function_doc(self, func) -> str:
        """Generate documentation for a function."""
        doc = f"### `{func.__name__}`\n\n"
        
        # Get signature
        sig = inspect.signature(func)
        doc += f"```python\n{func.__name__}{sig}\n```\n\n"
        
        # Get docstring
        if func.__doc__:
            doc += f"{func.__doc__.strip()}\n\n"
        
        # Parameters
        doc += "**Parameters:**\n\n"
        for param_name, param in sig.parameters.items():
            annotation = param.annotation if param.annotation != inspect.Parameter.empty else "Any"
            doc += f"- `{param_name}`: {annotation}\n"
        
        doc += "\n"
        return doc
    
    def generate_class_doc(self, cls) -> str:
        """Generate documentation for a class."""
        doc = f"## {cls.__name__}\n\n"
        
        if cls.__doc__:
            doc += f"{cls.__doc__.strip()}\n\n"
        
        # Methods
        methods = [m for m in dir(cls) if not m.startswith('_') and callable(getattr(cls, m))]
        
        if methods:
            doc += "### Methods\n\n"
            for method_name in methods[:5]:  # Limit to 5 methods
                try:
                    method = getattr(cls, method_name)
                    if callable(method):
                        doc += self.generate_function_doc(method)
                except:
                    pass
        
        return doc
    
    def generate_module_doc(self, module) -> str:
        """Generate documentation for a module."""
        markdown = MarkdownGenerator(f"{self.project_name} - {module.__name__}")
        
        if module.__doc__:
            markdown.add_section("Overview", module.__doc__.strip())
        
        # Find classes
        classes = [
            cls for name, cls in inspect.getmembers(module, inspect.isclass)
            if cls.__module__ == module.__name__
        ]
        
        for cls in classes[:3]:  # Limit to 3 classes
            markdown.sections.append(self.generate_class_doc(cls))
        
        return markdown.generate()

