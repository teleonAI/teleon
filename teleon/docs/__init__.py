"""
Documentation System - Auto-generate documentation.

Features:
- OpenAPI spec generation
- Markdown documentation
- Code examples extraction
- API reference
"""

from teleon.docs.generator import (
    DocGenerator,
    OpenAPIGenerator,
    MarkdownGenerator
)

__all__ = [
    "DocGenerator",
    "OpenAPIGenerator",
    "MarkdownGenerator",
]

