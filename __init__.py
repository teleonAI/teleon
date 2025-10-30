"""
Teleon - The Platform for Intelligent Agents

Deploy production-ready AI agents in minutes, not months.
"""

from teleon.__version__ import __version__, __title__, __description__
from teleon.decorators.agent import agent
from teleon.decorators.tool import tool
from teleon.client import TeleonClient, init_teleon

__all__ = [
    "__version__",
    "__title__",
    "__description__",
    "agent",
    "tool",
    "TeleonClient",
    "init_teleon",
]

