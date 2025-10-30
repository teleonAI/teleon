"""
Base Connector - Foundation for all data connectors.

Provides:
- Connection management
- Automatic retry
- Connection pooling
- Health checks
"""

from typing import Any, Optional
from abc import ABC, abstractmethod

from teleon.core import StructuredLogger, LogLevel, TeleonException


class ConnectionError(TeleonException):
    """Raised when connection fails."""
    pass


class BaseConnector(ABC):
    """
    Base class for all data connectors.
    
    Provides common functionality for connecting to external data sources.
    """
    
    def __init__(self, name: str):
        """
        Initialize connector.
        
        Args:
            name: Connector name
        """
        self.name = name
        self.logger = StructuredLogger(f"connector.{name}", LogLevel.INFO)
        self.connected = False
    
    @abstractmethod
    async def connect(self):
        """
        Establish connection to the data source.
        
        Raises:
            ConnectionError: If connection fails
        """
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close the connection to the data source."""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """
        Test if connection is alive.
        
        Returns:
            True if connection is healthy
        """
        pass
    
    async def ensure_connected(self):
        """Ensure connector is connected."""
        if not self.connected:
            await self.connect()
    
    async def __aenter__(self):
        """Context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.disconnect()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, connected={self.connected})"

