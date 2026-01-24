"""
WebSocket Support for Real-time Dashboard Updates.

Provides real-time updates for:
- Agent status changes
- Metric updates
- System notifications
- Log streaming
"""

from typing import Dict, Set, Any
from datetime import datetime, timezone
import asyncio
import json

from fastapi import WebSocket, WebSocketDisconnect
from teleon.core import StructuredLogger, LogLevel


logger = StructuredLogger("dashboard_websocket", LogLevel.INFO)


class ConnectionManager:
    """
    WebSocket connection manager.
    
    Manages multiple WebSocket connections and broadcasts updates.
    """
    
    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Set[WebSocket] = set()
        self.logger = StructuredLogger("connection_manager", LogLevel.INFO)
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)
        self.logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                self.logger.error(f"Error broadcasting to connection: {e}")
                disconnected.add(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_agent_update(self, agent_id: str, status: str, data: Dict[str, Any]):
        """Broadcast agent status update."""
        message = {
            "type": "agent_update",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "agent_id": agent_id,
                "status": status,
                **data
            }
        }
        await self.broadcast(message)
    
    async def broadcast_metric_update(self, metric_name: str, value: float):
        """Broadcast metric update."""
        message = {
            "type": "metric_update",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "metric": metric_name,
                "value": value
            }
        }
        await self.broadcast(message)
    
    async def broadcast_notification(self, level: str, title: str, message: str):
        """Broadcast system notification."""
        notification = {
            "type": "notification",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "level": level,  # info, warning, error
                "title": title,
                "message": message
            }
        }
        await self.broadcast(notification)
    
    async def broadcast_log(self, agent_id: str, log_entry: Dict[str, Any]):
        """Broadcast agent log entry."""
        message = {
            "type": "log",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "agent_id": agent_id,
                **log_entry
            }
        }
        await self.broadcast(message)


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint handler.
    
    Handles WebSocket connections and messages.
    """
    await manager.connect(websocket)
    
    try:
        # Send initial connection message
        await manager.send_personal_message({
            "type": "connected",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Connected to Teleon Dashboard"
        }, websocket)
        
        # Listen for messages from client
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                # Handle different message types
                if message_type == "ping":
                    await manager.send_personal_message({
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }, websocket)
                
                elif message_type == "subscribe":
                    # Handle subscription to specific events
                    event_type = message.get("event")
                    logger.info(f"Client subscribed to: {event_type}")
                
                else:
                    logger.warning(f"Unknown message type: {message_type}")
            
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def start_background_updates():
    """
    Start background task for periodic updates.
    
    Simulates real-time updates for demo purposes.
    """
    while True:
        try:
            # Broadcast random metric updates every 5 seconds
            await asyncio.sleep(5)
            
            # Simulate metric update
            await manager.broadcast_metric_update(
                "active_agents",
                3  # Example value
            )
            
        except Exception as e:
            logger.error(f"Background update error: {e}")
            await asyncio.sleep(5)


def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager instance."""
    return manager

