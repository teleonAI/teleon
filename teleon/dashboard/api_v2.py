"""
Dashboard API v2 - With WebSocket Support.

Enhanced version with real-time WebSocket updates.
"""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from teleon.dashboard.api import create_dashboard_app
from teleon.dashboard.websocket import websocket_endpoint, get_connection_manager


def create_enhanced_dashboard_app() -> FastAPI:
    """
    Create enhanced FastAPI application with WebSocket support.
    
    Returns:
        Configured FastAPI app with WebSocket
    """
    # Get base app
    app = create_dashboard_app()
    
    # Add WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_route(websocket: WebSocket):
        """WebSocket endpoint for real-time updates."""
        await websocket_endpoint(websocket)
    
    # Add endpoint to trigger test notifications
    @app.post("/api/test/notification")
    async def test_notification(level: str = "info"):
        """Send a test notification to all connected clients."""
        manager = get_connection_manager()
        await manager.broadcast_notification(
            level=level,
            title="Test Notification",
            message=f"This is a test {level} notification"
        )
        return {"message": "Notification sent"}
    
    # Add endpoint to simulate agent updates
    @app.post("/api/test/agent-update")
    async def test_agent_update():
        """Simulate an agent update."""
        manager = get_connection_manager()
        await manager.broadcast_agent_update(
            agent_id="agent-1",
            status="running",
            data={"requests": 100, "cost": 5.50}
        )
        return {"message": "Agent update sent"}
    
    return app


if __name__ == "__main__":
    import uvicorn
    
    app = create_enhanced_dashboard_app()
    
    print("🚀 Starting Teleon Dashboard API with WebSocket Support...")
    print("📊 Dashboard: http://localhost:3000")
    print("🔌 API: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    print("🔄 WebSocket: ws://localhost:8000/ws")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

