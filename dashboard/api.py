"""
Dashboard API - Backend endpoints for the Teleon dashboard.

Provides REST API for:
- Agent management
- Metrics and analytics
- Real-time monitoring
- Dashboard statistics
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from teleon.core import StructuredLogger, LogLevel


# Models
class AgentStatus(BaseModel):
    """Agent status update model."""
    status: str


class AgentInfo(BaseModel):
    """Agent information model."""
    id: str
    name: str
    status: str
    requests: int
    cost: float
    created_at: str
    last_active: Optional[str] = None


class DashboardStats(BaseModel):
    """Dashboard statistics model."""
    totalAgents: int
    activeAgents: int
    totalRequests: int
    totalCost: float


class MetricDataPoint(BaseModel):
    """Metric data point model."""
    timestamp: str
    requests: int
    cost: float


# In-memory storage (replace with actual database)
_agents_db: Dict[str, Dict[str, Any]] = {}
_metrics_db: List[Dict[str, Any]] = []

logger = StructuredLogger("dashboard_api", LogLevel.INFO)


def create_dashboard_app() -> FastAPI:
    """
    Create FastAPI application for the dashboard.
    
    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="Teleon Dashboard API",
        description="Backend API for the Teleon platform dashboard",
        version="1.0.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize sample data
    _initialize_sample_data()
    
    # Routes
    
    @app.get("/api/dashboard/stats", response_model=DashboardStats)
    async def get_dashboard_stats():
        """Get overall dashboard statistics."""
        active_count = sum(1 for a in _agents_db.values() if a["status"] == "running")
        total_requests = sum(a["requests"] for a in _agents_db.values())
        total_cost = sum(a["cost"] for a in _agents_db.values())
        
        return DashboardStats(
            totalAgents=len(_agents_db),
            activeAgents=active_count,
            totalRequests=total_requests,
            totalCost=total_cost
        )
    
    @app.get("/api/agents", response_model=List[AgentInfo])
    async def list_agents(limit: Optional[int] = None):
        """List all agents."""
        agents = list(_agents_db.values())
        if limit:
            agents = agents[:limit]
        return agents
    
    @app.get("/api/agents/{agent_id}", response_model=AgentInfo)
    async def get_agent(agent_id: str):
        """Get agent details."""
        if agent_id not in _agents_db:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent = _agents_db[agent_id]
        return AgentInfo(
            id=agent["id"],
            name=agent["name"],
            status=agent["status"],
            requests=agent["requests"],
            cost=agent["cost"],
            created_at=agent["created_at"],
            last_active=agent.get("last_active"),
            avgLatency=agent.get("avgLatency", 120)
        )
    
    @app.patch("/api/agents/{agent_id}/status")
    async def update_agent_status(agent_id: str, status: AgentStatus):
        """Update agent status (start/stop)."""
        if agent_id not in _agents_db:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        _agents_db[agent_id]["status"] = status.status
        _agents_db[agent_id]["last_active"] = datetime.utcnow().isoformat()
        
        logger.info(f"Agent {agent_id} status updated to {status.status}")
        
        return {"message": "Status updated", "status": status.status}
    
    @app.delete("/api/agents/{agent_id}")
    async def delete_agent(agent_id: str):
        """Delete an agent."""
        if agent_id not in _agents_db:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        del _agents_db[agent_id]
        logger.info(f"Agent {agent_id} deleted")
        
        return {"message": "Agent deleted"}
    
    @app.get("/api/metrics", response_model=List[MetricDataPoint])
    async def get_metrics():
        """Get platform metrics."""
        return _metrics_db
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
    
    return app


def _initialize_sample_data():
    """Initialize sample data for demo purposes."""
    global _agents_db, _metrics_db
    
    # Sample agents
    _agents_db = {
        "agent-1": {
            "id": "agent-1",
            "name": "customer-support",
            "status": "running",
            "requests": 1543,
            "cost": 24.56,
            "created_at": (datetime.utcnow() - timedelta(days=7)).isoformat(),
            "last_active": "2 minutes ago",
            "avgLatency": 120
        },
        "agent-2": {
            "id": "agent-2",
            "name": "data-processor",
            "status": "running",
            "requests": 2891,
            "cost": 38.92,
            "created_at": (datetime.utcnow() - timedelta(days=5)).isoformat(),
            "last_active": "5 minutes ago",
            "avgLatency": 85
        },
        "agent-3": {
            "id": "agent-3",
            "name": "research-assistant",
            "status": "stopped",
            "requests": 456,
            "cost": 12.34,
            "created_at": (datetime.utcnow() - timedelta(days=3)).isoformat(),
            "last_active": "1 hour ago",
            "avgLatency": 210
        },
        "agent-4": {
            "id": "agent-4",
            "name": "email-analyzer",
            "status": "running",
            "requests": 789,
            "cost": 19.75,
            "created_at": (datetime.utcnow() - timedelta(days=2)).isoformat(),
            "last_active": "10 minutes ago",
            "avgLatency": 95
        },
        "agent-5": {
            "id": "agent-5",
            "name": "content-generator",
            "status": "stopped",
            "requests": 234,
            "cost": 8.45,
            "created_at": (datetime.utcnow() - timedelta(days=1)).isoformat(),
            "last_active": "3 hours ago",
            "avgLatency": 180
        },
    }
    
    # Sample metrics (last 24 hours)
    base_time = datetime.utcnow() - timedelta(hours=24)
    _metrics_db = [
        {
            "timestamp": (base_time + timedelta(hours=i)).strftime("%H:%M"),
            "requests": 100 + (i * 20) + (i % 5) * 10,
            "cost": 1.5 + (i * 0.3) + (i % 3) * 0.2
        }
        for i in range(24)
    ]


# Helper function to get the app instance
def get_dashboard_app() -> FastAPI:
    """Get the dashboard FastAPI app instance."""
    return create_dashboard_app()


if __name__ == "__main__":
    import uvicorn
    
    app = create_dashboard_app()
    
    print("🚀 Starting Teleon Dashboard API...")
    print("📊 Dashboard: http://localhost:3000")
    print("🔌 API: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

