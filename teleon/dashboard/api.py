"""
Dashboard API - Backend endpoints for the Teleon dashboard.

Provides REST API for:
- Agent management
- Metrics and analytics
- Real-time monitoring
- Dashboard statistics
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import time
import base64
import json

from teleon.core import StructuredLogger, LogLevel
from teleon.cortex.registry import registry
from teleon.sentinel.registry import get_sentinel_registry
from teleon.sentinel.audit import SentinelAuditLogger
from teleon.client import TeleonClient


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
    has_sentinel: Optional[bool] = None
    has_cortex: Optional[bool] = None


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

# Security
security = HTTPBearer(auto_error=False)


async def get_current_user(
    authorization: Optional[str] = Header(None),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> str:
    """
    Extract user_id from Authorization header.
    
    Supports:
    - Bearer token (JWT or API key)
    - API key in Authorization header
    
    Args:
        authorization: Authorization header value
        credentials: HTTPBearer credentials
        
    Returns:
        user_id string
        
    Raises:
        HTTPException: If authentication fails
    """
    # Try to get token from HTTPBearer first
    token = None
    if credentials:
        token = credentials.credentials
    elif authorization:
        # Extract Bearer token
        if authorization.startswith("Bearer "):
            token = authorization.replace("Bearer ", "")
        else:
            token = authorization
    
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Missing authorization token. Please provide Authorization: Bearer <token>"
        )
    
    # Try to extract user_id from token
    user_id = None
    
    # Method 1: Try JWT decode (if it's a JWT)
    try:
        # Simple JWT decode (without verification for now - in production use proper JWT library)
        parts = token.split('.')
        if len(parts) == 3:
            # JWT format
            payload_part = parts[1]
            # Add padding if needed
            payload_part += '=' * (4 - len(payload_part) % 4)
            payload_json = base64.urlsafe_b64decode(payload_part)
            payload = json.loads(payload_json)
            user_id = payload.get('user_id') or payload.get('sub')
    except Exception:
        pass
    
    # Method 2: Try API key lookup (check if it's an API key)
    if not user_id:
        # Check if token is an API key format
        if token.startswith('tleon_'):
            # This is an API key - in production, look it up in database
            # For now, extract user_id from TeleonClient if available
            try:
                # Try to get user_id from client's API key validation
                # This is a placeholder - in production, validate against database
                # For demo, we'll use a simple mapping or extract from agent registry
                pass
            except Exception:
                pass
    
    # Method 3: Extract from agent registry (fallback for development)
    if not user_id:
        # Get all agents and try to find user_id from agent metadata
        all_agents = TeleonClient.get_all_agents()
        if all_agents:
            # Get user_id from first agent (for demo purposes)
            # In production, this should come from proper authentication
            first_agent = list(all_agents.values())[0]
            user_id = first_agent.get('user_id')
    
    # Method 4: Use token as user_id if it looks like a user ID (for development)
    if not user_id:
        # For development/demo: use token as user_id if it's short and alphanumeric
        if len(token) < 50 and token.replace('_', '').replace('-', '').isalnum():
            user_id = token
        else:
            # Generate a deterministic user_id from token hash for demo
            import hashlib
            user_id = f"user_{hashlib.sha256(token.encode()).hexdigest()[:16]}"
    
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token. Please provide a valid authorization token."
        )
    
    logger.debug("User authenticated", user_id=user_id)
    return user_id


def get_user_agents(user_id: str) -> Dict[str, Dict[str, Any]]:
    """
    Get all agents belonging to a specific user.
    
    Args:
        user_id: User ID
        
    Returns:
        Dictionary of agent_id -> agent_info
    """
    all_agents = TeleonClient.get_all_agents()
    user_agents = {
        agent_id: agent_info
        for agent_id, agent_info in all_agents.items()
        if agent_info.get('user_id') == user_id
    }
    return user_agents


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
        
        # Check registries for sentinel and cortex status
        try:
            sentinel_registry = await get_sentinel_registry()
            sentinel_agents = await sentinel_registry.list_all()  # Returns {agent_id: engine}
            sentinel_agent_ids = set(sentinel_agents.keys())
        except Exception as e:
            logger.warning(f"Error getting Sentinel registry: {e}")
            sentinel_agent_ids = set()
        
        try:
            cortex_agent_ids = set(await registry.list_agents())  # Returns [agent_id, ...]
        except Exception as e:
            logger.warning(f"Error getting Cortex registry: {e}")
            cortex_agent_ids = set()
        
        # Enhance agents with sentinel/cortex status
        enhanced_agents = []
        for agent in agents:
            agent_id = agent.get("id")
            agent_dict = dict(agent)
            agent_dict["has_sentinel"] = agent_id in sentinel_agent_ids if agent_id else False
            agent_dict["has_cortex"] = agent_id in cortex_agent_ids if agent_id else False
            enhanced_agents.append(AgentInfo(**agent_dict))
        
        return enhanced_agents
    
    @app.get("/api/agents/{agent_id}", response_model=AgentInfo)
    async def get_agent(agent_id: str):
        """Get agent details."""
        if agent_id not in _agents_db:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent = _agents_db[agent_id]
        
        # Check registries for sentinel and cortex status
        has_sentinel = False
        has_cortex = False
        
        try:
            sentinel_registry = await get_sentinel_registry()
            sentinel_agents = await sentinel_registry.list_all()
            has_sentinel = agent_id in sentinel_agents
        except Exception as e:
            logger.warning(f"Error checking Sentinel registry for {agent_id}: {e}")
        
        try:
            cortex_agent_ids = await registry.list_agents()
            has_cortex = agent_id in cortex_agent_ids
        except Exception as e:
            logger.warning(f"Error checking Cortex registry for {agent_id}: {e}")
        
        return AgentInfo(
            id=agent["id"],
            name=agent["name"],
            status=agent["status"],
            requests=agent["requests"],
            cost=agent["cost"],
            created_at=agent["created_at"],
            last_active=agent.get("last_active"),
            has_sentinel=has_sentinel,
            has_cortex=has_cortex
        )
    
    @app.patch("/api/agents/{agent_id}/status")
    async def update_agent_status(agent_id: str, status: AgentStatus):
        """Update agent status (start/stop)."""
        if agent_id not in _agents_db:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        _agents_db[agent_id]["status"] = status.status
        _agents_db[agent_id]["last_active"] = datetime.now(timezone.utc).isoformat()
        
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
    
    # ========================================
    # Cortex Memory Endpoints
    # ========================================
    
    @app.get("/api/v1/agents/{agent_id}/cortex/metrics")
    async def get_cortex_metrics(agent_id: str):
        """
        Get Cortex memory metrics for an agent.
        
        Returns real-time operation metrics, cache stats, and memory sizes.
        """
        try:
            cortex = await registry.get(agent_id)
            if not cortex:
                raise HTTPException(
                    status_code=404,
                    detail=f"Cortex instance not found for agent: {agent_id}"
                )
            
            metrics = cortex.get_metrics()
            return metrics
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting Cortex metrics for {agent_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/agents/{agent_id}/cortex/health")
    async def get_cortex_health(agent_id: str):
        """
        Get Cortex memory system health status.
        
        Returns health status for all memory types and storage backends.
        """
        try:
            cortex = await registry.get(agent_id)
            if not cortex:
                return {
                    "status": "not_found",
                    "agent_id": agent_id,
                    "message": "Cortex instance not registered",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            if not cortex._initialized:
                return {
                    "status": "not_initialized",
                    "agent_id": agent_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            # Perform health checks
            checks = {}
            latency_start = time.perf_counter()
            
            # Check episodic memory
            if cortex.episodic:
                try:
                    await cortex.episodic.get_recent(limit=1)
                    checks["episodic_memory"] = "healthy"
                except Exception as e:
                    checks["episodic_memory"] = f"unhealthy: {str(e)[:50]}"
            else:
                checks["episodic_memory"] = "disabled"
            
            # Check semantic memory
            if cortex.semantic:
                try:
                    await cortex.semantic.get_statistics()
                    checks["semantic_memory"] = "healthy"
                except Exception as e:
                    checks["semantic_memory"] = f"unhealthy: {str(e)[:50]}"
            else:
                checks["semantic_memory"] = "disabled"
            
            # Check procedural memory
            if cortex.procedural:
                try:
                    await cortex.procedural.get_statistics()
                    checks["procedural_memory"] = "healthy"
                except Exception as e:
                    checks["procedural_memory"] = f"unhealthy: {str(e)[:50]}"
            else:
                checks["procedural_memory"] = "disabled"
            
            # Check storage backend
            if cortex.storage:
                try:
                    await cortex.storage.get_statistics()
                    checks["storage_backend"] = "healthy"
                except Exception as e:
                    checks["storage_backend"] = f"unhealthy: {str(e)[:50]}"
            else:
                checks["storage_backend"] = "not_configured"
            
            latency_ms = (time.perf_counter() - latency_start) * 1000
            
            # Determine overall status
            overall_status = "healthy"
            if any("unhealthy" in str(v) for v in checks.values()):
                overall_status = "degraded"
            elif any("disabled" in str(v) or "not_configured" in str(v) for v in checks.values()):
                overall_status = "partial"
            
            return {
                "status": overall_status,
                "agent_id": agent_id,
                "checks": checks,
                "latency_ms": round(latency_ms, 2),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting Cortex health for {agent_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/agents/{agent_id}/cortex/performance")
    async def get_cortex_performance(agent_id: str, period: str = "24h"):
        """
        Get Cortex performance profiling data.
        
        Args:
            agent_id: Agent ID
            period: Time period (1h, 24h, 7d, 30d) - currently returns all data
        
        Returns:
            Performance statistics, slow operations, and recommendations
        """
        try:
            cortex = await registry.get(agent_id)
            if not cortex:
                raise HTTPException(
                    status_code=404,
                    detail=f"Cortex instance not found for agent: {agent_id}"
                )
            
            if not cortex._profiler:
                raise HTTPException(
                    status_code=503,
                    detail="Performance profiler not initialized"
                )
            
            report = cortex.get_performance_report()
            report["period"] = period
            report["agent_id"] = agent_id
            
            return report
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting Cortex performance for {agent_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/agents/{agent_id}/cortex/statistics")
    async def get_cortex_statistics(agent_id: str):
        """
        Get comprehensive Cortex memory statistics.
        
        Returns statistics from all memory types.
        """
        try:
            cortex = await registry.get(agent_id)
            if not cortex:
                raise HTTPException(
                    status_code=404,
                    detail=f"Cortex instance not found for agent: {agent_id}"
                )
            
            stats = await cortex.get_statistics()
            stats["agent_id"] = agent_id
            
            return stats
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting Cortex statistics for {agent_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    # ========================================
    # Sentinel Endpoints
    # ========================================
    
    @app.get("/api/v1/sentinel/violations")
    async def get_sentinel_violations(
        agent_id: Optional[str] = None,
        violation_type: Optional[str] = None,
        limit: int = 100,
        user_id: str = Depends(get_current_user)
    ):
        """
        Get Sentinel violations for user's agents only.
        
        Args:
            agent_id: Filter by agent ID (must belong to user)
            violation_type: Filter by violation type
            limit: Maximum records to return
            user_id: Current user ID (from auth)
        """
        try:
            # Get user's agents only
            user_agents = get_user_agents(user_id)
            if not user_agents:
                return {"violations": [], "total": 0}
            
            # Security check: verify agent_id belongs to user
            if agent_id and agent_id not in user_agents:
                raise HTTPException(
                    status_code=403,
                    detail=f"Access denied: Agent {agent_id} does not belong to your account"
                )
            
            sentinel_registry = await get_sentinel_registry()
            all_engines = await sentinel_registry.list_all()
            
            all_violations = []
            
            # Collect violations only from user's agents with Sentinel enabled
            for agent_id_key, engine in all_engines.items():
                # Security: Only process agents belonging to this user
                if agent_id_key not in user_agents:
                    continue
                
                # Only include agents with Sentinel enabled
                if not engine.config.enabled:
                    continue
                
                # Apply agent_id filter if specified
                if agent_id and agent_id_key != agent_id:
                    continue
                
                audit_logger = engine.get_audit_logger()
                if audit_logger:
                    violations = audit_logger.get_violations(
                        agent_id=agent_id_key,
                        violation_type=violation_type,
                        limit=limit
                    )
                    all_violations.extend([v.to_dict() for v in violations])
            
            # Sort by timestamp (newest first)
            all_violations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return {
                "violations": all_violations[:limit],
                "total": len(all_violations)
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting Sentinel violations: {e}", exc_info=True)
            # Return empty list on error
            return {"violations": [], "total": 0}
    
    @app.get("/api/v1/sentinel/stats")
    async def get_sentinel_stats(
        agent_id: Optional[str] = None,
        user_id: str = Depends(get_current_user)
    ):
        """
        Get Sentinel violation statistics for user's agents only.
        
        Args:
            agent_id: Filter by agent ID (must belong to user)
            user_id: Current user ID (from auth)
        """
        try:
            # Get user's agents only
            user_agents = get_user_agents(user_id)
            if not user_agents:
                return {
                    "total_violations": 0,
                    "by_type": {},
                    "by_action": {},
                    "recent_count": 0
                }
            
            # Security check: verify agent_id belongs to user
            if agent_id and agent_id not in user_agents:
                raise HTTPException(
                    status_code=403,
                    detail=f"Access denied: Agent {agent_id} does not belong to your account"
                )
            
            sentinel_registry = await get_sentinel_registry()
            all_engines = await sentinel_registry.list_all()
            
            if agent_id:
                # Get stats for specific agent (already verified it belongs to user)
                engine = all_engines.get(agent_id)
                if engine and engine.config.enabled:
                    audit_logger = engine.get_audit_logger()
                    if audit_logger:
                        return audit_logger.get_violation_stats(agent_id=agent_id)
                return {
                    "total_violations": 0,
                    "by_type": {},
                    "by_action": {},
                    "recent_count": 0
                }
            else:
                # Aggregate stats across user's agents with Sentinel enabled only
                total_violations = 0
                by_type = {}
                by_action = {}
                recent_count = 0
                
                for agent_id_key, engine in all_engines.items():
                    # Security: Only process agents belonging to this user
                    if agent_id_key not in user_agents:
                        continue
                    
                    # Only include agents with Sentinel enabled
                    if not engine.config.enabled:
                        continue
                    
                    audit_logger = engine.get_audit_logger()
                    if audit_logger:
                        stats = audit_logger.get_violation_stats(agent_id=agent_id_key)
                        total_violations += stats.get("total_violations", 0)
                        recent_count += stats.get("recent_count", 0)
                        
                        for v_type, count in stats.get("by_type", {}).items():
                            by_type[v_type] = by_type.get(v_type, 0) + count
                        
                        for action, count in stats.get("by_action", {}).items():
                            by_action[action] = by_action.get(action, 0) + count
                
                return {
                    "total_violations": total_violations,
                    "by_type": by_type,
                    "by_action": by_action,
                    "recent_count": recent_count
                }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting Sentinel stats: {e}", exc_info=True)
            return {
                "total_violations": 0,
                "by_type": {},
                "by_action": {},
                "recent_count": 0
            }
    
    @app.get("/api/v1/sentinel/agents")
    async def get_agents_with_sentinel(
        user_id: str = Depends(get_current_user)
    ):
        """
        Get user's agents with Sentinel enabled only.
        
        Args:
            user_id: Current user ID (from auth)
        """
        try:
            # Get user's agents only
            user_agents = get_user_agents(user_id)
            if not user_agents:
                return {"agents": []}
            
            sentinel_registry = await get_sentinel_registry()
            all_engines = await sentinel_registry.list_all()
            
            agents = []
            
            # Only process user's agents with Sentinel enabled
            for agent_id, engine in all_engines.items():
                # Security: Only process agents belonging to this user
                if agent_id not in user_agents:
                    continue
                
                config = engine.config
                
                # Only include agents with Sentinel enabled
                if not config.enabled:
                    continue
                
                audit_logger = engine.get_audit_logger()
                
                # Get violation stats for this agent
                violation_count = 0
                last_violation = None
                if audit_logger:
                    stats = audit_logger.get_violation_stats(agent_id=agent_id)
                    violation_count = stats.get("total_violations", 0)
                    violations = audit_logger.get_violations(agent_id=agent_id, limit=1)
                    if violations:
                        last_violation = violations[0].timestamp.isoformat()
                
                # Get agent name from user_agents or fallback
                agent_info = user_agents.get(agent_id, {})
                agent_name = agent_info.get("name", agent_id)
                
                agents.append({
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "enabled": config.enabled,
                    "features": {
                        "content_filtering": config.content_filtering,
                        "pii_detection": config.pii_detection,
                        "compliance": [c.value for c in config.compliance],
                        "custom_policies": config.custom_policies,
                    },
                    "violation_count": violation_count,
                    "last_violation": last_violation,
                })
            
            return {"agents": agents}
        except Exception as e:
            logger.error(f"Error getting agents with Sentinel: {e}", exc_info=True)
            return {"agents": []}
    
    @app.post("/api/v1/sentinel/violations/export")
    async def export_sentinel_violations(
        filters: Dict[str, Any],
        user_id: str = Depends(get_current_user)
    ):
        """
        Export Sentinel violations for user's agents only.
        
        Args:
            filters: Filter criteria
            user_id: Current user ID (from auth)
        """
        try:
            # Get user's agents only
            user_agents = get_user_agents(user_id)
            if not user_agents:
                return {"violations": []}
            
            # Security check: verify agent_id in filters belongs to user
            filter_agent_id = filters.get("agent_id")
            if filter_agent_id and filter_agent_id != "all" and filter_agent_id not in user_agents:
                raise HTTPException(
                    status_code=403,
                    detail=f"Access denied: Agent {filter_agent_id} does not belong to your account"
                )
            
            sentinel_registry = await get_sentinel_registry()
            all_engines = await sentinel_registry.list_all()
            
            all_violations = []
            
            # Only export violations from user's agents with Sentinel enabled
            for agent_id, engine in all_engines.items():
                # Security: Only process agents belonging to this user
                if agent_id not in user_agents:
                    continue
                
                # Only include agents with Sentinel enabled
                if not engine.config.enabled:
                    continue
                
                # Apply agent filter if specified
                if filter_agent_id and filter_agent_id != "all" and agent_id != filter_agent_id:
                    continue
                
                audit_logger = engine.get_audit_logger()
                if audit_logger:
                    violations = audit_logger.export_audit_trail(
                        agent_id=agent_id,
                        format="json"
                    )
                    all_violations.extend(violations)
            
            return {"violations": all_violations}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error exporting Sentinel violations: {e}", exc_info=True)
            return {"violations": []}
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
    
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
            "created_at": (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),
            "last_active": "2 minutes ago",
            "avgLatency": 120
        },
        "agent-2": {
            "id": "agent-2",
            "name": "data-processor",
            "status": "running",
            "requests": 2891,
            "cost": 38.92,
            "created_at": (datetime.now(timezone.utc) - timedelta(days=5)).isoformat(),
            "last_active": "5 minutes ago",
            "avgLatency": 85
        },
        "agent-3": {
            "id": "agent-3",
            "name": "research-assistant",
            "status": "stopped",
            "requests": 456,
            "cost": 12.34,
            "created_at": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(),
            "last_active": "1 hour ago",
            "avgLatency": 210
        },
        "agent-4": {
            "id": "agent-4",
            "name": "email-analyzer",
            "status": "running",
            "requests": 789,
            "cost": 19.75,
            "created_at": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
            "last_active": "10 minutes ago",
            "avgLatency": 95
        },
        "agent-5": {
            "id": "agent-5",
            "name": "content-generator",
            "status": "stopped",
            "requests": 234,
            "cost": 8.45,
            "created_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            "last_active": "3 hours ago",
            "avgLatency": 180
        },
    }
    
    # Sample metrics (last 24 hours)
    base_time = datetime.now(timezone.utc) - timedelta(hours=24)
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

