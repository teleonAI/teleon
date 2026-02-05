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
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
import time
import base64
import json
import os
import sys

from teleon.core import StructuredLogger, LogLevel
from teleon.cortex.manager import get_memory_manager, _managers as cortex_managers
from teleon.sentinel.registry import get_sentinel_registry
from teleon.sentinel.audit import SentinelAuditLogger
from teleon.client import TeleonClient

# Try to import database models from platform
# If running as part of platform, use platform models
# Otherwise, set up database connection directly
try:
    # Try importing from platform API
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..', 'teleon-platform-aws'))
    from apps.api.models import Agent, AgentStatus as AgentStatusEnum, AgentMetrics, User
    from shared.database import get_db as get_platform_db, AsyncSessionLocal
    DB_AVAILABLE = True
except ImportError:
    # If platform models not available, try to set up database connection directly
    DB_AVAILABLE = False
    logger = StructuredLogger("dashboard_api", LogLevel.WARNING)
    logger.warning("Platform database models not available. Dashboard API will use fallback methods.")


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


logger = StructuredLogger("dashboard_api", LogLevel.INFO)


async def get_db():
    """
    Get database session for dashboard API.
    
    Returns database session if available, otherwise returns None.
    This is an async generator for FastAPI dependency injection.
    """
    if not DB_AVAILABLE:
        yield None
        return
    
    try:
        async with AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    except Exception as e:
        logger.error(f"Database session error: {e}")
        yield None

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
    
    # Routes
    
    @app.get("/api/dashboard/stats", response_model=DashboardStats)
    async def get_dashboard_stats(db: Optional[AsyncSession] = Depends(get_db)):
        """Get overall dashboard statistics."""
        if DB_AVAILABLE and db is not None:
            try:
                # Get stats from database
                result = await db.execute(
                    select(
                        func.count(Agent.id).label("total_agents"),
                        func.sum(Agent.requests).label("total_requests"),
                        func.sum(Agent.cost).label("total_cost"),
                        func.sum(
                            func.case((Agent.status == AgentStatusEnum.RUNNING, 1), else_=0)
                        ).label("active_agents")
                    )
                )
                stats = result.first()
        
        return DashboardStats(
                    totalAgents=stats.total_agents or 0,
                    activeAgents=stats.active_agents or 0,
                    totalRequests=stats.total_requests or 0,
                    totalCost=float(stats.total_cost or 0.0)
                )
            except Exception as e:
                logger.error(f"Error getting dashboard stats from DB: {e}")
                # Fallback to empty stats
                return DashboardStats(
                    totalAgents=0,
                    activeAgents=0,
                    totalRequests=0,
                    totalCost=0.0
                )
        else:
            # Fallback when DB not available
            return DashboardStats(
                totalAgents=0,
                activeAgents=0,
                totalRequests=0,
                totalCost=0.0
        )
    
    @app.get("/api/agents", response_model=List[AgentInfo])
    async def list_agents(limit: Optional[int] = None, db: Optional[AsyncSession] = Depends(get_db)):
        """List all agents."""
        if DB_AVAILABLE and db is not None:
            try:
                # Get agents from database
                query = select(Agent).order_by(desc(Agent.created_at))
        if limit:
                    query = query.limit(limit)
                
                result = await db.execute(query)
                db_agents = result.scalars().all()
                
                # Check registries for sentinel and cortex status
                try:
                    sentinel_registry = await get_sentinel_registry()
                    sentinel_agents = await sentinel_registry.list_all()
                    sentinel_agent_ids = set(sentinel_agents.keys())
                except Exception as e:
                    logger.warning(f"Error getting Sentinel registry: {e}")
                    sentinel_agent_ids = set()
                
                try:
                    cortex_agent_ids = set(cortex_managers.keys())
                except Exception as e:
                    logger.warning(f"Error getting Cortex managers: {e}")
                    cortex_agent_ids = set()
                
                # Convert to AgentInfo
                # Use database values directly (they're the source of truth)
                # Registries are only for locally initialized agents
                enhanced_agents = []
                for agent in db_agents:
                    agent_dict = agent.to_dict()
                    # Database already has has_sentinel and has_cortex from deployment
                    # Only override if agent is in local registries (for real-time status)
                    if agent.agent_id in sentinel_agent_ids:
                        agent_dict["has_sentinel"] = True
                    if agent.agent_id in cortex_agent_ids:
                        agent_dict["has_cortex"] = True
                    enhanced_agents.append(AgentInfo(**agent_dict))
                
                return enhanced_agents
            except Exception as e:
                logger.error(f"Error listing agents from DB: {e}")
                return []
        else:
            # Fallback when DB not available
            return []
    
    @app.get("/api/agents/{agent_id}", response_model=AgentInfo)
    async def get_agent(agent_id: str, db: Optional[AsyncSession] = Depends(get_db)):
        """Get agent details."""
        if DB_AVAILABLE and db is not None:
            try:
                # Get agent from database by agent_id
                result = await db.execute(
                    select(Agent).where(Agent.agent_id == agent_id)
                )
                agent = result.scalar_one_or_none()
                
                if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
                # Use database values (source of truth)
                agent_dict = agent.to_dict()
                # Database already has has_sentinel and has_cortex from deployment
                # Only check registries for real-time status if needed (optional enhancement)
                
                return AgentInfo(**agent_dict)
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting agent from DB: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        else:
            raise HTTPException(status_code=503, detail="Database not available")
    
    @app.patch("/api/agents/{agent_id}/status")
    async def update_agent_status(agent_id: str, status: AgentStatus, db: Optional[AsyncSession] = Depends(get_db)):
        """Update agent status (start/stop)."""
        if DB_AVAILABLE and db is not None:
            try:
                # Get agent from database
                result = await db.execute(
                    select(Agent).where(Agent.agent_id == agent_id)
                )
                agent = result.scalar_one_or_none()
                
                if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
                # Update status
                try:
                    agent.status = AgentStatusEnum(status.status)
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid status: {status.status}")
                
                agent.last_active = datetime.now(timezone.utc)
                agent.updated_at = datetime.now(timezone.utc)
                
                await db.commit()
        
        logger.info(f"Agent {agent_id} status updated to {status.status}")
        
        return {"message": "Status updated", "status": status.status}
            except HTTPException:
                raise
            except Exception as e:
                await db.rollback()
                logger.error(f"Error updating agent status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        else:
            raise HTTPException(status_code=503, detail="Database not available")
    
    @app.delete("/api/agents/{agent_id}")
    async def delete_agent(agent_id: str, db: Optional[AsyncSession] = Depends(get_db)):
        """Delete an agent."""
        if DB_AVAILABLE and db is not None:
            try:
                # Get agent from database
                result = await db.execute(
                    select(Agent).where(Agent.agent_id == agent_id)
                )
                agent = result.scalar_one_or_none()
                
                if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
                await db.delete(agent)
                await db.commit()
                
        logger.info(f"Agent {agent_id} deleted")
        
        return {"message": "Agent deleted"}
            except HTTPException:
                raise
            except Exception as e:
                await db.rollback()
                logger.error(f"Error deleting agent: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        else:
            raise HTTPException(status_code=503, detail="Database not available")
    
    @app.get("/api/metrics", response_model=List[MetricDataPoint])
    async def get_metrics(db: Optional[AsyncSession] = Depends(get_db)):
        """Get platform metrics."""
        if DB_AVAILABLE and db is not None:
            try:
                # Get metrics from database (last 24 hours)
                start_time = datetime.now(timezone.utc) - timedelta(hours=24)
                
                result = await db.execute(
                    select(AgentMetrics)
                    .where(AgentMetrics.period_start >= start_time)
                    .where(AgentMetrics.aggregation_window == "1hour")
                    .order_by(AgentMetrics.period_start)
                )
                metrics = result.scalars().all()
                
                # Convert to MetricDataPoint format
                metric_points = []
                for metric in metrics:
                    metric_points.append(MetricDataPoint(
                        timestamp=metric.period_start.strftime("%H:%M"),
                        requests=metric.request_count,
                        cost=float(metric.llm_cost_usd or 0.0)
                    ))
                
                return metric_points
            except Exception as e:
                logger.error(f"Error getting metrics from DB: {e}")
                return []
        else:
            # Fallback when DB not available
            return []
    
    # ========================================
    # Cortex Memory Endpoints
    # ========================================
    
    @app.get("/api/v1/agents/{agent_id}/cortex/metrics")
    async def get_cortex_metrics(agent_id: str):
        """
        Get Cortex memory metrics for an agent.

        Returns memory entry counts and basic statistics.
        """
        try:
            manager = get_memory_manager(agent_id)
            if not manager:
                raise HTTPException(
                    status_code=404,
                    detail=f"Cortex not configured for agent: {agent_id}"
                )

            # Create memory instance to get stats
            memory = manager.create_memory({})
            total_entries = await memory.count()

            return {
                "agent_id": agent_id,
                "total_entries": total_entries,
                "memory_name": manager.memory_name,
                "scope": manager.config.scope,
                "auto_save": manager.config.auto,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting Cortex metrics for {agent_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/agents/{agent_id}/cortex/health")
    async def get_cortex_health(agent_id: str):
        """
        Get Cortex memory system health status.

        Returns health status for memory and storage backend.
        """
        try:
            manager = get_memory_manager(agent_id)
            if not manager:
                return {
                    "status": "not_configured",
                    "agent_id": agent_id,
                    "message": "Cortex not configured for this agent",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

            # Perform health checks
            checks = {}
            latency_start = time.perf_counter()

            # Check storage backend by performing a simple operation
            try:
                memory = manager.create_memory({})
                await memory.count()
                checks["storage"] = "healthy"
            except Exception as e:
                checks["storage"] = f"unhealthy: {str(e)[:50]}"

            # Check embedding service
            try:
                if manager._embedding:
                    checks["embeddings"] = "healthy"
                else:
                    checks["embeddings"] = "not_configured"
            except Exception as e:
                checks["embeddings"] = f"unhealthy: {str(e)[:50]}"

            latency_ms = (time.perf_counter() - latency_start) * 1000

            # Determine overall status
            overall_status = "healthy"
            if any("unhealthy" in str(v) for v in checks.values()):
                overall_status = "degraded"

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
        Get Cortex performance data.

        Args:
            agent_id: Agent ID
            period: Time period (1h, 24h, 7d, 30d)

        Returns:
            Basic performance statistics
        """
        try:
            manager = get_memory_manager(agent_id)
            if not manager:
                raise HTTPException(
                    status_code=404,
                    detail=f"Cortex not configured for agent: {agent_id}"
                )

            # Basic performance report
            memory = manager.create_memory({})
            total_entries = await memory.count()

            return {
                "agent_id": agent_id,
                "period": period,
                "total_entries": total_entries,
                "memory_name": manager.memory_name,
                "auto_save_enabled": manager.config.auto,
                "auto_context_enabled": manager.config.auto_context.enabled,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting Cortex performance for {agent_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/agents/{agent_id}/cortex/statistics")
    async def get_cortex_statistics(agent_id: str):
        """
        Get Cortex memory statistics.

        Returns entry counts and configuration.
        """
        try:
            manager = get_memory_manager(agent_id)
            if not manager:
                raise HTTPException(
                    status_code=404,
                    detail=f"Cortex not configured for agent: {agent_id}"
                )

            memory = manager.create_memory({})
            total = await memory.count()

            return {
                "agent_id": agent_id,
                "total_entries": total,
                "memory_name": manager.memory_name,
                "scope": manager.config.scope,
                "auto_save": manager.config.auto,
                "auto_context": {
                    "enabled": manager.config.auto_context.enabled,
                    "history_limit": manager.config.auto_context.history_limit,
                    "relevant_limit": manager.config.auto_context.relevant_limit
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
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
        
        Checks both:
        1. Sentinel registry (for locally initialized agents)
        2. TeleonClient agent registry (for all agents, including deployed ones)
        
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
            processed_agent_ids = set()
            
            # First, process agents from sentinel registry (locally initialized)
            for agent_id, engine in all_engines.items():
                # Security: Only process agents belonging to this user
                if agent_id not in user_agents:
                    continue
                
                config = engine.config
                
                # Only include agents with Sentinel enabled
                if not config.enabled:
                    continue
                
                processed_agent_ids.add(agent_id)
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
            
            # Second, check TeleonClient agents for sentinel config (includes deployed agents)
            for agent_id, agent_info in user_agents.items():
                # Skip if already processed from registry
                if agent_id in processed_agent_ids:
                    continue
                
                # Check if agent has sentinel configuration
                sentinel_config = agent_info.get("sentinel")
                if not sentinel_config:
                    continue
                
                # Check if sentinel is enabled
                # Handle both dict and boolean configs
                if isinstance(sentinel_config, dict):
                    enabled = sentinel_config.get("enabled", True)
                elif isinstance(sentinel_config, bool):
                    enabled = sentinel_config
                else:
                    enabled = True  # Default to enabled if config exists
                
                if not enabled:
                    continue
                
                # Extract features from config
                if isinstance(sentinel_config, dict):
                    features = {
                        "content_filtering": sentinel_config.get("content_filtering", False),
                        "pii_detection": sentinel_config.get("pii_detection", False),
                        "compliance": sentinel_config.get("compliance", []),
                        "custom_policies": sentinel_config.get("custom_policies", []),
                    }
                    # Normalize compliance list (handle enum values)
                    if features["compliance"]:
                        normalized_compliance = []
                        for comp in features["compliance"]:
                            if hasattr(comp, 'value'):
                                normalized_compliance.append(comp.value)
                            elif isinstance(comp, str):
                                normalized_compliance.append(comp.lower())
                            else:
                                normalized_compliance.append(str(comp))
                        features["compliance"] = normalized_compliance
                else:
                    features = {
                        "content_filtering": False,
                        "pii_detection": False,
                        "compliance": [],
                        "custom_policies": [],
                    }
                
                agent_name = agent_info.get("name", agent_id)
                
                agents.append({
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "enabled": enabled,
                    "features": features,
                    "violation_count": 0,  # No violations tracked if not initialized
                    "last_violation": None,
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


# Sample data initialization removed - all data now comes from database


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

