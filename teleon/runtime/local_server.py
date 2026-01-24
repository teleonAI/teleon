"""Local development server for Teleon agents."""

import asyncio
import importlib
import inspect
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from teleon.config.loader import load_config
from teleon.validation.schema import SchemaGenerator


class ExecuteRequest(BaseModel):
    """Request model for agent execution."""
    input: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None


class ExecuteResponse(BaseModel):
    """Response model for agent execution."""
    execution_id: str
    agent_name: str
    status: str
    output: Any = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None
    cost: Optional[float] = None


class LocalServer:
    """
    Local development server for running Teleon agents.
    
    Features:
    - Hot reload on file changes
    - API endpoints for agent execution
    - Agent discovery and registration
    - Request/response validation
    - Execution tracking
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        watch_dirs: Optional[List[str]] = None,
        hot_reload: bool = True
    ):
        """
        Initialize local development server.
        
        Args:
            host: Server host
            port: Server port
            watch_dirs: Directories to watch for changes
            hot_reload: Enable hot reload
        """
        self.host = host
        self.port = port
        self.watch_dirs = watch_dirs or ["."]
        self.hot_reload = hot_reload
        
        self.app = FastAPI(
            title="Teleon Local Server",
            description="Local development server for Teleon agents",
            version="0.1.0"
        )
        
        # Store registered agents
        self.agents: Dict[str, Any] = {}
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
    
    def _setup_middleware(self) -> None:
        """Setup FastAPI middleware."""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self) -> None:
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "name": "Teleon Local Server",
                "version": "0.1.0",
                "agents": list(self.agents.keys()),
                "docs": "/docs"
            }
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "agents_loaded": len(self.agents),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        @self.app.get("/agents")
        async def list_agents():
            """List all registered agents."""
            agents_info = []
            for name, func in self.agents.items():
                config = getattr(func, '_teleon_config', None)
                agents_info.append({
                    'name': name,
                    'memory': config.memory if config else False,
                    'scale': config.scale if config else None,
                    'schema': SchemaGenerator.from_function(func)
                })
            return {"agents": agents_info}
        
        @self.app.get("/agents/{agent_name}")
        async def get_agent(agent_name: str):
            """Get information about a specific agent."""
            if agent_name not in self.agents:
                raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
            
            func = self.agents[agent_name]
            config = getattr(func, '_teleon_config', None)
            
            return {
                'name': agent_name,
                'config': config.to_dict() if config else {},
                'schema': SchemaGenerator.from_function(func),
                'docstring': func.__doc__
            }
        
        @self.app.post("/agents/{agent_name}/execute")
        async def execute_agent(agent_name: str, request: ExecuteRequest):
            """Execute an agent."""
            if agent_name not in self.agents:
                raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
            
            func = self.agents[agent_name]
            
            # Generate execution ID
            import uuid
            execution_id = str(uuid.uuid4())
            
            start_time = datetime.now(timezone.utc)
            
            try:
                # Execute the agent
                if inspect.iscoroutinefunction(func):
                    result = await func(**request.input)
                else:
                    result = func(**request.input)
                
                # Calculate duration
                duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                # Track actual cost from execution context
                cost = 0.0
                if hasattr(result, '__dict__') and 'cost' in result.__dict__:
                    cost = result.__dict__['cost']
                elif isinstance(result, dict) and 'cost' in result:
                    cost = result['cost']
                else:
                    # Try to get cost from execution context
                    try:
                        from teleon.core import get_metrics
                        metrics = get_metrics()
                        # Get cost from metrics if available
                        if hasattr(metrics, '_counters'):
                            cost_key = f"llm_cost_{'provider': 'unknown', 'model': 'unknown'}"
                            cost = metrics._counters.get(cost_key, 0.0)
                    except (ImportError, Exception):
                        # Cost tracking not available
                        cost = 0.0
                
                return ExecuteResponse(
                    execution_id=execution_id,
                    agent_name=agent_name,
                    status="completed",
                    output=result,
                    duration_ms=int(duration),
                    cost=cost
                )
                
            except Exception as e:
                duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                return ExecuteResponse(
                    execution_id=execution_id,
                    agent_name=agent_name,
                    status="failed",
                    error=str(e),
                    duration_ms=int(duration)
                )
    
    def register_agent(self, name: str, func: Any) -> None:
        """
        Register an agent with the server.
        
        Args:
            name: Agent name
            func: Agent function
        """
        self.agents[name] = func
        print(f"✅ Registered agent: {name}")
    
    def discover_agents(self, module_path: str) -> None:
        """
        Discover and register agents from a Python module.
        
        Args:
            module_path: Python module path (e.g., 'my_agents')
        """
        try:
            module = importlib.import_module(module_path)
            
            # Find all functions with _teleon_agent attribute
            for name in dir(module):
                obj = getattr(module, name)
                if callable(obj) and getattr(obj, '_teleon_agent', False):
                    config = getattr(obj, '_teleon_config', None)
                    agent_name = config.name if config else name
                    self.register_agent(agent_name, obj)
                    
        except Exception as e:
            print(f"Warning: Failed to discover agents from {module_path}: {e}")
    
    def run(self) -> None:
        """Start the development server."""
        print("=" * 60)
        print("🚀 Teleon Local Development Server")
        print("=" * 60)
        print(f"📍 Server: http://{self.host}:{self.port}")
        print(f"📚 API Docs: http://{self.host}:{self.port}/docs")
        print(f"🔥 Hot Reload: {'Enabled' if self.hot_reload else 'Disabled'}")
        print(f"👀 Watching: {', '.join(self.watch_dirs)}")
        print(f"🤖 Agents Loaded: {len(self.agents)}")
        print("=" * 60)
        print("")
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            reload=self.hot_reload,
            reload_dirs=self.watch_dirs if self.hot_reload else None,
            log_level="info"
        )


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    **kwargs: Any
) -> LocalServer:
    """
    Create a local development server.
    
    Args:
        host: Server host
        port: Server port
        **kwargs: Additional server configuration
    
    Returns:
        LocalServer instance
    """
    return LocalServer(host=host, port=port, **kwargs)

