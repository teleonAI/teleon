"""
Teleon Development Server - Built-in dev server with automatic agent discovery.

This is the official Teleon dev server that's used by the `teleon dev` command.
It automatically discovers agents and provides:
- Dashboard UI
- Individual agent endpoints
- Shared endpoints
- OpenAPI documentation
- API key management (for testing)
"""

import asyncio
import hashlib
import json
import os
import secrets
import time
import uuid
import logging
import urllib.parse
from collections import deque, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Any, List

from fastapi import FastAPI, HTTPException, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from teleon.discovery import discover_agents

logger = logging.getLogger("teleon.dev_server")


class AgentRequest(BaseModel):
    """Request to an agent (shared endpoint)."""
    agent_name: str
    input: Any  # Can be string or dict with agent parameters
    temperature: float = 0.7
    max_tokens: int = 500


class IndividualAgentRequest(BaseModel):
    """Request to an individual agent endpoint."""
    input: Any  # Can be string or dict with agent parameters
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class AgentResponse(BaseModel):
    """Response from an agent."""
    agent_id: str
    agent_name: str
    output: Any  # Accept any return type: str, dict, list, etc. (will be JSON serialized)
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    latency_ms: Optional[float] = None
    timestamp: str


class APIKeyRequest(BaseModel):
    """Request to create an API key."""
    name: str
    description: Optional[str] = None


class APIKeyResponse(BaseModel):
    """API key creation response."""
    api_key: str
    name: str
    created_at: str
    message: str


class PlaygroundRequest(BaseModel):
    """Request from playground."""
    agent_id: str
    input: Any  # Can be string or dict with agent parameters
    temperature: float = 0.7
    max_tokens: int = 500


def create_dev_server(
    project_dir: Optional[Path] = None,
    discovered_agents: Optional[Dict[str, any]] = None
) -> FastAPI:
    """
    Create a development server with auto-discovered agents.

    Args:
        project_dir: Directory to scan for agents (defaults to current directory)
        discovered_agents: Pre-discovered agents (optional, will auto-discover if None)

    Returns:
        FastAPI app configured with all discovered agents
    """
    # Auto-discover agents if not provided
    if discovered_agents is None:
        if project_dir is None:
            project_dir = Path(".")
        discovered_agents = discover_agents(project_dir)

    # In-memory API key storage (for dev only)
    api_keys = {}
    
    # Dev mode check - these keys only work when TELEON_ENV is 'development' or not set
    is_dev_mode = os.getenv("TELEON_ENV", "development").lower() in ["development", "dev", "local"]

    require_login = os.getenv("TELEON_DEV_SERVER_REQUIRE_LOGIN", "").strip().lower() in ["1", "true", "yes", "on"]
    require_login = require_login or (not is_dev_mode)

    platform_url = os.getenv("TELEON_PLATFORM_URL", "https://api.teleon.ai")
    dashboard_url = os.getenv("TELEON_DASHBOARD_URL", "https://dashboard.teleon.ai")
    better_auth_url = os.getenv(
        "TELEON_AUTH_URL",
        os.getenv("BETTER_AUTH_URL", "https://auth.teleon.ai"),
    )
    auth_cookie_name = os.getenv("TELEON_DEV_SERVER_AUTH_COOKIE", "teleon_auth_token")
    auth_cache_ttl_seconds = int(os.getenv("TELEON_DEV_SERVER_AUTH_CACHE_TTL", "300"))
    auth_validation_cache: Dict[str, dict] = {}

    BETTER_AUTH_COOKIE_NAMES = (
        "__Secure-better-auth.session_token",
        "better-auth.session_token",
    )

    # Stable deployment ID set at deploy time (env var injected by the deployer).
    deployment_id = os.getenv("TELEON_DEPLOYMENT_ID", "")

    # Request history for debugging (max 500 entries)
    request_history = deque(maxlen=500)
    
    # Performance metrics
    performance_metrics = {
        "total_requests": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
        "total_bandwidth_bytes": 0,  # Track total bytes sent
        "requests_by_minute": defaultdict(int),  # {minute_timestamp: count}
        "latency_by_minute": defaultdict(list),   # {minute_timestamp: [latencies]}
        "bandwidth_by_minute": defaultdict(int),  # {minute_timestamp: bytes}
        "requests_by_agent": defaultdict(int),
        "tokens_by_agent": defaultdict(int),
        "cost_by_agent": defaultdict(float),
    }
    
    # Hot reload state
    reload_state = {
        "last_reload": datetime.now(timezone.utc),
        "reload_count": 0,
        "changed_files": [],
        "file_watcher_active": False,
        "needs_reload": False
    }

    # Create FastAPI app
    app = FastAPI(
        title="Teleon Dev Server",
        description="Development server with automatic agent discovery",
        version="2.0.0",
        docs_url="/docs",  # Explicit docs URL
        redoc_url=None     # Disable redoc
    )

    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request capture middleware for history and metrics
    class RequestCaptureMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            # Skip capturing for static endpoints and monitoring endpoints
            skip_paths = ["/health", "/api/history", "/api/metrics", "/api/reload/status"]
            if any(request.url.path.startswith(path) for path in skip_paths):
                return await call_next(request)
            
            # Capture request
            request_id = str(uuid.uuid4())
            start_time = time.time()
            request_timestamp = datetime.now(timezone.utc)
            
            # Try to read request body (for POST requests)
            request_body = None
            if request.method == "POST":
                try:
                    body_bytes = await request.body()
                    if body_bytes:
                        request_body = json.loads(body_bytes)
                    # Re-populate body for next middleware/endpoint
                    async def receive():
                        return {"type": "http.request", "body": body_bytes}
                    request._receive = receive
                except:
                    pass
            
            # Process request
            response = await call_next(request)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract agent info from request
            agent_id = None
            agent_name = None
            
            if request.url.path == "/invoke" and request_body:
                agent_name = request_body.get("agent_name")
                # Find agent_id from name
                for aid, info in discovered_agents.items():
                    if info["name"] == agent_name:
                        agent_id = aid
                        break
            elif "/invoke" in request.url.path and request.url.path != "/invoke":
                # Individual agent endpoint like /agent_xxx/invoke
                path_parts = request.url.path.split("/")
                if len(path_parts) >= 2:
                    agent_id = path_parts[1]
                    agent_info = discovered_agents.get(agent_id, {})
                    agent_name = agent_info.get("name")
            
            # Store in history (only for agent invocations)
            if agent_id or "/playground" in request.url.path:
                history_entry = {
                    "id": request_id,
                    "timestamp": request_timestamp.isoformat(),
                    "method": request.method,
                    "path": request.url.path,
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "request_body": request_body,
                    "status_code": response.status_code,
                    "latency_ms": round(latency_ms, 2),
                    "tokens_used": None,  # Will be updated by endpoint
                    "cost": None,  # Will be updated by endpoint
                    "error": None
                }
                request_history.append(history_entry)
                
                # Update performance metrics
                performance_metrics["total_requests"] += 1
                
                # Track bandwidth from response headers
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        bytes_sent = int(content_length)
                        performance_metrics["total_bandwidth_bytes"] += bytes_sent
                        minute_key_bw = request_timestamp.replace(second=0, microsecond=0).isoformat()
                        performance_metrics["bandwidth_by_minute"][minute_key_bw] += bytes_sent
                    except (ValueError, TypeError):
                        pass
                
                # Bucket by minute for time-series data
                minute_key = request_timestamp.replace(second=0, microsecond=0).isoformat()
                performance_metrics["requests_by_minute"][minute_key] += 1
                performance_metrics["latency_by_minute"][minute_key].append(latency_ms)
                
                if agent_name:
                    performance_metrics["requests_by_agent"][agent_name] += 1
            
            return response
    
    app.add_middleware(RequestCaptureMiddleware)

    async def validate_platform_api_key(api_key: str) -> Optional[dict]:
        api_key = (api_key or "").strip()
        if not api_key:
            return None

        now = datetime.now(timezone.utc)
        cached = auth_validation_cache.get(api_key)
        if cached and cached.get("expires_at") and cached["expires_at"] > now:
            return cached.get("data")

        if not HTTPX_AVAILABLE:
            return None

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{platform_url}/api/v1/api-keys/validate",
                    headers={"X-API-Key": api_key},
                )
            if resp.status_code != 200:
                return None

            data = resp.json()
            auth_validation_cache[api_key] = {
                "expires_at": now + timedelta(seconds=auth_cache_ttl_seconds),
                "data": data,
            }
            return data
        except Exception:
            return None

    def _get_session_token(request: Request) -> Optional[str]:
        """Extract the Better Auth session token from the request cookies."""
        for name in BETTER_AUTH_COOKIE_NAMES:
            val = request.cookies.get(name)
            if val:
                return val
        return None

    def _session_cookie_dict(session_token: str) -> dict:
        """Build the cookie dict to forward when calling Better Auth."""
        return {name: session_token for name in BETTER_AUTH_COOKIE_NAMES}

    async def validate_better_auth_session(session_token: str) -> Optional[dict]:
        """Validate a Better Auth session token against the auth service.

        Returns the session payload (including ``user`` dict) on success, or
        ``None`` if the session is invalid/expired.
        """
        if not session_token:
            return None

        cache_key = f"ba_session:{session_token[:16]}"
        now = datetime.now(timezone.utc)
        cached = auth_validation_cache.get(cache_key)
        if cached and cached.get("expires_at") and cached["expires_at"] > now:
            return cached.get("data")

        if not HTTPX_AVAILABLE:
            return None

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{better_auth_url}/api/auth/get-session",
                    cookies=_session_cookie_dict(session_token),
                )
            if resp.status_code != 200:
                return None

            data = resp.json()
            if not data or not data.get("user"):
                return None

            auth_validation_cache[cache_key] = {
                "expires_at": now + timedelta(seconds=auth_cache_ttl_seconds),
                "data": data,
            }
            return data
        except Exception as e:
            logger.warning(f"Better Auth session validation failed: {e}")
            return None

    def extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
        if not authorization:
            return None
        if authorization.startswith("Bearer "):
            return authorization.replace("Bearer ", "", 1).strip()
        return None

    def is_local_host(host: str) -> bool:
        if not host:
            return True
        host = host.lower().strip()
        host = host.split(":", 1)[0]
        return host in {"localhost", "127.0.0.1", "0.0.0.0"}

    class PlatformAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            forwarded_host = request.headers.get("x-forwarded-host")
            host = forwarded_host or request.headers.get("host") or ""
            deployed_host = (not is_local_host(host))
            enforce_login = require_login or deployed_host

            if not enforce_login:
                return await call_next(request)

            path = request.url.path
            if (path.startswith("/login") or path.startswith("/logout")
                    or path.startswith("/auth/callback") or path == "/health"):
                return await call_next(request)

            authenticated = False

            if deployed_host:
                # --- Deployed agents: use Better Auth session cookie ---
                session_token = _get_session_token(request)
                ownership_cookie = request.cookies.get(auth_cookie_name) or ""

                if session_token and ownership_cookie.startswith("verified:"):
                    expected = "verified:" + hashlib.sha256(session_token.encode()).hexdigest()[:16]
                    if ownership_cookie == expected:
                        session_data = await validate_better_auth_session(session_token)
                        if session_data:
                            authenticated = True

            if not authenticated:
                # --- Localhost / fallback: API key cookie or Authorization header ---
                token = request.cookies.get(auth_cookie_name)
                if not token or token.startswith("verified:"):
                    token = None
                if not token:
                    token = extract_bearer_token(request.headers.get("authorization"))

                if token:
                    valid = await validate_platform_api_key(token)
                    if valid:
                        authenticated = True

            if authenticated:
                return await call_next(request)

            accept = request.headers.get("accept", "")
            wants_html = "text/html" in accept
            if wants_html and request.method == "GET":
                return RedirectResponse(url="/login", status_code=302)

            return JSONResponse(status_code=401, content={"detail": "Authentication required"})

    app.add_middleware(PlatformAuthMiddleware)

    def verify_api_key(api_key: str) -> bool:
        """
        Verify if API key is valid.
        
        Dev API keys (starting with 'teleon_dev_') only work when TELEON_ENV
        is set to 'development', 'dev', or 'local'.
        """
        # Check if key exists
        if api_key not in api_keys:
            return False
        
        # If it's a dev key, ensure we're in dev mode
        if api_key.startswith("teleon_dev_"):
            if not is_dev_mode:
                return False
        
        return True

    def generate_api_key() -> str:
        """Generate a new dev-only API key."""
        return f"teleon_dev_{secrets.token_urlsafe(32)}"

    @app.get("/")
    async def root():
        """Root endpoint - shows dashboard."""
        no_agents_msg = '<div class="empty"><p>No agents found in this directory.</p><p>Create an agent using TeleonClient:</p><pre><code>from teleon import TeleonClient\n\nclient = TeleonClient(api_key="your-key")\n\n@client.agent(name="my-agent")\ndef my_agent(input: str) -> str:\n    return f"Hello, {input}!"</code></pre></div>'
        
        agents_html = ''.join([f'''
            <div class="agent">
                <h3>{agent_info["name"]}</h3>
                <p>{agent_info["description"]}</p>
                <p><strong>Agent ID:</strong> <code>{agent_id}</code></p>
                <p><strong>Model:</strong> {agent_info["model"]} | <strong>User:</strong> {agent_info["user_id"]}</p>

                <h4>Method 1 - Shared Endpoint:</h4>
                <p><code>POST /invoke</code> with <code>{{'agent_name': '{agent_info["name"]}', 'input': 'your input'}}</code></p>

                <h4>Method 2 - Individual Endpoint:</h4>
                <p><code>POST /{agent_id}/invoke</code> with <code>{{'input': 'your input'}}</code></p>
                <p><a href="/{agent_id}/docs">→ View {agent_info["name"]} API Docs</a></p>
                <p><a href="/{agent_id}/info">→ View {agent_info["name"]} Info</a></p>
            </div>
            ''' for agent_id, agent_info in discovered_agents.items()])
        
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Teleon Dev Server</title>
            <style>
                body {{ font-family: -apple-system, sans-serif; background: #000; color: #e0e0e0; padding: 40px; }}
                h1 {{ color: #00d4ff; }}
                .agent {{ background: #0a0a0a; border: 1px solid #333; padding: 20px; margin: 20px 0; border-radius: 8px; }}
                .agent h3 {{ margin-top: 0; color: #00d4ff; }}
                a {{ color: #00d4ff; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
                .method {{ display: inline-block; padding: 4px 12px; border-radius: 4px; font-size: 12px; font-weight: 600; }}
                .shared {{ background: #1a1a2e; color: #00d4ff; border: 1px solid #00d4ff; margin-right: 10px; }}
                .individual {{ background: #1a2e1a; color: #00ff88; border: 1px solid #00ff88; }}
                code {{ background: #1a1a1a; padding: 2px 8px; border-radius: 4px; }}
                .empty {{ color: #888; font-style: italic; }}
                .btn {{ background: #00d4ff; color: #000; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 14px; display: inline-block; text-decoration: none; }}
                .btn:hover {{ background: #00aacc; }}
                .btn-secondary {{ background: #333; color: #e0e0e0; }}
                .btn-secondary:hover {{ background: #444; }}
                .dev-warning {{ background: #2e1a1a; border: 1px solid #ff6b6b; padding: 10px 15px; border-radius: 6px; margin: 20px 0; color: #ff6b6b; font-size: 13px; }}
                .key-section {{ background: #0a0a0a; border: 1px solid #333; padding: 20px; margin: 20px 0; border-radius: 8px; }}
                #apiKeyResult {{ margin-top: 15px; padding: 15px; border-radius: 6px; display: none; }}
                .success {{ background: #1a2e1a; border: 1px solid #00ff88; color: #00ff88; }}
                input {{ background: #1a1a1a; color: #e0e0e0; border: 1px solid #333; padding: 10px; border-radius: 4px; width: 300px; margin-right: 10px; }}
                .key-display {{ background: #1a1a1a; padding: 10px; border-radius: 4px; margin: 10px 0; font-family: monospace; word-break: break-all; }}
                
                .features-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin: 30px 0; }}
                .feature-card {{ background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%); border: 1px solid #333; padding: 25px; border-radius: 10px; text-decoration: none; display: block; transition: all 0.3s ease; }}
                .feature-card:hover {{ border-color: #00d4ff; transform: translateY(-3px); box-shadow: 0 5px 20px rgba(0, 212, 255, 0.2); }}
                .feature-icon {{ font-size: 32px; margin-bottom: 10px; }}
                .feature-title {{ font-size: 18px; font-weight: 600; color: #00d4ff; margin: 10px 0; }}
                .feature-desc {{ font-size: 13px; color: #999; line-height: 1.5; }}
                .feature-badge {{ display: inline-block; background: #1a2e1a; color: #00ff88; padding: 3px 8px; border-radius: 3px; font-size: 11px; font-weight: 600; margin-top: 10px; }}
            </style>
        </head>
        <body>
            <h1>🚀 Teleon Dev Server</h1>
            <p><strong>__AGENTS_COUNT__</strong> agents discovered automatically</p>

            <div class="dev-warning">
                ⚠️ <strong>DEVELOPMENT MODE:</strong> This server is for local development only. API keys generated here are for testing purposes and will not work in production.
            </div>

            <div class="key-section">
                <h2>🔑 Generate Test API Key</h2>
                <p>Create a test API key to call your agents. These keys only work in development mode.</p>
                <div style="margin: 15px 0;">
                    <input type="text" id="keyName" placeholder="Enter key name (e.g., 'My Test Key')" />
                    <button class="btn" onclick="generateTestKey()">Generate Test API Key</button>
                    <a href="/api-keys" class="btn btn-secondary" style="margin-left: 10px;">Manage All Keys</a>
                </div>
                <div id="apiKeyResult"></div>
            </div>

            <h2>✨ Developer Tools</h2>
            <div class="features-grid">
                <a href="/playground" class="feature-card">
                    <div class="feature-icon">🎮</div>
                    <div class="feature-title">Agent Playground</div>
                    <div class="feature-desc">Test your agents interactively with a chat-like interface. No API keys or curl commands needed!</div>
                    <span class="feature-badge">NEW</span>
                </a>
                
                <a href="/performance" class="feature-card">
                    <div class="feature-icon">📊</div>
                    <div class="feature-title">Performance Dashboard</div>
                    <div class="feature-desc">Real-time metrics, charts, and analytics. Monitor latency, token usage, and costs at a glance.</div>
                    <span class="feature-badge">NEW</span>
                </a>
                
                <a href="/history" class="feature-card">
                    <div class="feature-icon">📜</div>
                    <div class="feature-title">Request History</div>
                    <div class="feature-desc">Debug requests with full request/response inspection. Search, filter, and export logs easily.</div>
                    <span class="feature-badge">NEW</span>
                </a>
                
                <a href="#" onclick="checkReloadStatus(); return false;" class="feature-card">
                    <div class="feature-icon">🔄</div>
                    <div class="feature-title">Hot Reload</div>
                    <div class="feature-desc">Automatic agent re-discovery when files change. Edit code and see updates instantly.</div>
                    <span class="feature-badge" id="reloadBadge">Active</span>
                </a>
            </div>

            __NO_AGENTS_MSG__

            <h2>Available Methods:</h2>
            <div class="agent">
                <span class="method shared">METHOD 1: SHARED</span>
                <p>All agents on one endpoint with agent name in body:</p>
                <p><code>POST /invoke</code> with <code>{{"agent_name": "...", "input": "..."}}</code></p>
                <p><a href="/docs">→ Shared API Documentation</a></p>
            </div>

            <div class="agent">
                <span class="method individual">METHOD 2: INDIVIDUAL</span>
                <p>Each agent has its own endpoint and docs:</p>
                <p><code>POST /{{agent_id}}/invoke</code> with <code>{{"input": "..."}}</code></p>
                <p>Each agent gets its own OpenAPI docs!</p>
            </div>

            <h2>Discovered Agents:</h2>
            __AGENTS_HTML__

            <h2>System Endpoints:</h2>
            <ul>
                <li><a href="/health">Health Check</a></li>
                <li><a href="/agents">List Agents (JSON)</a></li>
                <li><a href="/docs">Shared API Docs</a></li>
                <li><a href="/api-keys">API Key Management</a></li>
            </ul>

            <script>
                async function generateTestKey() {
                    const keyName = document.getElementById('keyName').value || 'Test Key';
                    const resultDiv = document.getElementById('apiKeyResult');

                    try {
                        const response = await fetch('/api-keys', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ 
                                name: keyName,
                                description: 'Dev-only test API key'
                            })
                        });

                        const data = await response.json();
                        
                        resultDiv.className = 'success';
                        resultDiv.style.display = 'block';
                        resultDiv.innerHTML = `
                            <p><strong>✓ Test API Key Generated!</strong></p>
                            <p><strong>Name:</strong> ${data.name}</p>
                            <p><strong>Your API Key:</strong></p>
                            <div class="key-display">${data.api_key}</div>
                            <p><em>⚠️ Save this key - you won't see it again! This key only works in development mode.</em></p>
                            <p style="margin-top: 15px;"><strong>Usage Example:</strong></p>
                            <div class="key-display" style="font-size: 12px;">
curl -X POST http://localhost:8000/invoke \\
  -H "Authorization: Bearer ${data.api_key}" \\
  -H "Content-Type: application/json" \\
  -d '{{"agent_name": "your-agent", "input": "Hello!"}}'
                            </div>
                        `;

                        document.getElementById('keyName').value = '';
                    } catch (error) {
                        resultDiv.style.display = 'block';
                        resultDiv.className = '';
                        resultDiv.style.background = '#2e1a1a';
                        resultDiv.style.border = '1px solid #ff6b6b';
                        resultDiv.style.color = '#ff6b6b';
                        resultDiv.innerHTML = `<p><strong>✗ Error:</strong> ${error.message}</p>`;
                    }
                }

                // Allow Enter key to submit
                document.getElementById('keyName').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        generateTestKey();
                    }
                });
                
                // Check hot reload status
                async function checkReloadStatus() {
                    try {
                        const response = await fetch('/api/reload/status');
                        const data = await response.json();
                        const message = `Hot Reload Status:\\n\\n` +
                            `✓ Watcher: ${data.watcher_active ? 'Active' : 'Inactive'}\\n` +
                            `✓ Total Agents: ${data.total_agents}\\n` +
                            `✓ Last Reload: ${data.seconds_ago}s ago\\n` +
                            `✓ Reload Count: ${data.reload_count}\\n\\n` +
                            `Files are monitored for changes. Edit any .py file to trigger auto-reload.`;
                        alert(message);
                    } catch (error) {
                        alert('Error checking reload status: ' + error.message);
                    }
                }
                
                // Update reload badge on load
                fetch('/api/reload/status')
                    .then(r => r.json())
                    .then(data => {
                        const badge = document.getElementById('reloadBadge');
                        if (data.watcher_active) {
                            badge.textContent = 'Active';
                            badge.style.background = '#1a2e1a';
                            badge.style.color = '#00ff88';
                        } else {
                            badge.textContent = 'Manual Only';
                            badge.style.background = '#2e1a1a';
                            badge.style.color = '#ff6b6b';
                        }
                    })
                    .catch(() => {});
            </script>
        </body>
        </html>
        """

        template = template.replace("{{", "{").replace("}}", "}")
        content = (
            template
            .replace("__AGENTS_COUNT__", str(len(discovered_agents)))
            .replace("__NO_AGENTS_MSG__", no_agents_msg if not discovered_agents else "")
            .replace("__AGENTS_HTML__", agents_html)
        )
        return HTMLResponse(content=content)

    def _build_dashboard_login_url(request: Request) -> str:
        """Build the URL that redirects the user to the Teleon dashboard to log in."""
        forwarded_proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        forwarded_host = request.headers.get("x-forwarded-host", request.headers.get("host", "localhost"))
        origin = f"{forwarded_proto}://{forwarded_host}"
        callback = f"{origin}/auth/callback"

        params = urllib.parse.urlencode({
            "devserver": "true",
            "callback": callback,
        })
        return f"{dashboard_url}/login?{params}"

    @app.get("/login")
    async def login_page(request: Request):
        login_url = _build_dashboard_login_url(request)
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Teleon - Login</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
background:#000;color:#e0e0e0;display:flex;align-items:center;justify-content:center;
min-height:100vh}}
.container{{text-align:center;max-width:420px;padding:40px 24px}}
.logo{{font-size:48px;margin-bottom:16px}}
h1{{font-size:22px;color:#fff;margin-bottom:8px}}
.sub{{color:#888;font-size:14px;margin-bottom:40px;line-height:1.5}}
.btn{{display:inline-flex;align-items:center;justify-content:center;gap:10px;
background:linear-gradient(135deg,#00d4ff 0%,#007cf0 100%);color:#000;
border:none;padding:16px 40px;border-radius:10px;cursor:pointer;font-weight:700;
font-size:16px;text-decoration:none;transition:all .2s ease;width:100%}}
.btn:hover{{transform:translateY(-2px);box-shadow:0 8px 30px rgba(0,212,255,.35)}}
.btn svg{{width:20px;height:20px}}
.footer{{margin-top:48px;color:#555;font-size:12px}}
</style>
</head>
<body>
<div class="container">
<div class="logo">&#x1F512;</div>
<h1>Agent Authentication Required</h1>
<p class="sub">Sign in with your Teleon account to access the agents running on this server.</p>
<a href="{login_url}" class="btn">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
stroke-linecap="round" stroke-linejoin="round">
<path d="M15 3h4a2 2 0 012 2v14a2 2 0 01-2 2h-4"/>
<polyline points="10 17 15 12 10 7"/>
<line x1="15" y1="12" x2="3" y2="12"/>
</svg>
Login with Teleon Account
</a>
<p class="footer">You will be redirected to the Teleon dashboard to sign in.</p>
</div>
</body>
</html>"""
        return HTMLResponse(content=html, status_code=200)

    def _render_violation_page(message: str, is_banned: bool) -> HTMLResponse:
        """Render a full-page warning or ban notice."""
        icon = "&#x1F6AB;" if is_banned else "&#x26A0;&#xFE0F;"
        title = "Account Banned" if is_banned else "Access Denied"
        btn_html = "" if is_banned else '<a href="/login" class="btn-back">Go Back</a>'
        color = "#ff4444" if is_banned else "#ffaa00"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Teleon - {title}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
background:#000;color:#e0e0e0;display:flex;align-items:center;justify-content:center;
min-height:100vh}}
.container{{text-align:center;max-width:480px;padding:40px 24px}}
.icon{{font-size:64px;margin-bottom:20px}}
h1{{font-size:24px;color:{color};margin-bottom:12px}}
.msg{{color:#ccc;font-size:15px;line-height:1.6;margin-bottom:32px;
background:#0a0a0a;border:1px solid #333;border-radius:10px;padding:20px}}
.btn-back{{display:inline-block;background:#333;color:#e0e0e0;border:none;
padding:14px 32px;border-radius:8px;cursor:pointer;font-weight:600;font-size:14px;
text-decoration:none;transition:all .2s ease}}
.btn-back:hover{{background:#444}}
</style>
</head>
<body>
<div class="container">
<div class="icon">{icon}</div>
<h1>{title}</h1>
<div class="msg">{message}</div>
{btn_html}
</div>
</body>
</html>"""
        return HTMLResponse(content=html, status_code=403)

    async def _check_ownership_and_record_violation(
        auth_headers: dict,
        auth_cookies: dict,
    ) -> tuple[bool, Optional[HTMLResponse]]:
        """Verify deployment ownership via the platform API.

        Returns ``(True, None)`` on success, or ``(False, error_response)`` on
        failure (including violation recording and ban pages).
        """
        if not deployment_id:
            logger.error("TELEON_DEPLOYMENT_ID not set — cannot verify ownership")
            return False, HTMLResponse(
                "<h2>Server misconfigured</h2>"
                "<p>TELEON_DEPLOYMENT_ID is not set. Cannot verify agent ownership.</p>",
                status_code=500,
            )

        ownership_ok = False
        if HTTPX_AVAILABLE:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        f"{platform_url}/api/v1/auth/verify-agent-ownership",
                        headers=auth_headers,
                        cookies=auth_cookies,
                        json={"deployment_id": deployment_id},
                    )
                if resp.status_code == 200:
                    ownership_ok = resp.json().get("owned", False)
                else:
                    logger.warning(f"Ownership check returned {resp.status_code}: {resp.text}")
            except Exception as e:
                logger.error(f"Ownership check failed (denying access): {e}")

        if ownership_ok:
            return True, None

        # Record the violation
        violation_data: dict = {"violations": 1, "is_banned": False, "message": "Ownership check failed."}
        if HTTPX_AVAILABLE:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    vresp = await client.post(
                        f"{platform_url}/api/v1/auth/record-violation",
                        headers=auth_headers,
                        cookies=auth_cookies,
                        json={"deployment_id": deployment_id},
                    )
                if vresp.status_code == 200:
                    violation_data = vresp.json()
            except Exception as e:
                logger.warning(f"Failed to record violation: {e}")

        return False, _render_violation_page(
            violation_data.get("message", ""),
            violation_data.get("is_banned", False),
        )

    @app.get("/auth/callback")
    async def auth_callback(request: Request):
        """Callback after the user logs in on the Teleon dashboard.

        Deployed agents (*.agents.teleon.ai) use the Better Auth session cookie
        that was set when the user logged into the dashboard (same parent domain
        ``.teleon.ai``).  No secrets travel in the URL.

        Local dev (localhost) falls back to an API key passed as a query param
        because cross-domain cookies are not available.
        """
        forwarded_host = request.headers.get("x-forwarded-host")
        host = forwarded_host or request.headers.get("host") or ""
        deployed = not is_local_host(host)
        is_https = (request.url.scheme == "https"
                    or request.headers.get("x-forwarded-proto") == "https")

        if deployed:
            # ── Deployed path: Better Auth session cookie ──
            session_token = _get_session_token(request)
            if not session_token:
                logger.warning("No Better Auth session cookie on /auth/callback")
                return RedirectResponse(url="/login", status_code=302)

            session_data = await validate_better_auth_session(session_token)
            if not session_data or not session_data.get("user"):
                logger.warning("Better Auth session invalid at /auth/callback")
                return RedirectResponse(url="/login", status_code=302)

            # Forward the session cookie to the platform for ownership check
            cookies_to_forward = _session_cookie_dict(session_token)
            owned, err_resp = await _check_ownership_and_record_violation(
                auth_headers={},
                auth_cookies=cookies_to_forward,
            )
            if not owned:
                cache_key = f"ba_session:{session_token[:16]}"
                auth_validation_cache.pop(cache_key, None)
                return err_resp

            # Ownership verified — set a local marker cookie tied to this
            # session so the middleware can skip the ownership check on
            # subsequent requests.
            session_hash = hashlib.sha256(session_token.encode()).hexdigest()[:16]
            resp = RedirectResponse(url="/", status_code=302)
            resp.set_cookie(
                auth_cookie_name,
                f"verified:{session_hash}",
                httponly=True,
                secure=is_https,
                samesite="lax",
                max_age=60 * 60 * 24 * 7,
            )
            return resp

        else:
            # ── Local dev path: API key in query param ──
            api_key = (request.query_params.get("api_key") or "").strip()
            if not api_key:
                return RedirectResponse(url="/login", status_code=302)

            valid = await validate_platform_api_key(api_key)
            if not valid:
                return RedirectResponse(url="/login", status_code=302)

            owned, err_resp = await _check_ownership_and_record_violation(
                auth_headers={"X-API-Key": api_key},
                auth_cookies={},
            )
            if not owned:
                auth_validation_cache.pop(api_key, None)
                return err_resp

            resp = RedirectResponse(url="/", status_code=302)
            resp.set_cookie(
                auth_cookie_name,
                api_key,
                httponly=True,
                secure=is_https,
                samesite="lax",
                max_age=60 * 60 * 24 * 7,
            )
            return resp

    @app.get("/logout")
    async def logout():
        resp = RedirectResponse(url="/login", status_code=302)
        resp.delete_cookie(auth_cookie_name)
        return resp

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agents_count": len(discovered_agents),
            "api_keys_active": len(api_keys),
            "environment": os.getenv("TELEON_ENV", "development"),
            "dev_mode": is_dev_mode
        }

    @app.get("/agents")
    async def list_agents():
        """List all discovered agents."""
        return {
            "agents": [
                {
                    "agent_id": agent_id,
                    "name": info["name"],
                    "description": info["description"],
                    "model": info["model"],
                    "user_id": info["user_id"],
                    "endpoints": {
                        "shared": "/invoke",
                        "individual": f"/{agent_id}/invoke",
                        "docs": f"/{agent_id}/docs",
                        "info": f"/{agent_id}/info"
                    }
                }
                for agent_id, info in discovered_agents.items()
            ]
        }

    # Dashboard compatibility endpoints
    @app.get("/api/dashboard/stats")
    async def dashboard_stats():
        """Dashboard stats endpoint."""
        return {
            "totalAgents": len(discovered_agents),
            "activeAgents": len(discovered_agents),  # All agents are active in dev mode
            "totalRequests": 0,  # Dev server doesn't track this
            "totalCost": 0.0
        }

    @app.get("/api/agents")
    async def list_agents_api():
        """List agents in dashboard format."""
        return [
            {
                "id": agent_id,
                "name": info["name"],
                "status": "running",
                "requests": 0,
                "cost": 0.0,
                "created_at": info["created_at"],
                "description": info["description"],
                "model": info["model"]
            }
            for agent_id, info in discovered_agents.items()
        ]

    @app.get("/api/agents/{agent_id}")
    async def get_agent_api(agent_id: str):
        """Get agent details in dashboard format."""
        agent_info = discovered_agents.get(agent_id)
        if not agent_info:
            raise HTTPException(status_code=404, detail="Agent not found")

        return {
            "id": agent_id,
            "name": agent_info["name"],
            "status": "running",
            "requests": 0,
            "cost": 0.0,
            "avgLatency": 0,
            "created_at": agent_info["created_at"],
            "description": agent_info["description"],
            "model": agent_info["model"],
            "temperature": agent_info["temperature"],
            "max_tokens": agent_info["max_tokens"]
        }

    # WebSocket endpoint for real-time updates (placeholder)
    @app.get("/ws")
    async def websocket_endpoint():
        """WebSocket endpoint placeholder."""
        return {
            "message": "WebSocket not implemented in dev server",
            "status": "placeholder",
            "note": "Dashboard will use polling instead for real-time updates"
        }

    # METHOD 1: Shared endpoint (original)
    @app.post("/invoke")
    async def invoke_agent_shared(
        request: AgentRequest,
        authorization: Optional[str] = Header(None)
    ):
        """
        METHOD 1: Invoke any agent by name (shared endpoint).

        Requires API key in Authorization header.
        """
        # Verify API key
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing authorization header")

        api_key = authorization.replace("Bearer ", "")
        if not verify_api_key(api_key):
            raise HTTPException(status_code=403, detail="Invalid API key")

        # Find agent by name
        agent_info = None
        agent_id = None
        for aid, info in discovered_agents.items():
            if info["name"] == request.agent_name:
                agent_info = info
                agent_id = aid
                break

        if not agent_info:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{request.agent_name}' not found"
            )

        # Invoke agent
        return await _invoke_agent(agent_id, agent_info, request.input, request.temperature, request.max_tokens)

    # METHOD 2: Individual agent endpoints
    @app.post("/{agent_id}/invoke")
    async def invoke_agent_individual(
        agent_id: str,
        request: IndividualAgentRequest,
        authorization: Optional[str] = Header(None)
    ):
        """
        METHOD 2: Invoke a specific agent by its ID.

        Each agent has its own endpoint with its own docs.
        """
        # Verify API key
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing authorization header")

        api_key = authorization.replace("Bearer ", "")
        if not verify_api_key(api_key):
            raise HTTPException(status_code=403, detail="Invalid API key")

        # Get agent
        agent_info = discovered_agents.get(agent_id)
        if not agent_info:
            raise HTTPException(status_code=404, detail=f"Agent ID '{agent_id}' not found")

        # Use agent's default config if not provided
        temperature = request.temperature if request.temperature is not None else agent_info["temperature"]
        max_tokens = request.max_tokens if request.max_tokens is not None else agent_info["max_tokens"]

        # Invoke agent
        return await _invoke_agent(agent_id, agent_info, request.input, temperature, max_tokens)

    async def _invoke_agent(agent_id: str, agent_info: dict, input_data: Any, temperature: float, max_tokens: int):
        """Internal function to invoke an agent."""
        try:
            start_time = datetime.now(timezone.utc)
            agent_func = agent_info["function"]

            # Determine how to call the agent based on input type
            if isinstance(input_data, dict):
                # Dict input: pass as keyword arguments to support multi-parameter agents
                # Filter out None values and reserved keys
                kwargs = {k: v for k, v in input_data.items() if v is not None and k not in ('temperature', 'max_tokens')}

                if asyncio.iscoroutinefunction(agent_func):
                    output = await agent_func(**kwargs)
                else:
                    output = agent_func(**kwargs)
            else:
                # String input: pass as positional argument (backward compatible)
                if asyncio.iscoroutinefunction(agent_func):
                    output = await agent_func(input_data)
                else:
                    output = agent_func(input_data)

            end_time = datetime.now(timezone.utc)
            latency_ms = (end_time - start_time).total_seconds() * 1000

            return AgentResponse(
                agent_id=agent_id,
                agent_name=agent_info["name"],
                output=output,
                tokens_used=None,
                cost=None,
                latency_ms=latency_ms,
                timestamp=end_time.isoformat()
            )

        except TypeError as e:
            # Provide helpful error message for parameter mismatches
            raise HTTPException(
                status_code=400,
                detail=f"Agent parameter error: {str(e)}. Check that your input matches the agent's expected parameters: {agent_info.get('parameters', 'unknown')}"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

    @app.get("/{agent_id}/info")
    async def get_agent_info(agent_id: str):
        """Get detailed information about a specific agent."""
        agent_info = discovered_agents.get(agent_id)
        if not agent_info:
            raise HTTPException(status_code=404, detail=f"Agent ID '{agent_id}' not found")

        return {
            "agent_id": agent_id,
            "name": agent_info["name"],
            "description": agent_info["description"],
            "user_id": agent_info["user_id"],
            "model": agent_info["model"],
            "temperature": agent_info["temperature"],
            "max_tokens": agent_info["max_tokens"],
            "parameters": agent_info["parameters"],
            "created_at": agent_info["created_at"],
            "endpoints": {
                "invoke": f"/{agent_id}/invoke",
                "docs": f"/{agent_id}/docs",
                "info": f"/{agent_id}/info"
            }
        }

    @app.get("/{agent_id}/docs", include_in_schema=False)
    async def get_agent_docs(agent_id: str):
        """Get OpenAPI documentation for a specific agent."""
        agent_info = discovered_agents.get(agent_id)
        if not agent_info:
            raise HTTPException(status_code=404, detail=f"Agent ID '{agent_id}' not found")

        return get_swagger_ui_html(
            openapi_url=f"/{agent_id}/openapi.json",
            title=f"{agent_info['name']} - API Docs"
        )

    @app.get("/{agent_id}/openapi.json", include_in_schema=False)
    async def get_agent_openapi(agent_id: str):
        """Get OpenAPI JSON spec for a specific agent."""
        agent_info = discovered_agents.get(agent_id)
        if not agent_info:
            raise HTTPException(status_code=404, detail=f"Agent ID '{agent_id}' not found")

        return {
            "openapi": "3.0.0",
            "info": {
                "title": f"{agent_info['name']} API",
                "description": agent_info['description'],
                "version": "1.0.0"
            },
            "servers": [
                {"url": f"http://localhost:8000/{agent_id}", "description": "Development server"}
            ],
            "paths": {
                "/invoke": {
                    "post": {
                        "summary": f"Invoke {agent_info['name']}",
                        "description": agent_info['description'],
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "input": {
                                                "oneOf": [
                                                    {"type": "string", "description": "Simple text input"},
                                                    {"type": "object", "description": "Structured input with agent parameters"}
                                                ],
                                                "description": "Input for the agent - can be a string or an object with parameters"
                                            },
                                            "temperature": {"type": "number", "default": agent_info["temperature"]},
                                            "max_tokens": {"type": "integer", "default": agent_info["max_tokens"]}
                                        },
                                        "required": ["input"]
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "Successful response",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "agent_id": {"type": "string"},
                                                "agent_name": {"type": "string"},
                                                "output": {
                                                    "description": "Agent output - can be string, object, array, or any JSON-serializable type"
                                                },
                                                "tokens_used": {"type": "integer"},
                                                "cost": {"type": "number"},
                                                "latency_ms": {"type": "number"},
                                                "timestamp": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "security": [{"bearerAuth": []}]
                    }
                }
            },
            "components": {
                "securitySchemes": {
                    "bearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "API Key"
                    }
                }
            }
        }

    @app.post("/api-keys")
    async def create_api_key(request: APIKeyRequest):
        """Create a new dev-only API key for testing."""
        if not is_dev_mode:
            raise HTTPException(
                status_code=403,
                detail="Test API key generation is only available in development mode. Set TELEON_ENV=development to enable."
            )
        
        api_key = generate_api_key()
        api_keys[api_key] = {
            "name": request.name,
            "description": request.description or "Dev-only test API key",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "environment": "development_only"
        }
        return APIKeyResponse(
            api_key=api_key,
            name=request.name,
            created_at=api_keys[api_key]["created_at"],
            message="Dev-only API key created successfully! This key will NOT work in production."
        )

    @app.get("/api-keys")
    async def list_api_keys():
        """List all API keys."""
        dev_mode_warning = """
            <div style="background: #2e1a1a; border: 1px solid #ff6b6b; padding: 10px 15px; border-radius: 6px; margin: 20px 0; color: #ff6b6b;">
                ⚠️ <strong>DEVELOPMENT MODE ONLY:</strong> API keys generated here will NOT work in production (when TELEON_ENV=production).
            </div>
        """ if is_dev_mode else """
            <div style="background: #2e1a1a; border: 1px solid #ff6b6b; padding: 10px 15px; border-radius: 6px; margin: 20px 0; color: #ff6b6b;">
                ⚠️ <strong>PRODUCTION MODE:</strong> Test API key generation is disabled. Set TELEON_ENV=development to enable.
            </div>
        """

        disabled_attr = "disabled" if not is_dev_mode else ""
        create_button_label = "Create Test API Key" if is_dev_mode else "Disabled in Production Mode"
        keys_html = "".join([
            f''' 
            <div class="key">
                <p><strong>{data["name"]}</strong> <span class="env-badge dev-badge">DEV ONLY</span></p>
                <p>{data.get("description", "No description")}</p>
                <p>Created: {data["created_at"]}</p>
                <p>Environment: {data.get("environment", "development_only")}</p>
                <p>Key: <code>{key[:40]}...</code></p>
            </div>
            '''
            for key, data in api_keys.items()
        ])

        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>API Keys - Teleon Dev Server</title>
            <style>
                body {{ font-family: -apple-system, sans-serif; background: #000; color: #e0e0e0; padding: 40px; }}
                h1 {{ color: #00d4ff; }}
                .key {{ background: #0a0a0a; border: 1px solid #333; padding: 15px; margin: 15px 0; border-radius: 8px; }}
                button {{ background: #00d4ff; color: #000; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-weight: 600; }}
                button:hover {{ background: #00aacc; }}
                button:disabled {{ background: #333; cursor: not-allowed; }}
                input {{ background: #1a1a1a; color: #e0e0e0; border: 1px solid #333; padding: 8px; border-radius: 4px; margin: 5px 0; width: 300px; }}
                code {{ background: #1a1a1a; padding: 4px 8px; border-radius: 4px; color: #00ff88; }}
                a {{ color: #00d4ff; text-decoration: none; }}
                .env-badge {{ display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 11px; font-weight: 600; margin-left: 10px; }}
                .dev-badge {{ background: #1a2e1a; color: #00ff88; border: 1px solid #00ff88; }}
            </style>
        </head>
        <body>
            <h1>🔑 API Key Management</h1>
            <p><a href="/">← Back to Dashboard</a></p>

            __DEV_MODE_WARNING__

            <h2>Create New Test API Key</h2>
            <div class="key">
                <form id="createKeyForm">
                    <div>
                        <label>Name: <input type="text" id="keyName" placeholder="My Test Key" required __DISABLED_ATTR__ /></label>
                    </div>
                    <div>
                        <label>Description: <input type="text" id="keyDesc" placeholder="Optional description" __DISABLED_ATTR__ /></label>
                    </div>
                    <div style="margin-top: 10px;">
                        <button type="submit" __DISABLED_ATTR__>
                            __CREATE_BUTTON_LABEL__
                        </button>
                        <span class="env-badge dev-badge">DEV ONLY</span>
                    </div>
                </form>
                <div id="result" style="margin-top: 15px;"></div>
            </div>

            <h2>Active API Keys (__KEYS_COUNT__)</h2>
            __KEYS_HTML__

            <script>
                document.getElementById('createKeyForm').addEventListener('submit', async (e) => {{
                    e.preventDefault();
                    const name = document.getElementById('keyName').value;
                    const description = document.getElementById('keyDesc').value;

                    try {{
                    const response = await fetch('/api-keys', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ name, description }})
                    }});

                        if (!response.ok) {{
                            const error = await response.json();
                            throw new Error(error.detail || 'Failed to create API key');
                        }}

                    const data = await response.json();
                    document.getElementById('result').innerHTML = `
                        <div style="background: #1a2e1a; border: 1px solid #00ff88; padding: 15px; border-radius: 4px;">
                                <p><strong>✓ Test API Key Created!</strong></p>
                            <p>Name: ${{data.name}}</p>
                            <p>Key: <code style="color: #00ff88;">${{data.api_key}}</code></p>
                                <p><em>⚠️ Save this key - you won't see it again! This key only works in development mode.</em></p>
                                <p style="margin-top: 10px; font-size: 12px; color: #ff6b6b;">
                                    <strong>Note:</strong> ${{data.message}}
                                </p>
                        </div>
                    `;

                        setTimeout(() => location.reload(), 4000);
                    }} catch (error) {{
                        document.getElementById('result').innerHTML = `
                            <div style="background: #2e1a1a; border: 1px solid #ff6b6b; padding: 15px; border-radius: 4px; color: #ff6b6b;">
                                <p><strong>✗ Error:</strong> ${{error.message}}</p>
                            </div>
                        `;
                    }}
                }});
            </script>
        </body>
        </html>

        """

        template = template.replace("{{", "{").replace("}}", "}")
        content = (
            template
            .replace("__DEV_MODE_WARNING__", dev_mode_warning)
            .replace("__DISABLED_ATTR__", disabled_attr)
            .replace("__CREATE_BUTTON_LABEL__", create_button_label)
            .replace("__KEYS_COUNT__", str(len(api_keys)))
            .replace("__KEYS_HTML__", keys_html)
        )
        return HTMLResponse(content=content)

    # ==================== REQUEST HISTORY ENDPOINTS ====================
    
    @app.get("/api/history")
    async def get_history(
        limit: int = 100,
        offset: int = 0,
        agent_id: Optional[str] = None,
        status_code: Optional[int] = None,
        search: Optional[str] = None
    ):
        """Get request history with optional filters."""
        # Convert deque to list for filtering
        history_list = list(request_history)
        
        # Apply filters
        if agent_id:
            history_list = [h for h in history_list if h.get("agent_id") == agent_id]
        
        if status_code:
            history_list = [h for h in history_list if h.get("status_code") == status_code]
        
        if search:
            search_lower = search.lower()
            history_list = [
                h for h in history_list 
                if (h.get("agent_name") and search_lower in h.get("agent_name", "").lower()) or
                   (h.get("path") and search_lower in h.get("path", "").lower())
            ]
        
        # Sort by timestamp (most recent first)
        history_list.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Apply pagination
        total = len(history_list)
        history_list = history_list[offset:offset + limit]
        
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "history": history_list
        }
    
    @app.post("/api/history/clear")
    async def clear_history():
        """Clear all request history."""
        request_history.clear()
        return {"message": "Request history cleared", "cleared_count": 0}
    
    @app.get("/history")
    async def history_dashboard():
        """Request history dashboard with interactive table."""
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Request History - Teleon Dev Server</title>
            <style>
                body { font-family: -apple-system, sans-serif; background: #000; color: #e0e0e0; padding: 20px; margin: 0; }
                h1 { color: #00d4ff; }
                a { color: #00d4ff; text-decoration: none; }
                a:hover { text-decoration: underline; }
                
                .controls { margin: 20px 0; display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
                .controls input { background: #1a1a1a; color: #e0e0e0; border: 1px solid #333; padding: 8px; border-radius: 4px; }
                .controls button { background: #00d4ff; color: #000; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-weight: 600; }
                .controls button:hover { background: #00aacc; }
                .controls button.secondary { background: #333; color: #e0e0e0; }
                .controls button.danger { background: #ff6b6b; color: #fff; }
                
                table { width: 100%; border-collapse: collapse; background: #0a0a0a; margin: 20px 0; }
                th { background: #1a1a1a; padding: 12px; text-align: left; border-bottom: 2px solid #333; color: #00d4ff; }
                td { padding: 12px; border-bottom: 1px solid #222; }
                tr:hover { background: #111; cursor: pointer; }
                
                .status-200 { color: #00ff88; }
                .status-300 { color: #ffaa00; }
                .status-400 { color: #ff6b6b; }
                .status-500 { color: #ff0000; }
                
                .detail { display: none; background: #0f0f0f; padding: 15px; margin: 10px 0; border-left: 3px solid #00d4ff; }
                .detail pre { background: #1a1a1a; padding: 10px; border-radius: 4px; overflow-x: auto; }
                
                .badge { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: 600; }
                .badge-post { background: #1a2e1a; color: #00ff88; }
                .badge-get { background: #1a1a2e; color: #00aaff; }
                
                .empty { text-align: center; padding: 40px; color: #666; font-style: italic; }
            </style>
        </head>
        <body>
            <h1>📜 Request History</h1>
            <p><a href="/">← Back to Dashboard</a></p>
            
            <div class="controls">
                <input type="text" id="searchInput" placeholder="Search by agent or path...">
                <button onclick="filterHistory('all')">All</button>
                <button onclick="filterHistory('errors')" class="secondary">Errors Only</button>
                <button onclick="refreshHistory()">🔄 Refresh</button>
                <button onclick="clearHistory()" class="danger">Clear History</button>
                <button onclick="exportHistory()" class="secondary">Export JSON</button>
                <span id="countBadge" style="margin-left: auto; color: #666;"></span>
            </div>
            
            <div id="historyContainer">
                <div class="empty">Loading...</div>
            </div>
            
            <script>
                let currentFilter = 'all';
                let currentSearch = '';
                let historyData = [];
                
                async function fetchHistory() {
                    const params = new URLSearchParams();
                    if (currentFilter === 'errors') {
                        // We'll filter client-side for now
                    }
                    if (currentSearch) {
                        params.set('search', currentSearch);
                    }
                    
                    const response = await fetch('/api/history?' + params);
                    const data = await response.json();
                    historyData = data.history;
                    renderHistory();
                }
                
                function filterHistory(filter) {
                    currentFilter = filter;
                    renderHistory();
                }
                
                function renderHistory() {
                    let filtered = historyData;
                    
                    if (currentFilter === 'errors') {
                        filtered = filtered.filter(h => h.status_code >= 400);
                    }
                    
                    const container = document.getElementById('historyContainer');
                    document.getElementById('countBadge').textContent = `${filtered.length} requests`;
                    
                    if (filtered.length === 0) {
                        container.innerHTML = '<div class="empty">No requests found. Try making some API calls!</div>';
                        return;
                    }
                    
                    let html = '<table><thead><tr>';
                    html += '<th>Time</th><th>Agent</th><th>Method</th><th>Status</th><th>Latency</th><th>Tokens</th><th>Cost</th>';
                    html += '</tr></thead><tbody>';
                    
                    filtered.forEach(item => {
                        const time = new Date(item.timestamp).toLocaleTimeString();
                        const statusClass = 'status-' + Math.floor(item.status_code / 100) + '00';
                        const method = item.method || 'POST';
                        const methodClass = method.toLowerCase() === 'post' ? 'badge-post' : 'badge-get';
                        
                        html += `<tr onclick="toggleDetail('${item.id}')">`;
                        html += `<td>${time}</td>`;
                        html += `<td>${item.agent_name || 'N/A'}</td>`;
                        html += `<td><span class="badge ${methodClass}">${method}</span></td>`;
                        html += `<td class="${statusClass}">${item.status_code}</td>`;
                        html += `<td>${item.latency_ms.toFixed(2)}ms</td>`;
                        html += `<td>${item.tokens_used || '-'}</td>`;
                        html += `<td>${item.cost ? '$' + item.cost.toFixed(4) : '-'}</td>`;
                        html += '</tr>';
                        
                        html += `<tr id="detail-${item.id}" class="detail-row"><td colspan="7">`;
                        html += `<div class="detail">`;
                        html += `<p><strong>Path:</strong> ${item.path}</p>`;
                        if (item.request_body) {
                            html += `<p><strong>Request Body:</strong></p>`;
                            html += `<pre>${JSON.stringify(item.request_body, null, 2)}</pre>`;
                        }
                        if (item.error) {
                            html += `<p><strong>Error:</strong> <span style="color: #ff6b6b;">${item.error}</span></p>`;
                        }
                        html += '</div></td></tr>';
                    });
                    
                    html += '</tbody></table>';
                    container.innerHTML = html;
                }
                
                function toggleDetail(id) {
                    const detail = document.getElementById('detail-' + id);
                    const detailDiv = detail.querySelector('.detail');
                    if (detailDiv.style.display === 'block') {
                        detailDiv.style.display = 'none';
                    } else {
                        detailDiv.style.display = 'block';
                    }
                }
                
                async function clearHistory() {
                    if (!confirm('Clear all request history?')) return;
                    await fetch('/api/history/clear', { method: 'POST' });
                    await fetchHistory();
                }
                
                async function refreshHistory() {
                    await fetchHistory();
                }
                
                function exportHistory() {
                    const dataStr = JSON.stringify(historyData, null, 2);
                    const dataBlob = new Blob([dataStr], { type: 'application/json' });
                    const url = URL.createObjectURL(dataBlob);
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = 'request-history-' + new Date().toISOString() + '.json';
                    link.click();
                }
                
                document.getElementById('searchInput').addEventListener('input', (e) => {
                    currentSearch = e.target.value;
                    setTimeout(() => fetchHistory(), 300);
                });
                
                // Initial load
                fetchHistory();
                
                // Auto-refresh every 5 seconds
                setInterval(refreshHistory, 5000);
            </script>
        </body>
        </html>
        """)

    # ==================== PERFORMANCE DASHBOARD ENDPOINTS ====================
    
    @app.get("/api/metrics")
    async def get_metrics():
        """Get performance metrics and statistics."""
        # Calculate average latency
        all_latencies = []
        for latencies in performance_metrics["latency_by_minute"].values():
            all_latencies.extend(latencies)
        
        avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
        
        # Get last 60 minutes of data
        now = datetime.now(timezone.utc)
        last_60_minutes = []
        for i in range(60, 0, -1):
            minute = (now - timedelta(minutes=i)).replace(second=0, microsecond=0)
            minute_key = minute.isoformat()
            last_60_minutes.append({
                "time": minute_key,
                "requests": performance_metrics["requests_by_minute"].get(minute_key, 0),
                "avg_latency": (sum(performance_metrics["latency_by_minute"].get(minute_key, [0])) / 
                               len(performance_metrics["latency_by_minute"].get(minute_key, [1])))
            })
        
        # Convert bandwidth to GB for readability
        bandwidth_gb = performance_metrics["total_bandwidth_bytes"] / (1024 * 1024 * 1024)
        
        return {
            "total_requests": performance_metrics["total_requests"],
            "avg_latency_ms": round(avg_latency, 2),
            "total_tokens": performance_metrics["total_tokens"],
            "total_cost": round(performance_metrics["total_cost"], 4),
            "total_bandwidth_bytes": performance_metrics["total_bandwidth_bytes"],
            "total_bandwidth_gb": round(bandwidth_gb, 4),
            "requests_per_minute": last_60_minutes,
            "requests_by_agent": dict(performance_metrics["requests_by_agent"]),
            "tokens_by_agent": dict(performance_metrics["tokens_by_agent"]),
            "cost_by_agent": {k: round(v, 4) for k, v in performance_metrics["cost_by_agent"].items()}
        }
    
    @app.get("/performance")
    async def performance_dashboard():
        """Performance metrics dashboard with charts."""
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Dashboard - Teleon Dev Server</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
            <style>
                body { font-family: -apple-system, sans-serif; background: #000; color: #e0e0e0; padding: 20px; margin: 0; }
                h1 { color: #00d4ff; }
                a { color: #00d4ff; text-decoration: none; }
                a:hover { text-decoration: underline; }
                
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
                .metric-card { background: #0a0a0a; border: 1px solid #333; padding: 20px; border-radius: 8px; }
                .metric-card h3 { margin: 0 0 10px 0; color: #00d4ff; font-size: 14px; text-transform: uppercase; }
                .metric-value { font-size: 32px; font-weight: 600; color: #00ff88; }
                .metric-label { font-size: 12px; color: #666; margin-top: 5px; }
                
                .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }
                .chart-container { background: #0a0a0a; border: 1px solid #333; padding: 20px; border-radius: 8px; }
                .chart-container h3 { margin: 0 0 15px 0; color: #00d4ff; }
                canvas { max-height: 300px; }
                
                .refresh-btn { background: #00d4ff; color: #000; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-weight: 600; margin: 10px 0; }
                .refresh-btn:hover { background: #00aacc; }
            </style>
        </head>
        <body>
            <h1>📊 Performance Dashboard</h1>
            <p><a href="/">← Back to Dashboard</a></p>
            
            <button class="refresh-btn" onclick="loadMetrics()">🔄 Refresh Now</button>
            <span style="color: #666; margin-left: 10px;">Auto-refresh every 5 seconds</span>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Total Requests</h3>
                    <div class="metric-value" id="totalRequests">0</div>
                    <div class="metric-label">All time</div>
                </div>
                <div class="metric-card">
                    <h3>Avg Latency</h3>
                    <div class="metric-value" id="avgLatency">0ms</div>
                    <div class="metric-label">Across all requests</div>
                </div>
                <div class="metric-card">
                    <h3>Total Tokens</h3>
                    <div class="metric-value" id="totalTokens">0</div>
                    <div class="metric-label">Cumulative usage</div>
                </div>
                <div class="metric-card">
                    <h3>Total Cost</h3>
                    <div class="metric-value" id="totalCost">$0.00</div>
                    <div class="metric-label">Estimated spend</div>
                </div>
            </div>
            
            <div class="charts-grid">
                <div class="chart-container">
                    <h3>Requests Per Minute (Last Hour)</h3>
                    <canvas id="requestsChart"></canvas>
                </div>
                <div class="chart-container">
                    <h3>Average Latency Over Time</h3>
                    <canvas id="latencyChart"></canvas>
                </div>
                <div class="chart-container">
                    <h3>Requests by Agent</h3>
                    <canvas id="agentRequestsChart"></canvas>
                </div>
                <div class="chart-container">
                    <h3>Token Usage by Agent</h3>
                    <canvas id="tokenUsageChart"></canvas>
                </div>
            </div>
            
            <script>
                let requestsChart, latencyChart, agentRequestsChart, tokenUsageChart;
                
                const chartOptions = {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: { display: false },
                    },
                    scales: {
                        y: { 
                            beginAtZero: true,
                            ticks: { color: '#666' },
                            grid: { color: '#1a1a1a' }
                        },
                        x: { 
                            ticks: { color: '#666' },
                            grid: { color: '#1a1a1a' }
                        }
                    }
                };
                
                async function loadMetrics() {
                    const response = await fetch('/api/metrics');
                    const data = await response.json();
                    
                    // Update metric cards
                    document.getElementById('totalRequests').textContent = data.total_requests;
                    document.getElementById('avgLatency').textContent = data.avg_latency_ms.toFixed(2) + 'ms';
                    document.getElementById('totalTokens').textContent = data.total_tokens.toLocaleString();
                    document.getElementById('totalCost').textContent = '$' + data.total_cost.toFixed(4);
                    
                    // Requests per minute chart
                    const times = data.requests_per_minute.map(d => new Date(d.time).toLocaleTimeString());
                    const requests = data.requests_per_minute.map(d => d.requests);
                    
                    if (requestsChart) requestsChart.destroy();
                    requestsChart = new Chart(document.getElementById('requestsChart'), {
                        type: 'line',
                        data: {
                            labels: times,
                            datasets: [{
                                label: 'Requests',
                                data: requests,
                                borderColor: '#00d4ff',
                                backgroundColor: 'rgba(0, 212, 255, 0.1)',
                                tension: 0.4,
                                fill: true
                            }]
                        },
                        options: chartOptions
                    });
                    
                    // Latency chart
                    const latencies = data.requests_per_minute.map(d => d.avg_latency);
                    
                    if (latencyChart) latencyChart.destroy();
                    latencyChart = new Chart(document.getElementById('latencyChart'), {
                        type: 'line',
                        data: {
                            labels: times,
                            datasets: [{
                                label: 'Latency (ms)',
                                data: latencies,
                                borderColor: '#00ff88',
                                backgroundColor: 'rgba(0, 255, 136, 0.1)',
                                tension: 0.4,
                                fill: true
                            }]
                        },
                        options: chartOptions
                    });
                    
                    // Requests by agent chart
                    const agents = Object.keys(data.requests_by_agent);
                    const agentRequests = Object.values(data.requests_by_agent);
                    
                    if (agentRequestsChart) agentRequestsChart.destroy();
                    agentRequestsChart = new Chart(document.getElementById('agentRequestsChart'), {
                        type: 'bar',
                        data: {
                            labels: agents,
                            datasets: [{
                                label: 'Requests',
                                data: agentRequests,
                                backgroundColor: '#00d4ff',
                            }]
                        },
                        options: chartOptions
                    });
                    
                    // Token usage by agent chart
                    const agentTokens = Object.values(data.tokens_by_agent);
                    
                    if (tokenUsageChart) tokenUsageChart.destroy();
                    tokenUsageChart = new Chart(document.getElementById('tokenUsageChart'), {
                        type: 'bar',
                        data: {
                            labels: agents,
                            datasets: [{
                                label: 'Tokens',
                                data: agentTokens,
                                backgroundColor: '#00ff88',
                            }]
                        },
                        options: chartOptions
                    });
                }
                
                // Initial load
                loadMetrics();
                
                // Auto-refresh every 5 seconds
                setInterval(loadMetrics, 5000);
            </script>
        </body>
        </html>
        """)

    # ==================== HOT RELOAD ENDPOINTS ====================
    
    def reload_agents():
        """Reload agents from project directory."""
        nonlocal discovered_agents
        try:
            new_agents = discover_agents(project_dir)
            
            # Track changes
            added = set(new_agents.keys()) - set(discovered_agents.keys())
            removed = set(discovered_agents.keys()) - set(new_agents.keys())
            
            # Update discovered agents
            discovered_agents.clear()
            discovered_agents.update(new_agents)
            
            # Update reload state
            reload_state["last_reload"] = datetime.now(timezone.utc)
            reload_state["reload_count"] += 1
            reload_state["changed_files"] = []
            
            return {
                "success": True,
                "total_agents": len(new_agents),
                "added": list(added),
                "removed": list(removed),
                "timestamp": reload_state["last_reload"].isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    @app.get("/api/reload")
    async def manual_reload():
        """Manually trigger agent reload."""
        return reload_agents()
    
    @app.get("/api/reload/status")
    async def reload_status():
        """Get hot reload status."""
        seconds_since_reload = (datetime.now(timezone.utc) - reload_state["last_reload"]).total_seconds()
        return {
            "last_reload": reload_state["last_reload"].isoformat(),
            "seconds_ago": round(seconds_since_reload),
            "reload_count": reload_state["reload_count"],
            "total_agents": len(discovered_agents),
            "watcher_active": reload_state["file_watcher_active"]
        }
    
    # Background task to check for file changes
    async def reload_checker():
        """Periodically check if reload is needed."""
        while True:
            await asyncio.sleep(0.5)  # Check every 500ms
            if reload_state.get("needs_reload", False):
                await asyncio.sleep(1)  # Wait 1 second for more changes
                if reload_state.get("needs_reload", False):
                    reload_state["needs_reload"] = False
                    reload_agents()
                    reload_state["changed_files"] = []
    
    # Start file watcher in background
    async def start_file_watcher():
        """Start watching for file changes."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            
            class AgentFileHandler(FileSystemEventHandler):
                def _trigger_reload(self, event):
                    """Common reload trigger logic."""
                    if event.src_path.endswith('.py') and not event.is_directory:
                        reload_state["changed_files"].append(event.src_path)
                        reload_state["needs_reload"] = True
                
                def on_modified(self, event):
                    self._trigger_reload(event)
                
                def on_created(self, event):
                    self._trigger_reload(event)
            
            event_handler = AgentFileHandler()
            observer = Observer()
            observer.schedule(event_handler, str(project_dir), recursive=True)
            observer.start()
            reload_state["file_watcher_active"] = True
            
            # Start background reload checker
            asyncio.create_task(reload_checker())
            
        except ImportError:
            # Watchdog not installed, skip file watching
            reload_state["file_watcher_active"] = False
        except Exception as e:
            reload_state["file_watcher_active"] = False
    
    # ==================== PLATFORM METRICS REPORTER ====================
    
    async def report_metrics_to_platform():
        """
        Background task that periodically reports metrics to Teleon platform.
        
        This runs every 60 seconds and batches all metrics, so it has
        ZERO impact on request latency. The agent never waits for this.
        """
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available - platform metrics reporting disabled")
            return
        
        platform_url = os.getenv("TELEON_PLATFORM_URL")
        deployment_id = os.getenv("TELEON_DEPLOYMENT_ID")
        platform_api_key = os.getenv("TELEON_API_KEY") or os.getenv("TELEON_PLATFORM_API_KEY")
        
        if not all([platform_url, deployment_id, platform_api_key]):
            logger.debug(
                "Platform metrics reporting disabled - missing env vars: "
                "TELEON_PLATFORM_URL, TELEON_DEPLOYMENT_ID, or TELEON_API_KEY"
            )
            return
        
        logger.info(f"Platform metrics reporter started for deployment {deployment_id}")
        
        report_interval = 60  # Report every 60 seconds
        last_reported = {
            "requests": 0,
            "bandwidth_bytes": 0,
        }
        
        while True:
            try:
                await asyncio.sleep(report_interval)
                
                # Get current totals
                current_requests = performance_metrics["total_requests"]
                current_bandwidth = performance_metrics["total_bandwidth_bytes"]
                
                # Calculate deltas since last report
                new_requests = current_requests - last_reported["requests"]
                new_bandwidth = current_bandwidth - last_reported["bandwidth_bytes"]
                
                if new_requests == 0:
                    # No activity, skip this report
                    continue
                
                # Calculate average latency from recent data
                all_latencies = []
                for latencies in performance_metrics["latency_by_minute"].values():
                    all_latencies.extend(latencies)
                avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
                
                # Calculate success/error counts (approximation: assume 95% success in dev)
                # In production, middleware would track actual success/error
                success_count = int(new_requests * 0.95)
                error_count = new_requests - success_count
                
                # Build payload
                payload = {
                    "deployment_id": deployment_id,
                    "metrics": {
                        "request_count": new_requests,
                        "success_count": success_count,
                        "error_count": error_count,
                        "avg_latency_ms": round(avg_latency, 2),
                        "bandwidth_bytes": new_bandwidth,
                        "period_start": (datetime.now(timezone.utc) - timedelta(seconds=report_interval)).isoformat(),
                        "period_end": datetime.now(timezone.utc).isoformat(),
                    }
                }
                
                # Send to platform (fire-and-forget style - don't wait long)
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.post(
                        f"{platform_url.rstrip('/')}/api/v1/agents/ingest",
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {platform_api_key}",
                            "Content-Type": "application/json",
                            "X-Deployment-ID": deployment_id,
                        }
                    )
                    
                    if response.status_code in (200, 201, 202):
                        logger.debug(f"Reported {new_requests} requests, {new_bandwidth} bytes to platform")
                        last_reported["requests"] = current_requests
                        last_reported["bandwidth_bytes"] = current_bandwidth
                    else:
                        logger.warning(
                            f"Platform metrics report failed: {response.status_code} - {response.text[:100]}"
                        )
                        
            except asyncio.CancelledError:
                logger.info("Platform metrics reporter shutting down")
                break
            except Exception as e:
                # Never let metrics reporting crash the agent
                logger.debug(f"Platform metrics report error (non-fatal): {e}")
                # Continue running - don't stop on errors
    
    # Start file watcher on startup
    @app.on_event("startup")
    async def startup_event():
        await start_file_watcher()
        # Start platform metrics reporter (non-blocking background task)
        asyncio.create_task(report_metrics_to_platform())
    
    # ==================== PLAYGROUND ENDPOINTS ====================
    
    @app.post("/api/playground/invoke")
    async def playground_invoke(request: PlaygroundRequest):
        """Invoke agent from playground (no API key required)."""
        agent_info = discovered_agents.get(request.agent_id)
        if not agent_info:
            raise HTTPException(status_code=404, detail=f"Agent ID '{request.agent_id}' not found")
        
        # Invoke agent (reuse existing logic)
        return await _invoke_agent(request.agent_id, agent_info, request.input, request.temperature, request.max_tokens)
    
    @app.get("/playground")
    async def playground_ui():
        """Interactive agent playground."""
        agents_options = ''.join([
            f'<option value="{agent_id}">{info["name"]} ({info["model"]})</option>'
            for agent_id, info in discovered_agents.items()
        ])
        
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Agent Playground - Teleon Dev Server</title>
            <style>
                body {{ font-family: -apple-system, sans-serif; background: #000; color: #e0e0e0; padding: 20px; margin: 0; }}
                h1 {{ color: #00d4ff; }}
                a {{ color: #00d4ff; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
                
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .controls {{ background: #0a0a0a; border: 1px solid #333; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .controls label {{ display: block; margin: 10px 0 5px 0; color: #00d4ff; font-weight: 600; }}
                .controls select, .controls input {{ background: #1a1a1a; color: #e0e0e0; border: 1px solid #333; padding: 10px; border-radius: 4px; width: 100%; }}
                .controls button {{ background: #00d4ff; color: #000; border: none; padding: 12px 24px; border-radius: 4px; cursor: pointer; font-weight: 600; margin-top: 10px; }}
                .controls button:hover {{ background: #00aacc; }}
                .controls button:disabled {{ background: #333; cursor: not-allowed; }}
                
                .chat-container {{ background: #0a0a0a; border: 1px solid #333; border-radius: 8px; padding: 20px; min-height: 400px; max-height: 600px; overflow-y: auto; margin: 20px 0; }}
                .message {{ margin: 15px 0; padding: 15px; border-radius: 8px; }}
                .message-user {{ background: #1a2e2e; border-left: 3px solid #00d4ff; }}
                .message-agent {{ background: #1a1a1a; border-left: 3px solid #00ff88; }}
                .message-header {{ font-size: 12px; color: #666; margin-bottom: 8px; }}
                .message-content {{ color: #e0e0e0; line-height: 1.6; }}
                .message-meta {{ font-size: 11px; color: #666; margin-top: 8px; }}
                
                .input-area {{ background: #0a0a0a; border: 1px solid #333; padding: 20px; border-radius: 8px; }}
                .input-area textarea {{ background: #1a1a1a; color: #e0e0e0; border: 1px solid #333; padding: 10px; border-radius: 4px; width: 100%; min-height: 100px; resize: vertical; font-family: inherit; }}
                
                .params-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
                
                .empty {{ text-align: center; padding: 40px; color: #666; font-style: italic; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎮 Agent Playground</h1>
                <p><a href="/">← Back to Dashboard</a></p>
                
                <div class="controls">
                    <label for="agentSelect">Select Agent</label>
                    <select id="agentSelect" onchange="saveSelection()">
                        <option value="">Choose an agent...</option>
                        __AGENTS_OPTIONS__
                    </select>
                    
                    <div class="params-grid" style="margin-top: 15px;">
                        <div>
                            <label for="temperature">Temperature: <span id="tempValue">0.7</span></label>
                            <input type="range" id="temperature" min="0" max="2" step="0.1" value="0.7" 
                                   oninput="document.getElementById('tempValue').textContent = this.value">
                        </div>
                        <div>
                            <label for="maxTokens">Max Tokens</label>
                            <input type="number" id="maxTokens" value="500" min="1" max="4000">
                        </div>
                    </div>
                </div>
                
                <div class="chat-container" id="chatContainer">
                    <div class="empty">Select an agent and start chatting!</div>
                </div>
                
                <div class="input-area">
                    <textarea id="inputText" placeholder="Type your message here..."></textarea>
                    <button onclick="sendMessage()" id="sendBtn" disabled>Send Message</button>
                    <button onclick="clearChat()" style="background: #333; margin-left: 10px;">Clear Chat</button>
                </div>
            </div>
            
            <script>
                let conversationHistory = [];
                
                // Load last selected agent from localStorage
                const savedAgent = localStorage.getItem('playground_agent');
                if (savedAgent) {{
                    document.getElementById('agentSelect').value = savedAgent;
                    document.getElementById('sendBtn').disabled = false;
                }}
                
                function saveSelection() {{
                    const agent = document.getElementById('agentSelect').value;
                    if (agent) {{
                        localStorage.setItem('playground_agent', agent);
                        document.getElementById('sendBtn').disabled = false;
                    }} else {{
                        document.getElementById('sendBtn').disabled = true;
                    }}
                }}
                
                async function sendMessage() {{
                    const agentId = document.getElementById('agentSelect').value;
                    const input = document.getElementById('inputText').value.trim();
                    const temperature = parseFloat(document.getElementById('temperature').value);
                    const maxTokens = parseInt(document.getElementById('maxTokens').value);
                    
                    if (!agentId || !input) return;
                    
                    // Add user message to chat
                    addMessage('user', input);
                    document.getElementById('inputText').value = '';
                    document.getElementById('sendBtn').disabled = true;
                    document.getElementById('sendBtn').textContent = 'Sending...';
                    
                    try {{
                        const response = await fetch('/api/playground/invoke', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify({{ agent_id: agentId, input, temperature, max_tokens: maxTokens }})
                        }});
                        
                        const data = await response.json();
                        
                        if (response.ok) {{
                            addMessage('agent', data.output, {{
                                latency: data.latency_ms,
                                tokens: data.tokens_used,
                                cost: data.cost
                            }});
                        }} else {{
                            addMessage('error', data.detail || 'An error occurred');
                        }}
                    }} catch (error) {{
                        addMessage('error', error.message);
                    }} finally {{
                        document.getElementById('sendBtn').disabled = false;
                        document.getElementById('sendBtn').textContent = 'Send Message';
                    }}
                }}
                
                function addMessage(type, content, meta = {{}}) {{
                    const container = document.getElementById('chatContainer');
                    if (container.querySelector('.empty')) {{
                        container.innerHTML = '';
                    }}
                    
                    const messageDiv = document.createElement('div');
                    messageDiv.className = type === 'user' ? 'message message-user' : 'message message-agent';
                    
                    let header = type === 'user' ? '👤 You' : '🤖 Agent';
                    let metaHtml = '';
                    if (meta.latency) {{
                        metaHtml = `<div class="message-meta">⏱️ ${{meta.latency.toFixed(2)}}ms`;
                        if (meta.tokens) metaHtml += ` | 🎫 ${{meta.tokens}} tokens`;
                        if (meta.cost) metaHtml += ` | 💰 $${{meta.cost.toFixed(4)}}`;
                        metaHtml += '</div>';
                    }}
                    
                    messageDiv.innerHTML = `
                        <div class="message-header">${{header}} • ${{new Date().toLocaleTimeString()}}</div>
                        <div class="message-content">${{content}}</div>
                        ${{metaHtml}}
                    `;
                    
                    container.appendChild(messageDiv);
                    container.scrollTop = container.scrollHeight;
                    
                    conversationHistory.push({{ type, content, meta, timestamp: new Date().toISOString() }});
                }}
                
                function clearChat() {{
                    if (!confirm('Clear conversation history?')) return;
                    document.getElementById('chatContainer').innerHTML = '<div class="empty">Conversation cleared. Start a new chat!</div>';
                    conversationHistory = [];
                }}
                
                // Allow Enter to send (Shift+Enter for new line)
                document.getElementById('inputText').addEventListener('keydown', (e) => {{
                    if (e.key === 'Enter' && !e.shiftKey) {{
                        e.preventDefault();
                        sendMessage();
                    }}
                }});
            </script>
        </body>
        </html>
        """

        template = template.replace("{{", "{").replace("}}", "}")
        content = template.replace("__AGENTS_OPTIONS__", agents_options)
        return HTMLResponse(content=content)

    return app
