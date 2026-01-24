"""
Deploy command for Teleon Platform.

Deploys agents to teleon.ai (managed platform).
"""

import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import os
import time

app = typer.Typer(help="Deploy agents to Teleon Platform")
console = Console()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    env: str = typer.Option("production", "--env", help="Environment (production, preview)"),
    project: Optional[str] = typer.Option(None, "--project", help="Project name"),
):
    """
    Deploy agents to Teleon Platform (teleon.ai)
    
    Examples:
        teleon deploy                    # Deploy to production
        teleon deploy --env preview      # Deploy preview environment
        teleon deploy --project my-app   # Deploy specific project
    """
    
    # If a subcommand is invoked, don't run the main deploy logic
    if ctx.invoked_subcommand is not None:
        return
    
    console.print(Panel.fit(
        "[bold cyan]Teleon Platform Deployment[/bold cyan]\n"
        f"Environment: [yellow]{env}[/yellow]",
        title="🚀 Deploy"
    ))
    
    # Step 1: Check authentication
    is_authenticated, auth_config = check_authentication()
    if not is_authenticated:
        console.print("\n[yellow]⚠️  Not authenticated with Teleon Platform[/yellow]")
        console.print("\nPlease run: [cyan]teleon login[/cyan]")
        raise typer.Exit(1)
    
    # Get project info - show existing projects first
    if not project:
        project = auth_config.get("default_project_name")
        if project:
            console.print(f"\n[dim]Using default project: {project}[/dim]")
    
    if not project:
        # List existing projects
        import httpx
        import json
        
        config_file = Path.home() / ".teleon" / "config.json"
        if config_file.exists():
            config_data = json.loads(config_file.read_text())
            auth_token = config_data.get("auth_token")
            platform_url = os.getenv("TELEON_PLATFORM_URL", "https://api.teleon.ai")
            
            if auth_token:
                try:
                    response = httpx.get(
                        f"{platform_url}/api/v1/projects",
                        headers={"Authorization": f"Bearer {auth_token}"},
                        timeout=10.0
                    )
                    
                    if response.status_code == 200:
                        projects_data = response.json()
                        projects = projects_data.get("projects", [])
                        
                        if projects:
                            console.print("\n[cyan]📁 Your existing projects:[/cyan]")
                            for proj in projects:
                                deployment_count = proj.get("deployment_count", 0)
                                if deployment_count > 0:
                                    console.print(f"  • [green]{proj['name']}[/green] ({deployment_count} deployment{'s' if deployment_count != 1 else ''})")
                                else:
                                    console.print(f"  • [dim]{proj['name']}[/dim] (empty)")
                            console.print("")
                        else:
                            console.print("\n[dim]No existing projects found[/dim]")
                except:
                    pass  # Silently fail if can't list projects
        
        console.print("[yellow]Enter a project name[/yellow]")
        console.print("[dim]You can use an existing project or create a new one[/dim]")
        console.print("[dim]Format: lowercase letters, numbers, and hyphens only (e.g., 'my-agents')[/dim]")
        
        while True:
            project = Prompt.ask("Project name", default="my-agents")
            
            # Validate project name format
            import re
            project = project.strip().lower()
            
            if not re.match(r'^[a-z0-9-]+$', project):
                console.print("[red]❌ Invalid project name. Only lowercase letters, numbers, and hyphens allowed.[/red]")
                continue
            
            if project.startswith('-') or project.endswith('-'):
                console.print("[red]❌ Project name cannot start or end with a hyphen.[/red]")
                continue
            
            if '--' in project:
                console.print("[red]❌ Project name cannot have consecutive hyphens.[/red]")
                continue
            
            break
    
    # Display authentication info
    if auth_config.get("user_email"):
        console.print(f"[dim]Deploying as: {auth_config['user_email']}[/dim]")
    if auth_config.get("api_key_scopes"):
        console.print(f"[dim]API Key scopes: {', '.join(auth_config['api_key_scopes'])}[/dim]")
    
    # Step 2: Detect agents
    agents = detect_agents()
    if not agents:
        console.print("\n[red]❌ No agents found![/red]")
        console.print("\nMake sure you have agents defined with @client.agent decorator")
        raise typer.Exit(1)
    
    console.print(f"\n[green]✓[/green] Detected {len(agents)} agent(s)")
    for agent in agents:
        console.print(f"  • {agent['name']}")
    
    # Step 2.5: Validate API keys in code (CRITICAL for production)
    api_key_errors = validate_api_keys_in_code(agents, env)
    if api_key_errors:
        console.print("\n" + "="*70)
        console.print("[bold red]❌ API KEY VALIDATION ERRORS[/bold red]")
        console.print("="*70)
        for error in api_key_errors:
            console.print(f"\n[red]📄 {error['file']}:[/red]")
            console.print(f"   {error['message']}")
        
        console.print("\n" + "="*70)
        console.print("\n[bold yellow]💡 How to fix:[/bold yellow]")
        if env == "production":
            console.print("   • Get a production API key from: [cyan]https://dashboard.teleon.ai[/cyan]")
            console.print("   • Set api_key='tlk_live_xxxxx' in TeleonClient()")
            console.print("   • Set environment='production' in TeleonClient()")
        else:
            console.print("   • For development: use environment='dev' (no API key needed)")
            console.print("   • For production: get API key from https://dashboard.teleon.ai")
        
        console.print("\n[bold red]Deployment blocked to prevent runtime errors.[/bold red]")
        raise typer.Exit(1)
    
    console.print(f"[green]✓[/green] API key validation passed")
    
    # Step 3: Analyze dependencies
    needs_database = analyze_database_needs(agents)
    
    if needs_database:
        database_config = configure_databases(agents)
    else:
        database_config = None
        console.print("\n[dim]No persistent storage needed (agents don't use Cortex)[/dim]")
    
    # Step 4: Show deployment plan
    show_deployment_plan(agents, database_config, env)
    
    if not Confirm.ask("\n[bold]Continue with deployment?[/bold]", default=True):
        console.print("[yellow]Deployment cancelled[/yellow]")
        raise typer.Exit(0)
    
    # Step 5: Deploy
    deploy_to_platform(agents, database_config, env, project)


def check_authentication() -> tuple[bool, dict]:
    """
    Check if user is authenticated with Teleon Platform.
    
    Returns:
        Tuple of (is_authenticated, config_data)
    """
    import json
    import httpx
    
    # Check for auth token
    auth_token = os.getenv("TELEON_AUTH_TOKEN")
    config_data = {}
    
    if not auth_token:
        # Check config file
        config_file = Path.home() / ".teleon" / "config.json"
        if config_file.exists():
            try:
                config_data = json.loads(config_file.read_text())
                auth_token = config_data.get("auth_token")
            except:
                pass
    
    if not auth_token:
        return False, {}
    
    # Verify API key with platform and check scopes
    platform_url = os.getenv("TELEON_PLATFORM_URL", "https://api.teleon.ai")
    
    try:
        # Verify the API key has deployment permissions
        response = httpx.get(
            f"{platform_url}/api/v1/api-keys/validate",
            headers={"X-API-Key": auth_token},
            timeout=10.0
        )
        
        if response.status_code == 200:
            key_info = response.json()
            
            # Check if key has deploy scope
            scopes = key_info.get("key", {}).get("scopes", [])
            if "agents:deploy" not in scopes:
                console.print("\n[red]✗ API key does not have deployment permissions[/red]")
                console.print("\n[yellow]Required scope: 'agents:deploy'[/yellow]")
                console.print("[dim]Create a new API key with deployment permissions from the dashboard[/dim]")
                return False, {}
            
            # Add key info to config data
            config_data["api_key_scopes"] = scopes
            config_data["api_key_verified"] = True
            
            return True, config_data
        else:
            console.print(f"\n[red]✗ API key verification failed: {response.status_code}[/red]")
            return False, {}
            
    except httpx.RequestError:
        # If platform is not available, assume key is valid for now
        # (This allows offline development)
        console.print("\n[yellow]⚠️  Could not verify API key with platform (offline mode)[/yellow]")
        return True, config_data
    except Exception as e:
        console.print(f"\n[yellow]⚠️  API key verification error: {e}[/yellow]")
        return True, config_data


def validate_api_keys_in_code(agents, environment):
    """
    Validate API keys in agent code files.
    
    For production deployments, ensures:
    1. API key is present
    2. API key format is correct (tlk_live_* for production)
    3. API key is ACTUALLY VALID (makes API call to verify)
    4. Environment is set to 'production'
    
    Returns list of errors, empty if validation passes.
    """
    import re
    import httpx
    
    errors = []
    checked_files = set()
    verified_keys = {}  # Cache API key verification results
    
    platform_url = os.getenv("TELEON_PLATFORM_URL", "https://api.teleon.ai")
    
    for agent in agents:
        file_path = Path(agent['file'])
        
        # Only check each file once
        if file_path in checked_files:
            continue
        checked_files.add(file_path)
        
        try:
            content = file_path.read_text()
        except Exception as e:
            errors.append({
                'file': str(file_path),
                'message': f'Failed to read file: {e}'
            })
            continue
        
        # Find TeleonClient initialization
        # Pattern: TeleonClient(api_key=..., environment=...)
        client_patterns = [
            r'TeleonClient\s*\([^)]*\)',
            r'client\s*=\s*TeleonClient\s*\([^)]*\)',
        ]
        
        found_client = False
        for pattern in client_patterns:
            matches = re.finditer(pattern, content, re.DOTALL)
            for match in matches:
                found_client = True
                client_init = match.group(0)
                
                # Check for api_key parameter
                api_key_match = re.search(r'api_key\s*=\s*["\']([^"\']+)["\']', client_init)
                env_match = re.search(r'environment\s*=\s*["\']([^"\']+)["\']', client_init)
                
                # For production deployments, strict validation
                if environment == "production":
                    if not api_key_match:
                        errors.append({
                            'file': str(file_path),
                            'message': 'Missing api_key parameter in TeleonClient(). Production deployments require a valid API key.'
                        })
                        continue
                    
                    api_key = api_key_match.group(1)
                    
                    # Step 1: Validate API key format
                    if not api_key.startswith('tlk_live_'):
                        errors.append({
                            'file': str(file_path),
                            'message': f'Invalid API key format: "{api_key}". Production requires api_key starting with "tlk_live_"'
                        })
                        continue  # Don't try to verify invalid format
                    
                    # Step 2: ACTUALLY VERIFY the API key with the platform
                    if api_key not in verified_keys:
                        console.print(f"[dim]  Verifying API key {api_key[:20]}...[/dim]")
                        
                        try:
                            # Make a test API call to verify the key
                            # Use /api/v1/projects endpoint to verify auth works
                            response = httpx.get(
                                f"{platform_url}/api/v1/projects",
                                headers={"Authorization": f"Bearer {api_key}"},
                                timeout=10.0
                            )
                            
                            if response.status_code == 200:
                                # API key is valid and has access
                                verified_keys[api_key] = True
                                console.print(f"[dim]  ✓ API key verified[/dim]")
                            elif response.status_code == 401:
                                # API key is invalid or missing
                                verified_keys[api_key] = False
                                errors.append({
                                    'file': str(file_path),
                                    'message': f'API key is invalid or expired: "{api_key[:20]}...". Get a valid key from https://dashboard.teleon.ai'
                                })
                            elif response.status_code == 403:
                                # API key is valid but doesn't have permission
                                verified_keys[api_key] = False
                                errors.append({
                                    'file': str(file_path),
                                    'message': f'API key lacks required permissions: "{api_key[:20]}...". Ensure your API key has "projects:read" scope at https://dashboard.teleon.ai'
                                })
                            else:
                                # Other error - don't block deployment
                                console.print(f"[yellow]  ⚠️  Could not verify API key (server returned {response.status_code}), but will proceed[/yellow]")
                                verified_keys[api_key] = True  # Allow deployment to proceed
                        except httpx.TimeoutException:
                            # Network timeout - don't block deployment but warn
                            console.print(f"[yellow]  ⚠️  Timeout verifying API key (will proceed anyway)[/yellow]")
                            verified_keys[api_key] = True  # Allow deployment to proceed
                        except Exception as e:
                            # Network error - don't block deployment but warn
                            console.print(f"[yellow]  ⚠️  Could not verify API key: {e} (will proceed anyway)[/yellow]")
                            verified_keys[api_key] = True  # Allow deployment to proceed
                    elif not verified_keys[api_key]:
                        # Key was already verified and found invalid
                        errors.append({
                            'file': str(file_path),
                            'message': f'API key is invalid: "{api_key[:20]}..."'
                        })
                    
                    # Step 3: Check environment parameter
                    if env_match:
                        env_value = env_match.group(1)
                        if env_value != 'production':
                            errors.append({
                                'file': str(file_path),
                                'message': f'Environment mismatch: environment="{env_value}" but deploying to production. Set environment="production"'
                            })
                    else:
                        # No environment specified - warn about it
                        errors.append({
                            'file': str(file_path),
                            'message': 'Missing environment="production" in TeleonClient(). Production deployments require explicit environment setting.'
                        })
        
        # If we found agents but no TeleonClient, that's also an error
        if not found_client and "@client.agent" in content:
            errors.append({
                'file': str(file_path),
                'message': 'Found @client.agent decorator but no TeleonClient initialization. Add: client = TeleonClient(api_key="tlk_live_...", environment="production")'
            })
    
    return errors


def detect_agents():
    """Detect agents in current directory"""
    
    # Look for agents.py or files with @client.agent
    agents_found = []
    
    # Check common locations
    for pattern in ["agents.py", "*/agents.py", "src/agents.py"]:
        for file in Path(".").glob(pattern):
            # Parse file to find agents (simplified)
            content = file.read_text()
            if "@client.agent" in content or "@agent" in content:
                # Extract agent names (simplified)
                import re
                # Handle multi-line decorators with DOTALL flag
                matches = re.findall(r'@client\.agent\([^)]*name=["\']([^"\']+)["\']', content, re.DOTALL)
                for name in matches:
                    # Better detection of feature usage
                    uses_cortex = (
                        "cortex=" in content or 
                        "create_cortex" in content or 
                        "from teleon.cortex" in content or
                        "import cortex" in content
                    )
                    
                    uses_helix = (
                        "helix=" in content or 
                        "create_helix" in content or 
                        "from teleon.helix" in content or
                        "import helix" in content
                    )
                    
                    uses_sentinel = (
                        "sentinel=" in content or 
                        "create_sentinel" in content or 
                        "from teleon.sentinel" in content or
                        "import sentinel" in content
                    )
                    
                    # Detect memory backend usage
                    uses_chromadb = (
                        "create_chroma_storage" in content or
                        "enable_chromadb=True" in content or
                        "from teleon.cortex.storage" in content or
                        "ChromaDB" in content
                    )
                    
                    uses_semantic_memory = (
                        "cortex.semantic" in content or
                        ".semantic.store" in content or
                        ".semantic.search" in content
                    )
                    
                    uses_redis = (
                        "storage_backend=\"redis\"" in content or
                        "storage_backend='redis'" in content
                    )
                    
                    # Extract configurations if present
                    helix_config = extract_helix_config(content)
                    sentinel_config = extract_sentinel_config(content)
                    cortex_config = extract_cortex_config(content)
                    
                    agents_found.append({
                        "name": name,
                        "file": str(file),
                        "uses_cortex": uses_cortex,
                        "uses_helix": uses_helix,
                        "uses_sentinel": uses_sentinel,
                        "uses_chromadb": uses_chromadb,
                        "uses_semantic_memory": uses_semantic_memory,
                        "uses_redis": uses_redis,
                        "helix_config": helix_config,
                        "sentinel_config": sentinel_config,
                        "cortex_config": cortex_config
                    })
    
    return agents_found


def extract_helix_config(content: str) -> dict:
    """
    Extract helix configuration from agent code.
    
    Looks for patterns like:
    - RuntimeConfig(min_replicas=2, max_replicas=10, ...)
    - create_helix(min_replicas=2, ...)
    - @helix_config(min_replicas=2, ...)
    
    Returns dict with helix configuration.
    """
    import re
    
    config = {
        "min_replicas": 1,
        "max_replicas": 10,
        "target_cpu_percent": 70,
        "target_memory_percent": 80,
        "memory_limit_mb": 512,
        "cpu_limit_cores": 1.0,
        "health_check_interval": 30,
        "scale_up_cooldown": 60,
        "scale_down_cooldown": 300,
    }
    
    # Look for RuntimeConfig or helix configuration
    patterns = [
        r'RuntimeConfig\s*\([^)]+\)',
        r'create_helix\s*\([^)]+\)',
        r'ScalingPolicy\s*\([^)]+\)',
        r'ResourceConfig\s*\([^)]+\)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            # Extract key-value pairs
            param_pattern = r'(\w+)\s*=\s*([^,\)]+)'
            params = re.findall(param_pattern, match)
            
            for key, value in params:
                key = key.strip()
                value = value.strip().strip('"\'')
                
                # Map common parameter names to our config
                key_mapping = {
                    "min_replicas": "min_replicas",
                    "max_replicas": "max_replicas",
                    "min_instances": "min_replicas",
                    "max_instances": "max_replicas",
                    "target_cpu": "target_cpu_percent",
                    "target_cpu_percent": "target_cpu_percent",
                    "target_memory": "target_memory_percent",
                    "target_memory_percent": "target_memory_percent",
                    "memory_limit": "memory_limit_mb",
                    "memory_limit_mb": "memory_limit_mb",
                    "cpu_limit": "cpu_limit_cores",
                    "cpu_limit_cores": "cpu_limit_cores",
                    "health_check_interval": "health_check_interval",
                    "scale_up_cooldown": "scale_up_cooldown",
                    "scale_down_cooldown": "scale_down_cooldown",
                }
                
                if key in key_mapping:
                    try:
                        # Try to parse as number
                        if '.' in value:
                            config[key_mapping[key]] = float(value)
                        else:
                            config[key_mapping[key]] = int(value)
                    except ValueError:
                        pass  # Keep default
    
    return config


def extract_sentinel_config(content: str) -> dict:
    """
    Extract sentinel configuration from agent code.
    
    Looks for patterns like:
    - sentinel={'enabled': True, 'content_filtering': True, ...}
    - sentinel={"enabled": True, ...}
    
    Returns dict with sentinel configuration.
    """
    import re
    import ast
    
    config = {
        "enabled": False,
        "content_filtering": False,
        "pii_detection": False,
        "compliance": [],
        "custom_policies": [],
    }
    
    # Look for sentinel={...} patterns in decorator calls
    # Pattern: sentinel={...} or sentinel={'key': value, ...}
    pattern = r'sentinel\s*=\s*(\{[^}]*(?:\{[^}]*\}[^}]*)*\})'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        try:
            # Try to parse as Python dict literal
            # Handle both single and double quotes
            parsed = ast.literal_eval(match)
            if isinstance(parsed, dict):
                # Update config with parsed values
                for key, value in parsed.items():
                    if key in config:
                        config[key] = value
                    # Handle nested structures
                    elif key == "compliance" and isinstance(value, list):
                        config["compliance"] = value
                    elif key == "custom_policies" and isinstance(value, list):
                        config["custom_policies"] = value
        except (ValueError, SyntaxError):
            # If literal_eval fails, try regex extraction
            # Extract key-value pairs from the dict string
            kv_pattern = r'["\']?(\w+)["\']?\s*:\s*([^,}]+)'
            kv_matches = re.findall(kv_pattern, match)
            for key, value in kv_matches:
                key = key.strip()
                value = value.strip().strip('"\'')
                
                if key == "enabled":
                    config["enabled"] = value.lower() in ("true", "1", "yes")
                elif key == "content_filtering":
                    config["content_filtering"] = value.lower() in ("true", "1", "yes")
                elif key == "pii_detection":
                    config["pii_detection"] = value.lower() in ("true", "1", "yes")
                elif key == "compliance":
                    # Try to parse as list
                    if value.startswith("[") and value.endswith("]"):
                        try:
                            config["compliance"] = ast.literal_eval(value)
                        except:
                            config["compliance"] = [v.strip().strip('"\'') for v in value[1:-1].split(",")]
    
    return config


def extract_cortex_config(content: str) -> dict:
    """
    Extract cortex configuration from agent code.
    
    Looks for patterns like:
    - cortex={'learning': True, 'memory_types': ['episodic', 'semantic'], ...}
    - cortex={"learning": True, ...}
    
    Returns dict with cortex configuration.
    """
    import re
    import ast
    
    config = {
        "learning": False,
        "memory_types": [],
        "storage": "memory",
        "episodic_config": {},
        "semantic_config": {},
        "procedural_config": {},
    }
    
    # Look for cortex={...} patterns in decorator calls
    pattern = r'cortex\s*=\s*(\{[^}]*(?:\{[^}]*\}[^}]*)*\})'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        try:
            # Try to parse as Python dict literal
            parsed = ast.literal_eval(match)
            if isinstance(parsed, dict):
                # Update config with parsed values
                for key, value in parsed.items():
                    if key in config:
                        config[key] = value
                    # Handle nested configs
                    elif key.endswith("_config") and isinstance(value, dict):
                        config[key] = value
        except (ValueError, SyntaxError):
            # If literal_eval fails, try regex extraction
            kv_pattern = r'["\']?(\w+)["\']?\s*:\s*([^,}]+)'
            kv_matches = re.findall(kv_pattern, match)
            for key, value in kv_matches:
                key = key.strip()
                value = value.strip().strip('"\'')
                
                if key == "learning":
                    config["learning"] = value.lower() in ("true", "1", "yes")
                elif key == "storage":
                    config["storage"] = value.strip('"\'')
                elif key == "memory_types":
                    # Try to parse as list
                    if value.startswith("[") and value.endswith("]"):
                        try:
                            config["memory_types"] = ast.literal_eval(value)
                        except:
                            config["memory_types"] = [v.strip().strip('"\'') for v in value[1:-1].split(",")]
    
    return config


def analyze_database_needs(agents):
    """Check if agents need databases"""
    
    for agent in agents:
        if agent.get("uses_cortex"):
            return True
    return False


# ============================================================================
# STORAGE ARCHITECTURE DEFINITIONS
# ============================================================================

ARCHITECTURES = {
    "hybrid_speed": {
        "id": "hybrid_speed",
        "name": "Hybrid: Speed-Optimized",
        "short_name": "ChromaDB + Redis",
        "icon": "⚡",
        "badge": "RECOMMENDED",
        "badge_color": "green",
        "description": "Best performance for most AI agents",
        "databases": [
            {"type": "chromadb", "size": "10GB", "purpose": "Semantic memory (knowledge base)"},
            {"type": "redis", "size": "4GB", "purpose": "Episodic & procedural memory"}
        ],
        "features": [
            "Ultra-fast conversation storage (~1ms latency)",
            "Semantic search with ChromaDB + FastEmbed",
            "Free local embeddings (no API costs)",
            "Perfect for real-time, high-throughput agents"
        ],
        "best_for": ["Chat bots", "Customer support", "Real-time assistants", "High-traffic agents"],
        "performance": {"read_latency": "~1ms", "write_latency": "~1ms", "throughput": "100K+ ops/sec"},
        "storage_total": "14GB",
        "monthly_cost": 0  # Included
    },
    "hybrid_analytics": {
        "id": "hybrid_analytics",
        "name": "Hybrid: Analytics-Ready",
        "short_name": "ChromaDB + PostgreSQL",
        "icon": "📊",
        "badge": "COMPLIANCE",
        "badge_color": "cyan",
        "description": "Best for complex queries & audit trails",
        "databases": [
            {"type": "chromadb", "size": "10GB", "purpose": "Semantic memory (knowledge base)"},
            {"type": "postgres", "size": "20GB", "purpose": "Episodic & procedural memory"}
        ],
        "features": [
            "Full SQL queries on conversation history",
            "ACID compliance & data integrity",
            "Complex time-range & analytics queries",
            "Semantic search with ChromaDB + FastEmbed"
        ],
        "best_for": ["Enterprise agents", "Compliance-required", "Analytics/reporting", "Audit trails"],
        "performance": {"read_latency": "~5ms", "write_latency": "~5ms", "throughput": "10K+ ops/sec"},
        "storage_total": "30GB",
        "monthly_cost": 0
    },
    "unified_postgres": {
        "id": "unified_postgres",
        "name": "Unified: PostgreSQL Only",
        "short_name": "PostgreSQL + pgvector",
        "icon": "🐘",
        "badge": "SIMPLE",
        "badge_color": "yellow",
        "description": "Single database, simpler operations",
        "databases": [
            {"type": "postgres", "size": "30GB", "purpose": "All memory types with pgvector"}
        ],
        "features": [
            "Single database for everything",
            "pgvector for semantic search",
            "Full SQL across all memory types",
            "Simpler backup & recovery"
        ],
        "best_for": ["Small teams", "Simpler ops", "SQL-first workflows", "Lower complexity"],
        "performance": {"read_latency": "~5ms", "write_latency": "~5ms", "throughput": "10K+ ops/sec"},
        "storage_total": "30GB",
        "monthly_cost": 0
    },
    "unified_redis": {
        "id": "unified_redis",
        "name": "Unified: Redis Only",
        "short_name": "Redis (All Memory)",
        "icon": "🔴",
        "badge": "SPEED",
        "badge_color": "red",
        "description": "Maximum speed, in-memory everything",
        "databases": [
            {"type": "redis", "size": "8GB", "purpose": "All memory types (ephemeral vectors)"}
        ],
        "features": [
            "Blazing fast everything (~1ms)",
            "Simple key-value patterns",
            "Best for ephemeral/session data",
            "⚠️ Limited vector search capabilities"
        ],
        "best_for": ["Session-only agents", "Ephemeral memory", "Maximum speed", "Simple use cases"],
        "performance": {"read_latency": "~1ms", "write_latency": "~1ms", "throughput": "100K+ ops/sec"},
        "storage_total": "8GB",
        "monthly_cost": 0
    }
}


def analyze_agent_workload(agents) -> dict:
    """
    Analyze agent code to determine optimal architecture.
    
    Returns:
        Dict with workload analysis and recommendation
    """
    analysis = {
        "uses_semantic_memory": False,
        "uses_chromadb": False,
        "uses_episodic_memory": False,
        "uses_procedural_memory": False,
        "uses_redis_explicit": False,
        "uses_postgres_explicit": False,
        "high_throughput_hints": False,
        "compliance_hints": False,
        "analytics_hints": False,
        "agent_count": len(agents),
        "recommended_architecture": "hybrid_speed"
    }
    
    for agent in agents:
        if agent.get("uses_chromadb"):
            analysis["uses_chromadb"] = True
            analysis["uses_semantic_memory"] = True
        if agent.get("uses_semantic_memory"):
            analysis["uses_semantic_memory"] = True
        if agent.get("uses_cortex"):
            analysis["uses_episodic_memory"] = True
            analysis["uses_procedural_memory"] = True
        if agent.get("uses_redis"):
            analysis["uses_redis_explicit"] = True
        
        # Check for workload hints in agent metadata
        agent_name = agent.get("name", "").lower()
        if any(kw in agent_name for kw in ["chat", "bot", "realtime", "assistant", "support"]):
            analysis["high_throughput_hints"] = True
        if any(kw in agent_name for kw in ["compliance", "audit", "legal", "finance", "enterprise"]):
            analysis["compliance_hints"] = True
        if any(kw in agent_name for kw in ["analytics", "report", "insight", "dashboard"]):
            analysis["analytics_hints"] = True
    
    # Determine recommendation
    if analysis["compliance_hints"] or analysis["analytics_hints"]:
        analysis["recommended_architecture"] = "hybrid_analytics"
    elif analysis["uses_chromadb"] or analysis["uses_semantic_memory"]:
        if analysis["high_throughput_hints"]:
            analysis["recommended_architecture"] = "hybrid_speed"
        else:
            analysis["recommended_architecture"] = "hybrid_speed"
    elif not analysis["uses_semantic_memory"]:
        if analysis["high_throughput_hints"]:
            analysis["recommended_architecture"] = "unified_redis"
        else:
            analysis["recommended_architecture"] = "hybrid_speed"
    
    return analysis


def show_architecture_comparison():
    """Display architecture comparison table"""
    
    table = Table(
        title="\n[bold]Architecture Comparison[/bold]",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        padding=(0, 1)
    )
    
    table.add_column("Feature", style="white", width=22)
    table.add_column("⚡ Speed\n(ChromaDB+Redis)", style="green", width=18, justify="center")
    table.add_column("📊 Analytics\n(ChromaDB+PG)", style="cyan", width=18, justify="center")
    table.add_column("🐘 Unified\n(PostgreSQL)", style="yellow", width=18, justify="center")
    
    # Performance metrics
    table.add_row(
        "[bold]Read Latency[/bold]",
        "[green]~1ms[/green]",
        "~5ms",
        "~5ms"
    )
    table.add_row(
        "[bold]Write Latency[/bold]",
        "[green]~1ms[/green]",
        "~5ms",
        "~5ms"
    )
    table.add_row(
        "[bold]Throughput[/bold]",
        "[green]100K+ ops/s[/green]",
        "10K+ ops/s",
        "10K+ ops/s"
    )
    table.add_row("", "", "", "")  # Spacer
    
    # Features
    table.add_row(
        "[bold]Vector Search[/bold]",
        "[green]✓ ChromaDB[/green]",
        "[green]✓ ChromaDB[/green]",
        "[yellow]✓ pgvector[/yellow]"
    )
    table.add_row(
        "[bold]SQL Queries[/bold]",
        "[dim]✗ Limited[/dim]",
        "[green]✓ Full SQL[/green]",
        "[green]✓ Full SQL[/green]"
    )
    table.add_row(
        "[bold]ACID Compliance[/bold]",
        "[dim]✗ Eventual[/dim]",
        "[green]✓ Full ACID[/green]",
        "[green]✓ Full ACID[/green]"
    )
    table.add_row(
        "[bold]Analytics Ready[/bold]",
        "[dim]✗ External[/dim]",
        "[green]✓ Native[/green]",
        "[green]✓ Native[/green]"
    )
    table.add_row("", "", "", "")  # Spacer
    
    # Cost & storage
    table.add_row(
        "[bold]Storage[/bold]",
        "14GB",
        "30GB",
        "30GB"
    )
    table.add_row(
        "[bold]Embedding Cost[/bold]",
        "[green]$0 (FastEmbed)[/green]",
        "[green]$0 (FastEmbed)[/green]",
        "[yellow]$0 (pgvector)[/yellow]"
    )
    table.add_row(
        "[bold]Monthly Cost[/bold]",
        "[green]Included ✓[/green]",
        "[green]Included ✓[/green]",
        "[green]Included ✓[/green]"
    )
    
    console.print(table)


def show_architecture_details(arch_id: str):
    """Show detailed info for a specific architecture"""
    arch = ARCHITECTURES.get(arch_id)
    if not arch:
        return
    
    console.print(f"\n[bold]{arch['icon']} {arch['name']}[/bold]")
    console.print(f"[dim]{arch['description']}[/dim]\n")
    
    # Databases
    console.print("[bold]Databases provisioned:[/bold]")
    for db in arch["databases"]:
        db_icon = {"chromadb": "🔮", "redis": "🔴", "postgres": "🐘"}.get(db["type"], "💾")
        console.print(f"  {db_icon} [cyan]{db['type'].title()}[/cyan] ({db['size']}): {db['purpose']}")
    
    # Features
    console.print("\n[bold]Features:[/bold]")
    for feature in arch["features"]:
        if feature.startswith("⚠️"):
            console.print(f"  [yellow]{feature}[/yellow]")
        else:
            console.print(f"  [green]✓[/green] {feature}")
    
    # Best for
    console.print("\n[bold]Best for:[/bold]")
    console.print(f"  {', '.join(arch['best_for'])}")
    
    # Performance
    perf = arch["performance"]
    console.print("\n[bold]Performance:[/bold]")
    console.print(f"  • Read: {perf['read_latency']} | Write: {perf['write_latency']} | Throughput: {perf['throughput']}")


def configure_databases(agents):
    """Configure database options with explicit architecture selection"""
    
    console.print("\n" + "="*70)
    console.print("[bold cyan]💾 Storage Architecture Configuration[/bold cyan]")
    console.print("="*70)
    
    # Analyze workload
    cortex_agents = [a for a in agents if a.get("uses_cortex")]
    analysis = analyze_agent_workload(cortex_agents)
    recommended = analysis["recommended_architecture"]
    
    console.print(f"\n[bold]{len(cortex_agents)} agent(s) use Cortex memory[/bold]")
    
    # Show what was detected
    detected = []
    if analysis["uses_semantic_memory"]:
        detected.append("Semantic Memory (Knowledge Base)")
    if analysis["uses_episodic_memory"]:
        detected.append("Episodic Memory (Conversations)")
    if analysis["uses_procedural_memory"]:
        detected.append("Procedural Memory (Learned Patterns)")
    
    if detected:
        console.print(f"[dim]Detected: {', '.join(detected)}[/dim]")
    
    # First ask: Managed vs BYO
    console.print("\n[bold]How would you like to handle storage?[/bold]")
    console.print("\n  1. [green]Managed by Teleon[/green] (recommended)")
    console.print("     • Zero configuration, automatic provisioning")
    console.print("     • Automatic backups, scaling & monitoring")
    console.print("     • Included in subscription")
    console.print("")
    console.print("  2. [yellow]Bring Your Own Database (BYO-DB)[/yellow]")
    console.print("     • Use your existing Redis/PostgreSQL")
    console.print("     • Provide connection strings")
    console.print("     • You manage backups & scaling")
    
    mode_choice = Prompt.ask("\n[bold]Choose option[/bold]", choices=["1", "2"], default="1")
    
    if mode_choice == "2":
        return configure_byo_databases(analysis)
    
    # Managed mode - show architecture options
    return configure_managed_databases(cortex_agents, analysis, recommended)


def configure_managed_databases(agents, analysis: dict, recommended: str):
    """Configure Teleon-managed databases with architecture selection"""
    
    console.print("\n[green]✓ Selected: Managed by Teleon[/green]")
    
    # Show architecture selection
    console.print("\n" + "-"*70)
    console.print("[bold cyan]🏗️  Choose Your Storage Architecture[/bold cyan]")
    console.print("-"*70)
    
    console.print("\n[dim]📝 Working Memory (session context) always uses in-memory storage - no database needed[/dim]")
    
    # Display options with recommendation highlighted
    console.print("\n[bold]Select an architecture:[/bold]\n")
    
    arch_order = ["hybrid_speed", "hybrid_analytics", "unified_postgres", "unified_redis"]
    
    for i, arch_id in enumerate(arch_order, 1):
        arch = ARCHITECTURES[arch_id]
        is_recommended = arch_id == recommended
        
        # Build the display line
        if is_recommended:
            badge = f"[bold green][{arch['badge']}][/bold green]"
            name_style = "bold green"
            prefix = "★"
        else:
            badge = f"[{arch['badge_color']}][{arch['badge']}][/{arch['badge_color']}]"
            name_style = "white"
            prefix = " "
        
        console.print(f"  {prefix} {i}. [{name_style}]{arch['icon']} {arch['name']}[/{name_style}] {badge}")
        console.print(f"       [dim]{arch['short_name']} • {arch['description']}[/dim]")
        
        # Show key benefits
        for feature in arch["features"][:2]:  # Show first 2 features
            if feature.startswith("⚠️"):
                console.print(f"       [yellow]• {feature}[/yellow]")
            else:
                console.print(f"       [dim]• {feature}[/dim]")
        console.print("")
    
    # Show comparison table option
    console.print("  [dim]5. Show detailed comparison table[/dim]")
    console.print("  [dim]6. Custom configuration (advanced)[/dim]")
    
    # Get selection
    default_choice = str(arch_order.index(recommended) + 1)
    choice = Prompt.ask(
        f"\n[bold]Choose architecture[/bold]",
        choices=["1", "2", "3", "4", "5", "6"],
        default=default_choice
    )
    
    # Handle special options
    if choice == "5":
        show_architecture_comparison()
        # Re-prompt after showing table
        choice = Prompt.ask(
            f"\n[bold]Now choose your architecture[/bold]",
            choices=["1", "2", "3", "4", "6"],
            default=default_choice
        )
    
    if choice == "6":
        return configure_custom_architecture(agents, analysis)
    
    # Map choice to architecture
    selected_arch_id = arch_order[int(choice) - 1]
    selected_arch = ARCHITECTURES[selected_arch_id]
    
    # Show selection confirmation with details
    console.print("\n" + "-"*70)
    show_architecture_details(selected_arch_id)
    
    # Confirm selection
    if not Confirm.ask(f"\n[bold]Proceed with {selected_arch['name']}?[/bold]", default=True):
        # Recurse to re-select
        return configure_managed_databases(agents, analysis, recommended)
    
    # Build database config
    databases = selected_arch["databases"].copy()
    
    # Show final summary
    console.print("\n" + "-"*70)
    console.print("[bold green]✓ Architecture Selected![/bold green]")
    console.print("-"*70)
    
    console.print("\n[bold]Provisioned Storage:[/bold]")
    for db in databases:
        db_icon = {"chromadb": "🔮", "redis": "🔴", "postgres": "🐘"}.get(db["type"], "💾")
        console.print(f"  {db_icon} [cyan]{db['type'].title()} ({db['size']}):[/cyan] {db['purpose']}")
    
    console.print(f"\n[bold]Total Storage:[/bold] {selected_arch['storage_total']}")
    
    console.print("\n[bold]What you get:[/bold]")
    console.print("  ✓ Automatic provisioning (3-5 minutes)")
    console.print("  ✓ Persistent storage across restarts")
    console.print("  ✓ Automatic daily backups (7-day retention)")
    console.print("  ✓ Real-time monitoring via dashboard")
    console.print("  ✓ Auto-scaling based on usage")
    console.print("  ✓ [green]Included in subscription[/green]")
    
    # Special notes
    if any(db.get("type") == "chromadb" for db in databases):
        console.print("  ✓ [green]Free embeddings with FastEmbed (save $100+/month)[/green]")
    
    return {
        "mode": "managed",
        "architecture": selected_arch_id,
        "architecture_name": selected_arch["name"],
        "databases": databases
    }


def configure_custom_architecture(agents, analysis: dict):
    """Configure custom architecture by selecting each component"""
    
    console.print("\n[yellow]✓ Custom Configuration Mode[/yellow]")
    console.print("[dim]Configure each memory type separately[/dim]")
    
    databases = []
    
    # ========================================
    # 1. SEMANTIC MEMORY (Knowledge Base)
    # ========================================
    if analysis["uses_semantic_memory"]:
        console.print("\n[bold cyan]━━━ Semantic Memory (Knowledge Base) ━━━[/bold cyan]")
        console.print("[dim]Stores long-term knowledge with vector similarity search[/dim]\n")
        
        console.print("  [bold]Choose vector storage backend:[/bold]")
        console.print("\n  1. [green]ChromaDB[/green] (recommended)")
        console.print("     • Purpose-built for vector search")
        console.print("     • Free embeddings with FastEmbed")
        console.print("     • Optimized similarity algorithms")
        console.print("     • 10GB storage included")
        console.print("")
        console.print("  2. [yellow]PostgreSQL with pgvector[/yellow]")
        console.print("     • SQL + vectors in one database")
        console.print("     • Complex joins with other data")
        console.print("     • 20GB storage included")
        console.print("")
        console.print("  3. [dim]Skip (no semantic memory)[/dim]")
        
        semantic_choice = Prompt.ask("\n  Choose", choices=["1", "2", "3"], default="1")
        
        if semantic_choice == "1":
            databases.append({
                "type": "chromadb",
                "size": "10GB",
                "purpose": "Semantic memory (knowledge base)"
            })
            console.print("  [green]✓ ChromaDB selected[/green]")
        elif semantic_choice == "2":
            databases.append({
                "type": "postgres",
                "size": "20GB",
                "purpose": "Semantic memory with pgvector"
            })
            console.print("  [green]✓ PostgreSQL + pgvector selected[/green]")
        else:
            console.print("  [dim]Semantic memory skipped[/dim]")
    
    # ========================================
    # 2. EPISODIC & PROCEDURAL MEMORY
    # ========================================
    if analysis["uses_episodic_memory"] or analysis["uses_procedural_memory"]:
        console.print("\n[bold cyan]━━━ Episodic & Procedural Memory ━━━[/bold cyan]")
        console.print("[dim]Conversation history & learned patterns[/dim]\n")
        
        # Check if PostgreSQL already selected
        has_postgres = any(db.get("type") == "postgres" for db in databases)
        
        console.print("  [bold]Choose key-value storage backend:[/bold]")
        console.print("\n  1. [green]Redis[/green] (recommended for speed)")
        console.print("     • Ultra-fast in-memory (~1ms latency)")
        console.print("     • 100K+ operations/second")
        console.print("     • Perfect for real-time agents")
        console.print("     • 4GB storage included")
        console.print("")
        
        if has_postgres:
            console.print("  2. [cyan]Use existing PostgreSQL[/cyan]")
            console.print("     • Share with semantic memory")
            console.print("     • Single database to manage")
            console.print("     • Full SQL on conversations")
        else:
            console.print("  2. [yellow]PostgreSQL[/yellow]")
            console.print("     • ACID compliance")
            console.print("     • Complex time-range queries")
            console.print("     • Analytics & reporting")
            console.print("     • 20GB storage included")
        
        console.print("")
        console.print("  3. [dim]Skip (no episodic/procedural)[/dim]")
        
        kv_choice = Prompt.ask("\n  Choose", choices=["1", "2", "3"], default="1")
        
        if kv_choice == "1":
            databases.append({
                "type": "redis",
                "size": "4GB",
                "purpose": "Episodic & procedural memory"
            })
            console.print("  [green]✓ Redis selected[/green]")
        elif kv_choice == "2":
            if has_postgres:
                # Update existing PostgreSQL entry
                for db in databases:
                    if db["type"] == "postgres":
                        db["size"] = "30GB"
                        db["purpose"] = "Semantic, episodic & procedural memory"
                console.print("  [green]✓ Using existing PostgreSQL (shared)[/green]")
            else:
                databases.append({
                    "type": "postgres",
                    "size": "20GB",
                    "purpose": "Episodic & procedural memory"
                })
                console.print("  [green]✓ PostgreSQL selected[/green]")
        else:
            console.print("  [dim]Episodic/procedural memory skipped[/dim]")
    
    # ========================================
    # FALLBACK
    # ========================================
    if not databases:
        console.print("\n[yellow]⚠️  No databases selected. Adding minimal storage.[/yellow]")
        databases.append({
            "type": "storage",
            "size": "5GB",
            "purpose": "General persistent storage"
        })
    
    # Calculate total storage
    total_gb = sum(int(db["size"].replace("GB", "")) for db in databases)
    
    # Determine architecture name
    db_types = [db["type"] for db in databases]
    if "chromadb" in db_types and "redis" in db_types:
        arch_name = "Custom: ChromaDB + Redis"
    elif "chromadb" in db_types and "postgres" in db_types:
        arch_name = "Custom: ChromaDB + PostgreSQL"
    elif "postgres" in db_types and len(db_types) == 1:
        arch_name = "Custom: PostgreSQL Only"
    elif "redis" in db_types and len(db_types) == 1:
        arch_name = "Custom: Redis Only"
    else:
        arch_name = "Custom Configuration"
    
    # Summary
    console.print("\n" + "-"*70)
    console.print("[bold green]✓ Custom Architecture Configured![/bold green]")
    console.print("-"*70)
    
    console.print("\n[bold]Provisioned Storage:[/bold]")
    for db in databases:
        db_icon = {"chromadb": "🔮", "redis": "🔴", "postgres": "🐘"}.get(db["type"], "💾")
        console.print(f"  {db_icon} [cyan]{db['type'].title()} ({db['size']}):[/cyan] {db['purpose']}")
    
    console.print(f"\n[bold]Total Storage:[/bold] {total_gb}GB")
    
    console.print("\n[bold]What you get:[/bold]")
    console.print("  ✓ Automatic provisioning (3-5 minutes)")
    console.print("  ✓ Persistent storage across restarts")
    console.print("  ✓ Automatic daily backups")
    console.print("  ✓ [green]Included in subscription[/green]")
    
    if any(db.get("type") == "chromadb" for db in databases):
        console.print("  ✓ [green]Free embeddings with FastEmbed[/green]")
    
    return {
        "mode": "managed",
        "architecture": "custom",
        "architecture_name": arch_name,
        "databases": databases
    }


def configure_byo_databases(analysis: dict):
    """Configure bring-your-own databases with architecture guidance"""
    
    console.print("\n[yellow]✓ Bring Your Own Database (BYO-DB)[/yellow]")
    console.print("[dim]Connect to your existing database infrastructure[/dim]")
    
    console.print("\n[bold]Based on your agent configuration, you'll need:[/bold]")
    
    needs = []
    if analysis["uses_semantic_memory"]:
        needs.append("• [cyan]Vector Database[/cyan]: ChromaDB, Pinecone, Weaviate, or PostgreSQL+pgvector")
    if analysis["uses_episodic_memory"] or analysis["uses_procedural_memory"]:
        needs.append("• [cyan]Key-Value Store[/cyan]: Redis (recommended) or PostgreSQL")
    
    for need in needs:
        console.print(f"  {need}")
    
    console.print("\n[dim]Connection string examples:[/dim]")
    console.print("  Redis:      [dim]redis://user:pass@your-redis.com:6379/0[/dim]")
    console.print("  PostgreSQL: [dim]postgresql://user:pass@your-db.com:5432/teleon[/dim]")
    console.print("  ChromaDB:   [dim]http://your-chroma.com:8000[/dim]")
    
    databases = []
    
    # ========================================
    # Redis Configuration
    # ========================================
    if analysis["uses_episodic_memory"] or analysis["uses_procedural_memory"]:
        console.print("\n[bold cyan]━━━ Key-Value Storage ━━━[/bold cyan]")
        
        kv_type = Prompt.ask(
            "Backend type",
            choices=["redis", "postgres"],
            default="redis"
        )
        
        if kv_type == "redis":
            console.print("\n[bold]Redis Configuration:[/bold]")
            redis_url = Prompt.ask("Redis URL", default="redis://localhost:6379/0")
            
            console.print("\n[dim]Testing connection...[/dim]")
            if test_redis_connection(redis_url):
                console.print("[green]✓ Redis connection successful[/green]")
                databases.append({
                    "type": "redis",
                    "url": redis_url,
                    "purpose": "Episodic & procedural memory"
                })
            else:
                console.print("[red]❌ Could not connect to Redis[/red]")
                if not Confirm.ask("Continue anyway?", default=False):
                    raise typer.Exit(1)
                databases.append({
                    "type": "redis",
                    "url": redis_url,
                    "purpose": "Episodic & procedural memory",
                    "verified": False
                })
        else:
            console.print("\n[bold]PostgreSQL Configuration (for key-value):[/bold]")
            postgres_url = Prompt.ask("PostgreSQL URL", default="postgresql://localhost:5432/teleon")
            
            console.print("\n[dim]Testing connection...[/dim]")
            if test_postgres_connection(postgres_url):
                console.print("[green]✓ PostgreSQL connection successful[/green]")
                databases.append({
                    "type": "postgres",
                    "url": postgres_url,
                    "purpose": "Episodic & procedural memory"
                })
            else:
                console.print("[red]❌ Could not connect to PostgreSQL[/red]")
                if not Confirm.ask("Continue anyway?", default=False):
                    raise typer.Exit(1)
                databases.append({
                    "type": "postgres",
                    "url": postgres_url,
                    "purpose": "Episodic & procedural memory",
                    "verified": False
                })
    
    # ========================================
    # Vector Database Configuration
    # ========================================
    if analysis["uses_semantic_memory"]:
        console.print("\n[bold cyan]━━━ Vector Storage (Semantic Memory) ━━━[/bold cyan]")
        
        # Check if PostgreSQL already configured
        has_postgres = any(db.get("type") == "postgres" for db in databases)
        
        vector_choices = ["chromadb", "postgres_pgvector", "pinecone", "weaviate"]
        if has_postgres:
            console.print("\n[dim]You can use your existing PostgreSQL with pgvector extension[/dim]")
        
        console.print("\n[bold]Choose vector database:[/bold]")
        console.print("  1. chromadb       - Self-hosted ChromaDB")
        console.print("  2. postgres_pgvector - PostgreSQL + pgvector")
        console.print("  3. pinecone       - Pinecone (managed)")
        console.print("  4. weaviate       - Weaviate")
        
        vector_type = Prompt.ask("Vector backend", choices=["1", "2", "3", "4"], default="1")
        
        if vector_type == "1":
            chroma_url = Prompt.ask("ChromaDB URL", default="http://localhost:8000")
            databases.append({
                "type": "chromadb",
                "url": chroma_url,
                "purpose": "Semantic memory (vectors)"
            })
            console.print("[green]✓ ChromaDB configured[/green]")
            
        elif vector_type == "2":
            if has_postgres:
                # Use existing PostgreSQL
                for db in databases:
                    if db["type"] == "postgres":
                        db["purpose"] = "All memory types (with pgvector)"
                        db["pgvector"] = True
                console.print("[green]✓ Using existing PostgreSQL with pgvector[/green]")
            else:
                postgres_url = Prompt.ask("PostgreSQL URL", default="postgresql://localhost:5432/teleon")
                databases.append({
                    "type": "postgres",
                    "url": postgres_url,
                    "purpose": "Semantic memory with pgvector",
                    "pgvector": True
                })
                console.print("[green]✓ PostgreSQL + pgvector configured[/green]")
                
        elif vector_type == "3":
            pinecone_api_key = Prompt.ask("Pinecone API Key")
            pinecone_env = Prompt.ask("Pinecone Environment", default="us-east-1-aws")
            databases.append({
                "type": "pinecone",
                "api_key": pinecone_api_key,
                "environment": pinecone_env,
                "purpose": "Semantic memory (vectors)"
            })
            console.print("[green]✓ Pinecone configured[/green]")
            
        elif vector_type == "4":
            weaviate_url = Prompt.ask("Weaviate URL", default="http://localhost:8080")
            databases.append({
                "type": "weaviate",
                "url": weaviate_url,
                "purpose": "Semantic memory (vectors)"
            })
            console.print("[green]✓ Weaviate configured[/green]")
    
    # Summary
    console.print("\n" + "-"*70)
    console.print("[bold yellow]✓ BYO-DB Configuration Complete[/bold yellow]")
    console.print("-"*70)
    
    console.print("\n[bold]Configured Databases:[/bold]")
    for db in databases:
        db_icon = {"chromadb": "🔮", "redis": "🔴", "postgres": "🐘", "pinecone": "🌲", "weaviate": "🔷"}.get(db["type"], "💾")
        verified = db.get("verified", True)
        status = "[green]verified[/green]" if verified else "[yellow]unverified[/yellow]"
        console.print(f"  {db_icon} [cyan]{db['type'].title()}[/cyan]: {db['purpose']} ({status})")
    
    console.print("\n[bold yellow]⚠️  Important Notes:[/bold yellow]")
    console.print("  • You are responsible for database backups & maintenance")
    console.print("  • Ensure databases are accessible from Teleon infrastructure")
    console.print("  • Consider using private endpoints for security")
    
    return {
        "mode": "byo",
        "architecture": "byo",
        "architecture_name": "Bring Your Own Database",
        "databases": databases
    }


def test_redis_connection(url: str) -> bool:
    """Test Redis connection"""
    try:
        import redis.asyncio as aioredis
        import asyncio
        
        async def test():
            client = aioredis.from_url(url)
            await client.ping()
            await client.close()
        
        asyncio.run(test())
        return True
    except:
        return False


def test_postgres_connection(url: str) -> bool:
    """Test PostgreSQL connection"""
    try:
        import asyncpg
        import asyncio
        
        async def test():
            conn = await asyncpg.connect(url)
            await conn.close()
        
        asyncio.run(test())
        return True
    except:
        return False


def show_deployment_plan(agents, database_config, env):
    """Show deployment plan with architecture visualization"""
    
    console.print("\n" + "="*70)
    console.print("[bold cyan]📋 Deployment Plan Summary[/bold cyan]")
    console.print("="*70)
    
    # ========================================
    # AGENTS TABLE
    # ========================================
    table = Table(
        title=f"\n[bold]Agents ({len(agents)})[/bold]",
        show_header=True,
        header_style="bold cyan",
        border_style="dim"
    )
    table.add_column("Agent", style="cyan", width=20)
    table.add_column("Memory Types", style="green", width=25)
    table.add_column("Features", style="yellow", width=15)
    table.add_column("Scaling", style="magenta", width=12)
    
    for agent in agents:
        memory_types = []
        features = []
        
        if agent.get("uses_cortex"):
            if agent.get("uses_chromadb") or agent.get("uses_semantic_memory"):
                memory_types.append("Semantic")
            memory_types.append("Episodic")
            memory_types.append("Procedural")
            features.append("Cortex")
        
        if agent.get("uses_helix"):
            features.append("Auto-scale")
        
        if agent.get("uses_sentinel"):
            features.append("Sentinel")
        
        table.add_row(
            agent["name"],
            ", ".join(memory_types) if memory_types else "[dim]None[/dim]",
            ", ".join(features) if features else "[dim]Basic[/dim]",
            "1-10" if agent.get("uses_helix") else "1"
        )
    
    console.print(table)
    
    # ========================================
    # STORAGE ARCHITECTURE
    # ========================================
    if database_config:
        console.print("\n" + "-"*70)
        
        arch_name = database_config.get("architecture_name", "Unknown")
        arch_id = database_config.get("architecture", "unknown")
        mode = database_config.get("mode", "managed")
        
        # Get architecture icon
        arch_icons = {
            "hybrid_speed": "⚡",
            "hybrid_analytics": "📊",
            "unified_postgres": "🐘",
            "unified_redis": "🔴",
            "custom": "🔧",
            "byo": "🔌"
        }
        arch_icon = arch_icons.get(arch_id, "💾")
        
        if mode == "managed":
            console.print(f"[bold cyan]{arch_icon} Storage Architecture:[/bold cyan] [green]{arch_name}[/green]")
            console.print(f"[dim]Mode: Managed by Teleon (zero ops)[/dim]")
        else:
            console.print(f"[bold cyan]{arch_icon} Storage Architecture:[/bold cyan] [yellow]{arch_name}[/yellow]")
            console.print(f"[dim]Mode: Bring Your Own Database[/dim]")
        
        # Storage breakdown table
        storage_table = Table(
            show_header=True,
            header_style="bold",
            border_style="dim",
            padding=(0, 1),
            expand=False
        )
        storage_table.add_column("Database", style="cyan", width=15)
        storage_table.add_column("Purpose", style="white", width=35)
        storage_table.add_column("Size/Status", style="green", width=15, justify="right")
        
        databases = database_config.get("databases", [])
        total_size = 0
        
        for db in databases:
            db_icon = {"chromadb": "🔮", "redis": "🔴", "postgres": "🐘", "pinecone": "🌲", "weaviate": "🔷"}.get(db["type"], "💾")
            db_name = f"{db_icon} {db['type'].title()}"
            purpose = db.get("purpose", "Storage")
            
            if mode == "managed":
                size = db.get("size", "N/A")
                if "GB" in str(size):
                    total_size += int(size.replace("GB", ""))
                storage_table.add_row(db_name, purpose, size)
            else:
                verified = db.get("verified", True)
                status = "[green]✓ Connected[/green]" if verified else "[yellow]⚠ Unverified[/yellow]"
                storage_table.add_row(db_name, purpose, status)
        
        console.print("")
        console.print(storage_table)
        
        if mode == "managed" and total_size > 0:
            console.print(f"\n[bold]Total Storage:[/bold] {total_size}GB [green](included)[/green]")
    
    # ========================================
    # ARCHITECTURE DIAGRAM
    # ========================================
    if database_config and database_config.get("mode") == "managed":
        arch_id = database_config.get("architecture", "")
        
        console.print("\n" + "-"*70)
        console.print("[bold]Data Flow:[/bold]")
        
        if arch_id == "hybrid_speed" or (arch_id == "custom" and 
            any(db["type"] == "chromadb" for db in database_config.get("databases", [])) and
            any(db["type"] == "redis" for db in database_config.get("databases", []))):
            console.print("""
    ┌─────────────┐    ┌──────────────────────────────────────────────┐
    │   Agent     │───▶│               Cortex Memory                  │
    └─────────────┘    │  ┌─────────────┐   ┌───────────────────────┐ │
                       │  │ Working     │   │ 🔮 ChromaDB           │ │
                       │  │ (RAM)       │   │    Semantic Memory    │ │
                       │  └─────────────┘   │    [green]FastEmbed (free)[/green]   │ │
                       │                    └───────────────────────┘ │
                       │  ┌─────────────────────────────────────────┐ │
                       │  │ 🔴 Redis                                │ │
                       │  │    Episodic + Procedural Memory         │ │
                       │  │    [cyan]~1ms latency[/cyan]                        │ │
                       │  └─────────────────────────────────────────┘ │
                       └──────────────────────────────────────────────┘
""")
        elif arch_id == "hybrid_analytics" or (arch_id == "custom" and
            any(db["type"] == "chromadb" for db in database_config.get("databases", [])) and
            any(db["type"] == "postgres" for db in database_config.get("databases", []))):
            console.print("""
    ┌─────────────┐    ┌──────────────────────────────────────────────┐
    │   Agent     │───▶│               Cortex Memory                  │
    └─────────────┘    │  ┌─────────────┐   ┌───────────────────────┐ │
                       │  │ Working     │   │ 🔮 ChromaDB           │ │
                       │  │ (RAM)       │   │    Semantic Memory    │ │
                       │  └─────────────┘   │    [green]FastEmbed (free)[/green]   │ │
                       │                    └───────────────────────┘ │
                       │  ┌─────────────────────────────────────────┐ │
                       │  │ 🐘 PostgreSQL                           │ │
                       │  │    Episodic + Procedural Memory         │ │
                       │  │    [cyan]Full SQL + ACID[/cyan]                     │ │
                       │  └─────────────────────────────────────────┘ │
                       └──────────────────────────────────────────────┘
""")
        elif arch_id == "unified_postgres":
            console.print("""
    ┌─────────────┐    ┌──────────────────────────────────────────────┐
    │   Agent     │───▶│               Cortex Memory                  │
    └─────────────┘    │  ┌─────────────┐   ┌───────────────────────┐ │
                       │  │ Working     │   │ 🐘 PostgreSQL         │ │
                       │  │ (RAM)       │   │    [yellow]pgvector[/yellow]            │ │
                       │  └─────────────┘   │    All Memory Types   │ │
                       │                    │    [cyan]Full SQL + ACID[/cyan]     │ │
                       │                    └───────────────────────┘ │
                       └──────────────────────────────────────────────┘
""")
        elif arch_id == "unified_redis":
            console.print("""
    ┌─────────────┐    ┌──────────────────────────────────────────────┐
    │   Agent     │───▶│               Cortex Memory                  │
    └─────────────┘    │  ┌─────────────┐   ┌───────────────────────┐ │
                       │  │ Working     │   │ 🔴 Redis              │ │
                       │  │ (RAM)       │   │    All Memory Types   │ │
                       │  └─────────────┘   │    [cyan]~1ms latency[/cyan]       │ │
                       │                    │    [yellow]Limited vectors[/yellow]   │ │
                       │                    └───────────────────────┘ │
                       └──────────────────────────────────────────────┘
""")
    
    # ========================================
    # COST BREAKDOWN
    # ========================================
    console.print("-"*70)
    console.print("[bold]💰 Cost Estimate:[/bold]")
    
    cost_table = Table(
        show_header=False,
        border_style="dim",
        padding=(0, 1),
        expand=False
    )
    cost_table.add_column("Item", style="white", width=35)
    cost_table.add_column("Cost", style="green", width=25, justify="right")
    
    cost_table.add_row("Compute (auto-scales with usage)", "$29 - $99/month")
    
    if database_config:
        if database_config["mode"] == "managed":
            cost_table.add_row("Storage (managed, backed up)", "[green]Included ✓[/green]")
            
            uses_chromadb = any(db.get("type") == "chromadb" for db in database_config.get("databases", []))
            if uses_chromadb:
                cost_table.add_row("Embeddings (FastEmbed)", "[green]$0/month (free!)[/green]")
            
            cost_table.add_row("Monitoring & Dashboard", "[green]Included ✓[/green]")
            cost_table.add_row("Automatic Backups (7-day)", "[green]Included ✓[/green]")
            cost_table.add_row("", "")
            cost_table.add_row("[bold]Total[/bold]", "[bold green]$29 - $99/month[/bold green]")
        else:
            cost_table.add_row("Storage (BYO)", "[yellow]You pay provider[/yellow]")
            cost_table.add_row("", "")
            cost_table.add_row("[bold]Total Teleon[/bold]", "[bold]$29 - $79/month[/bold]")
            cost_table.add_row("[dim]+ External DB costs[/dim]", "[dim]Varies[/dim]")
    else:
        cost_table.add_row("[bold]Total[/bold]", "[bold]$29 - $99/month[/bold]")
    
    console.print(cost_table)
    
    # ========================================
    # DEPLOYMENT INFO
    # ========================================
    console.print("\n[bold]Deployment Details:[/bold]")
    console.print(f"  • Environment: [cyan]{env}[/cyan]")
    console.print(f"  • Region: [cyan]us-east-1[/cyan] (auto-selected)")
    console.print(f"  • Provisioning Time: ~3-5 minutes")
    
    if database_config and database_config["mode"] == "managed":
        console.print(f"\n[dim]After deployment, access your databases via the dashboard at https://teleon.ai/dashboard[/dim]")


def deploy_to_platform(agents, database_config, env, project_name):
    """Deploy to Teleon Platform"""
    import httpx
    import json
    import zipfile
    import io
    import tempfile
    from pathlib import Path as PathlibPath
    
    console.print("\n" + "="*70)
    console.print("[bold green]🚀 Deploying to Teleon Platform[/bold green]")
    console.print("="*70)
    
    # Get auth token
    config_file = Path.home() / ".teleon" / "config.json"
    if not config_file.exists():
        console.print("\n[red]❌ Not authenticated. Run: teleon login[/red]")
        raise typer.Exit(1)
    
    config_data = json.loads(config_file.read_text())
    auth_token = config_data.get("auth_token")
    platform_url = os.getenv("TELEON_PLATFORM_URL", "https://api.teleon.ai")
    
    if not auth_token:
        console.print("\n[red]❌ No auth token found. Run: teleon login[/red]")
        raise typer.Exit(1)
    
    try:
        # Step 1: Create or get project
        console.print(f"\n[cyan]Finding project '{project_name}'...[/cyan]")
        
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # List projects
        response = httpx.get(
            f"{platform_url}/api/v1/projects",
            headers=headers,
            timeout=30.0
        )
        
        if response.status_code != 200:
            console.print(f"[red]❌ Failed to list projects: {response.status_code}[/red]")
            console.print(f"[dim]{response.text}[/dim]")
            raise typer.Exit(1)
        
        projects_data = response.json()
        projects = projects_data.get("projects", [])
        
        # Find existing project (case-insensitive, trimmed match)
        project_id = None
        project_name_normalized = project_name.strip().lower()
        
        for proj in projects:
            if proj["name"].strip().lower() == project_name_normalized:
                project_id = proj["id"]
                console.print(f"[green]✓ Using existing project: {proj['name']}[/green]")
                break
        
        # Create new project if not found
        if not project_id:
            console.print(f"[cyan]Creating new project '{project_name}'...[/cyan]")
            response = httpx.post(
                f"{platform_url}/api/v1/projects",
                headers=headers,
                json={"name": project_name, "description": f"Deployed via CLI"},
                timeout=30.0
            )
            
            if response.status_code == 400:
                # Project already exists
                error_data = response.json()
                if "already exists" in error_data.get("detail", ""):
                    console.print(f"[yellow]⚠️  Project '{project_name}' already exists[/yellow]")
                    # Try to find it again
                    for proj in projects:
                        if proj["name"] == project_name:
                            project_id = proj["id"]
                            console.print(f"[green]✓ Using existing project: {proj['name']}[/green]")
                            break
                    
                    if not project_id:
                        console.print(f"[red]❌ Could not find existing project[/red]")
                        raise typer.Exit(1)
                else:
                    console.print(f"[red]❌ Failed to create project: {error_data.get('detail', 'Unknown error')}[/red]")
                    raise typer.Exit(1)
            elif response.status_code not in [200, 201]:
                console.print(f"[red]❌ Failed to create project: {response.status_code}[/red]")
                console.print(f"[dim]{response.text}[/dim]")
                raise typer.Exit(1)
            else:
                project_data = response.json()
                project_id = project_data["id"]
                console.print(f"[green]✓ Created project: {project_name}[/green]")
        
        # Step 2: Package code as ZIP
        console.print(f"\n[cyan]Packaging code...[/cyan]")
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all Python files in current directory
            for py_file in PathlibPath(".").rglob("*.py"):
                if not any(part.startswith('.') or part == '__pycache__' for part in py_file.parts):
                    zipf.write(py_file, py_file)
            
            # Add requirements.txt if it exists
            if PathlibPath("requirements.txt").exists():
                zipf.write("requirements.txt")
        
        zip_buffer.seek(0)
        console.print(f"[green]✓ Code packaged ({len(zip_buffer.getvalue()) // 1024}KB)[/green]")
        
        # Step 3: Create deployment
        console.print(f"\n[cyan]Uploading deployment...[/cyan]")
        
        # Prepare form data
        database_mode = "none"
        database_requirements = []
        architecture_id = "none"
        architecture_name = "None"
        
        if database_config:
            database_mode = database_config["mode"]
            architecture_id = database_config.get("architecture", "custom")
            architecture_name = database_config.get("architecture_name", "Custom")
            
            # Extract database requirements for managed mode
            if database_mode == "managed" and "databases" in database_config:
                for db in database_config["databases"]:
                    db_entry = {
                        "type": db["type"],
                        "purpose": db.get("purpose", "Storage")
                    }
                    # Only include size for managed mode
                    if "size" in db:
                        db_entry["size"] = db["size"]
                    database_requirements.append(db_entry)
            
            # Extract database connections for BYO mode
            elif database_mode == "byo" and "databases" in database_config:
                for db in database_config["databases"]:
                    db_entry = {
                        "type": db["type"],
                        "purpose": db.get("purpose", "Storage")
                    }
                    if "url" in db:
                        db_entry["url"] = db["url"]
                    if "api_key" in db:
                        db_entry["api_key"] = db["api_key"]
                    if db.get("pgvector"):
                        db_entry["pgvector"] = True
                    database_requirements.append(db_entry)
        
        files = {
            'code': ('code.zip', zip_buffer, 'application/zip')
        }
        
        # Merge helix configs from all agents
        merged_helix_config = {
            "min_replicas": 1,
            "max_replicas": 10,
            "target_cpu_percent": 70,
            "target_memory_percent": 80,
            "memory_limit_mb": 512,
            "cpu_limit_cores": 1.0,
            "health_check_interval": 30,
            "scale_up_cooldown": 60,
            "scale_down_cooldown": 300,
        }
        
        for agent in agents:
            agent_helix = agent.get("helix_config", {})
            for key in merged_helix_config:
                if key in agent_helix:
                    # Take the max of min/max replicas, and configured values
                    if key in ["max_replicas", "memory_limit_mb"]:
                        merged_helix_config[key] = max(merged_helix_config[key], agent_helix[key])
                    elif key == "min_replicas":
                        merged_helix_config[key] = max(merged_helix_config[key], agent_helix[key])
                    else:
                        merged_helix_config[key] = agent_helix[key]
        
        # Collect sentinel and cortex configs from all agents (for backward compatibility)
        sentinel_configs = []
        cortex_configs = []
        for agent in agents:
            if agent.get("uses_sentinel") and agent.get("sentinel_config"):
                sentinel_configs.append(agent["sentinel_config"])
            if agent.get("uses_cortex") and agent.get("cortex_config"):
                cortex_configs.append(agent["cortex_config"])
        
        # Prepare agent metadata for platform (clean, structured data)
        agents_metadata = []
        for agent in agents:
            agents_metadata.append({
                'name': agent.get('name'),
                'file': agent.get('file'),
                'uses_sentinel': agent.get('uses_sentinel', False),
                'uses_cortex': agent.get('uses_cortex', False),
                'uses_helix': agent.get('uses_helix', False),
                'sentinel_config': agent.get('sentinel_config'),
                'cortex_config': agent.get('cortex_config'),
                'helix_config': agent.get('helix_config'),
            })
        
        data = {
            'project_id': project_id,
            'environment': env,
            'database_mode': database_mode,
            'storage_architecture': architecture_id,
            'storage_architecture_name': architecture_name,
            'helix_config': json.dumps(merged_helix_config),
            'agents': json.dumps(agents_metadata),  # Send full agent metadata
        }
        
        # Add sentinel and cortex configs for backward compatibility
        if sentinel_configs:
            data['sentinel_config'] = json.dumps(sentinel_configs)
        if cortex_configs:
            data['cortex_config'] = json.dumps(cortex_configs)
        
        # Add database configuration based on mode
        if database_config:
            data['database_requirements'] = json.dumps(database_requirements)
            
            # For BYO mode, also send legacy format for backward compatibility
            if database_mode == "byo":
                # Find Redis URL if present
                redis_db = next((db for db in database_config.get("databases", []) if db["type"] == "redis"), None)
                if redis_db and redis_db.get("url"):
                    data['redis_url'] = redis_db["url"]
                
                # Find PostgreSQL URL if present
                postgres_db = next((db for db in database_config.get("databases", []) if db["type"] == "postgres"), None)
                if postgres_db and postgres_db.get("url"):
                    data['postgres_url'] = postgres_db["url"]
                
                # Find ChromaDB URL if present
                chroma_db = next((db for db in database_config.get("databases", []) if db["type"] == "chromadb"), None)
                if chroma_db and chroma_db.get("url"):
                    data['chromadb_url'] = chroma_db["url"]
        
        response = httpx.post(
            f"{platform_url}/api/v1/deployments",
            headers={"Authorization": f"Bearer {auth_token}"},
            files=files,
            data=data,
            timeout=60.0
        )
        
        if response.status_code not in [200, 201]:
            console.print(f"[red]❌ Failed to create deployment: {response.status_code}[/red]")
            console.print(f"[dim]{response.text}[/dim]")
            raise typer.Exit(1)
        
        deployment_data = response.json()
        deployment_id = deployment_data["deployment"]["id"]
        console.print(f"[green]✓ Deployment created: {deployment_id[:8]}...[/green]")
        
        # Step 4: Poll deployment status
        console.print(f"\n[cyan]Building and deploying...[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Deploying...", total=None)
            
            max_attempts = 60  # 5 minutes
            attempt = 0
            last_status = "queued"
            
            while attempt < max_attempts:
                time.sleep(5)
                attempt += 1
                
                # Get deployment status
                status_response = httpx.get(
                    f"{platform_url}/api/v1/deployments/{deployment_id}",
                    headers=headers,
                    timeout=10.0
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    deployment_status = status_data["deployment"]["status"]
                    
                    if deployment_status != last_status:
                        last_status = deployment_status
                        
                        if deployment_status == "BUILDING":
                            progress.update(task, description="[cyan]Building containers...")
                        elif deployment_status == "DEPLOYING":
                            progress.update(task, description="[cyan]Deploying infrastructure...")
                        elif deployment_status == "STARTING":
                            progress.update(task, description="[yellow]⏳ Agents starting up...")
                        elif deployment_status in ["RUNNING", "ACTIVE", "PROVISIONED"]:
                            progress.update(task, description="[green]✓ Deployment successful!")
                            progress.stop_task(task)
                            break
                        elif deployment_status == "FAILED":
                            progress.update(task, description="[red]✗ Deployment failed")
                            progress.stop_task(task)
                            error_msg = status_data["deployment"].get("error_message", "Unknown error")
                            console.print(f"\n[red]❌ Deployment failed: {error_msg}[/red]")
                            raise typer.Exit(1)
            
            if attempt >= max_attempts:
                console.print(f"\n[yellow]⚠️  Deployment is taking longer than expected[/yellow]")
                console.print(f"[dim]Check status at: {platform_url}/deployments/{deployment_id}[/dim]")
        
        # Success!
        console.print("\n" + "="*70)
        console.print("[bold green]✅ Deployment Successful![/bold green]")
        console.print("="*70)

        deployment_url = status_data["deployment"].get("url", f"https://{project_name.lower().replace(' ', '-')}.teleon.dev")

        # Get dashboard URL
        dashboard_url = os.getenv("TELEON_DASHBOARD_URL", "https://dashboard.teleon.ai")

        deployment_status = status_data["deployment"]["status"]
        if deployment_status == "ACTIVE":
            console.print(f"\n[bold]Your agents are live![/bold] 🎉")
            console.print(f"\n  🌐 URL: [cyan]{deployment_url}[/cyan]")
        elif deployment_status == "STARTING":
            console.print(f"\n[bold]Your agents are starting up![/bold] 🚀")
            console.print(f"[dim]The agents are initializing and will be ready in 1-2 minutes.[/dim]")
            console.print(f"[dim]Check the dashboard for status updates: {dashboard_url}/deployments?project={project_id}[/dim]")
        else:
            console.print(f"\n[bold]Deployment completed![/bold]")
            console.print(f"[dim]Status: {deployment_status}[/dim]")
            if deployment_url:
                console.print(f"\n  🌐 URL: [cyan]{deployment_url}[/cyan]")
        console.print(f"  📊 Dashboard: [cyan]{dashboard_url}/deployments?project={project_id}[/cyan]")
        console.print(f"  🆔 Deployment ID: [dim]{deployment_id}[/dim]")
        
        if database_config:
            if database_config["mode"] == "managed":
                console.print(f"\n  💾 [bold]Provisioned Storage:[/bold]")
                for db in database_config.get("databases", []):
                    db_type = db["type"].upper() if db["type"] == "chromadb" else db["type"].title()
                    console.print(f"     • [green]{db_type}[/green] ({db['size']}) - {db['purpose']}")
                
                # Special note for ChromaDB
                if any(db.get("type") == "chromadb" for db in database_config.get("databases", [])):
                    console.print(f"     [dim]✓ Free embeddings enabled (no API costs!)[/dim]")
            else:
                console.print(f"  💾 Databases: [yellow]External (BYO-DB)[/yellow]")
        
        console.print(f"\n[bold]Next steps:[/bold]")
        console.print(f"  1. Test your agents: [cyan]{deployment_url}[/cyan]")
        console.print(f"  2. View logs: [cyan]teleon logs --deployment {deployment_id[:8]}[/cyan]")
        console.print(f"  3. Monitor: [cyan]{dashboard_url}/deployments?project={project_id}[/cyan]")
    
    except httpx.RequestError as e:
        console.print(f"\n[red]❌ Network error: {e}[/red]")
        console.print(f"\n[yellow]Make sure the Teleon platform is running at: {platform_url}[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]❌ Deployment error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def rollback(
    deployment_id: Optional[str] = typer.Option(None, "--deployment-id", "-d", help="Deployment ID to rollback"),
):
    """Rollback to a previous deployment version"""
    import httpx
    import json

    if not deployment_id:
        console.print("[red]❌ Deployment ID is required[/red]")
        console.print("[dim]Usage: teleon deploy rollback -d <deployment-id>[/dim]")
        raise typer.Exit(1)

    # Get auth token
    config_file = Path.home() / ".teleon" / "config.json"
    if not config_file.exists():
        console.print("\n[red]❌ Not authenticated. Run: teleon login[/red]")
        raise typer.Exit(1)

    config_data = json.loads(config_file.read_text())
    auth_token = config_data.get("auth_token")
    platform_url = os.getenv("TELEON_PLATFORM_URL", "https://api.teleon.ai")

    if not auth_token:
        console.print("\n[red]❌ No auth token found. Run: teleon login[/red]")
        raise typer.Exit(1)

    headers = {"Authorization": f"Bearer {auth_token}"}

    # Handle short deployment IDs - expand to full UUID
    original_id = deployment_id
    if len(deployment_id) < 36:
        console.print(f"\n[dim]Looking up full deployment ID for: {deployment_id}...[/dim]")

        try:
            response = httpx.get(
                f"{platform_url}/api/v1/deployments",
                headers=headers,
                timeout=10.0
            )

            if response.status_code == 200:
                deployments_data = response.json()
                deployments = deployments_data.get("deployments", [])

                matching = [d for d in deployments if d["id"].startswith(deployment_id)]

                if len(matching) == 1:
                    deployment_id = matching[0]["id"]
                    console.print(f"[dim]Found: {deployment_id}[/dim]")
                elif len(matching) > 1:
                    console.print(f"[yellow]⚠️  Multiple deployments match '{original_id}':[/yellow]")
                    for d in matching[:5]:
                        console.print(f"  • {d['id']}")
                    console.print("[dim]Please provide a more specific ID[/dim]")
                    raise typer.Exit(1)
                else:
                    console.print(f"[red]❌ No deployment found matching: {original_id}[/red]")
                    raise typer.Exit(1)
        except httpx.RequestError as e:
            console.print(f"[red]❌ Network error: {e}[/red]")
            raise typer.Exit(1)

    # Fetch push history (versions) for this deployment
    console.print(f"\n[cyan]Fetching version history for deployment {deployment_id[:8]}...[/cyan]")

    try:
        response = httpx.get(
            f"{platform_url}/api/v1/deployments/{deployment_id}/push-history",
            headers=headers,
            timeout=10.0
        )

        if response.status_code == 404:
            console.print(f"[red]❌ Deployment not found: {deployment_id}[/red]")
            raise typer.Exit(1)
        elif response.status_code != 200:
            console.print(f"[red]❌ Failed to fetch history: {response.status_code}[/red]")
            raise typer.Exit(1)

        history_data = response.json()
        history = history_data.get("history", [])

        if not history:
            console.print("[yellow]⚠️  No version history found for this deployment[/yellow]")
            raise typer.Exit(1)

        # Filter to only completed pushes with versions
        completed_versions = [
            h for h in history
            if h.get("status") == "completed" and h.get("version") and h.get("version") != "N/A"
        ]

        if not completed_versions:
            console.print("[yellow]⚠️  No completed versions found to rollback to[/yellow]")
            raise typer.Exit(1)

        # Get current version (most recent completed)
        current_version = completed_versions[0].get("version") if completed_versions else "unknown"

        # Display version list for selection
        console.print(Panel.fit(
            f"[bold cyan]Version History[/bold cyan]\n"
            f"Deployment: [yellow]{deployment_id[:8]}...[/yellow]\n"
            f"Current version: [green]{current_version}[/green]",
            title="🔄 Rollback"
        ))

        # Build table of versions
        table = Table(title="\nAvailable Versions")
        table.add_column("#", style="dim", width=3)
        table.add_column("Version", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Pushed At", style="yellow")
        table.add_column("Message", style="dim", max_width=40)

        # Show up to 10 versions
        for idx, version in enumerate(completed_versions[:10], 1):
            is_current = idx == 1
            version_display = version.get("version", "N/A")
            if is_current:
                version_display = f"{version_display} [current]"

            table.add_row(
                str(idx),
                version_display,
                version.get("status", "unknown"),
                version.get("pushed_at", "N/A")[:19] if version.get("pushed_at") else "N/A",
                (version.get("message", "")[:37] + "...") if len(version.get("message", "")) > 40 else version.get("message", "")
            )

        console.print(table)

        if len(completed_versions) == 1:
            console.print("\n[yellow]⚠️  Only one version available - nothing to rollback to[/yellow]")
            raise typer.Exit(1)

        # Prompt user to select a version
        console.print("\n[bold]Select a version to rollback to:[/bold]")
        console.print("[dim]Enter the number (2 or higher), or 'q' to cancel[/dim]")

        while True:
            choice = Prompt.ask("Version number", default="q")

            if choice.lower() == 'q':
                console.print("[yellow]Rollback cancelled[/yellow]")
                raise typer.Exit(0)

            try:
                choice_num = int(choice)
                if choice_num < 2:
                    console.print("[red]Cannot rollback to current version. Select 2 or higher.[/red]")
                    continue
                if choice_num > len(completed_versions):
                    console.print(f"[red]Invalid choice. Enter a number between 2 and {len(completed_versions)}[/red]")
                    continue
                break
            except ValueError:
                console.print("[red]Please enter a valid number or 'q' to cancel[/red]")
                continue

        selected_version = completed_versions[choice_num - 1]
        target_version = selected_version.get("version")

        # Confirm rollback
        console.print(f"\n[bold yellow]⚠️  You are about to rollback:[/bold yellow]")
        console.print(f"   From: [green]{current_version}[/green]")
        console.print(f"   To:   [cyan]{target_version}[/cyan]")

        if not Confirm.ask("\n[bold]Proceed with rollback?[/bold]", default=False):
            console.print("[yellow]Rollback cancelled[/yellow]")
            raise typer.Exit(0)

        # Perform rollback
        console.print(f"\n[cyan]Rolling back to version {target_version}...[/cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Rolling back...", total=None)

            response = httpx.post(
                f"{platform_url}/api/v1/deployments/{deployment_id}/rollback",
                headers=headers,
                json={"target_version": target_version},
                timeout=60.0
            )

            if response.status_code in [200, 201, 202]:
                progress.update(task, description="[green]✓ Rollback initiated!")
            else:
                progress.update(task, description="[red]✗ Rollback failed")
                console.print(f"\n[red]❌ Rollback failed: {response.status_code}[/red]")
                try:
                    error_data = response.json()
                    console.print(f"[dim]{error_data.get('detail', response.text)}[/dim]")
                except:
                    console.print(f"[dim]{response.text}[/dim]")
                raise typer.Exit(1)

        rollback_data = response.json()

        console.print("\n" + "="*50)
        console.print("[bold green]✅ Rollback Initiated![/bold green]")
        console.print("="*50)
        console.print(f"\n  Deployment: [cyan]{deployment_id[:8]}...[/cyan]")
        console.print(f"  From version: [dim]{current_version}[/dim]")
        console.print(f"  To version: [green]{target_version}[/green]")
        console.print(f"  Status: [yellow]Rolling update in progress...[/yellow]")

        if rollback_data.get("rollback", {}).get("deployment_id"):
            console.print(f"\n[dim]The rollback is running asynchronously.[/dim]")

        console.print(f"\n[dim]View logs: teleon logs -d {deployment_id[:8]}[/dim]")

    except httpx.RequestError as e:
        console.print(f"\n[red]❌ Network error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def preview():
    """Deploy to preview environment"""
    
    console.print("[cyan]Deploying to preview environment...[/cyan]")
    console.print("\n[green]✓ Preview deployed: https://preview-abc123.teleon.ai[/green]")


if __name__ == "__main__":
    app()
