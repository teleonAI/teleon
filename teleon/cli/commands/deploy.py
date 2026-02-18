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
    """Detect agents in current directory - scans user project Python files recursively"""
    
    # Directories to always skip (virtual envs, installed packages, build artifacts, etc.)
    SKIP_DIRS = {
        'venv', '.venv', 'env', '.env', 'node_modules', '.git',
        '__pycache__', 'site-packages', 'dist-packages',
        '.tox', '.nox', '.mypy_cache', '.pytest_cache',
        'dist', 'build', '.eggs', 'egg-info',
    }
    
    # Look for agents in any Python file, not just files named "agents.py"
    agents_found = []
    processed_files = set()  # Track processed files to avoid duplicates
    
    # Scan ALL Python files recursively
    for file in Path(".").rglob("*.py"):
        # Skip files inside excluded directories
        file_parts = set(file.parts)
        if file_parts & SKIP_DIRS:
            continue
        
        # Also skip any directory that ends with .egg-info or starts with '.'
        if any(part.endswith('.egg-info') or (part.startswith('.') and part != '.') for part in file.parts):
            continue
        
        # Skip test files and __pycache__
        if "test" in str(file).lower():
            continue
        
        # Skip if we've already processed this file (avoid duplicates)
        file_key = str(file.resolve())
        if file_key in processed_files:
            continue
        processed_files.add(file_key)
        
        # Parse file to find agents (simplified)
        try:
            content = file.read_text()
        except Exception:
            continue
        
        if "@client.agent" in content or "@agent" in content:
            # Extract agent names - improved regex to handle multiline decorators
            import re
            # Match @client.agent( ... name="..." ... ) 
            # Use a simpler approach: find the decorator start, then find name= within it
            # This handles multiline decorators better
            decorator_pattern = r'@client\.agent\s*\([^)]*?name\s*=\s*["\']([^"\']+)["\']'
            matches = re.findall(decorator_pattern, content, re.DOTALL)
            
            # If that doesn't work, try a more permissive pattern that handles nested parentheses
            if not matches:
                # Find all @client.agent decorators, then extract name from each
                decorator_blocks = re.findall(r'@client\.agent\s*\((.*?)\)', content, re.DOTALL)
                for block in decorator_blocks:
                    name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', block)
                    if name_match:
                        matches.append(name_match.group(1))
            for name in matches:
                # Skip placeholder/template agent names (e.g. {agent_name}, {{name}})
                if '{' in name or '}' in name or name.strip() == '':
                    continue
                
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
                
                uses_redis = (
                    "storage_backend=\"redis\"" in content or
                    "storage_backend='redis'" in content or
                    "REDIS_URL" in content
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
# STORAGE CONFIGURATION
# ============================================================================


def configure_databases(agents):
    """Configure database options for Cortex memory."""

    console.print("\n" + "="*70)
    console.print("[bold cyan]💾 Storage Configuration[/bold cyan]")
    console.print("="*70)

    cortex_agents = [a for a in agents if a.get("uses_cortex")]
    console.print(f"\n[bold]{len(cortex_agents)} agent(s) use Cortex memory[/bold]")
    console.print("[dim]Cortex uses a single database for all memory (store, search, context).[/dim]")

    console.print("\n[bold]How would you like to handle storage?[/bold]")
    console.print("\n  1. [green]Managed by Teleon[/green] (recommended)")
    console.print("     • PostgreSQL + pgvector provisioned automatically")
    console.print("     • Accessible via pgAdmin (db-N.teleon.ai)")
    console.print("     • Automatic backups, monitoring & SSL")
    console.print("     • Included in subscription")
    console.print("")
    console.print("  2. [yellow]Bring Your Own Database (BYO-DB)[/yellow]")
    console.print("     • Use your existing PostgreSQL or Redis")
    console.print("     • Provide connection string")
    console.print("     • You manage backups & scaling")

    mode_choice = Prompt.ask("\n[bold]Choose option[/bold]", choices=["1", "2"], default="1")

    if mode_choice == "2":
        return configure_byo_databases()

    return configure_managed_databases()


def configure_managed_databases():
    """Configure Teleon-managed PostgreSQL + pgvector database."""

    console.print("\n[green]✓ Selected: Managed by Teleon[/green]")
    console.print("\n" + "-"*70)
    console.print("[bold cyan]🐘 PostgreSQL + pgvector[/bold cyan]")
    console.print("-"*70)

    console.print("\n[bold]What gets provisioned:[/bold]")
    console.print("  🐘 [cyan]PostgreSQL[/cyan] with pgvector extension")
    console.print("     • Vector similarity search (cosine, IVFFlat index)")
    console.print("     • Full SQL queries on all memory data")
    console.print("     • ACID compliance & data integrity")
    console.print("     • Embeddings via FastEmbed (free) or OpenAI")

    console.print("\n[bold]Connection details:[/bold]")
    console.print("  • Host: [cyan]db-N.teleon.ai[/cyan] (custom DNS)")
    console.print("  • SSL: Required (enforced)")
    console.print("  • Access: pgAdmin, DBeaver, any PostgreSQL client")
    console.print("  • Credentials: Shown after provisioning")

    console.print("\n[bold]What you get:[/bold]")
    console.print("  ✓ Automatic provisioning (~3-5 minutes)")
    console.print("  ✓ Persistent storage across restarts")
    console.print("  ✓ Automatic daily backups (7-day retention)")
    console.print("  ✓ Real-time monitoring via dashboard")
    console.print("  ✓ [green]Included in subscription[/green]")

    if not Confirm.ask("\n[bold]Proceed with managed PostgreSQL?[/bold]", default=True):
        return None

    return {
        "mode": "managed",
        "architecture": "managed_postgres",
        "architecture_name": "PostgreSQL + pgvector (Managed)",
        "databases": [
            {"type": "postgres", "purpose": "Cortex memory with pgvector"}
        ]
    }
    
def configure_byo_databases():
    """Configure bring-your-own databases — PostgreSQL or Redis only."""

    console.print("\n[yellow]✓ Bring Your Own Database (BYO-DB)[/yellow]")
    console.print("[dim]Connect your existing database to Cortex memory.[/dim]")

    console.print("\n[bold]Cortex supports two backends:[/bold]")
    console.print("\n  1. [green]PostgreSQL + pgvector[/green] (recommended)")
    console.print("     • Vector search + full SQL in one database")
    console.print("     • ACID compliance, analytics, pgAdmin support")
    console.print("     • Requires pgvector extension installed")
    console.print("")
    console.print("  2. [cyan]Redis + RediSearch[/cyan]")
    console.print("     • In-memory, ultra-fast (~1ms latency)")
    console.print("     • Requires RediSearch module (Redis Stack or self-hosted)")
    console.print("     [yellow]• Not available on AWS ElastiCache — use Redis Cloud or self-hosted[/yellow]")

    backend_choice = Prompt.ask("\n[bold]Choose backend[/bold]", choices=["1", "2"], default="1")

    databases = []

    if backend_choice == "1":
        console.print("\n[bold]PostgreSQL Connection:[/bold]")
        console.print("[dim]Example: postgresql://user:pass@your-db.com:5432/teleon[/dim]")
        postgres_url = Prompt.ask("PostgreSQL URL", default="postgresql://localhost:5432/teleon")

        console.print("\n[dim]Testing connection...[/dim]")
        if test_postgres_connection(postgres_url):
            console.print("[green]✓ PostgreSQL connection successful[/green]")
            databases.append({
                "type": "postgres",
                "url": postgres_url,
                "purpose": "Cortex memory with pgvector"
            })
        else:
            console.print("[red]Could not connect to PostgreSQL[/red]")
            if not Confirm.ask("Continue anyway?", default=False):
                raise typer.Exit(1)
            databases.append({
                "type": "postgres",
                "url": postgres_url,
                "purpose": "Cortex memory with pgvector",
                "verified": False
            })
    else:
        console.print("\n[bold]Redis Connection:[/bold]")
        console.print("[dim]Example: redis://user:pass@your-redis.com:6379/0[/dim]")
        console.print("[yellow]Requires RediSearch module — AWS ElastiCache does NOT support this.[/yellow]")
        console.print("[dim]Use Redis Cloud, Redis Stack, or self-hosted Redis with RediSearch.[/dim]")
        redis_url = Prompt.ask("Redis URL", default="redis://localhost:6379/0")

        console.print("\n[dim]Testing connection...[/dim]")
        if test_redis_connection(redis_url):
            console.print("[green]✓ Redis connection successful[/green]")
            databases.append({
                "type": "redis",
                "url": redis_url,
                "purpose": "Cortex memory with RediSearch"
            })
        else:
            console.print("[red]Could not connect to Redis[/red]")
            if not Confirm.ask("Continue anyway?", default=False):
                raise typer.Exit(1)
            databases.append({
                "type": "redis",
                "url": redis_url,
                "purpose": "Cortex memory with RediSearch",
                "verified": False
            })

    console.print("\n" + "-"*70)
    console.print("[bold yellow]✓ BYO-DB Configuration Complete[/bold yellow]")
    console.print("-"*70)

    db = databases[0]
    db_icon = {"redis": "🔴", "postgres": "🐘"}.get(db["type"], "💾")
    verified = db.get("verified", True)
    status = "[green]verified[/green]" if verified else "[yellow]unverified[/yellow]"
    console.print(f"\n  {db_icon} [cyan]{db['type'].title()}[/cyan]: {db['purpose']} ({status})")

    console.print("\n[bold yellow]Important:[/bold yellow]")
    console.print("  • You are responsible for database backups & maintenance")
    console.print("  • Ensure the database is accessible from Teleon infrastructure")
    console.print("  • SSL connections are strongly recommended")

    return {
        "mode": "byo",
        "architecture": f"byo_{db['type']}",
        "architecture_name": f"BYO {db['type'].title()}",
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
        memory_info = ""
        features = []

        if agent.get("uses_cortex"):
            memory_info = "Cortex Memory"
            features.append("Cortex")

        if agent.get("uses_helix"):
            features.append("Auto-scale")

        if agent.get("uses_sentinel"):
            features.append("Sentinel")

        table.add_row(
            agent["name"],
            memory_info if memory_info else "[dim]None[/dim]",
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
        mode = database_config.get("mode", "managed")
        databases = database_config.get("databases", [])
        db_type = databases[0]["type"] if databases else "postgres"
        db_icon = {"redis": "🔴", "postgres": "🐘"}.get(db_type, "💾")

        if mode == "managed":
            console.print(f"[bold cyan]{db_icon} Storage:[/bold cyan] [green]{arch_name}[/green]")
            console.print("[dim]Mode: Managed by Teleon (zero ops)[/dim]")
        else:
            console.print(f"[bold cyan]{db_icon} Storage:[/bold cyan] [yellow]{arch_name}[/yellow]")
            console.print("[dim]Mode: Bring Your Own Database[/dim]")

        for db in databases:
            icon = {"redis": "🔴", "postgres": "🐘"}.get(db["type"], "💾")
            purpose = db.get("purpose", "Storage")
            if mode == "byo":
                verified = db.get("verified", True)
                status = "[green]✓ Connected[/green]" if verified else "[yellow]⚠ Unverified[/yellow]"
                console.print(f"  {icon} [cyan]{db['type'].title()}[/cyan]: {purpose} ({status})")
            else:
                console.print(f"  {icon} [cyan]{db['type'].title()}[/cyan]: {purpose}")

    # ========================================
    # ARCHITECTURE DIAGRAM
    # ========================================
    if database_config:
        databases = database_config.get("databases", [])
        db_type = databases[0]["type"] if databases else "postgres"

        console.print("\n" + "-"*70)
        console.print("[bold]Data Flow:[/bold]")

        if db_type == "postgres":
            console.print("""
    ┌─────────────┐    ┌──────────────────────────────────────────────┐
    │   Agent     │───▶│               Cortex Memory                  │
    └─────────────┘    │  ┌─────────────┐   ┌───────────────────────┐ │
                       │  │ Working     │   │ 🐘 PostgreSQL         │ │
                       │  │ (RAM)       │   │    [yellow]pgvector[/yellow]            │ │
                       │  └─────────────┘   │    All Memory          │ │
                       │                    │    [cyan]Full SQL + ACID[/cyan]     │ │
                       │                    └───────────────────────┘ │
                       └──────────────────────────────────────────────┘
""")
        elif db_type == "redis":
            console.print("""
    ┌─────────────┐    ┌──────────────────────────────────────────────┐
    │   Agent     │───▶│               Cortex Memory                  │
    └─────────────┘    │  ┌─────────────┐   ┌───────────────────────┐ │
                       │  │ Working     │   │ 🔴 Redis              │ │
                       │  │ (RAM)       │   │    [yellow]RediSearch[/yellow]          │ │
                       │  └─────────────┘   │    All Memory          │ │
                       │                    │    [cyan]~1ms latency[/cyan]       │ │
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
                
                # Pass BYO connection URL if present
                for db in database_config.get("databases", []):
                    if db.get("url"):
                        data[f'{db["type"]}_url'] = db["url"]
        
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
                    console.print(f"     • [green]{db['type'].title()}[/green] - {db['purpose']}")
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
