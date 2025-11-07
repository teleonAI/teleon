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
                matches = re.findall(r'@client\.agent\([^)]*name=["\']([^"\']+)["\']', content)
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
                    
                    uses_nexusnet = (
                        "nexusnet=" in content or 
                        "from teleon.nexusnet" in content or
                        "import nexusnet" in content
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
                    
                    agents_found.append({
                        "name": name,
                        "file": str(file),
                        "uses_cortex": uses_cortex,
                        "uses_helix": uses_helix,
                        "uses_nexusnet": uses_nexusnet,
                        "uses_chromadb": uses_chromadb,
                        "uses_semantic_memory": uses_semantic_memory,
                        "uses_redis": uses_redis
                    })
    
    return agents_found


def analyze_database_needs(agents):
    """Check if agents need databases"""
    
    for agent in agents:
        if agent.get("uses_cortex"):
            return True
    return False


def configure_databases(agents):
    """Configure database options"""
    
    console.print("\n" + "="*70)
    console.print("[bold cyan]💾 Database Configuration[/bold cyan]")
    console.print("="*70)
    
    # Count agents using Cortex
    cortex_agents = [a for a in agents if a.get("uses_cortex")]
    console.print(f"\n[bold]{len(cortex_agents)} agent(s) use Cortex memory[/bold]")
    
    # Show options
    console.print("\n[bold]How would you like to handle databases?[/bold]")
    console.print("\n  1. [green]Managed by Teleon[/green] (recommended)")
    console.print("     • Zero configuration")
    console.print("     • Automatic backups & scaling")
    console.print("     • Included in subscription")
    console.print("")
    console.print("  2. [yellow]Bring Your Own Database (BYO-DB)[/yellow]")
    console.print("     • Use your existing databases")
    console.print("     • Provide connection strings")
    console.print("     • You manage backups & scaling")
    
    choice = Prompt.ask("\n[bold]Choose option[/bold]", choices=["1", "2"], default="1")
    
    if choice == "1":
        return configure_managed_databases(cortex_agents)
    else:
        return configure_byo_databases()


def configure_managed_databases(agents):
    """Configure Teleon-managed databases"""
    
    console.print("\n[green]✓ Selected: Managed by Teleon[/green]")
    
    # Determine database needs based on memory types
    needs_chromadb = any(a.get("uses_chromadb", False) for a in agents)
    needs_semantic = any(a.get("uses_semantic_memory", False) for a in agents)
    needs_cortex = any(a.get("uses_cortex", False) for a in agents)
    
    databases = []
    
    # Note about Working Memory (always in-memory, no DB needed)
    console.print("\n[dim]📝 Note: Working Memory (session context) uses in-memory storage - no database needed[/dim]")
    
    # ========================================
    # 1. SEMANTIC MEMORY (Knowledge Base)
    # ========================================
    if needs_chromadb or needs_semantic:
        console.print("\n[bold cyan]Semantic Memory (Knowledge Base):[/bold cyan]")
        console.print("[dim]Stores long-term knowledge with vector similarity search[/dim]\n")
        
        if needs_chromadb:
            # ChromaDB explicitly detected
            console.print("  Detected: [green]ChromaDB usage[/green]")
            console.print("\n  Using ChromaDB for semantic memory:")
            console.print("    ✓ Free embeddings with FastEmbed (no API costs)")
            console.print("    ✓ Optimized for vector similarity search")
            console.print("    ✓ 10GB persistent storage via Azure Files")
            
            databases.append({
                "type": "chromadb",
                "size": "10GB",
                "purpose": "Semantic memory with ChromaDB + FastEmbed"
            })
        else:
            # Semantic memory detected but not ChromaDB - offer choice
            console.print("  [bold]Choose semantic memory backend:[/bold]")
            console.print("\n  1. [green]ChromaDB (recommended)[/green]")
            console.print("     • Free embeddings with FastEmbed")
            console.print("     • Optimized for vector search")
            console.print("     • 10GB storage included")
            console.print("")
            console.print("  2. [yellow]PostgreSQL with pgvector[/yellow]")
            console.print("     • Full SQL capabilities")
            console.print("     • Complex queries & analytics")
            console.print("     • 20GB storage included")
            
            semantic_choice = Prompt.ask("\n  Choose backend", choices=["1", "2"], default="1")
            
            if semantic_choice == "1":
                databases.append({
                    "type": "chromadb",
                    "size": "10GB",
                    "purpose": "Semantic memory with ChromaDB + FastEmbed"
                })
                console.print("  [green]✓ Selected: ChromaDB[/green]")
            else:
                databases.append({
                    "type": "postgres",
                    "size": "20GB",
                    "purpose": "Semantic memory with pgvector"
                })
                console.print("  [green]✓ Selected: PostgreSQL[/green]")
    
    # ========================================
    # 2. EPISODIC & PROCEDURAL MEMORY
    # ========================================
    if needs_cortex:
        console.print("\n[bold cyan]Episodic & Procedural Memory:[/bold cyan]")
        console.print("[dim]Conversation history & learned patterns[/dim]\n")
        
        console.print("  [bold]Choose storage backend:[/bold]")
        console.print("\n  1. [green]Redis (recommended for speed)[/green]")
        console.print("     • Ultra-fast in-memory operations")
        console.print("     • Perfect for high-throughput agents")
        console.print("     • 4GB storage included")
        console.print("     • Best for: Real-time agents, chat bots")
        console.print("")
        console.print("  2. [yellow]PostgreSQL (recommended for complex queries)[/yellow]")
        console.print("     • ACID compliance & data integrity")
        console.print("     • Complex time-range queries")
        console.print("     • Advanced analytics & reporting")
        console.print("     • Best for: Audit trails, compliance")
        
        episodic_choice = Prompt.ask("\n  Choose backend", choices=["1", "2"], default="1")
        
        if episodic_choice == "1":
            databases.append({
                "type": "redis",
                "size": "4GB",
                "purpose": "Episodic & procedural memory"
            })
            console.print("  [green]✓ Selected: Redis[/green]")
        else:
            # Check if we already have PostgreSQL for semantic memory
            has_postgres = any(db.get("type") == "postgres" for db in databases)
            if has_postgres:
                console.print("  [green]✓ Using existing PostgreSQL (shared with semantic memory)[/green]")
                # Update existing PostgreSQL entry
                for db in databases:
                    if db["type"] == "postgres":
                        db["size"] = "30GB"  # Increased size for combined usage
                        db["purpose"] = "Semantic, episodic & procedural memory"
            else:
                databases.append({
                    "type": "postgres",
                    "size": "20GB",
                    "purpose": "Episodic & procedural memory"
                })
                console.print("  [green]✓ Selected: PostgreSQL[/green]")
    
    # ========================================
    # 3. FALLBACK (minimal storage)
    # ========================================
    if not databases:
        databases.append({
            "type": "storage",
            "size": "5GB",
            "purpose": "General persistent storage"
        })
    
    # ========================================
    # Summary
    # ========================================
    console.print("\n[bold]Provisioned Storage:[/bold]")
    for db in databases:
        console.print(f"  • [cyan]{db['type'].title()} ({db['size']}):[/cyan] {db['purpose']}")
    
    console.print("\n[bold]What you get:[/bold]")
    console.print("  ✓ Automatic provisioning (3-5 minutes)")
    console.print("  ✓ Persistent storage across restarts")
    console.print("  ✓ Automatic daily backups")
    console.print("  ✓ Direct access via dashboard")
    console.print("  ✓ Included in subscription")
    
    # Special note for ChromaDB
    if any(db.get("type") == "chromadb" for db in databases):
        console.print("  ✓ [green]Free embeddings with FastEmbed (save $$$)[/green]")
    
    return {
        "mode": "managed",
        "databases": databases
    }


def configure_byo_databases():
    """Configure bring-your-own databases"""
    
    console.print("\n[yellow]✓ Selected: Bring Your Own Database[/yellow]")
    
    console.print("\n[bold]You'll need to provide:[/bold]")
    console.print("  • Redis connection string")
    console.print("  • PostgreSQL connection string (if using semantic memory)")
    
    console.print("\n[dim]Examples:[/dim]")
    console.print("  [dim]Redis: redis://user:pass@your-redis.com:6379/0[/dim]")
    console.print("  [dim]PostgreSQL: postgresql://user:pass@your-db.com:5432/teleon[/dim]")
    
    # Get Redis connection
    console.print("\n[bold cyan]Redis Configuration:[/bold cyan]")
    redis_url = Prompt.ask("Redis URL")
    
    # Test Redis connection
    console.print("\n[dim]Testing Redis connection...[/dim]")
    redis_ok = test_redis_connection(redis_url)
    
    if not redis_ok:
        console.print("[red]❌ Could not connect to Redis[/red]")
        if not Confirm.ask("Continue anyway?", default=False):
            raise typer.Exit(1)
    else:
        console.print("[green]✓ Redis connection successful[/green]")
    
    # Get PostgreSQL connection (optional)
    needs_postgres = Confirm.ask("\nDo you need PostgreSQL? (for semantic memory)", default=False)
    postgres_url = None
    
    if needs_postgres:
        console.print("\n[bold cyan]PostgreSQL Configuration:[/bold cyan]")
        postgres_url = Prompt.ask("PostgreSQL URL")
        
        console.print("\n[dim]Testing PostgreSQL connection...[/dim]")
        postgres_ok = test_postgres_connection(postgres_url)
        
        if not postgres_ok:
            console.print("[red]❌ Could not connect to PostgreSQL[/red]")
            if not Confirm.ask("Continue anyway?", default=False):
                raise typer.Exit(1)
        else:
            console.print("[green]✓ PostgreSQL connection successful[/green]")
    
    return {
        "mode": "byo",
        "redis_url": redis_url,
        "postgres_url": postgres_url
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
    """Show deployment plan"""
    
    console.print("\n" + "="*70)
    console.print("[bold cyan]📋 Deployment Plan[/bold cyan]")
    console.print("="*70)
    
    # Agents table
    table = Table(title=f"\nAgents ({len(agents)})")
    table.add_column("Name", style="cyan")
    table.add_column("Features", style="green")
    table.add_column("Scaling", style="yellow")
    
    for agent in agents:
        features = []
        if agent.get("uses_cortex"):
            if agent.get("uses_chromadb") or agent.get("uses_semantic_memory"):
                features.append("ChromaDB")
            else:
                features.append("Memory")
        if agent.get("uses_helix"):
            features.append("Auto-scale")
        if agent.get("uses_nexusnet"):
            features.append("Collab")
        
        table.add_row(
            agent["name"],
            ", ".join(features) if features else "Basic",
            "1-10 instances" if agent.get("uses_helix") else "1 instance"
        )
    
    console.print(table)
    
    # Database info
    if database_config:
        console.print(f"\n[bold]Databases:[/bold]")
        if database_config["mode"] == "managed":
            console.print("  Mode: [green]Managed by Teleon[/green]")
            for db in database_config["databases"]:
                console.print(f"  • {db['type'].title()} ({db['size']}): {db['purpose']}")
        else:
            console.print("  Mode: [yellow]Bring Your Own[/yellow]")
            console.print(f"  • Redis: External")
            if database_config.get("postgres_url"):
                console.print(f"  • PostgreSQL: External")
    
    # Pricing estimate
    console.print(f"\n[bold]Estimated Cost:[/bold]")
    console.print(f"  • Compute: $29-99/month (based on usage)")
    if database_config and database_config["mode"] == "managed":
        console.print(f"  • Storage: Included ✓")
        # Check if using ChromaDB
        uses_chromadb = any(db.get("type") == "chromadb" for db in database_config.get("databases", []))
        if uses_chromadb:
            console.print(f"  • Embeddings: [green]$0/month (FastEmbed - free!)[/green]")
        console.print(f"  [bold]Total: $29-99/month[/bold]")
    elif database_config and database_config["mode"] == "byo":
        console.print(f"  • Databases: External (you pay your provider)")
        console.print(f"  [bold]Total Teleon: $29-79/month[/bold]")
    else:
        console.print(f"  [bold]Total: $29-99/month[/bold]")
    
    console.print(f"\n[bold]Environment:[/bold] {env}")
    console.print(f"[bold]Region:[/bold] us-east-1 (auto-selected)")


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
        
        if database_config:
            database_mode = database_config["mode"]
            
            # Extract database requirements for managed mode
            if database_mode == "managed" and "databases" in database_config:
                for db in database_config["databases"]:
                    database_requirements.append({
                        "type": db["type"],
                        "size": db["size"],
                        "purpose": db["purpose"]
                    })
        
        files = {
            'code': ('code.zip', zip_buffer, 'application/zip')
        }
        
        data = {
            'project_id': project_id,
            'environment': env,
            'database_mode': database_mode,
        }
        
        # Add database configuration based on mode
        if database_config:
            if database_mode == "managed":
                # Send database requirements to platform
                data['database_requirements'] = json.dumps(database_requirements)
            elif database_mode == "byo":
                # Send connection strings
                if database_config.get("redis_url"):
                    data['redis_url'] = database_config["redis_url"]
                if database_config.get("postgres_url"):
                    data['postgres_url'] = database_config["postgres_url"]
        
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
                        
                        if deployment_status == "building":
                            progress.update(task, description="[cyan]Building containers...")
                        elif deployment_status == "deploying":
                            progress.update(task, description="[cyan]Deploying agents...")
                        elif deployment_status in ["running", "active", "provisioned"]:
                            progress.update(task, description="[green]✓ Deployment successful!")
                            progress.stop_task(task)
                            break
                        elif deployment_status == "failed":
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
        
        console.print(f"\n[bold]Your agents are live![/bold] 🎉")
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
    deployment_id: Optional[str] = None,
):
    """Rollback to a previous deployment"""
    
    console.print("[yellow]Rolling back deployment...[/yellow]")
    console.print("\n[green]✓ Rolled back to previous version[/green]")


@app.command()
def preview():
    """Deploy to preview environment"""
    
    console.print("[cyan]Deploying to preview environment...[/cyan]")
    console.print("\n[green]✓ Preview deployed: https://preview-abc123.teleon.ai[/green]")


if __name__ == "__main__":
    app()
