"""
Teleon CLI - Command Line Interface

Main entry point for the `teleon` command.
"""

import os
import typer
from rich.console import Console
from rich.panel import Panel
from typing import Optional

from teleon.__version__ import __version__
from teleon.cli.commands.dev import dev as dev_cmd
from teleon.cli.commands.exec import exec_app
from teleon.cli.commands.deploy import app as deploy_app
from teleon.cli.commands.agents import app as agents_app
from teleon.cli.commands.helix import app as helix_app
from teleon.cli.commands.cortex import app as cortex_app
from teleon.cli.commands.nexusnet import app as nexusnet_app
from teleon.cli.commands.auth import app as auth_app

# Initialize Typer app
app = typer.Typer(
    name="teleon",
    help="The Platform for Intelligent Agents - Deploy production-ready AI agents in minutes",
    add_completion=False,
)

# Add subcommands
app.add_typer(auth_app, name="auth")
app.add_typer(dev_cmd, name="dev")
app.add_typer(exec_app, name="exec")
app.add_typer(deploy_app, name="deploy")
app.add_typer(agents_app, name="agents")
app.add_typer(helix_app, name="helix")
app.add_typer(cortex_app, name="cortex")
app.add_typer(nexusnet_app, name="nexusnet")

# Add login as a top-level command (alias to auth login)
@app.command()
def login(api_key: str = typer.Option(None, "--api-key", help="Your Teleon API key")):
    """Authenticate with Teleon Platform (alias for 'teleon auth login')"""
    from teleon.cli.commands.auth import login as auth_login_cmd
    auth_login_cmd(api_key=api_key)

console = Console()


@app.command()
def version():
    """Show Teleon version."""
    console.print(f"[bold green]Teleon[/bold green] version [cyan]{__version__}[/cyan]")


@app.command()
def init(
    name: Optional[str] = None,
    template: str = "basic"
):
    """
    Initialize a new Teleon project.
    
    Example:
        teleon init my-agent
        teleon init --template advanced my-agent
    
    Args:
        name: Project name
        template: Project template (basic, advanced, production)
    """
    import os
    from pathlib import Path
    
    project_name = name or "teleon-project"
    
    console.print(Panel.fit(
        f"[bold green]Creating Teleon project:[/bold green] [cyan]{project_name}[/cyan]\n"
        f"Template: [yellow]{template}[/yellow]",
        title="🚀 Teleon Init"
    ))
    
    # Implement project scaffolding
    project_path = Path(project_name)
    
    if project_path.exists():
        console.print(f"[red]✗ Directory '{project_name}' already exists[/red]")
        return
    
    try:
        # Create project structure
        project_path.mkdir(parents=True)
        (project_path / "agents").mkdir()
        (project_path / "tools").mkdir()
        (project_path / "tests").mkdir()
        
        # Create teleon.yaml
        yaml_content = f"""# Teleon Configuration
project: {project_name}
environment: development

llm:
  default_provider: openai
  default_model: gpt-4
  max_retries: 3
  timeout: 30.0

memory:
  enabled: true
  working_ttl: 3600

tools:
  enabled: true
  max_concurrent: 5

observability:
  log_level: INFO
  metrics_enabled: true
"""
        (project_path / "teleon.yaml").write_text(yaml_content)
        
        # Create basic agent template
        if template == "basic":
            agent_content = '''"""Basic Teleon Agent Example"""

from teleon.decorators.agent import agent


@agent(name="hello-agent", memory=True)
async def hello_agent(name: str) -> str:
    """
    A simple greeting agent.
    
    Args:
        name: Name to greet
    
    Returns:
        Greeting message
    """
    return f"Hello, {name}! Welcome to Teleon!"


if __name__ == "__main__":
    import asyncio
    
    async def main():
        result = await hello_agent("World")
        print(result)
    
    asyncio.run(main())
'''
        elif template == "advanced":
            agent_content = '''"""Advanced Teleon Agent Example"""

from teleon.decorators.agent import agent
from teleon.decorators.tool import tool


@tool(name="calculate", description="Perform calculations")
async def calculate(expression: str) -> float:
    """Simple calculator tool."""
    return eval(expression)


@agent(
    name="advanced-agent",
    memory=True,
    scale={'min': 1, 'max': 10},
    tools=[calculate],
    collaborate=True
)
async def advanced_agent(task: str) -> dict:
    """
    An advanced agent with scaling and tools.
    
    Args:
        task: Task to perform
    
    Returns:
        Task result
    """
    return {
        "status": "completed",
        "task": task,
        "result": "Advanced processing complete"
    }


if __name__ == "__main__":
    import asyncio
    
    async def main():
        result = await advanced_agent("Process complex data")
        print(result)
    
    asyncio.run(main())
'''
        else:  # production
            agent_content = '''"""Production Teleon Agent Example"""

from teleon.decorators.agent import agent
from teleon.core import get_config, StructuredLogger, LogLevel


@agent(
    name="production-agent",
    memory=True,
    scale={'min': 2, 'max': 100, 'target_cpu': 70},
    timeout=60.0
)
async def production_agent(data: dict) -> dict:
    """
    Production-grade agent with full observability.
    
    Args:
        data: Input data to process
    
    Returns:
        Processed result
    """
    logger = StructuredLogger("production-agent", LogLevel.INFO)
    logger.info("Processing request", data_keys=list(data.keys()))
    
    # Your production logic here
    result = {
        "status": "success",
        "processed": True,
        "data": data
    }
    
    logger.info("Request completed", result_status=result["status"])
    return result


if __name__ == "__main__":
    import asyncio
    
    async def main():
        result = await production_agent({"key": "value"})
        print(result)
    
    asyncio.run(main())
'''
        
        (project_path / "agents" / "main.py").write_text(agent_content)
        
        # Create README
        readme_content = f"""# {project_name}

A Teleon AI Agent Project

## Getting Started

1. Install dependencies:
   ```bash
   pip install teleon
   ```

2. Run your agent:
   ```bash
   cd {project_name}
   python agents/main.py
   ```

3. Or use the Teleon CLI:
   ```bash
   teleon exec run agents/main.py
   ```

4. Start development server:
   ```bash
   teleon dev start agents/
   ```

## Project Structure

- `agents/` - Your AI agents
- `tools/` - Custom tools
- `tests/` - Test files
- `teleon.yaml` - Configuration

## Documentation

Visit https://teleon.ai/docs for complete documentation.
"""
        (project_path / "README.md").write_text(readme_content)
        
        # Create .gitignore
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.venv

# Teleon
.teleon/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo
"""
        (project_path / ".gitignore").write_text(gitignore_content)
        
        # Create requirements.txt
        requirements_content = """teleon>=0.1.0
"""
        (project_path / "requirements.txt").write_text(requirements_content)
        
        console.print(f"\n[green]✓ Project '{project_name}' created successfully![/green]")
        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"  1. cd {project_name}")
        console.print("  2. pip install -r requirements.txt")
        console.print("  3. python agents/main.py")
        console.print("\n[dim]Happy building! 🚀[/dim]")
        
    except Exception as e:
        console.print(f"[red]✗ Failed to create project: {e}[/red]")




@app.command()
def logs(
    agent: Optional[str] = None,
    deployment: Optional[str] = typer.Option(None, "--deployment", "-d", help="Deployment ID to view logs for"),
    follow: bool = False,
    lines: int = 100,
    level: str = "INFO"
):
    """
    View agent logs.
    
    Example:
        teleon logs my-agent
        teleon logs --deployment abc12345
        teleon logs my-agent --follow
        teleon logs --deployment abc12345 --lines 50 --level ERROR
    
    Args:
        agent: Agent name to view logs for (local logs)
        deployment: Deployment ID to view logs for (platform logs)
        follow: Follow log output (like tail -f)
        lines: Number of lines to show
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
    """
    import time
    from pathlib import Path
    from datetime import datetime
    import httpx
    import json
    
    # If deployment ID is provided, fetch logs from platform
    if deployment:
        console.print(Panel.fit(
            f"[bold green]Deployment Logs[/bold green]\n"
            f"Deployment ID: [cyan]{deployment}[/cyan]\n"
            f"Lines: [yellow]{lines}[/yellow] | Level: [yellow]{level}[/yellow]",
            title="📋 Teleon Logs"
        ))
        
        # Get auth token and platform URL
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
            # If deployment ID looks like a short ID (8 chars), try to expand it
            full_deployment_id = deployment
            
            if len(deployment) == 8 and '-' not in deployment:
                console.print(f"\n[dim]Looking up full deployment ID for: {deployment}...[/dim]")
                
                # Get user's deployments to find the matching one
                deployments_response = httpx.get(
                    f"{platform_url}/api/v1/deployments",
                    headers={"Authorization": f"Bearer {auth_token}"},
                    timeout=30.0
                )
                
                if deployments_response.status_code == 200:
                    deployments = deployments_response.json().get("deployments", [])
                    
                    # Find deployment that starts with the short ID
                    matching = [d for d in deployments if d.get("id", "").startswith(deployment)]
                    
                    if len(matching) == 1:
                        full_deployment_id = matching[0]["id"]
                        console.print(f"[dim]Found: {full_deployment_id}[/dim]")
                    elif len(matching) > 1:
                        console.print(f"[yellow]⚠️  Multiple deployments match '{deployment}':[/yellow]")
                        for d in matching[:5]:
                            console.print(f"  • {d['id']}")
                        console.print("\n[dim]Please use a more specific deployment ID[/dim]")
                        return
                    else:
                        console.print(f"[yellow]⚠️  No deployment found starting with '{deployment}'[/yellow]")
                        return
            
            # Fetch logs from platform API
            console.print(f"\n[dim]Fetching logs from {platform_url}...[/dim]\n")
            
            response = httpx.get(
                f"{platform_url}/api/v1/deployments/{full_deployment_id}/logs",
                headers={"Authorization": f"Bearer {auth_token}"},
                params={"lines": lines, "level": level},
                timeout=30.0
            )
            
            if response.status_code == 404:
                console.print(f"[yellow]⚠️  Deployment '{full_deployment_id}' not found[/yellow]")
                console.print("[dim]Make sure the deployment ID is correct[/dim]")
                return
            elif response.status_code == 403:
                console.print("[red]❌ Access denied to this deployment[/red]")
                return
            elif response.status_code != 200:
                console.print(f"[red]❌ Failed to fetch logs: {response.text}[/red]")
                return
            
            logs_data = response.json()
            log_lines = logs_data.get("logs", [])
            
            if not log_lines:
                console.print("[yellow]⚠️  No logs available for this deployment yet[/yellow]")
                console.print("\n[dim]Logs may take a few moments to appear after deployment[/dim]")
                return
            
            # Display logs
            for log_entry in log_lines:
                timestamp = log_entry.get("timestamp", "")
                log_level = log_entry.get("level", "INFO")
                message = log_entry.get("message", "")
                
                # Color code by level
                level_colors = {
                    "DEBUG": "dim",
                    "INFO": "cyan",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold red"
                }
                color = level_colors.get(log_level, "white")
                
                console.print(f"[dim]{timestamp}[/dim] [{color}]{log_level}[/{color}] {message}")
            
            console.print(f"\n[dim]Showing {len(log_lines)} log entries[/dim]")
            
        except httpx.RequestError as e:
            console.print(f"[red]❌ Network error: {e}[/red]")
            console.print(f"\n[yellow]Make sure the Teleon platform is running at: {platform_url}[/yellow]")
            raise typer.Exit(1)
        
        return
    
    # Otherwise, show local logs
    console.print(Panel.fit(
        f"[bold green]Logs for:[/bold green] [cyan]{agent or 'all agents'}[/cyan]\n"
        f"Follow: [yellow]{follow}[/yellow] | Lines: [yellow]{lines}[/yellow] | Level: [yellow]{level}[/yellow]",
        title="📋 Teleon Logs"
    ))
    
    # Implement log streaming
    log_dir = Path(".teleon/logs")
    
    if not log_dir.exists():
        console.print("[yellow]⚠️  No logs found. Run an agent first to generate logs.[/yellow]")
        console.print("\n[dim]Logs will be stored in .teleon/logs/[/dim]")
        console.print("\n[bold]To view deployment logs:[/bold]")
        console.print("[cyan]  teleon logs --deployment <deployment-id>[/cyan]")
        return
    
    # Find log files
    if agent:
        log_files = list(log_dir.glob(f"{agent}*.log"))
    else:
        log_files = list(log_dir.glob("*.log"))
    
    if not log_files:
        console.print(f"[yellow]⚠️  No logs found for '{agent or 'any agent'}'[/yellow]")
        console.print("\n[bold]To view deployment logs:[/bold]")
        console.print("[cyan]  teleon logs --deployment <deployment-id>[/cyan]")
        return
    
    # Sort by modification time (newest first)
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    console.print(f"\n[dim]Found {len(log_files)} log file(s)[/dim]\n")
    
    try:
        if follow:
            # Follow mode - tail -f like behavior
            console.print("[dim]Following logs (Ctrl+C to stop)...[/dim]\n")
            
            log_file = log_files[0]  # Most recent
            with open(log_file, 'r') as f:
                # Seek to end
                f.seek(0, 2)
                
                while True:
                    line = f.readline()
                    if line:
                        # Parse and format log line
                        if level.upper() in line or level == "DEBUG":
                            console.print(line.strip())
                    else:
                        time.sleep(0.1)
        else:
            # Static mode - show last N lines
            all_lines = []
            for log_file in log_files:
                with open(log_file, 'r') as f:
                    file_lines = f.readlines()
                    all_lines.extend([
                        (log_file.name, line.strip())
                        for line in file_lines
                        if level.upper() in line or level == "DEBUG"
                    ])
            
            # Show last N lines
            for filename, line in all_lines[-lines:]:
                console.print(f"[dim]{filename}:[/dim] {line}")
            
            console.print(f"\n[dim]Showing last {min(lines, len(all_lines))} lines[/dim]")
    
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped following logs[/dim]")
    except Exception as e:
        console.print(f"[red]✗ Error reading logs: {e}[/red]")


# CLI entry point for Poetry
cli = app


if __name__ == "__main__":
    app()

