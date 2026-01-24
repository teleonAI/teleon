"""
Push command for Teleon Platform.

Push code changes to already deployed agents without creating a new deployment.
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

app = typer.Typer(help="Push updates to deployed agents")
console = Console()


@app.callback(invoke_without_command=True)
def push(
    ctx: typer.Context,
    deployment_id: Optional[str] = typer.Option(None, "--deployment", "-d", help="Deployment ID to push to"),
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Agent name to push"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
    message: Optional[str] = typer.Option(None, "--message", "-m", help="Deployment message/note"),
):
    """
    Push code changes to deployed agents.
    
    Examples:
        teleon push                           # Interactive - select from your deployments
        teleon push --deployment abc12345     # Push to specific deployment
        teleon push --agent my-agent          # Push by agent name
        teleon push -f                        # Skip confirmation
    """
    # If a subcommand was invoked, don't run the main push logic
    if ctx.invoked_subcommand is not None:
        return
    import httpx
    import json
    import zipfile
    import io
    from pathlib import Path as PathlibPath
    
    console.print(Panel.fit(
        "[bold cyan]Teleon Push[/bold cyan]\n"
        "Push code changes to your deployed agents",
        title="🚀 Push"
    ))
    
    # Step 1: Check authentication
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
    
    # Step 2: Get user's deployments
    console.print("\n[dim]Fetching your deployments...[/dim]")
    
    try:
        response = httpx.get(
            f"{platform_url}/api/v1/deployments",
            headers=headers,
            timeout=30.0
        )
        
        if response.status_code == 401:
            console.print("\n[red]❌ Authentication failed. Run: teleon login[/red]")
            raise typer.Exit(1)
        
        if response.status_code != 200:
            console.print(f"\n[red]❌ Failed to fetch deployments: {response.text}[/red]")
            raise typer.Exit(1)
        
        deployments_data = response.json()
        all_deployments = deployments_data.get("deployments", [])
        
    except httpx.RequestError as e:
        console.print(f"\n[red]❌ Network error: {e}[/red]")
        console.print(f"\n[yellow]Make sure the Teleon platform is running at: {platform_url}[/yellow]")
        raise typer.Exit(1)
    
    # Filter to only active/running deployments
    active_deployments = [
        d for d in all_deployments 
        if d.get("status") in ["ACTIVE", "STARTING", "active", "running", "provisioned"]
    ]
    
    if not active_deployments:
        console.print("\n[yellow]⚠️  No active deployments found.[/yellow]")
        console.print("\n[dim]Deploy your agents first with: teleon deploy[/dim]")
        raise typer.Exit(0)
    
    # Step 3: Select deployment
    selected_deployment = None
    
    # If deployment_id provided, find it
    if deployment_id:
        # Support short IDs (first 8 chars)
        for d in active_deployments:
            if d["id"] == deployment_id or d["id"].startswith(deployment_id):
                selected_deployment = d
                break
        
        if not selected_deployment:
            console.print(f"\n[red]❌ Deployment '{deployment_id}' not found or not active[/red]")
            raise typer.Exit(1)
    
    # If agent name provided, find deployments with that agent
    elif agent:
        matching = []
        for d in active_deployments:
            agents_info = d.get("agents", {})
            agent_list = agents_info.get("agents", [])
            
            for a in agent_list:
                if a.get("name") == agent or agent in a.get("name", ""):
                    matching.append(d)
                    break
        
        if not matching:
            console.print(f"\n[yellow]⚠️  No active deployment found with agent '{agent}'[/yellow]")
            raise typer.Exit(1)
        elif len(matching) == 1:
            selected_deployment = matching[0]
        else:
            # Multiple matches - show selection
            console.print(f"\n[cyan]Found {len(matching)} deployments with agent '{agent}':[/cyan]")
            active_deployments = matching  # Filter to just matching
    
    # Interactive selection if no deployment selected yet
    if not selected_deployment:
        console.print("\n[bold cyan]📦 Your Active Deployments:[/bold cyan]\n")
        
        # Create table
        table = Table(
            show_header=True,
            header_style="bold cyan",
            border_style="dim"
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Project", style="green", width=20)
        table.add_column("Deployment ID", style="cyan", width=12)
        table.add_column("Status", style="yellow", width=12)
        table.add_column("URL", style="cyan", width=30)
        table.add_column("Deployed", style="dim", width=16)
        
        for idx, d in enumerate(active_deployments, 1):
            # Format deployed time
            deployed_at = d.get("deployed_at", "")
            if deployed_at:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(deployed_at.replace("Z", "+00:00"))
                    deployed_at = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    deployed_at = deployed_at[:16]
            
            status_color = {
                "active": "[green]active[/green]",
                "running": "[green]running[/green]",
                "provisioned": "[cyan]provisioned[/cyan]",
            }.get(d.get("status"), d.get("status", "unknown"))
            
            # Show URL or N/A
            url = d.get("url", "")
            if url:
                # Truncate long URLs
                if len(url) > 28:
                    url = url[:25] + "..."
            else:
                url = "[dim]N/A[/dim]"
            
            table.add_row(
                str(idx),
                d.get("project_name", "Unknown"),
                d.get("id", "")[:8] + "...",
                status_color,
                url,
                deployed_at
            )
        
        console.print(table)
        
        # Prompt for selection
        while True:
            choice = Prompt.ask(
                "\n[bold]Select deployment to push to[/bold]",
                default="1"
            )
            
            try:
                idx = int(choice)
                if 1 <= idx <= len(active_deployments):
                    selected_deployment = active_deployments[idx - 1]
                    break
                else:
                    console.print(f"[red]Please enter a number between 1 and {len(active_deployments)}[/red]")
            except ValueError:
                # Maybe they entered a deployment ID
                for d in active_deployments:
                    if d["id"].startswith(choice):
                        selected_deployment = d
                        break
                
                if selected_deployment:
                    break
                console.print("[red]Invalid selection. Enter a number or deployment ID.[/red]")
    
    # Step 4: Show selected deployment and confirm
    console.print(f"\n[green]✓ Selected:[/green] [bold]{selected_deployment['project_name']}[/bold]")
    console.print(f"  Deployment: [cyan]{selected_deployment['id'][:8]}...[/cyan]")
    console.print(f"  URL: [cyan]{selected_deployment.get('url', 'N/A')}[/cyan]")
    
    agents_info = selected_deployment.get("agents", {})
    agent_list = agents_info.get("agents", [])
    if agent_list:
        console.print(f"  Agents: {', '.join([a.get('name', '?') for a in agent_list])}")
    
    # Step 5: Detect local agents
    console.print("\n[dim]Scanning for local agents...[/dim]")
    
    from teleon.cli.commands.deploy import detect_agents
    local_agents = detect_agents()
    
    if not local_agents:
        console.print("\n[red]❌ No agents found in current directory![/red]")
        console.print("[dim]Make sure you have agents defined with @client.agent decorator[/dim]")
        raise typer.Exit(1)
    
    console.print(f"\n[green]✓[/green] Found {len(local_agents)} local agent(s):")
    for a in local_agents:
        console.print(f"  • {a['name']}")
    
    # Step 6: Confirm push
    if not force:
        console.print("\n[bold yellow]⚠️  Warning:[/bold yellow] This will update the running deployment.")
        console.print("[dim]The changes will be applied with zero downtime (rolling update).[/dim]")
        
        if not Confirm.ask("\n[bold]Continue with push?[/bold]", default=True):
            console.print("[yellow]Push cancelled[/yellow]")
            raise typer.Exit(0)
    
    # Step 7: Package code
    console.print(f"\n[cyan]Packaging code...[/cyan]")
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all Python files in current directory
        for py_file in PathlibPath(".").rglob("*.py"):
            if not any(part.startswith('.') or part == '__pycache__' or part == 'venv' for part in py_file.parts):
                zipf.write(py_file, py_file)
        
        # Add requirements.txt if it exists
        if PathlibPath("requirements.txt").exists():
            zipf.write("requirements.txt")
    
    zip_buffer.seek(0)
    console.print(f"[green]✓[/green] Code packaged ({len(zip_buffer.getvalue()) // 1024}KB)")
    
    # Step 8: Push to platform
    console.print(f"\n[cyan]Pushing to deployment {selected_deployment['id'][:8]}...[/cyan]")
    
    try:
        files = {
            'code': ('code.zip', zip_buffer, 'application/zip')
        }
        
        data = {
            'message': message or f"Push from CLI at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Uploading...", total=None)
            
            response = httpx.post(
                f"{platform_url}/api/v1/deployments/{selected_deployment['id']}/push",
                headers={"Authorization": f"Bearer {auth_token}"},
                files=files,
                data=data,
                timeout=120.0
            )
            
            progress.update(task, description="[cyan]Processing...")
            
            if response.status_code == 404:
                console.print("\n[red]❌ Push endpoint not available[/red]")
                console.print("[dim]Your platform may need to be updated to support push.[/dim]")
                raise typer.Exit(1)
            
            if response.status_code not in [200, 201, 202]:
                error_detail = response.json().get("detail", response.text) if response.headers.get("content-type", "").startswith("application/json") else response.text
                console.print(f"\n[red]❌ Push failed: {error_detail}[/red]")
                raise typer.Exit(1)
            
            result = response.json()
            progress.update(task, description="[green]✓ Push successful!")
        
        # Success!
        console.print("\n" + "="*60)
        console.print("[bold green]✅ Push Successful![/bold green]")
        console.print("="*60)
        
        console.print(f"\n[bold]Changes pushed to:[/bold]")
        console.print(f"  Project: [green]{selected_deployment['project_name']}[/green]")
        console.print(f"  URL: [cyan]{selected_deployment.get('url', 'N/A')}[/cyan]")
        
        push_info = result.get("push", {})
        if push_info.get("version"):
            console.print(f"  Version: [cyan]{push_info['version']}[/cyan]")
        if push_info.get("rollout_status"):
            console.print(f"  Rollout: [yellow]{push_info['rollout_status']}[/yellow]")
        
        console.print(f"\n[bold]Next steps:[/bold]")
        console.print(f"  • Monitor deployment: [cyan]teleon logs -d {selected_deployment['id'][:8]}[/cyan]")
        console.print(f"  • Rollback if needed: [cyan]teleon deploy rollback -d {selected_deployment['id'][:8]}[/cyan]")
        
    except httpx.RequestError as e:
        console.print(f"\n[red]❌ Network error: {e}[/red]")
        raise typer.Exit(1)


@app.command("status")
def push_status(
    deployment_id: str = typer.Option(..., "--deployment", "-d", help="Deployment ID"),
):
    """
    Check the status of a push operation.
    
    Example:
        teleon push status --deployment abc12345
    """
    import httpx
    import json
    
    config_file = Path.home() / ".teleon" / "config.json"
    if not config_file.exists():
        console.print("\n[red]❌ Not authenticated. Run: teleon login[/red]")
        raise typer.Exit(1)
    
    config_data = json.loads(config_file.read_text())
    auth_token = config_data.get("auth_token")
    platform_url = os.getenv("TELEON_PLATFORM_URL", "https://api.teleon.ai")
    
    try:
        # If deployment ID looks like a short ID (8 chars), try to expand it
        full_deployment_id = deployment_id
        
        if len(deployment_id) == 8 and '-' not in deployment_id:
            console.print(f"\n[dim]Looking up full deployment ID for: {deployment_id}...[/dim]")
            
            # Get user's deployments to find the matching one
            deployments_response = httpx.get(
                f"{platform_url}/api/v1/deployments",
                headers={"Authorization": f"Bearer {auth_token}"},
                timeout=30.0
            )
            
            if deployments_response.status_code == 200:
                deployments = deployments_response.json().get("deployments", [])
                
                # Find deployment that starts with the short ID
                matching = [d for d in deployments if d.get("id", "").startswith(deployment_id)]
                
                if len(matching) == 1:
                    full_deployment_id = matching[0]["id"]
                    console.print(f"[dim]Found: {full_deployment_id}[/dim]")
                elif len(matching) > 1:
                    console.print(f"[yellow]⚠️  Multiple deployments match '{deployment_id}':[/yellow]")
                    for d in matching[:5]:
                        console.print(f"  • {d['id']}")
                    console.print("\n[dim]Please use a more specific deployment ID[/dim]")
                    return
                else:
                    console.print(f"[yellow]⚠️  No deployment found starting with '{deployment_id}'[/yellow]")
                    return
        
        response = httpx.get(
            f"{platform_url}/api/v1/deployments/{full_deployment_id}",
            headers={"Authorization": f"Bearer {auth_token}"},
            timeout=30.0
        )
        
        if response.status_code != 200:
            console.print(f"\n[red]❌ Failed to get status: {response.text}[/red]")
            raise typer.Exit(1)
        
        data = response.json()
        deployment = data.get("deployment", {})
        
        console.print(Panel.fit(
            f"[bold]Deployment Status[/bold]\n"
            f"Project: [green]{deployment.get('project_name', 'Unknown')}[/green]\n"
            f"Status: [cyan]{deployment.get('status', 'unknown')}[/cyan]\n"
            f"URL: [cyan]{deployment.get('url', 'N/A')}[/cyan]",
            title="📊 Status"
        ))
        
    except httpx.RequestError as e:
        console.print(f"\n[red]❌ Network error: {e}[/red]")
        raise typer.Exit(1)


@app.command("history")
def push_history(
    deployment_id: str = typer.Option(..., "--deployment", "-d", help="Deployment ID"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of entries to show"),
):
    """
    View push history for a deployment.
    
    Example:
        teleon push history --deployment abc12345
        teleon push history -d abc12345 -n 20
    """
    import httpx
    import json
    
    config_file = Path.home() / ".teleon" / "config.json"
    if not config_file.exists():
        console.print("\n[red]❌ Not authenticated. Run: teleon login[/red]")
        raise typer.Exit(1)
    
    config_data = json.loads(config_file.read_text())
    auth_token = config_data.get("auth_token")
    platform_url = os.getenv("TELEON_PLATFORM_URL", "https://api.teleon.ai")
    
    try:
        # If deployment ID looks like a short ID (8 chars), try to expand it
        full_deployment_id = deployment_id
        
        if len(deployment_id) == 8 and '-' not in deployment_id:
            console.print(f"\n[dim]Looking up full deployment ID for: {deployment_id}...[/dim]")
            
            # Get user's deployments to find the matching one
            deployments_response = httpx.get(
                f"{platform_url}/api/v1/deployments",
                headers={"Authorization": f"Bearer {auth_token}"},
                timeout=30.0
            )
            
            if deployments_response.status_code == 200:
                deployments = deployments_response.json().get("deployments", [])
                
                # Find deployment that starts with the short ID
                matching = [d for d in deployments if d.get("id", "").startswith(deployment_id)]
                
                if len(matching) == 1:
                    full_deployment_id = matching[0]["id"]
                    console.print(f"[dim]Found: {full_deployment_id}[/dim]")
                elif len(matching) > 1:
                    console.print(f"[yellow]⚠️  Multiple deployments match '{deployment_id}':[/yellow]")
                    for d in matching[:5]:
                        console.print(f"  • {d['id']}")
                    console.print("\n[dim]Please use a more specific deployment ID[/dim]")
                    return
                else:
                    console.print(f"[yellow]⚠️  No deployment found starting with '{deployment_id}'[/yellow]")
                    return
        
        response = httpx.get(
            f"{platform_url}/api/v1/deployments/{full_deployment_id}/push-history",
            headers={"Authorization": f"Bearer {auth_token}"},
            params={"limit": limit},
            timeout=30.0
        )
        
        if response.status_code == 404:
            console.print("\n[yellow]⚠️  Push history endpoint not available[/yellow]")
            console.print("[dim]View deployment logs instead: teleon logs -d {deployment_id}[/dim]")
            raise typer.Exit(0)
        
        if response.status_code != 200:
            console.print(f"\n[red]❌ Failed to get history: {response.text}[/red]")
            raise typer.Exit(1)
        
        data = response.json()
        history = data.get("history", [])
        
        if not history:
            console.print("\n[yellow]No push history found[/yellow]")
            return
        
        console.print(Panel.fit(
            f"[bold cyan]Push History[/bold cyan]\n"
            f"Deployment: {deployment_id[:8]}...",
            title="📜 History"
        ))
        
        table = Table(show_header=True, header_style="bold cyan", border_style="dim")
        table.add_column("Version", style="cyan", width=10)
        table.add_column("Status", style="yellow", width=12)
        table.add_column("Message", style="white", width=30)
        table.add_column("Pushed At", style="dim", width=20)
        
        for entry in history:
            table.add_row(
                entry.get("version", "N/A"),
                entry.get("status", "unknown"),
                entry.get("message", "No message")[:30],
                entry.get("pushed_at", "")[:19]
            )
        
        console.print(table)
        
    except httpx.RequestError as e:
        console.print(f"\n[red]❌ Network error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

