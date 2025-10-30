"""
Helix Runtime CLI Commands.

Commands for managing the Helix runtime system.
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional
import asyncio

app = typer.Typer(
    name="helix",
    help="Manage Helix runtime and scaling",
    add_completion=False
)

console = Console()


@app.command()
def status():
    """
    Show Helix runtime status.
    
    Example:
        teleon helix status
    """
    from teleon.helix.runtime import get_runtime
    
    console.print(Panel.fit(
        "[bold green]Helix Runtime Status[/bold green]",
        title="🏭 Helix"
    ))
    
    try:
        runtime = get_runtime()
        
        console.print(f"\n[bold]Runtime:[/bold] {'Running' if runtime.running else 'Stopped'}")
        console.print(f"[bold]Environment:[/bold] {runtime.config.environment}")
        console.print(f"[bold]Debug Mode:[/bold] {runtime.config.debug}")
        console.print(f"[bold]Hot Reload:[/bold] {runtime.config.hot_reload}")
        
        # List agents
        agents = asyncio.run(runtime.list_agents())
        
        if agents:
            console.print(f"\n[bold]Registered Agents:[/bold] {len(agents)}")
            
            table = Table()
            table.add_column("Agent ID", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Instances", justify="center")
            table.add_column("Min", justify="center")
            table.add_column("Max", justify="center")
            
            for agent in agents:
                status_emoji = "✓" if agent.get('status') == 'running' else "○"
                table.add_row(
                    agent['agent_id'][:16] + "...",
                    f"{status_emoji} {agent.get('status', 'unknown')}",
                    str(agent.get('instances', 0)),
                    str(agent.get('resources', {}).get('min_instances', 1)),
                    str(agent.get('resources', {}).get('max_instances', 1))
                )
            
            console.print(table)
        else:
            console.print("\n[yellow]No agents registered with Helix[/yellow]")
        
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {str(e)}")


@app.command()
def scale(
    agent_id: str = typer.Argument(..., help="Agent ID to scale"),
    instances: int = typer.Argument(..., help="Desired number of instances")
):
    """
    Scale an agent to specific number of instances.
    
    Example:
        teleon helix scale agent_abc123 5
    """
    from teleon.helix.integration import scale_agent
    
    console.print(f"[bold]Scaling agent {agent_id[:16]}... to {instances} instances...[/bold]")
    
    try:
        success = asyncio.run(scale_agent(agent_id, instances))
        
        if success:
            console.print(f"[green]✓[/green] Agent scaled to {instances} instances")
        else:
            console.print("[red]✗[/red] Failed to scale agent")
    
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {str(e)}")


@app.command()
def restart(
    agent_id: str = typer.Argument(..., help="Agent ID to restart")
):
    """
    Restart an agent.
    
    Example:
        teleon helix restart agent_abc123
    """
    from teleon.helix.integration import restart_agent
    
    console.print(f"[bold]Restarting agent {agent_id[:16]}...[/bold]")
    
    try:
        success = asyncio.run(restart_agent(agent_id))
        
        if success:
            console.print("[green]✓[/green] Agent restarted successfully")
        else:
            console.print("[red]✗[/red] Failed to restart agent")
    
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {str(e)}")


@app.command()
def stop(
    agent_id: str = typer.Argument(..., help="Agent ID to stop"),
    force: bool = typer.Option(False, "--force", "-f", help="Force kill")
):
    """
    Stop an agent.
    
    Example:
        teleon helix stop agent_abc123
        teleon helix stop agent_abc123 --force
    """
    from teleon.helix.integration import stop_agent
    
    action = "Force stopping" if force else "Stopping"
    console.print(f"[bold]{action} agent {agent_id[:16]}...[/bold]")
    
    try:
        success = asyncio.run(stop_agent(agent_id, force=force))
        
        if success:
            console.print(f"[green]✓[/green] Agent stopped")
        else:
            console.print("[red]✗[/red] Failed to stop agent")
    
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {str(e)}")


@app.command()
def health(
    agent_id: str = typer.Argument(..., help="Agent ID to check")
):
    """
    Check agent health.
    
    Example:
        teleon helix health agent_abc123
    """
    from teleon.helix.integration import get_agent_status
    
    console.print(f"[bold]Checking health for {agent_id[:16]}...[/bold]")
    
    try:
        status = asyncio.run(get_agent_status(agent_id))
        
        if status.get('status') == 'not_found':
            console.print(f"[red]✗[/red] Agent not found")
            return
        
        health = status.get('health', {})
        health_status = health.get('status', 'unknown')
        
        color = "green" if health_status == "healthy" else "yellow"
        console.print(f"\n[{color}]Health:[/{color}] {health_status.upper()}")
        
        if status.get('processes'):
            console.print(f"\n[bold]Processes:[/bold]")
            for proc in status['processes']:
                console.print(f"  • {proc.get('process_id')}: {proc.get('status')}")
                console.print(f"    CPU: {proc.get('cpu_percent', 0):.1f}%, Memory: {proc.get('memory_mb', 0):.1f} MB")
    
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {str(e)}")


if __name__ == "__main__":
    app()

