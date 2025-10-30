"""
NexusNet Collaboration CLI Commands.

Commands for managing multi-agent collaboration.
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional
import asyncio

app = typer.Typer(
    name="nexusnet",
    help="Manage multi-agent collaboration",
    add_completion=False
)

console = Console()


@app.command()
def status():
    """
    Show NexusNet collaboration status.
    
    Example:
        teleon nexusnet status
    """
    from teleon.nexusnet import get_registry
    
    console.print(Panel.fit(
        "[bold green]NexusNet Status[/bold green]",
        title="🌐 Multi-Agent Collaboration"
    ))
    
    try:
        registry = get_registry()
        agents = asyncio.run(registry.list_all())
        
        if not agents:
            console.print("\n[yellow]No agents registered with NexusNet[/yellow]")
            console.print("\n[dim]Enable collaboration with:[/dim]")
            console.print("[dim]  @client.agent([/dim]")
            console.print("[dim]    nexusnet={'capabilities': ['research'], 'collaborate': True}[/dim]")
            console.print("[dim]  )[/dim]")
            return
        
        console.print(f"\n[bold]Registered Agents:[/bold] {len(agents)}")
        
        # Group by status
        healthy = [a for a in agents if a.status.value == "healthy"]
        unhealthy = [a for a in agents if a.status.value != "healthy"]
        
        console.print(f"  [green]✓ Healthy:[/green] {len(healthy)}")
        console.print(f"  [yellow]○ Other:[/yellow] {len(unhealthy)}")
        
        # Create table
        table = Table(title="Collaborative Agents")
        table.add_column("Agent ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Capabilities", style="blue")
        
        for agent in agents[:10]:  # Show top 10
            caps = [c.name for c in agent.capabilities[:3]]
            caps_str = ", ".join(caps)
            if len(agent.capabilities) > 3:
                caps_str += f" +{len(agent.capabilities)-3}"
            
            status_emoji = "✓" if agent.status.value == "healthy" else "○"
            
            table.add_row(
                agent.agent_id[:16] + "...",
                agent.name,
                f"{status_emoji} {agent.status.value}",
                caps_str
            )
        
        console.print("\n", table)
        
        if len(agents) > 10:
            console.print(f"\n[dim]... and {len(agents)-10} more[/dim]")
    
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {str(e)}")


@app.command()
def discover(
    capability: str = typer.Argument(..., help="Capability to search for")
):
    """
    Discover agents by capability.
    
    Example:
        teleon nexusnet discover analysis
        teleon nexusnet discover research
    """
    from teleon.nexusnet import get_registry
    
    console.print(f"[bold]Discovering agents with capability: [cyan]{capability}[/cyan][/bold]")
    
    try:
        registry = get_registry()
        agents = asyncio.run(registry.discover(capability))
        
        if not agents:
            console.print(f"\n[yellow]No agents found with capability '{capability}'[/yellow]")
            return
        
        console.print(f"\n[green]✓ Found {len(agents)} agent(s)[/green]")
        
        for i, agent in enumerate(agents, 1):
            console.print(f"\n[bold]{i}. {agent.name}[/bold]")
            console.print(f"   ID: [cyan]{agent.agent_id}[/cyan]")
            console.print(f"   Status: {agent.status.value}")
            console.print(f"   Description: [dim]{agent.description}[/dim]")
            
            caps = [c.name for c in agent.capabilities]
            console.print(f"   Capabilities: {', '.join(caps)}")
    
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {str(e)}")


@app.command()
def capabilities():
    """
    List all available capabilities.
    
    Example:
        teleon nexusnet capabilities
    """
    from teleon.nexusnet import get_registry
    
    console.print(Panel.fit(
        "[bold green]Available Capabilities[/bold green]",
        title="🎯 NexusNet Capabilities"
    ))
    
    try:
        registry = get_registry()
        agents = asyncio.run(registry.list_all())
        
        # Collect all unique capabilities
        capabilities = {}
        for agent in agents:
            for cap in agent.capabilities:
                if cap.name not in capabilities:
                    capabilities[cap.name] = {
                        'description': cap.description,
                        'agents': []
                    }
                capabilities[cap.name]['agents'].append(agent.name)
        
        if not capabilities:
            console.print("\n[yellow]No capabilities registered[/yellow]")
            return
        
        console.print(f"\n[bold]Found {len(capabilities)} unique capabilities:[/bold]\n")
        
        for cap_name, info in sorted(capabilities.items()):
            console.print(f"[cyan]• {cap_name}[/cyan]")
            console.print(f"  [dim]{info['description']}[/dim]")
            console.print(f"  Agents: {', '.join(info['agents'][:3])}")
            if len(info['agents']) > 3:
                console.print(f"  [dim]... and {len(info['agents'])-3} more[/dim]")
            console.print()
    
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {str(e)}")


@app.command()
def messages(
    agent_id: Optional[str] = typer.Option(None, help="Filter by agent ID")
):
    """
    View collaboration messages.
    
    Example:
        teleon nexusnet messages
        teleon nexusnet messages --agent-id agent_abc123
    """
    console.print(Panel.fit(
        "[bold green]Collaboration Messages[/bold green]",
        title="💬 NexusNet Messages"
    ))
    
    if agent_id:
        console.print(f"\n[dim]Filtering for agent: {agent_id}[/dim]")
    
    console.print("\n[yellow]Note:[/yellow] Message viewing available through CollaborationAPI")
    console.print("\n[dim]In agent code:[/dim]")
    console.print("[dim]  messages = await collaboration.receive_messages(limit=10)[/dim]")
    console.print("[dim]  for msg in messages:[/dim]")
    console.print("[dim]    print(f'From {msg.from_agent}: {msg.content}')[/dim]")


if __name__ == "__main__":
    app()

