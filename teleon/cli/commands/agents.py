"""
Agent Management CLI Commands.

Commands for listing, inspecting, testing, and managing agents.
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from typing import Optional

app = typer.Typer(
    name="agents",
    help="Manage and inspect Teleon agents",
    add_completion=False
)

console = Console()


@app.command()
def list(
    user_id: Optional[str] = typer.Option(None, help="Filter by user ID"),
    status: Optional[str] = typer.Option(None, help="Filter by status"),
):
    """
    List all registered agents.
    
    Example:
        teleon agents list
        teleon agents list --user-id abc123
        teleon agents list --status running
    """
    from teleon.client import TeleonClient
    
    console.print(Panel.fit(
        "[bold green]Teleon Agents[/bold green]",
        title="📋 Agent List"
    ))
    
    # Get all agents
    all_agents = TeleonClient.get_all_agents()
    
    if not all_agents:
        console.print("[yellow]No agents registered yet.[/yellow]")
        console.print("\n[dim]Create an agent with:[/dim]")
        console.print("[dim]  @client.agent(name='my-agent')[/dim]")
        console.print("[dim]  def my_agent(input): ...[/dim]")
        return
    
    # Apply filters
    agents = all_agents
    if user_id:
        agents = {k: v for k, v in agents.items() if v.get('user_id') == user_id}
    
    # Create table
    table = Table(title=f"Found {len(agents)} agent(s)")
    table.add_column("Agent ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("User ID", style="yellow", no_wrap=True)
    table.add_column("Model", style="magenta")
    table.add_column("Features", style="blue")
    table.add_column("Created", style="dim")
    
    for agent_id, info in agents.items():
        # Build features list
        features = []
        if info.get('helix'):
            features.append("🏭 Helix")
        if info.get('cortex'):
            features.append("🧠 Cortex")
        features_str = " ".join(features) if features else "Basic"
        
        table.add_row(
            agent_id[:16] + "...",
            info.get('name', 'Unknown'),
            info.get('user_id', 'Unknown')[:8] + "...",
            info.get('model', 'N/A'),
            features_str,
            info.get('created_at', 'Unknown')[:10]
        )
    
    console.print(table)
    console.print(f"\n[dim]Total: {len(agents)} agent(s)[/dim]")


@app.command()
def inspect(
    agent_id: str = typer.Argument(..., help="Agent ID to inspect")
):
    """
    Inspect detailed information about an agent.
    
    Example:
        teleon agents inspect agent_abc123
    """
    from teleon.client import TeleonClient
    
    agent = TeleonClient.get_agent(agent_id)
    
    if not agent:
        console.print(f"[red]✗[/red] Agent '{agent_id}' not found")
        return
    
    console.print(Panel.fit(
        f"[bold green]{agent.get('name', 'Unknown')}[/bold green]\n"
        f"[dim]{agent.get('description', 'No description')}[/dim]",
        title=f"🔍 Agent: {agent_id[:16]}..."
    ))
    
    # Basic info
    console.print("\n[bold]Basic Information:[/bold]")
    console.print(f"  Agent ID: [cyan]{agent_id}[/cyan]")
    console.print(f"  Name: [green]{agent.get('name')}[/green]")
    console.print(f"  User ID: [yellow]{agent.get('user_id')}[/yellow]")
    console.print(f"  Model: [magenta]{agent.get('model')}[/magenta]")
    console.print(f"  Temperature: {agent.get('temperature')}")
    console.print(f"  Max Tokens: {agent.get('max_tokens')}")
    console.print(f"  Created: [dim]{agent.get('created_at')}[/dim]")
    
    # Helix configuration
    if agent.get('helix'):
        console.print("\n[bold]🏭 Helix Configuration:[/bold]")
        helix = agent['helix']
        console.print(f"  Min Instances: {helix.get('min_instances', helix.get('min', 1))}")
        console.print(f"  Max Instances: {helix.get('max_instances', helix.get('max', 1))}")
        if 'target_cpu' in helix:
            console.print(f"  Target CPU: {helix['target_cpu']}%")
        if 'memory_limit_mb' in helix:
            console.print(f"  Memory Limit: {helix['memory_limit_mb']} MB")
    
    # Cortex configuration
    if agent.get('cortex'):
        console.print("\n[bold]🧠 Cortex Configuration:[/bold]")
        cortex = agent['cortex']
        if isinstance(cortex, dict):
            for key, value in cortex.items():
                console.print(f"  {key}: {value}")
        else:
            console.print(f"  Enabled: {cortex}")
    
    # Parameters
    if agent.get('parameters'):
        console.print("\n[bold]Parameters:[/bold]")
        for param, details in agent['parameters'].items():
            required = "✓" if details.get('required') else "○"
            console.print(f"  {required} {param}: [dim]{details.get('type')}[/dim]")


@app.command()
def test(
    agent_id: str = typer.Argument(..., help="Agent ID to test"),
    input_data: Optional[str] = typer.Option(None, "--input", "-i", help="Input data (JSON)"),
):
    """
    Test an agent interactively.
    
    Example:
        teleon agents test agent_abc123
        teleon agents test agent_abc123 --input '{"query": "hello"}'
    """
    from teleon.client import TeleonClient
    import json
    import asyncio
    
    agent = TeleonClient.get_agent(agent_id)
    
    if not agent:
        console.print(f"[red]✗[/red] Agent '{agent_id}' not found")
        return
    
    console.print(Panel.fit(
        f"[bold green]Testing Agent: {agent.get('name')}[/bold green]",
        title="🧪 Agent Test"
    ))
    
    # Get input
    if input_data:
        try:
            test_input = json.loads(input_data)
        except json.JSONDecodeError:
            console.print("[red]✗[/red] Invalid JSON input")
            return
    else:
        console.print("\n[bold]Enter test input (JSON format):[/bold]")
        console.print("[dim]Example: {\"query\": \"hello\"}[/dim]")
        input_str = console.input("[green]>[/green] ")
        try:
            test_input = json.loads(input_str)
        except json.JSONDecodeError:
            console.print("[red]✗[/red] Invalid JSON input")
            return
    
    # Execute agent
    console.print("\n[bold]Executing agent...[/bold]")
    
    try:
        func = agent.get('function')
        if asyncio.iscoroutinefunction(func):
            result = asyncio.run(func(**test_input))
        else:
            result = func(**test_input)
        
        console.print("\n[bold green]✓ Success![/bold green]")
        console.print("\n[bold]Result:[/bold]")
        console.print(f"[cyan]{json.dumps(result, indent=2)}[/cyan]")
        
    except Exception as e:
        console.print(f"\n[red]✗ Error:[/red] {str(e)}")


@app.command()
def stats():
    """
    Show agent statistics.
    
    Example:
        teleon agents stats
    """
    from teleon.client import TeleonClient
    
    all_agents = TeleonClient.get_all_agents()
    
    console.print(Panel.fit(
        "[bold green]Agent Statistics[/bold green]",
        title="📊 Stats"
    ))
    
    if not all_agents:
        console.print("[yellow]No agents registered.[/yellow]")
        return
    
    # Calculate stats
    total = len(all_agents)
    with_helix = sum(1 for a in all_agents.values() if a.get('helix'))
    with_cortex = sum(1 for a in all_agents.values() if a.get('cortex'))
    
    # Count by model
    models = {}
    for agent in all_agents.values():
        model = agent.get('model', 'unknown')
        models[model] = models.get(model, 0) + 1
    
    # Display
    console.print(f"\n[bold]Total Agents:[/bold] {total}")
    console.print(f"\n[bold]Features Enabled:[/bold]")
    console.print(f"  🏭 Helix: {with_helix} ({with_helix/total*100:.1f}%)")
    console.print(f"  🧠 Cortex: {with_cortex} ({with_cortex/total*100:.1f}%)")
    
    console.print(f"\n[bold]Models Used:[/bold]")
    for model, count in sorted(models.items(), key=lambda x: x[1], reverse=True):
        console.print(f"  {model}: {count}")


if __name__ == "__main__":
    app()

