"""
Cortex Memory CLI Commands.

Commands for inspecting and managing Cortex memory system.
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional
import asyncio
import json

app = typer.Typer(
    name="cortex",
    help="Manage Cortex memory and learning",
    add_completion=False
)

console = Console()


@app.command()
def stats(
    agent_id: str = typer.Argument(..., help="Agent ID")
):
    """
    Show Cortex memory statistics for an agent.
    
    Example:
        teleon cortex stats agent_abc123
    """
    console.print(Panel.fit(
        f"[bold green]Cortex Statistics[/bold green]",
        title="🧠 Memory & Learning"
    ))
    
    console.print(f"\n[dim]Agent: {agent_id}[/dim]")
    console.print("\n[yellow]Note:[/yellow] Initialize Cortex with agent to view stats")
    console.print("\n[dim]Example in code:[/dim]")
    console.print("[dim]  cortex = await create_cortex(agent_id='{agent_id}')[/dim]")
    console.print("[dim]  stats = await cortex.get_statistics()[/dim]")


@app.command()
def inspect(
    agent_id: str = typer.Argument(..., help="Agent ID"),
    memory_type: str = typer.Option("all", help="Memory type (working, episodic, semantic, procedural, all)")
):
    """
    Inspect agent memory contents.
    
    Example:
        teleon cortex inspect agent_abc123
        teleon cortex inspect agent_abc123 --memory-type episodic
    """
    console.print(Panel.fit(
        f"[bold green]Memory Inspection[/bold green]\n"
        f"[dim]Agent: {agent_id}[/dim]\n"
        f"[dim]Type: {memory_type}[/dim]",
        title="🔍 Cortex Inspector"
    ))
    
    console.print("\n[yellow]Note:[/yellow] Memory inspection available through agent runtime")
    console.print("\n[dim]To inspect memory programmatically:[/dim]")
    console.print("[dim]  cortex = await create_cortex(agent_id)[/dim]")
    
    if memory_type in ["episodic", "all"]:
        console.print("[dim]  episodes = await cortex.episodic.get_recent(10)[/dim]")
    if memory_type in ["semantic", "all"]:
        console.print("[dim]  knowledge = await cortex.semantic.get_by_category('category')[/dim]")
    if memory_type in ["procedural", "all"]:
        console.print("[dim]  patterns = await cortex.procedural.get_top_patterns(10)[/dim]")


@app.command()
def clear(
    agent_id: str = typer.Argument(..., help="Agent ID"),
    memory_type: str = typer.Option("all", help="Memory type to clear"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")
):
    """
    Clear agent memory.
    
    Example:
        teleon cortex clear agent_abc123
        teleon cortex clear agent_abc123 --memory-type episodic --yes
    """
    if not confirm:
        console.print(f"[yellow]⚠️  This will clear {memory_type} memory for agent {agent_id}[/yellow]")
        confirm_input = console.input("Continue? [y/N]: ")
        if confirm_input.lower() != 'y':
            console.print("[dim]Cancelled[/dim]")
            return
    
    console.print(f"[bold]Clearing {memory_type} memory for {agent_id}...[/bold]")
    console.print("\n[yellow]Note:[/yellow] Memory clearing available through agent runtime")
    console.print("[dim]Use: await cortex.clear_all()[/dim]")


@app.command()
def learning(
    agent_id: str = typer.Argument(..., help="Agent ID")
):
    """
    Show learning metrics for an agent.
    
    Example:
        teleon cortex learning agent_abc123
    """
    console.print(Panel.fit(
        f"[bold green]Learning Metrics[/bold green]",
        title="🎓 Continuous Learning"
    ))
    
    console.print(f"\n[dim]Agent: {agent_id}[/dim]")
    console.print("\n[bold]Learning Status:[/bold] Active")
    console.print("[dim]Agents automatically learn from interactions[/dim]")
    
    console.print("\n[bold]Capabilities:[/bold]")
    console.print("  • Pattern recognition from successful interactions")
    console.print("  • Cost optimization through model selection")
    console.print("  • Latency improvements")
    console.print("  • Knowledge extraction")
    
    console.print("\n[dim]View detailed metrics in code:[/dim]")
    console.print("[dim]  from teleon.cortex.learning import LearningEngine[/dim]")
    console.print("[dim]  metrics = engine.get_metrics()[/dim]")


if __name__ == "__main__":
    app()

