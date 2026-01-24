"""
Cortex Memory CLI Commands.

Commands for inspecting and managing Cortex memory system.
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Optional
from pathlib import Path
import asyncio
import json
import time
from datetime import datetime

app = typer.Typer(
    name="cortex",
    help="Manage Cortex memory and learning",
    add_completion=False
)

console = Console()


@app.command()
def stats(
    agent_id: str = typer.Argument(..., help="Agent ID"),
    format: str = typer.Option("table", "--format", "-f", help="Output format (table, json)")
):
    """
    Show Cortex memory statistics for an agent.
    
    Example:
        teleon cortex stats agent_abc123
        teleon cortex stats agent_abc123 --format json
    """
    console.print(Panel.fit(
        f"[bold green]Cortex Statistics[/bold green]",
        title="🧠 Memory & Learning"
    ))
    
    console.print(f"\n[dim]Agent: {agent_id}[/dim]")
    
    # Try to get stats from local registry first
    try:
        from teleon.cortex.registry import registry
        
        async def get_stats():
            cortex = await registry.get(agent_id)
            if cortex:
                return await cortex.get_statistics()
            return None
        
        stats = asyncio.run(get_stats())
        if stats:
            _display_statistics(stats, format)
            return
    except Exception as e:
        console.print(f"[dim]Local registry check failed: {e}[/dim]")
    
    # Try to get stats from API if available
    try:
        import httpx
        import os
        
        platform_url = os.getenv("TELEON_PLATFORM_URL", "https://api.teleon.ai")
        config_file = Path.home() / ".teleon" / "config.json"
        
        if config_file.exists():
            config_data = json.loads(config_file.read_text())
            auth_token = config_data.get("auth_token")
            
            if auth_token:
                try:
                    response = httpx.get(
                        f"{platform_url}/api/v1/agents/{agent_id}/cortex/statistics",
                        headers={"Authorization": f"Bearer {auth_token}"},
                        timeout=10.0
                    )
                    
                    if response.status_code == 200:
                        stats = response.json()
                        _display_statistics(stats, format)
                        return
                except Exception:
                    pass  # Fall back to code example
    except Exception:
        pass
    
    # Fallback to code example
    console.print("\n[yellow]Note:[/yellow] Initialize Cortex with agent to view stats")
    console.print("\n[dim]Example in code:[/dim]")
    console.print(f"[dim]  from teleon.cortex import CortexMemory[/dim]")
    console.print(f"[dim]  cortex = CortexMemory(agent_id='{agent_id}')[/dim]")
    console.print(f"[dim]  await cortex.initialize()[/dim]")
    console.print(f"[dim]  stats = await cortex.get_statistics()[/dim]")


def _display_statistics(stats: dict, format: str):
    """Display statistics in table or JSON format."""
    if format == "json":
        console.print(json.dumps(stats, indent=2))
        return
    
    # Display as table
    table = Table(title="\n[bold]Memory Statistics[/bold]")
    table.add_column("Memory Type", style="cyan", width=15)
    table.add_column("Metric", style="white", width=25)
    table.add_column("Value", style="green", width=20)
    
    # Episodic memory
    if "episodic" in stats:
        ep = stats["episodic"]
        table.add_row("Episodic", "Total Episodes", str(ep.get("total_episodes", 0)))
        table.add_row("", "Success Rate", f"{ep.get('success_rate', 0):.1f}%")
        table.add_row("", "Avg Duration", f"{ep.get('avg_duration_ms', 0):.1f}ms")
        table.add_row("", "Unique Sessions", str(ep.get("unique_sessions", 0)))
    
    # Semantic memory
    if "semantic" in stats:
        sem = stats["semantic"]
        table.add_row("Semantic", "Total Entries", str(sem.get("total_entries", 0)))
        table.add_row("", "Categories", str(len(sem.get("categories", []))))
        table.add_row("", "Avg Importance", f"{sem.get('avg_importance', 0):.2f}")
    
    # Procedural memory
    if "procedural" in stats:
        proc = stats["procedural"]
        table.add_row("Procedural", "Total Patterns", str(proc.get("total_patterns", 0)))
        table.add_row("", "Avg Success Rate", f"{proc.get('avg_success_rate', 0):.1f}%")
        table.add_row("", "Total Usage", str(proc.get("total_usage", 0)))
    
    console.print(table)


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


@app.command()
def monitor(
    agent_id: str = typer.Argument(..., help="Agent ID"),
    interval: int = typer.Option(5, "--interval", "-i", help="Update interval in seconds"),
    format: str = typer.Option("table", "--format", "-f", help="Output format (table, json)")
):
    """
    Monitor Cortex memory metrics in real-time.
    
    Example:
        teleon cortex monitor agent_abc123
        teleon cortex monitor agent_abc123 --interval 10
    """
    console.print(Panel.fit(
        f"[bold cyan]Cortex Memory Monitor[/bold cyan]\n"
        f"Agent: [yellow]{agent_id}[/yellow]\n"
        f"Update interval: [dim]{interval}s[/dim]",
        title="📊 Real-Time Monitoring"
    ))
    
    # Try local registry first
    try:
        from teleon.cortex.registry import registry
        
        async def monitor_local():
            cortex = await registry.get(agent_id)
            if not cortex:
                return None
            return cortex.get_metrics()
        
        while True:
            try:
                metrics = asyncio.run(monitor_local())
                if metrics:
                    _display_metrics(metrics, format)
                else:
                    console.print(f"[yellow]Cortex instance not found for agent: {agent_id}[/yellow]")
                    console.print("[dim]Make sure Cortex is initialized for this agent[/dim]")
                    break
                
                time.sleep(interval)
                console.print("\n" + "─" * 70 + "\n")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Monitoring stopped[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                time.sleep(interval)
        
        return
    except Exception as e:
        console.print(f"[dim]Local monitoring failed: {e}[/dim]")
    
    # Fallback to API
    try:
        import httpx
        import os
        
        platform_url = os.getenv("TELEON_PLATFORM_URL", "https://api.teleon.ai")
        config_file = Path.home() / ".teleon" / "config.json"
        
        auth_token = None
        if config_file.exists():
            config_data = json.loads(config_file.read_text())
            auth_token = config_data.get("auth_token")
        
        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
        
        while True:
            try:
                response = httpx.get(
                    f"{platform_url}/api/v1/agents/{agent_id}/cortex/metrics",
                    headers=headers,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    metrics = response.json()
                    _display_metrics(metrics, format)
                else:
                    console.print(f"[red]Error: {response.status_code}[/red]")
                
                time.sleep(interval)
                console.print("\n" + "─" * 70 + "\n")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Monitoring stopped[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                time.sleep(interval)
                
    except ImportError:
        console.print("\n[yellow]Note:[/yellow] Real-time monitoring requires Cortex instance or API connection")
        console.print(f"[dim]Use: cortex = CortexMemory(agent_id='{agent_id}')[/dim]")
        console.print("[dim]await cortex.initialize()[/dim]")
        console.print("[dim]metrics = cortex.get_metrics()[/dim]")


def _display_metrics(metrics: dict, format: str):
    """Display metrics in table or JSON format."""
    if format == "json":
        console.print(json.dumps(metrics, indent=2))
        return
    
    # Display as table
    table = Table(title=f"\n[bold]Cortex Metrics - {datetime.now().strftime('%H:%M:%S')}[/bold]")
    table.add_column("Category", style="cyan", width=20)
    table.add_column("Metric", style="white", width=25)
    table.add_column("Value", style="green", width=20)
    
    # Operations
    if "operations" in metrics:
        for memory_type, ops in metrics["operations"].items():
            for op_name, op_stats in ops.items():
                table.add_row(
                    f"{memory_type}.{op_name}",
                    "Count",
                    str(op_stats.get("count", 0))
                )
                table.add_row(
                    "",
                    "Avg Latency",
                    f"{op_stats.get('avg_latency_ms', 0):.2f}ms"
                )
                table.add_row(
                    "",
                    "P95 Latency",
                    f"{op_stats.get('p95_latency_ms', 0):.2f}ms"
                )
                if op_stats.get("errors", 0) > 0:
                    table.add_row(
                        "",
                        "Errors",
                        f"[red]{op_stats.get('errors', 0)}[/red]"
                    )
    
    # Cache
    if "cache" in metrics:
        cache = metrics["cache"]
        table.add_row("Cache", "Hit Rate", f"{cache.get('hit_rate', 0):.1f}%")
        table.add_row("", "Hits", str(cache.get("hits", 0)))
        table.add_row("", "Misses", str(cache.get("misses", 0)))
    
    console.print(table)


@app.command()
def profile(
    agent_id: str = typer.Argument(..., help="Agent ID"),
    period: str = typer.Option("24h", "--period", "-p", help="Time period (1h, 24h, 7d, 30d)"),
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Export to file (JSON)")
):
    """
    Get performance profiling report for Cortex memory.
    
    Example:
        teleon cortex profile agent_abc123
        teleon cortex profile agent_abc123 --period 7d --export performance.json
    """
    console.print(Panel.fit(
        f"[bold cyan]Performance Profile[/bold cyan]\n"
        f"Agent: [yellow]{agent_id}[/yellow]\n"
        f"Period: [dim]{period}[/dim]",
        title="📊 Performance Analysis"
    ))
    
    # Try local registry first
    try:
        from teleon.cortex.registry import registry
        
        async def get_perf():
            cortex = await registry.get(agent_id)
            if not cortex:
                return None
            if not cortex._profiler:
                return None
            return cortex.get_performance_report()
        
        perf_data = asyncio.run(get_perf())
        if perf_data:
            perf_data["period"] = period
            perf_data["agent_id"] = agent_id
            
            # Export if requested
            if export:
                with open(export, 'w') as f:
                    json.dump(perf_data, f, indent=2)
                console.print(f"\n[green]✓ Exported to {export}[/green]")
            
            _display_performance(perf_data)
            return
    except Exception as e:
        console.print(f"[dim]Local profiling failed: {e}[/dim]")
    
    # Fallback to API
    try:
        import httpx
        import os
        
        platform_url = os.getenv("TELEON_PLATFORM_URL", "https://api.teleon.ai")
        config_file = Path.home() / ".teleon" / "config.json"
        
        auth_token = None
        if config_file.exists():
            config_data = json.loads(config_file.read_text())
            auth_token = config_data.get("auth_token")
        
        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
        
        response = httpx.get(
            f"{platform_url}/api/v1/agents/{agent_id}/cortex/performance",
            headers=headers,
            params={"period": period},
            timeout=10.0
        )
        
        if response.status_code == 200:
            perf_data = response.json()
            
            # Export if requested
            if export:
                with open(export, 'w') as f:
                    json.dump(perf_data, f, indent=2)
                console.print(f"\n[green]✓ Exported to {export}[/green]")
            
            _display_performance(perf_data)
        else:
            console.print(f"[red]Error: {response.status_code}[/red]")
            console.print(f"[dim]{response.text}[/dim]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("\n[yellow]Note:[/yellow] Performance profiling requires Cortex instance or API connection")
        console.print(f"[dim]Use: cortex = CortexMemory(agent_id='{agent_id}')[/dim]")
        console.print("[dim]await cortex.initialize()[/dim]")
        console.print("[dim]report = cortex.get_performance_report()[/dim]")


def _display_performance(perf_data: dict):
    """Display performance profiling data."""
    # Operations table
    if "operations" in perf_data:
        table = Table(title="\n[bold]Operation Performance[/bold]")
        table.add_column("Operation", style="cyan", width=25)
        table.add_column("Count", style="white", width=10, justify="right")
        table.add_column("Avg (ms)", style="green", width=12, justify="right")
        table.add_column("P95 (ms)", style="yellow", width=12, justify="right")
        table.add_column("P99 (ms)", style="red", width=12, justify="right")
        
        for op_name, op_data in perf_data["operations"].items():
            latency = op_data.get("latency_ms", {})
            table.add_row(
                op_name,
                str(op_data.get("count", 0)),
                f"{latency.get('avg', 0):.2f}",
                f"{latency.get('p95', 0):.2f}",
                f"{latency.get('p99', 0):.2f}"
            )
        
        console.print(table)
    
    # Slow operations
    if perf_data.get("slow_operations"):
        console.print("\n[bold yellow]⚠️  Slow Operations (>100ms):[/bold yellow]")
        for slow_op in perf_data["slow_operations"][:10]:
            timestamp = slow_op.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    timestamp = dt.strftime("%H:%M:%S")
                except:
                    pass
            console.print(
                f"  • [red]{slow_op.get('operation', 'unknown')}[/red]: "
                f"{slow_op.get('duration_ms', 0):.1f}ms at {timestamp}"
            )
    
    # Recommendations
    if perf_data.get("recommendations"):
        console.print("\n[bold cyan]💡 Recommendations:[/bold cyan]")
        for rec in perf_data["recommendations"]:
            console.print(f"  • {rec}")


if __name__ == "__main__":
    app()

