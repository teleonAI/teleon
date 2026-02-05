"""
Cortex Memory CLI Commands.

Commands for inspecting and managing Cortex memory system.
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional
from pathlib import Path
import asyncio
import json
import time
from datetime import datetime

app = typer.Typer(
    name="cortex",
    help="Manage Cortex memory system",
    add_completion=False
)

console = Console()


@app.command()
def stats(
    agent_name: str = typer.Argument(..., help="Agent name"),
    scope: Optional[str] = typer.Option(None, "--scope", "-s", help="Scope filter (JSON format, e.g., '{\"customer_id\": \"alice\"}')"),
    format: str = typer.Option("table", "--format", "-f", help="Output format (table, json)")
):
    """
    Show Cortex memory statistics for an agent.

    Example:
        teleon cortex stats support-agent
        teleon cortex stats support-agent --scope '{"customer_id": "alice"}'
        teleon cortex stats support-agent --format json
    """
    console.print(Panel.fit(
        f"[bold green]Cortex Memory Statistics[/bold green]",
        title="🧠 Memory"
    ))

    console.print(f"\n[dim]Agent: {agent_name}[/dim]")

    # Parse scope filter
    scope_values = {}
    if scope:
        try:
            scope_values = json.loads(scope)
        except json.JSONDecodeError:
            console.print(f"[red]Invalid scope JSON: {scope}[/red]")
            raise typer.Exit(1)

    # Try to get stats from local memory manager
    try:
        from teleon.cortex import get_memory_manager, InMemoryBackend, get_embedding_service, Memory

        async def get_stats():
            # Create or get memory manager for this agent
            manager = get_memory_manager(agent_name)

            if manager is None:
                # Try to create a temporary memory instance
                backend = InMemoryBackend()
                embedding_service = get_embedding_service(is_paid_tier=False)

                memory = Memory(
                    backend=backend,
                    embedding_service=embedding_service,
                    memory_name=agent_name,
                    scope=list(scope_values.keys()),
                    scope_values=scope_values
                )

                total = await memory.count()
                return {"total_entries": total, "scope": scope_values}

            # Get stats from manager's memory
            memory = manager.create_memory(scope_values)
            total = await memory.count()

            return {"total_entries": total, "scope": scope_values, "agent": agent_name}

        stats = asyncio.run(get_stats())
        _display_stats(stats, format)
        return

    except Exception as e:
        console.print(f"[dim]Local stats failed: {e}[/dim]")

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
                    params = {"scope": json.dumps(scope_values)} if scope_values else {}
                    response = httpx.get(
                        f"{platform_url}/api/v1/agents/{agent_name}/cortex/stats",
                        headers={"Authorization": f"Bearer {auth_token}"},
                        params=params,
                        timeout=10.0
                    )

                    if response.status_code == 200:
                        stats = response.json()
                        _display_stats(stats, format)
                        return
                except Exception:
                    pass
    except Exception:
        pass

    # Fallback to code example
    console.print("\n[yellow]Note:[/yellow] No active memory found for this agent")
    console.print("\n[dim]Example usage in code:[/dim]")
    console.print(f"""[dim]
from teleon import TeleonClient

client = TeleonClient(api_key="...")

@client.agent(name="{agent_name}", cortex=True)
async def {agent_name.replace('-', '_')}(query: str, customer_id: str, cortex):
    # Get stats
    total = await cortex.count()
    queries = await cortex.count(filter={{"type": "query"}})
    print(f"Total: {{total}}, Queries: {{queries}}")
[/dim]""")


def _display_stats(stats: dict, format: str):
    """Display statistics in table or JSON format."""
    if format == "json":
        console.print(json.dumps(stats, indent=2))
        return

    # Display as table
    table = Table(title="\n[bold]Memory Statistics[/bold]")
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Value", style="green", width=30)

    table.add_row("Total Entries", str(stats.get("total_entries", 0)))

    if stats.get("scope"):
        table.add_row("Scope", json.dumps(stats["scope"]))

    if stats.get("agent"):
        table.add_row("Agent", stats["agent"])

    # Additional stats if available
    for key, value in stats.items():
        if key not in ["total_entries", "scope", "agent"]:
            table.add_row(key.replace("_", " ").title(), str(value))

    console.print(table)


@app.command()
def search(
    agent_name: str = typer.Argument(..., help="Agent name"),
    query: str = typer.Argument(..., help="Search query"),
    scope: Optional[str] = typer.Option(None, "--scope", "-s", help="Scope filter (JSON)"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results"),
    format: str = typer.Option("table", "--format", "-f", help="Output format (table, json)")
):
    """
    Search agent memory semantically.

    Example:
        teleon cortex search support-agent "billing issues"
        teleon cortex search support-agent "refund" --scope '{"customer_id": "alice"}'
    """
    console.print(Panel.fit(
        f"[bold green]Cortex Memory Search[/bold green]\n"
        f"[dim]Query: {query}[/dim]",
        title="🔍 Search"
    ))

    # Parse scope filter
    scope_values = {}
    if scope:
        try:
            scope_values = json.loads(scope)
        except json.JSONDecodeError:
            console.print(f"[red]Invalid scope JSON: {scope}[/red]")
            raise typer.Exit(1)

    # Try local search
    try:
        from teleon.cortex import get_memory_manager, InMemoryBackend, get_embedding_service, Memory

        async def do_search():
            manager = get_memory_manager(agent_name)

            if manager:
                memory = manager.create_memory(scope_values)
            else:
                backend = InMemoryBackend()
                embedding_service = get_embedding_service(is_paid_tier=False)
                memory = Memory(
                    backend=backend,
                    embedding_service=embedding_service,
                    memory_name=agent_name,
                    scope=list(scope_values.keys()),
                    scope_values=scope_values
                )

            results = await memory.search(query=query, limit=limit)
            return results

        results = asyncio.run(do_search())

        if not results:
            console.print("\n[yellow]No results found[/yellow]")
            return

        _display_search_results(results, format)
        return

    except Exception as e:
        console.print(f"[dim]Local search failed: {e}[/dim]")

    # Fallback
    console.print("\n[yellow]Note:[/yellow] Search requires active memory instance")
    console.print("[dim]Use cortex.search(query='...') in your agent code[/dim]")


def _display_search_results(results: list, format: str):
    """Display search results."""
    if format == "json":
        console.print(json.dumps([
            {
                "id": r.id,
                "content": r.content,
                "score": r.score,
                "fields": r.fields,
                "created_at": r.created_at.isoformat() if r.created_at else None
            }
            for r in results
        ], indent=2))
        return

    console.print(f"\n[bold]Found {len(results)} results:[/bold]\n")

    for i, entry in enumerate(results, 1):
        score_str = f"[green]{entry.score:.3f}[/green]" if entry.score else "[dim]N/A[/dim]"
        console.print(f"[bold]{i}.[/bold] Score: {score_str}")
        console.print(f"   Content: {entry.content[:100]}{'...' if len(entry.content) > 100 else ''}")
        if entry.fields:
            fields_str = ", ".join(f"{k}={v}" for k, v in entry.fields.items() if not k.startswith("_"))
            if fields_str:
                console.print(f"   Fields: [dim]{fields_str}[/dim]")
        console.print()


@app.command()
def history(
    agent_name: str = typer.Argument(..., help="Agent name"),
    scope: Optional[str] = typer.Option(None, "--scope", "-s", help="Scope filter (JSON)"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum entries"),
    type_filter: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by type field"),
    format: str = typer.Option("table", "--format", "-f", help="Output format (table, json)")
):
    """
    View recent memory entries.

    Example:
        teleon cortex history support-agent
        teleon cortex history support-agent --scope '{"customer_id": "alice"}' --limit 10
        teleon cortex history support-agent --type query
    """
    console.print(Panel.fit(
        f"[bold green]Memory History[/bold green]",
        title="📋 History"
    ))

    console.print(f"\n[dim]Agent: {agent_name}[/dim]")

    # Parse scope filter
    scope_values = {}
    if scope:
        try:
            scope_values = json.loads(scope)
        except json.JSONDecodeError:
            console.print(f"[red]Invalid scope JSON: {scope}[/red]")
            raise typer.Exit(1)

    # Build filter
    filter_dict = dict(scope_values)
    if type_filter:
        filter_dict["type"] = type_filter

    # Try local retrieval
    try:
        from teleon.cortex import get_memory_manager, InMemoryBackend, get_embedding_service, Memory

        async def get_history():
            manager = get_memory_manager(agent_name)

            if manager:
                memory = manager.create_memory(scope_values)
            else:
                backend = InMemoryBackend()
                embedding_service = get_embedding_service(is_paid_tier=False)
                memory = Memory(
                    backend=backend,
                    embedding_service=embedding_service,
                    memory_name=agent_name,
                    scope=list(scope_values.keys()),
                    scope_values=scope_values
                )

            entries = await memory.get(filter=filter_dict if filter_dict else {}, limit=limit)
            return entries

        entries = asyncio.run(get_history())

        if not entries:
            console.print("\n[yellow]No entries found[/yellow]")
            return

        _display_history(entries, format)
        return

    except Exception as e:
        console.print(f"[dim]Local history failed: {e}[/dim]")

    # Fallback
    console.print("\n[yellow]Note:[/yellow] History requires active memory instance")
    console.print("[dim]Use cortex.get(filter={...}) in your agent code[/dim]")


def _display_history(entries: list, format: str):
    """Display history entries."""
    if format == "json":
        console.print(json.dumps([
            {
                "id": e.id,
                "content": e.content,
                "fields": e.fields,
                "created_at": e.created_at.isoformat() if e.created_at else None
            }
            for e in entries
        ], indent=2))
        return

    table = Table(title=f"\n[bold]Recent Entries ({len(entries)})[/bold]")
    table.add_column("Time", style="dim", width=20)
    table.add_column("Type", style="cyan", width=12)
    table.add_column("Content", style="white", width=50)

    for entry in entries:
        time_str = entry.created_at.strftime("%Y-%m-%d %H:%M:%S") if entry.created_at else "N/A"
        type_str = entry.fields.get("type", "-") if entry.fields else "-"
        content_str = entry.content[:50] + "..." if len(entry.content) > 50 else entry.content

        table.add_row(time_str, type_str, content_str)

    console.print(table)


@app.command()
def clear(
    agent_name: str = typer.Argument(..., help="Agent name"),
    scope: Optional[str] = typer.Option(None, "--scope", "-s", help="Scope filter (JSON) - required for safety"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")
):
    """
    Clear memory entries.

    Example:
        teleon cortex clear support-agent --scope '{"customer_id": "alice"}'
        teleon cortex clear support-agent --scope '{"customer_id": "alice"}' --yes
    """
    if not scope:
        console.print("[red]❌ --scope is required for safety[/red]")
        console.print("[dim]Example: --scope '{\"customer_id\": \"alice\"}'[/dim]")
        raise typer.Exit(1)

    # Parse scope filter
    try:
        scope_values = json.loads(scope)
    except json.JSONDecodeError:
        console.print(f"[red]Invalid scope JSON: {scope}[/red]")
        raise typer.Exit(1)

    if not confirm:
        console.print(f"[yellow]⚠️  This will delete all entries matching:[/yellow]")
        console.print(f"   Agent: {agent_name}")
        console.print(f"   Scope: {json.dumps(scope_values)}")
        confirm_input = console.input("\nContinue? [y/N]: ")
        if confirm_input.lower() != 'y':
            console.print("[dim]Cancelled[/dim]")
            return

    # Try local deletion
    try:
        from teleon.cortex import get_memory_manager, InMemoryBackend, get_embedding_service, Memory

        async def do_clear():
            manager = get_memory_manager(agent_name)

            if manager:
                memory = manager.create_memory(scope_values)
            else:
                backend = InMemoryBackend()
                embedding_service = get_embedding_service(is_paid_tier=False)
                memory = Memory(
                    backend=backend,
                    embedding_service=embedding_service,
                    memory_name=agent_name,
                    scope=list(scope_values.keys()),
                    scope_values=scope_values
                )

            deleted = await memory.delete(filter={})
            return deleted

        deleted = asyncio.run(do_clear())
        console.print(f"\n[green]✓ Deleted {deleted} entries[/green]")
        return

    except Exception as e:
        console.print(f"[red]Clear failed: {e}[/red]")

    # Fallback
    console.print("\n[yellow]Note:[/yellow] Clear requires active memory instance")
    console.print("[dim]Use cortex.delete(filter={...}) in your agent code[/dim]")


@app.command()
def info():
    """
    Show Cortex system information.

    Example:
        teleon cortex info
    """
    console.print(Panel.fit(
        "[bold cyan]Cortex Memory System[/bold cyan]",
        title="ℹ️ Info"
    ))

    # Check what's available
    try:
        from teleon.cortex import (
            POSTGRES_AVAILABLE,
            REDIS_AVAILABLE,
            FASTEMBED_AVAILABLE,
            OPENAI_EMBED_AVAILABLE
        )

        console.print("\n[bold]Storage Backends:[/bold]")
        console.print(f"  • InMemory: [green]✓ Available[/green]")
        console.print(f"  • PostgreSQL: {'[green]✓ Available[/green]' if POSTGRES_AVAILABLE else '[dim]Not installed (pip install asyncpg)[/dim]'}")
        console.print(f"  • Redis: {'[green]✓ Available[/green]' if REDIS_AVAILABLE else '[dim]Not installed (pip install redis)[/dim]'}")

        console.print("\n[bold]Embedding Models:[/bold]")
        console.print(f"  • FastEmbed: {'[green]✓ Available (free)[/green]' if FASTEMBED_AVAILABLE else '[dim]Not installed (pip install fastembed)[/dim]'}")
        console.print(f"  • OpenAI: {'[green]✓ Available (paid)[/green]' if OPENAI_EMBED_AVAILABLE else '[dim]Not installed (pip install openai)[/dim]'}")

        console.print("\n[bold]API Methods:[/bold]")
        console.print("  • store() - Save content with any fields")
        console.print("  • search() - Semantic search with optional filter")
        console.print("  • get() - Get by filter (no semantic search)")
        console.print("  • update() - Update entries matching filter")
        console.print("  • delete() - Delete entries matching filter")
        console.print("  • count() - Count entries matching filter")

        console.print("\n[bold]Documentation:[/bold]")
        console.print("  https://docs.teleon.ai/cortex")

    except ImportError as e:
        console.print(f"[red]Cortex not available: {e}[/red]")


if __name__ == "__main__":
    app()
