"""
Sentinel Safety & Compliance CLI Commands.

Commands for inspecting and managing Sentinel safety system.
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
    name="sentinel",
    help="Manage Sentinel safety and compliance",
    add_completion=False
)

console = Console()


@app.command()
def status():
    """
    Show Sentinel status across all agents.
    
    Example:
        teleon sentinel status
    """
    console.print(Panel.fit(
        "[bold green]Sentinel Status[/bold green]",
        title="🛡️ Safety & Compliance"
    ))
    
    try:
        from teleon.sentinel.registry import get_sentinel_registry
        
        async def get_status():
            registry = await get_sentinel_registry()
            engines = await registry.list_all()
            
            if not engines:
                console.print("\n[yellow]No agents registered with Sentinel[/yellow]")
                console.print("[dim]    sentinel={'content_filtering': True, 'pii_detection': True}[/dim]")
                return
            
            # Display status table
            table = Table(title="Sentinel Status")
            table.add_column("Agent ID", style="cyan")
            table.add_column("Enabled", style="green")
            table.add_column("Content Filtering", style="yellow")
            table.add_column("PII Detection", style="yellow")
            table.add_column("Compliance", style="blue")
            
            for agent_id, engine in engines.items():
                config = engine.config
                compliance_str = ", ".join([c.value.upper() for c in config.compliance]) if config.compliance else "None"
                
                table.add_row(
                    agent_id[:20] + "..." if len(agent_id) > 20 else agent_id,
                    "✓" if config.enabled else "✗",
                    "✓" if config.content_filtering else "✗",
                    "✓" if config.pii_detection else "✗",
                    compliance_str
                )
            
            console.print("\n")
            console.print(table)
        
        asyncio.run(get_status())
        
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")
        console.print("[dim]Make sure agents are running with Sentinel enabled[/dim]")


@app.command()
def violations(
    agent_id: str = typer.Argument(..., help="Agent ID"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum violations to show"),
    format: str = typer.Option("table", "--format", "-f", help="Output format (table, json)")
):
    """
    List Sentinel violations for an agent.
    
    Example:
        teleon sentinel violations agent_abc123
        teleon sentinel violations agent_abc123 --limit 50 --format json
    """
    console.print(Panel.fit(
        f"[bold green]Sentinel Violations[/bold green]",
        title="🛡️ Safety & Compliance"
    ))
    
    console.print(f"\n[dim]Agent: {agent_id}[/dim]")
    
    try:
        from teleon.sentinel.registry import get_sentinel_registry
        
        async def get_violations():
            registry = await get_sentinel_registry()
            engine = await registry.get(agent_id)
            
            if not engine:
                console.print(f"\n[yellow]Agent {agent_id} not found or Sentinel not enabled[/yellow]")
                return
            
            audit_logger = engine.get_audit_logger()
            if not audit_logger:
                console.print("\n[yellow]Audit logging not enabled for this agent[/yellow]")
                return
            
            violations = audit_logger.get_violations(agent_id=agent_id, limit=limit)
            
            if not violations:
                console.print("\n[green]✓ No violations found[/green]")
                return
            
            if format == "json":
                console.print(json.dumps([v.to_dict() for v in violations], indent=2))
                return
            
            # Display violations table
            table = Table(title=f"Violations for {agent_id}")
            table.add_column("Timestamp", style="dim")
            table.add_column("Type", style="yellow")
            table.add_column("Action", style="cyan")
            table.add_column("Details", style="white")
            
            for violation in violations:
                timestamp = violation.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                details = str(violation.details.get('message', ''))[:50]
                
                table.add_row(
                    timestamp,
                    violation.violation_type,
                    violation.action_taken,
                    details
                )
            
            console.print("\n")
            console.print(table)
            
            # Show statistics
            stats = audit_logger.get_violation_stats(agent_id=agent_id)
            console.print(f"\n[dim]Total violations: {stats['total_violations']}[/dim]")
            console.print(f"[dim]Recent (last hour): {stats['recent_count']}[/dim]")
        
        asyncio.run(get_violations())
        
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")


@app.command()
def test(
    agent_id: str = typer.Argument(..., help="Agent ID"),
    input_text: str = typer.Option(..., "--input", "-i", help="Input text to test")
):
    """
    Test Sentinel validation on input text.
    
    Example:
        teleon sentinel test agent_abc123 --input "test@example.com"
        teleon sentinel test agent_abc123 --input "This is toxic content"
    """
    console.print(Panel.fit(
        f"[bold green]Sentinel Test[/bold green]",
        title="🛡️ Safety & Compliance"
    ))
    
    console.print(f"\n[dim]Agent: {agent_id}[/dim]")
    console.print(f"[dim]Input: {input_text[:100]}...[/dim]" if len(input_text) > 100 else f"[dim]Input: {input_text}[/dim]")
    
    try:
        from teleon.sentinel.registry import get_sentinel_registry
        
        async def test_validation():
            registry = await get_sentinel_registry()
            engine = await registry.get(agent_id)
            
            if not engine:
                console.print(f"\n[yellow]Agent {agent_id} not found or Sentinel not enabled[/yellow]")
                return
            
            # Test input validation
            result = await engine.validate_input(input_text, agent_id)
            
            if result.passed:
                console.print("\n[green]✓ Validation passed[/green]")
            else:
                console.print(f"\n[red]✗ Validation failed[/red]")
                console.print(f"[yellow]Action: {result.action.value}[/yellow]")
                console.print(f"[yellow]Violations: {len(result.violations)}[/yellow]")
                
                for violation in result.violations:
                    console.print(f"\n  [red]• {violation.get('type', 'unknown')}[/red]")
                    console.print(f"    {violation.get('message', 'No message')}")
                
                if result.redacted_content:
                    console.print(f"\n[dim]Redacted content:[/dim]")
                    console.print(f"[dim]{result.redacted_content}[/dim]")
        
        asyncio.run(test_validation())
        
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


@app.command()
def config(
    agent_id: str = typer.Argument(..., help="Agent ID"),
    format: str = typer.Option("table", "--format", "-f", help="Output format (table, json)")
):
    """
    Show Sentinel configuration for an agent.
    
    Example:
        teleon sentinel config agent_abc123
        teleon sentinel config agent_abc123 --format json
    """
    console.print(Panel.fit(
        f"[bold green]Sentinel Configuration[/bold green]",
        title="🛡️ Safety & Compliance"
    ))
    
    console.print(f"\n[dim]Agent: {agent_id}[/dim]")
    
    try:
        from teleon.sentinel.registry import get_sentinel_registry
        
        async def get_config():
            registry = await get_sentinel_registry()
            engine = await registry.get(agent_id)
            
            if not engine:
                console.print(f"\n[yellow]Agent {agent_id} not found or Sentinel not enabled[/yellow]")
                return
            
            config = engine.config
            
            if format == "json":
                console.print(json.dumps(config.to_dict(), indent=2))
                return
            
            # Display config table
            table = Table(title=f"Configuration for {agent_id}")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Enabled", "✓" if config.enabled else "✗")
            table.add_row("Content Filtering", "✓" if config.content_filtering else "✗")
            table.add_row("PII Detection", "✓" if config.pii_detection else "✗")
            table.add_row("Moderation Threshold", str(config.moderation_threshold))
            table.add_row("Action on Violation", config.action_on_violation.value)
            table.add_row("Log Violations", "✓" if config.log_violations else "✗")
            table.add_row("Audit Enabled", "✓" if config.audit_enabled else "✗")
            
            compliance_str = ", ".join([c.value.upper() for c in config.compliance]) if config.compliance else "None"
            table.add_row("Compliance Standards", compliance_str)
            
            policies_str = ", ".join(config.custom_policies) if config.custom_policies else "None"
            table.add_row("Custom Policies", policies_str)
            
            console.print("\n")
            console.print(table)
        
        asyncio.run(get_config())
        
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")


@app.command()
def audit(
    agent_id: str = typer.Argument(..., help="Agent ID"),
    format: str = typer.Option("json", "--format", "-f", help="Export format (json, csv)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path")
):
    """
    Export Sentinel audit log for an agent.
    
    Example:
        teleon sentinel audit agent_abc123
        teleon sentinel audit agent_abc123 --format csv --output audit.csv
    """
    console.print(Panel.fit(
        f"[bold green]Sentinel Audit Export[/bold green]",
        title="🛡️ Safety & Compliance"
    ))
    
    console.print(f"\n[dim]Agent: {agent_id}[/dim]")
    
    try:
        from teleon.sentinel.registry import get_sentinel_registry
        
        async def export_audit():
            registry = await get_sentinel_registry()
            engine = await registry.get(agent_id)
            
            if not engine:
                console.print(f"\n[yellow]Agent {agent_id} not found or Sentinel not enabled[/yellow]")
                return
            
            audit_logger = engine.get_audit_logger()
            if not audit_logger:
                console.print("\n[yellow]Audit logging not enabled for this agent[/yellow]")
                return
            
            # Export audit trail
            audit_data = audit_logger.export_audit_trail(agent_id=agent_id, format=format)
            
            if output:
                # Write to file
                output_path = Path(output)
                if format == "json":
                    output_path.write_text(json.dumps(audit_data, indent=2))
                else:
                    output_path.write_text(audit_data)
                
                console.print(f"\n[green]✓ Audit log exported to {output}[/green]")
                console.print(f"[dim]Records: {len(audit_data) if isinstance(audit_data, list) else 'N/A'}[/dim]")
            else:
                # Print to console
                if format == "json":
                    console.print(json.dumps(audit_data, indent=2))
                else:
                    console.print(audit_data)
        
        asyncio.run(export_audit())
        
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")

