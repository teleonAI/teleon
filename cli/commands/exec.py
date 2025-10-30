"""Exec command for Teleon CLI."""

import typer
import asyncio
import json
import importlib.util
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional

console = Console()

exec_app = typer.Typer(help="Execute agent commands")


@exec_app.command(name="run")
def run(
    agent_file: str,
    agent_name: Optional[str] = None,
    input_data: Optional[str] = None,
    input_file: Optional[str] = None
):
    """
    Execute an agent from a Python file.
    
    Example:
        teleon exec run agent.py --input-data '{"message": "hello"}'
        teleon exec run agent.py --input-file input.json
        teleon exec run agent.py --agent-name my-agent --input-data '{"x": 5}'
    
    Args:
        agent_file: Path to agent file (e.g., agent.py)
        agent_name: Agent name to execute
        input_data: Input data as JSON string
        input_file: Input data from JSON file
    """
    console.print(Panel.fit(
        f"[bold green]Executing Agent[/bold green]\n"
        f"File: [cyan]{agent_file}[/cyan]",
        title="🚀 Teleon Exec"
    ))
    
    # Load the agent file
    agent_path = Path(agent_file)
    if not agent_path.exists():
        console.print(f"[red]Error: File not found: {agent_file}[/red]")
        raise typer.Exit(1)
    
    # Import the module
    spec = importlib.util.spec_from_file_location("agent_module", agent_path)
    if spec is None or spec.loader is None:
        console.print(f"[red]Error: Could not load module from {agent_file}[/red]")
        raise typer.Exit(1)
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_module"] = module
    spec.loader.exec_module(module)
    
    # Find agents in the module
    agents = {}
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and getattr(obj, '_teleon_agent', False):
            config = getattr(obj, '_teleon_config', None)
            agent_key = config.name if config else name
            agents[agent_key] = obj
    
    if not agents:
        console.print("[red]Error: No agents found in file[/red]")
        raise typer.Exit(1)
    
    # Select agent to execute
    if agent_name:
        if agent_name not in agents:
            console.print(f"[red]Error: Agent '{agent_name}' not found[/red]")
            console.print(f"Available agents: {', '.join(agents.keys())}")
            raise typer.Exit(1)
        selected_agent = agents[agent_name]
    elif len(agents) == 1:
        selected_agent = list(agents.values())[0]
        agent_name = list(agents.keys())[0]
    else:
        console.print("[red]Error: Multiple agents found. Use --agent-name to specify[/red]")
        console.print(f"Available agents: {', '.join(agents.keys())}")
        raise typer.Exit(1)
    
    # Parse input data
    if input_data:
        try:
            input_dict = json.loads(input_data)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error: Invalid JSON in input: {e}[/red]")
            raise typer.Exit(1)
    elif input_file:
        try:
            with open(input_file, 'r') as f:
                input_dict = json.load(f)
        except Exception as e:
            console.print(f"[red]Error reading input file: {e}[/red]")
            raise typer.Exit(1)
    else:
        input_dict = {}
    
    # Execute the agent
    console.print(f"\n[bold]Executing agent:[/bold] [cyan]{agent_name}[/cyan]")
    console.print(f"[bold]Input:[/bold] {json.dumps(input_dict, indent=2)}")
    console.print("")
    
    try:
        # Run async agent
        result = asyncio.run(selected_agent(**input_dict))
        
        # Display result
        console.print("[bold green]✅ Execution successful![/bold green]")
        console.print("\n[bold]Output:[/bold]")
        console.print(Panel(
            json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result),
            title="Result",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[red]❌ Execution failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    exec_app()

