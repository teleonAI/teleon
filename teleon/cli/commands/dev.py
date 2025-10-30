"""Dev command for Teleon CLI."""

import typer
from rich.console import Console
from pathlib import Path
from typing import Optional
import asyncio
import importlib.util
import sys

console = Console()

dev = typer.Typer(help="Development server commands")


@dev.command()
def start():
    """
    Start local development server with automatic agent discovery.

    The dev server automatically discovers all Teleon agents in your project
    and provides:
    - Dashboard UI at http://127.0.0.1:8000
    - Individual agent endpoints with dedicated docs
    - Shared endpoints for all agents
    - API key management for testing

    Examples:
        teleon dev start        # Start server with auto-discovery

    Features:
    - Automatic agent discovery (no code required!)
    - Each agent gets its own endpoint and API docs
    - Shared endpoint for all agents
    - Beautiful dashboard interface

    Note: You can also run directly with:
        python -m teleon.dev_server.run
    """
    # Just run the dev server runner module
    import subprocess
    import sys
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "teleon.dev_server.run"],
            cwd=Path.cwd()
        )
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        console.print("\n[yellow]🛑 Server stopped[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    dev()

