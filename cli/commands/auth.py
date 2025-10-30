"""
Authentication commands for Teleon CLI.

Commands for logging in and managing authentication.
"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from pathlib import Path
import json
import os
import webbrowser
import http.server
import socketserver
import threading
from urllib.parse import parse_qs, urlparse
import time

app = typer.Typer(
    name="auth",
    help="Authentication commands",
    add_completion=False
)

console = Console()

# Global variable to store received API key
_received_api_key = None
_received_data = None


class CallbackHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler for OAuth callback"""
    
    def do_GET(self):
        """Handle GET request from callback"""
        global _received_api_key, _received_data
        
        # Parse query parameters
        parsed_path = urlparse(self.path)
        params = parse_qs(parsed_path.query)
        
        # Check if we got an API key
        if 'api_key' in params:
            _received_api_key = params['api_key'][0]
            _received_data = {
                'api_key': _received_api_key,
                'user_email': params.get('email', [''])[0],
                'user_name': params.get('name', [''])[0],
                'project_id': params.get('project_id', [''])[0],
                'project_name': params.get('project_name', [''])[0],
            }
            
            # Send success response
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            success_html = """
            <html>
            <head><title>Teleon Authentication</title></head>
            <body style="font-family: -apple-system, sans-serif; text-align: center; padding: 50px; background: #0a0a0a; color: #e0e0e0;">
                <h1 style="color: #00d4ff;">✓ Authentication Successful!</h1>
                <p>You can now close this window and return to your terminal.</p>
                <p style="color: #888;">Your API key has been securely saved.</p>
            </body>
            </html>
            """
            self.wfile.write(success_html.encode())
        else:
            # Send error response
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            error_html = """
            <html>
            <head><title>Teleon Authentication</title></head>
            <body style="font-family: -apple-system, sans-serif; text-align: center; padding: 50px; background: #0a0a0a; color: #e0e0e0;">
                <h1 style="color: #ff4444;">✗ Authentication Failed</h1>
                <p>No API key received. Please try again.</p>
            </body>
            </html>
            """
            self.wfile.write(error_html.encode())
    
    def log_message(self, format, *args):
        """Suppress log messages"""
        pass


def start_callback_server(port=8765):
    """Start local callback server"""
    handler = CallbackHandler
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    
    # Run in thread
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    return httpd


@app.command()
def login(
    api_key: str = typer.Option(None, "--api-key", help="Your Teleon API key (skip browser login)"),
    browser: bool = typer.Option(True, "--browser/--no-browser", help="Open browser for login"),
):
    """
    Authenticate with Teleon Platform.
    
    By default, opens your browser to log in to the platform.
    Alternatively, you can provide an API key directly.
    
    Examples:
        teleon login                          # Open browser login
        teleon login --api-key tlk_live_xxx   # Manual API key
        teleon login --no-browser             # Skip browser, manual entry
    """
    console.print(Panel.fit(
        "[bold cyan]Teleon Authentication[/bold cyan]",
        title="🔐 Login"
    ))
    
    # If API key provided directly, skip browser flow
    if api_key:
        _save_credentials(api_key, {})
        return
    
    # Browser-based OAuth flow
    if browser:
        global _received_api_key, _received_data
        _received_api_key = None
        _received_data = None
        
        # Get login URL from environment or use default (deployed platform)
        login_base_url = os.getenv('TELEON_LOGIN_URL', 'https://dashboard.teleon.ai/login')
        callback_port = 8765
        callback_url = f"http://127.0.0.1:{callback_port}/callback"
        
        # Start local callback server
        console.print("\n[dim]Starting local callback server...[/dim]")
        try:
            httpd = start_callback_server(callback_port)
        except OSError as e:
            console.print(f"[red]✗ Failed to start callback server on port {callback_port}[/red]")
            console.print(f"[yellow]Try manual login: teleon login --no-browser[/yellow]")
            raise typer.Exit(1)
        
        # Build login URL with callback
        login_url = f"{login_base_url}?callback={callback_url}&cli=true"
        
        console.print(f"\n[bold]Opening browser for authentication...[/bold]")
        console.print(f"[dim]URL: {login_url}[/dim]\n")
        
        # Open browser
        if not webbrowser.open(login_url):
            console.print("[yellow]⚠️  Could not open browser automatically[/yellow]")
            console.print(f"\n[bold]Please open this URL in your browser:[/bold]")
            console.print(f"[cyan]{login_url}[/cyan]\n")
        
        console.print("[dim]Waiting for authentication... (Press Ctrl+C to cancel)[/dim]\n")
        
        # Wait for callback
        timeout = 300  # 5 minutes
        start_time = time.time()
        
        try:
            while _received_api_key is None:
                if time.time() - start_time > timeout:
                    console.print("[red]✗ Authentication timed out[/red]")
                    httpd.shutdown()
                    raise typer.Exit(1)
                
                time.sleep(0.5)
            
            # Got the API key!
            httpd.shutdown()
            
            console.print("[green]✓ Received API key from browser![/green]\n")
            
            # Save credentials
            _save_credentials(_received_api_key, _received_data)
            
        except KeyboardInterrupt:
            httpd.shutdown()
            console.print("\n[yellow]Authentication cancelled[/yellow]")
            raise typer.Exit(0)
    
    else:
        # Manual entry
        console.print("\n[bold]Enter your Teleon API key:[/bold]")
        console.print(f"[dim]Get your API key from: {os.getenv('TELEON_LOGIN_URL', 'https://dashboard.teleon.ai')}[/dim]\n")
        api_key = Prompt.ask("API Key", password=True)
        
        if not api_key:
            console.print("[red]✗ No API key provided[/red]")
            raise typer.Exit(1)
        
        _save_credentials(api_key, {})


def _save_credentials(api_key: str, extra_data: dict):
    """Save API key and credentials to config file"""
    
    # Validate format
    if not api_key.startswith(('tlk_live_', 'tlk_test_', 'teleon_')):
        console.print("[yellow]⚠️  Warning: API key format doesn't match expected pattern[/yellow]")
        console.print("[dim]Expected: tlk_live_xxx or tlk_test_xxx[/dim]")
        
        if not Confirm.ask("Continue anyway?", default=False):
            raise typer.Exit(0)
    
    # Save to config
    config_dir = Path.home() / ".teleon"
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "config.json"
    
    # Load existing config or create new
    if config_file.exists():
        try:
            config = json.loads(config_file.read_text())
        except:
            config = {}
    else:
        config = {}
    
    # Update with new token and data
    config["auth_token"] = api_key
    if extra_data.get('user_email'):
        config["user_email"] = extra_data['user_email']
    if extra_data.get('user_name'):
        config["user_name"] = extra_data['user_name']
    if extra_data.get('project_id'):
        config["default_project_id"] = extra_data['project_id']
    if extra_data.get('project_name'):
        config["default_project_name"] = extra_data['project_name']
    
    # Save
    try:
        config_file.write_text(json.dumps(config, indent=2))
        config_file.chmod(0o600)  # Secure permissions
        
        console.print("\n[green]✓ Successfully authenticated![/green]")
        console.print(f"[dim]Config saved to: {config_file}[/dim]")
        
        if extra_data.get('user_email'):
            console.print(f"\n[bold]Logged in as:[/bold] [cyan]{extra_data['user_email']}[/cyan]")
        if extra_data.get('project_name'):
            console.print(f"[bold]Default project:[/bold] [cyan]{extra_data['project_name']}[/cyan]")
        
        console.print("\n[bold]You can now:[/bold]")
        console.print("  • Deploy agents: [cyan]teleon deploy[/cyan]")
        console.print("  • View status: [cyan]teleon auth status[/cyan]")
        
    except Exception as e:
        console.print(f"[red]✗ Failed to save config: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def logout():
    """
    Log out from Teleon Platform.
    
    This removes your stored API key.
    
    Example:
        teleon logout
    """
    config_file = Path.home() / ".teleon" / "config.json"
    
    if not config_file.exists():
        console.print("[yellow]Not logged in[/yellow]")
        return
    
    try:
        config = json.loads(config_file.read_text())
        if "auth_token" in config:
            del config["auth_token"]
            config_file.write_text(json.dumps(config, indent=2))
        
        console.print("[green]✓ Logged out successfully[/green]")
        
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def status():
    """
    Check authentication status.
    
    Example:
        teleon auth status
    """
    console.print(Panel.fit(
        "[bold cyan]Authentication Status[/bold cyan]",
        title="🔐 Auth"
    ))
    
    # Check environment variable
    env_token = os.getenv("TELEON_AUTH_TOKEN")
    if env_token:
        console.print("\n[green]✓ Authenticated via environment variable[/green]")
        console.print(f"[dim]Token: {env_token[:20]}...[/dim]")
        return
    
    # Check config file
    config_file = Path.home() / ".teleon" / "config.json"
    if config_file.exists():
        try:
            config = json.loads(config_file.read_text())
            if config.get("auth_token"):
                token = config["auth_token"]
                console.print("\n[green]✓ Authenticated[/green]")
                console.print(f"[dim]Token: {token[:20]}...[/dim]")
                
                if config.get("user_email"):
                    console.print(f"[bold]User:[/bold] {config['user_email']}")
                if config.get("default_project_name"):
                    console.print(f"[bold]Default Project:[/bold] {config['default_project_name']}")
                
                console.print(f"\n[dim]Config: {config_file}[/dim]")
                return
        except:
            pass
    
    console.print("\n[yellow]✗ Not authenticated[/yellow]")
    console.print("\nRun: [cyan]teleon login[/cyan]")


@app.command(name="whoami")
def whoami():
    """
    Show current authenticated user info.
    
    Example:
        teleon auth whoami
    """
    console.print("[dim]Checking authentication...[/dim]\n")
    
    # Get token
    token = os.getenv("TELEON_AUTH_TOKEN")
    if not token:
        config_file = Path.home() / ".teleon" / "config.json"
        if config_file.exists():
            try:
                config = json.loads(config_file.read_text())
                token = config.get("auth_token")
            except:
                pass
    
    if not token:
        console.print("[yellow]Not authenticated[/yellow]")
        console.print("\nRun: [cyan]teleon login[/cyan]")
        return
    
    console.print("[bold]Authenticated as:[/bold]")
    console.print(f"  API Key: [cyan]{token[:20]}...[/cyan]")
    
    # Could make API call here to get user info
    console.print("\n[dim]To see full user details, use the dashboard[/dim]")


if __name__ == "__main__":
    app()

