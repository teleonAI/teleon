#!/usr/bin/env python3
"""
Simple runner script for Teleon Dev Server.

This is a fallback in case the CLI has issues.
Users can run this directly: python -m teleon.dev_server.run
"""

import sys
import uvicorn
from pathlib import Path
from teleon.dev_server import create_dev_server
from teleon.discovery import discover_agents


def main():
    """Run the Teleon development server."""
    print("\n" + "=" * 80)
    print("🚀 TELEON DEVELOPMENT SERVER")
    print("=" * 80)
    print()
    
    # Auto-discover agents (with clean output)
    try:
        agents = discover_agents()
    except ValueError as e:
        # API key validation failed - error already shown by discover_agents()
        print("\n💡 Quick fix:")
        print("   • For local development: use environment='dev'")
        print("   • For production: get API key from https://dashboard.teleon.ai")
        print()
        return 1
    
    if not agents:
        print("=" * 80)
        print("ℹ️  NO AGENTS FOUND")
        print("=" * 80)
        print()
        print("Create agents in your Python files:")
        print()
        print("   from teleon import TeleonClient")
        print("   client = TeleonClient(api_key='your-key', environment='dev')")
        print()
        print("   @client.agent(name='my-agent')")
        print("   def my_agent(input: str) -> str:")
        print("       return 'Hello!'")
        print()
        print("=" * 80)
        print()
    
    print("=" * 80)
    print("🌐 SERVER ENDPOINTS")
    print("=" * 80)
    print()
    print("   📍 Dashboard:   http://127.0.0.1:8000")
    print("   📖 API Docs:    http://127.0.0.1:8000/docs")
    
    if agents:
        print("\n   🤖 Agent Endpoints:")
        # Group agents by file for organized display
        agents_by_file = {}
        for agent_id, info in agents.items():
            source = info.get('source_file', 'unknown')
            if source not in agents_by_file:
                agents_by_file[source] = []
            agents_by_file[source].append((agent_id, info))
        
        for filename, file_agents in sorted(agents_by_file.items()):
            for agent_id, info in file_agents:
                short_id = agent_id.replace('agent_', '')[:12]
                # Neon cyan for agent IDs, neon green for URLs
                print(f"      • {info['name']} \033[96m[{short_id}]\033[0m")
                print(f"        \033[92mhttp://127.0.0.1:8000/{agent_id}/docs\033[0m")
    
    print()
    print("=" * 80)
    print()
    
    # Create and run server (pass pre-discovered agents to avoid double scanning)
    app = create_dev_server(discovered_agents=agents)
    
    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)

