"""
Agent Scanner - Automatically discover Teleon agents in a project.

This module scans Python files to find TeleonClient instances and their registered agents.
Users don't need to write any discovery code - it's all automatic!
"""

import importlib.util
import inspect
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set


def discover_agents(
    directory: Optional[Path] = None,
    exclude_files: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Auto-discover Teleon agents from Python files.

    Scans all Python files in the given directory (or current directory) for
    TeleonClient agents and returns them as a registry.

    Args:
        directory: Directory to scan (defaults to current directory)
        exclude_files: List of file names to exclude from scanning
        verbose: Show detailed scanning output (default: False)

    Returns:
        Dictionary of discovered agents {agent_id: agent_info}

    Example:
        >>> from teleon.discovery import discover_agents
        >>> agents = discover_agents()
        >>> print(f"Found {len(agents)} agents")
    """
    if directory is None:
        directory = Path(".")

    if exclude_files is None:
        exclude_files = [
            "setup.py",
            "test_*.py",
            "conftest.py",
            "__pycache__",
        ]

    discovered_agents = {}
    api_key_usage = {}  # Track {api_key: [files]}
    scanned_files = []
    failed_files = []
    
    # Suppress TeleonClient verbose output during discovery
    old_teleon_quiet = os.environ.get('TELEON_QUIET')
    os.environ['TELEON_QUIET'] = '1'
    
    try:
        # Find all Python files
        python_files = list(directory.glob("**/*.py"))

        # Filter out excluded files
        python_files = [
            f for f in python_files
            if not (
                f.name.startswith("__") or
                f.name.startswith("test_") or
                any(exclude in str(f) for exclude in exclude_files)
            )
        ]

        print(f"📂 Scanning {len(python_files)} Python files...")
        
        for py_file in python_files:
            try:
                if verbose:
                    print(f"  • {py_file.name}...", end=" ")
                
                # Load the module
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                if spec is None or spec.loader is None:
                    if verbose:
                        print("⏭️  skipped")
                    continue

                module = importlib.util.module_from_spec(spec)
                
                # Add module's directory to sys.path temporarily
                module_dir = str(py_file.parent.absolute())
                if module_dir not in sys.path:
                    sys.path.insert(0, module_dir)

                try:
                    spec.loader.exec_module(module)
                finally:
                    # Clean up sys.path
                    if module_dir in sys.path:
                        sys.path.remove(module_dir)

                # Look for TeleonClient instances and their agents
                file_has_agents = False
                for name, obj in inspect.getmembers(module):
                    if (hasattr(obj, '__class__') and
                        obj.__class__.__name__ == 'TeleonClient' and
                        hasattr(obj, 'agents')):
                        
                        file_has_agents = True
                        
                        # Track API key usage
                        api_key = getattr(obj, 'api_key', None)
                        if api_key:
                            key_preview = f"{api_key[:15]}..." if len(api_key) > 15 else api_key
                            if key_preview not in api_key_usage:
                                api_key_usage[key_preview] = {
                                    'files': [],
                                    'environment': getattr(obj, 'environment', 'unknown'),
                                    'scopes': getattr(obj, 'scopes', []),
                                    'user_id': getattr(obj, 'user_id', 'unknown')
                                }
                            api_key_usage[key_preview]['files'].append(py_file.name)

                        # Add all agents from this client
                        for agent_id, agent_info in obj.agents.items():
                            discovered_agents[agent_id] = agent_info
                            agent_info['source_file'] = py_file.name
                
                if file_has_agents:
                    scanned_files.append(py_file.name)
                    if verbose:
                        print("✅")
                elif verbose:
                    print("⏭️  no agents")

            except ValueError as e:
                # API key validation errors should be shown to the user
                error_msg = str(e)
                if "API key" in error_msg or "api_key" in error_msg.lower():
                    failed_files.append((py_file.name, error_msg))
                    if verbose:
                        print("❌")
                    # Store the error but continue scanning other files
                    continue
                # Other ValueErrors can be skipped
                if verbose:
                    print("⚠️  error")
                continue
            except Exception as e:
                # Silently skip files that can't be imported
                if verbose:
                    print(f"⚠️  {type(e).__name__}")
                continue
        
        # Print clean summary
        print()
        print("=" * 80)
        
        if failed_files:
            print("❌ API KEY VALIDATION ERRORS")
            print("=" * 80)
            for filename, error in failed_files:
                print(f"\n📄 {filename}:")
                # Extract just the main error message
                first_line = error.split('\n')[0]
                print(f"   {first_line}")
            print("\n" + "=" * 80)
            raise ValueError("API key validation failed. Fix the errors above and try again.")
        
        if discovered_agents:
            print(f"✅ DISCOVERY COMPLETE")
            print("=" * 80)
            print(f"\n📊 Summary:")
            print(f"   • Files scanned: {len(scanned_files)}")
            print(f"   • Agents found: {len(discovered_agents)}")
            print(f"   • API keys used: {len(api_key_usage)}")
            
            # Show API key usage details
            if len(api_key_usage) > 1:
                print(f"\n⚠️  Multiple API keys detected:")
                for key_preview, info in api_key_usage.items():
                    env = info['environment']
                    files = info['files']
                    scopes = info['scopes']
                    print(f"\n   🔑 {key_preview}")
                    print(f"      Environment: {env}")
                    if scopes:
                        print(f"      Scopes: {', '.join(scopes[:3])}{'...' if len(scopes) > 3 else ''}")
                    print(f"      Used in: {', '.join(files)}")
            else:
                # Single API key
                for key_preview, info in api_key_usage.items():
                    env = info['environment']
                    scopes = info['scopes']
                    print(f"\n   🔑 API Key: {key_preview}")
                    print(f"      Environment: {env}")
                    if scopes:
                        print(f"      Scopes: {', '.join(scopes)}")
            
            # Show agents by file
            print(f"\n📦 Discovered Agents:")
            agents_by_file = {}
            for agent_id, info in discovered_agents.items():
                source = info.get('source_file', 'unknown')
                if source not in agents_by_file:
                    agents_by_file[source] = []
                agents_by_file[source].append(info)
            
            for filename, agents in sorted(agents_by_file.items()):
                print(f"\n   📄 {filename} ({len(agents)} agent{'s' if len(agents) != 1 else ''})")
                for agent in agents:
                    # Get agent ID (find it from discovered_agents)
                    agent_id = None
                    for aid, info in discovered_agents.items():
                        if info['name'] == agent['name'] and info.get('source_file') == filename:
                            agent_id = aid
                            break
                    
                    # Neon cyan color for agent IDs: \033[96m text \033[0m (reset)
                    if agent_id:
                        # Shorten the agent_id for display
                        short_id = agent_id.replace('agent_', '')[:12]
                        print(f"      • {agent['name']} \033[96m[{short_id}]\033[0m ({agent['model']})")
                    else:
                        print(f"      • {agent['name']} ({agent['model']})")
        
        print()
        
    finally:
        # Restore original environment
        if old_teleon_quiet is None:
            os.environ.pop('TELEON_QUIET', None)
        else:
            os.environ['TELEON_QUIET'] = old_teleon_quiet

    return discovered_agents


def get_agent_summary(agents: Dict[str, Any]) -> str:
    """
    Get a human-readable summary of discovered agents.

    Args:
        agents: Dictionary of agents from discover_agents()

    Returns:
        Formatted string summary
    """
    if not agents:
        return "No agents found"

    lines = [f"Found {len(agents)} agent(s):\n"]
    for agent_id, info in agents.items():
        lines.append(f"  • {info['name']}")
        lines.append(f"    ID: {agent_id}")
        lines.append(f"    Model: {info['model']}")
        lines.append(f"    Description: {info['description'][:60]}...")
        lines.append("")

    return "\n".join(lines)

