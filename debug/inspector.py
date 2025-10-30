"""
State Inspector - Runtime state inspection and breakpoints.

Features:
- State inspection
- Breakpoint management
- Variable watching
- Conditional breakpoints
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import asyncio

from teleon.core import StructuredLogger, LogLevel


@dataclass
class Breakpoint:
    """Breakpoint configuration."""
    
    name: str
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    enabled: bool = True
    hit_count: int = 0


class StateInspector:
    """
    State inspector for runtime inspection.
    
    Features:
    - Inspect agent state
    - Variable watching
    - State snapshots
    """
    
    def __init__(self):
        """Initialize state inspector."""
        self.snapshots: List[Dict[str, Any]] = []
        self.watched_vars: Dict[str, Any] = {}
        self.logger = StructuredLogger("inspector", LogLevel.DEBUG)
    
    def capture_snapshot(self, state: Dict[str, Any], label: str = ""):
        """
        Capture state snapshot.
        
        Args:
            state: State to capture
            label: Snapshot label
        """
        snapshot = {
            "label": label,
            "timestamp": asyncio.get_event_loop().time(),
            "state": dict(state)  # Copy state
        }
        self.snapshots.append(snapshot)
        
        self.logger.debug(f"Snapshot captured: {label}")
    
    def get_snapshot(self, index: int = -1) -> Optional[Dict[str, Any]]:
        """
        Get snapshot by index.
        
        Args:
            index: Snapshot index
        
        Returns:
            Snapshot or None
        """
        if 0 <= index < len(self.snapshots) or -len(self.snapshots) <= index < 0:
            return self.snapshots[index]
        return None
    
    def watch_variable(self, name: str, value: Any):
        """
        Watch a variable.
        
        Args:
            name: Variable name
            value: Variable value
        """
        old_value = self.watched_vars.get(name)
        self.watched_vars[name] = value
        
        if old_value != value:
            self.logger.info(f"Variable '{name}' changed: {old_value} → {value}")
    
    def get_watched_variables(self) -> Dict[str, Any]:
        """Get all watched variables."""
        return dict(self.watched_vars)
    
    def inspect(self, obj: Any) -> Dict[str, Any]:
        """
        Inspect an object.
        
        Args:
            obj: Object to inspect
        
        Returns:
            Inspection result
        """
        result = {
            "type": type(obj).__name__,
            "value": str(obj),
        }
        
        # Add attributes if object has __dict__
        if hasattr(obj, '__dict__'):
            result["attributes"] = {
                k: str(v)[:100]  # Limit length
                for k, v in obj.__dict__.items()
                if not k.startswith('_')
            }
        
        return result
    
    def print_snapshot(self, index: int = -1):
        """Print snapshot."""
        snapshot = self.get_snapshot(index)
        if not snapshot:
            print("No snapshot found")
            return
        
        print("\n" + "="*80)
        print(f"STATE SNAPSHOT: {snapshot['label']}")
        print("="*80)
        print(f"Timestamp: {snapshot['timestamp']:.6f}")
        print("\nState:")
        for key, value in snapshot['state'].items():
            print(f"  {key}: {value}")
        print("="*80 + "\n")
    
    def clear(self):
        """Clear all data."""
        self.snapshots.clear()
        self.watched_vars.clear()


class BreakpointManager:
    """
    Breakpoint manager for debugging.
    
    Features:
    - Named breakpoints
    - Conditional breakpoints
    - Hit counting
    - Enable/disable
    """
    
    def __init__(self):
        """Initialize breakpoint manager."""
        self.breakpoints: Dict[str, Breakpoint] = {}
        self.logger = StructuredLogger("breakpoints", LogLevel.DEBUG)
    
    def add_breakpoint(
        self,
        name: str,
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    ):
        """
        Add a breakpoint.
        
        Args:
            name: Breakpoint name
            condition: Optional condition function
        """
        self.breakpoints[name] = Breakpoint(
            name=name,
            condition=condition
        )
        self.logger.info(f"Breakpoint added: {name}")
    
    def remove_breakpoint(self, name: str):
        """Remove a breakpoint."""
        if name in self.breakpoints:
            del self.breakpoints[name]
            self.logger.info(f"Breakpoint removed: {name}")
    
    def enable_breakpoint(self, name: str):
        """Enable a breakpoint."""
        if name in self.breakpoints:
            self.breakpoints[name].enabled = True
            self.logger.info(f"Breakpoint enabled: {name}")
    
    def disable_breakpoint(self, name: str):
        """Disable a breakpoint."""
        if name in self.breakpoints:
            self.breakpoints[name].enabled = False
            self.logger.info(f"Breakpoint disabled: {name}")
    
    def check_breakpoint(
        self,
        name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if breakpoint should trigger.
        
        Args:
            name: Breakpoint name
            context: Context for condition evaluation
        
        Returns:
            True if breakpoint triggered
        """
        if name not in self.breakpoints:
            return False
        
        bp = self.breakpoints[name]
        
        if not bp.enabled:
            return False
        
        # Check condition
        if bp.condition and context:
            if not bp.condition(context):
                return False
        
        # Increment hit count
        bp.hit_count += 1
        
        self.logger.info(
            f"Breakpoint hit: {name} (count: {bp.hit_count})"
        )
        
        return True
    
    def list_breakpoints(self) -> List[Dict[str, Any]]:
        """List all breakpoints."""
        return [
            {
                "name": bp.name,
                "enabled": bp.enabled,
                "hit_count": bp.hit_count,
                "has_condition": bp.condition is not None
            }
            for bp in self.breakpoints.values()
        ]
    
    def print_breakpoints(self):
        """Print all breakpoints."""
        print("\n" + "="*80)
        print("BREAKPOINTS")
        print("="*80)
        
        if not self.breakpoints:
            print("No breakpoints set")
        else:
            for bp in self.breakpoints.values():
                status = "✓" if bp.enabled else "✗"
                cond = "conditional" if bp.condition else "unconditional"
                print(f"  {status} {bp.name:30} {cond:15} (hits: {bp.hit_count})")
        
        print("="*80 + "\n")

