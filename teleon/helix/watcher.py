"""
File Watcher - Hot reload for development.

Features:
- Watch file changes
- Automatic agent restart
- Configurable watch paths
- Debouncing
"""

from typing import List, Set, Optional, Dict
from pathlib import Path
from pydantic import BaseModel, Field
import asyncio
import time
import hashlib

from teleon.core import (
    StructuredLogger,
    LogLevel,
)


class WatcherConfig(BaseModel):
    """File watcher configuration."""
    
    watch_paths: List[str] = Field(default_factory=list, description="Paths to watch")
    ignore_patterns: List[str] = Field(
        default_factory=lambda: ["__pycache__", "*.pyc", ".git", "venv"],
        description="Patterns to ignore"
    )
    debounce_seconds: float = Field(1.0, ge=0.1, description="Debounce delay")


class FileWatcher:
    """
    Hot reload file watcher.
    
    Features:
    - Recursive directory watching
    - Change debouncing
    - Automatic restart trigger
    """
    
    def __init__(self, config: WatcherConfig, runtime):
        """
        Initialize file watcher.
        
        Args:
            config: Watcher configuration
            runtime: AgentRuntime instance
        """
        self.config = config
        self.runtime = runtime
        
        # File hashes for change detection
        self.file_hashes: Dict[Path, str] = {}
        
        # Pending changes (for debouncing)
        self.pending_changes: Set[Path] = set()
        self.last_change_time: Optional[float] = None
        
        self.logger = StructuredLogger("file_watcher", LogLevel.INFO)
        self.running = False
    
    def _should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored."""
        path_str = str(path)
        
        for pattern in self.config.ignore_patterns:
            if pattern.startswith("*"):
                if path_str.endswith(pattern[1:]):
                    return True
            else:
                if pattern in path_str:
                    return True
        
        return False
    
    def _get_file_hash(self, path: Path) -> str:
        """Get file content hash."""
        try:
            with open(path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""
    
    def _scan_files(self) -> List[Path]:
        """Scan all watched files."""
        files = []
        
        for watch_path_str in self.config.watch_paths:
            watch_path = Path(watch_path_str)
            
            if not watch_path.exists():
                continue
            
            if watch_path.is_file():
                if not self._should_ignore(watch_path):
                    files.append(watch_path)
            else:
                # Recursively scan directory
                for file_path in watch_path.rglob("*.py"):
                    if not self._should_ignore(file_path):
                        files.append(file_path)
        
        return files
    
    async def _check_changes(self) -> Set[Path]:
        """Check for file changes."""
        changed_files = set()
        
        current_files = self._scan_files()
        
        # Check for changes in existing files
        for file_path in current_files:
            current_hash = self._get_file_hash(file_path)
            previous_hash = self.file_hashes.get(file_path)
            
            if previous_hash and current_hash != previous_hash:
                changed_files.add(file_path)
            
            self.file_hashes[file_path] = current_hash
        
        # Check for deleted files
        deleted_files = set(self.file_hashes.keys()) - set(current_files)
        for file_path in deleted_files:
            del self.file_hashes[file_path]
            changed_files.add(file_path)
        
        return changed_files
    
    async def _handle_changes(self, changed_files: Set[Path]):
        """Handle file changes."""
        if not changed_files:
            return
        
        self.logger.info(
            "Files changed, triggering reload",
            files=[str(f) for f in changed_files]
        )
        
        # Restart all running agents
        for agent_id in list(self.runtime.agents.keys()):
            agent_info = self.runtime.agents[agent_id]
            if agent_info["processes"]:
                self.logger.info(f"Restarting agent: {agent_id}")
                await self.runtime.restart_agent(agent_id)
    
    async def start(self):
        """Start watching for changes."""
        if self.running:
            return
        
        self.running = True
        self.logger.info(
            "File watcher started",
            watch_paths=self.config.watch_paths
        )
        
        # Initial scan
        self._scan_files()
        for file_path in self.file_hashes.keys():
            self.file_hashes[file_path] = self._get_file_hash(file_path)
        
        # Watch loop
        while self.running:
            try:
                # Check for changes
                changed_files = await self._check_changes()
                
                if changed_files:
                    # Add to pending changes
                    self.pending_changes.update(changed_files)
                    self.last_change_time = time.time()
                
                # Check if debounce period has passed
                if self.pending_changes and self.last_change_time:
                    time_since_change = time.time() - self.last_change_time
                    
                    if time_since_change >= self.config.debounce_seconds:
                        # Handle pending changes
                        await self._handle_changes(self.pending_changes)
                        self.pending_changes.clear()
                        self.last_change_time = None
                
                # Wait before next check
                await asyncio.sleep(0.5)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Watcher error: {e}")
                await asyncio.sleep(1)
        
        self.logger.info("File watcher stopped")
    
    async def stop(self):
        """Stop watching."""
        self.running = False

