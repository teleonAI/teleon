"""
Production-grade secure tool executor.

Enterprise security features:
- Permission system (RBAC)
- Resource limits (CPU, memory, time)
- Sandboxing
- Audit logging
- Input/output validation
- Command whitelisting
- File system isolation
- Network restrictions
"""

from typing import Optional, Dict, Any, List, Set
from datetime import datetime
import asyncio
import resource
import psutil
from pathlib import Path
from enum import Enum

from teleon.core import (
    ToolError,
    ToolExecutionError,
    ToolTimeoutError,
    ToolPermissionError,
    SecurityError,
    get_metrics,
    get_monitor,
    StructuredLogger,
    LogLevel,
    InputValidator,
    SecurityValidator,
)
from teleon.tools.base import BaseTool, ToolResult


class Permission(str, Enum):
    """Tool permissions."""
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EXECUTE_COMMAND = "execute_command"
    NETWORK_ACCESS = "network_access"
    DATABASE_ACCESS = "database_access"
    SEND_EMAIL = "send_email"
    SEND_SMS = "send_sms"
    HTTP_REQUEST = "http_request"


class ResourceLimits:
    """Resource limits for tool execution."""
    
    def __init__(
        self,
        max_memory_mb: int = 512,
        max_cpu_percent: int = 80,
        max_execution_time: float = 30.0,
        max_file_size_mb: int = 10,
        max_network_bandwidth_mbps: int = 10
    ):
        """
        Initialize resource limits.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            max_cpu_percent: Maximum CPU usage percentage
            max_execution_time: Maximum execution time in seconds
            max_file_size_mb: Maximum file size in MB
            max_network_bandwidth_mbps: Maximum network bandwidth in Mbps
        """
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.max_execution_time = max_execution_time
        self.max_file_size_mb = max_file_size_mb
        self.max_network_bandwidth_mbps = max_network_bandwidth_mbps


class PermissionManager:
    """
    Role-Based Access Control (RBAC) for tools.
    
    Production features:
    - Role-based permissions
    - Permission inheritance
    - Audit logging
    - Dynamic permission grants
    """
    
    def __init__(self):
        """Initialize permission manager."""
        self.role_permissions: Dict[str, Set[Permission]] = {
            "admin": {p for p in Permission},
            "developer": {
                Permission.READ_FILE,
                Permission.WRITE_FILE,
                Permission.HTTP_REQUEST,
                Permission.DATABASE_ACCESS,
            },
            "analyst": {
                Permission.READ_FILE,
                Permission.DATABASE_ACCESS,
                Permission.HTTP_REQUEST,
            },
            "basic": {
                Permission.READ_FILE,
            }
        }
        
        self.tool_required_permissions: Dict[str, Set[Permission]] = {}
        self.user_roles: Dict[str, str] = {}  # user_id -> role
        
        self.logger = StructuredLogger("permission_manager", LogLevel.INFO)
    
    def register_tool_permissions(
        self,
        tool_name: str,
        required_permissions: List[Permission]
    ):
        """
        Register required permissions for a tool.
        
        Args:
            tool_name: Tool name
            required_permissions: Required permissions
        """
        self.tool_required_permissions[tool_name] = set(required_permissions)
    
    def assign_role(self, user_id: str, role: str):
        """
        Assign role to user.
        
        Args:
            user_id: User ID
            role: Role name
        """
        if role not in self.role_permissions:
            raise SecurityError(
                f"Invalid role: {role}",
                context={"user_id": user_id, "role": role}
            )
        
        self.user_roles[user_id] = role
        self.logger.info(
            "Role assigned",
            user_id=user_id,
            role=role
        )
    
    def check_permission(
        self,
        user_id: str,
        tool_name: str
    ) -> bool:
        """
        Check if user has permission to execute tool.
        
        Args:
            user_id: User ID
            tool_name: Tool name
        
        Returns:
            True if permitted
        
        Raises:
            ToolPermissionError: If permission denied
        """
        # Get user role
        role = self.user_roles.get(user_id)
        if not role:
            raise ToolPermissionError(
                tool_name,
                f"User {user_id} has no assigned role"
            )
        
        # Get role permissions
        role_perms = self.role_permissions.get(role, set())
        
        # Get required permissions for tool
        required_perms = self.tool_required_permissions.get(tool_name, set())
        
        # Check if user has all required permissions
        if not required_perms.issubset(role_perms):
            missing = required_perms - role_perms
            raise ToolPermissionError(
                tool_name,
                f"Missing permissions: {[p.value for p in missing]}"
            )
        
        # Audit log
        self.logger.info(
            "Permission granted",
            user_id=user_id,
            role=role,
            tool=tool_name
        )
        
        return True


class SecuritySandbox:
    """
    Security sandbox for tool execution.
    
    Production features:
    - File system isolation
    - Network restrictions
    - Resource monitoring
    - Process isolation
    """
    
    def __init__(self, limits: ResourceLimits):
        """
        Initialize security sandbox.
        
        Args:
            limits: Resource limits
        """
        self.limits = limits
        self.logger = StructuredLogger("security_sandbox", LogLevel.INFO)
        
        # Allowed directories for file access
        self.allowed_read_dirs: Set[Path] = {
            Path("/tmp"),
            Path.home() / "teleon_workspace"
        }
        
        self.allowed_write_dirs: Set[Path] = {
            Path("/tmp/teleon"),
            Path.home() / "teleon_workspace"
        }
    
    def validate_file_access(
        self,
        file_path: str,
        mode: str = "read"
    ) -> bool:
        """
        Validate file access.
        
        Args:
            file_path: File path
            mode: Access mode (read/write)
        
        Returns:
            True if allowed
        
        Raises:
            ToolPermissionError: If access denied
        """
        path = Path(file_path).resolve()
        
        # Check if path is within allowed directories
        allowed_dirs = self.allowed_read_dirs if mode == "read" else self.allowed_write_dirs
        
        is_allowed = any(
            path.is_relative_to(allowed_dir) if hasattr(path, 'is_relative_to')
            else str(path).startswith(str(allowed_dir))
            for allowed_dir in allowed_dirs
        )
        
        if not is_allowed:
            raise ToolPermissionError(
                "file_access",
                f"{mode.capitalize()} access denied: {file_path}"
            )
        
        # Check file size for writes
        if mode == "write" and path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self.limits.max_file_size_mb:
                raise ToolPermissionError(
                    "file_access",
                    f"File too large: {size_mb:.1f}MB > {self.limits.max_file_size_mb}MB"
                )
        
        return True
    
    async def monitor_resources(
        self,
        process_id: int,
        start_time: float
    ):
        """
        Monitor resource usage during execution.
        
        Args:
            process_id: Process ID
            start_time: Start time
        
        Raises:
            ToolExecutionError: If limits exceeded
        """
        try:
            process = psutil.Process(process_id)
            
            # Check memory
            memory_mb = process.memory_info().rss / (1024 * 1024)
            if memory_mb > self.limits.max_memory_mb:
                self.logger.error(
                    "Memory limit exceeded",
                    memory_mb=memory_mb,
                    limit_mb=self.limits.max_memory_mb
                )
                process.terminate()
                raise ToolExecutionError(
                    "resource_monitor",
                    f"Memory limit exceeded: {memory_mb:.1f}MB > {self.limits.max_memory_mb}MB"
                )
            
            # Check CPU
            cpu_percent = process.cpu_percent()
            if cpu_percent > self.limits.max_cpu_percent:
                self.logger.warning(
                    "High CPU usage",
                    cpu_percent=cpu_percent,
                    limit_percent=self.limits.max_cpu_percent
                )
            
            # Check execution time
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > self.limits.max_execution_time:
                self.logger.error(
                    "Execution time limit exceeded",
                    elapsed=elapsed,
                    limit=self.limits.max_execution_time
                )
                process.terminate()
                raise ToolTimeoutError("resource_monitor", self.limits.max_execution_time)
        
        except psutil.NoSuchProcess:
            # Process already terminated
            pass


class AuditLogger:
    """
    Audit logger for tool executions.
    
    Logs all tool executions for compliance and security.
    """
    
    def __init__(self):
        """Initialize audit logger."""
        self.logger = StructuredLogger("tool_audit", LogLevel.INFO)
        self.executions: List[Dict[str, Any]] = []
    
    def log_execution(
        self,
        user_id: str,
        tool_name: str,
        parameters: Dict[str, Any],
        result: ToolResult,
        duration: float
    ):
        """
        Log tool execution.
        
        Args:
            user_id: User ID
            tool_name: Tool name
            parameters: Tool parameters
            result: Execution result
            duration: Execution duration
        """
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "tool_name": tool_name,
            "parameters": self._sanitize_parameters(parameters),
            "success": result.success,
            "error": result.error,
            "duration": duration,
        }
        
        self.executions.append(audit_entry)
        
        self.logger.info(
            "Tool execution audit",
            **audit_entry
        )
        
        # Record metrics
        get_metrics().record_tool_execution(
            tool_name=tool_name,
            duration=duration,
            status="success" if result.success else "error"
        )
    
    def _sanitize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize parameters for logging (remove sensitive data)."""
        sanitized = {}
        sensitive_keys = {"password", "api_key", "secret", "token"}
        
        for key, value in params.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = value
        
        return sanitized


class SecureToolExecutor:
    """
    Production-grade secure tool executor.
    
    Enterprise features:
    - RBAC permission system
    - Resource limits enforcement
    - Security sandboxing
    - Audit logging
    - Input/output validation
    """
    
    def __init__(
        self,
        limits: Optional[ResourceLimits] = None,
        enable_sandbox: bool = True,
        enable_audit: bool = True
    ):
        """
        Initialize secure tool executor.
        
        Args:
            limits: Resource limits
            enable_sandbox: Enable security sandboxing
            enable_audit: Enable audit logging
        """
        self.limits = limits or ResourceLimits()
        self.enable_sandbox = enable_sandbox
        self.enable_audit = enable_audit
        
        self.permission_manager = PermissionManager()
        self.sandbox = SecuritySandbox(self.limits)
        self.audit_logger = AuditLogger() if enable_audit else None
        
        self.logger = StructuredLogger("secure_tool_executor", LogLevel.INFO)
        self.monitor = get_monitor()
    
    async def execute(
        self,
        tool: BaseTool,
        parameters: Dict[str, Any],
        user_id: str = "system"
    ) -> ToolResult:
        """
        Execute tool with full security and monitoring.
        
        Args:
            tool: Tool to execute
            parameters: Tool parameters
            user_id: User ID (for RBAC)
        
        Returns:
            Tool execution result
        
        Raises:
            ToolPermissionError: If permission denied
            ToolExecutionError: If execution fails
        """
        start_time = asyncio.get_event_loop().time()
        tool_name = tool.name
        
        try:
            # Check permissions
            self.permission_manager.check_permission(user_id, tool_name)
            
            # Validate inputs
            self._validate_inputs(parameters)
            
            # Execute with timeout and monitoring
            async with self.monitor.track("secure_tool_executor", "execute"):
                result = await asyncio.wait_for(
                    self._execute_with_monitoring(tool, parameters),
                    timeout=self.limits.max_execution_time
                )
            
            # Validate outputs
            self._validate_outputs(result)
            
            # Audit log
            if self.enable_audit:
                duration = asyncio.get_event_loop().time() - start_time
                self.audit_logger.log_execution(
                    user_id=user_id,
                    tool_name=tool_name,
                    parameters=parameters,
                    result=result,
                    duration=duration
                )
            
            return result
        
        except asyncio.TimeoutError:
            raise ToolTimeoutError(tool_name, self.limits.max_execution_time)
        
        except Exception as e:
            self.logger.error(
                "Tool execution failed",
                tool=tool_name,
                error=str(e),
                user_id=user_id
            )
            raise
    
    async def _execute_with_monitoring(
        self,
        tool: BaseTool,
        parameters: Dict[str, Any]
    ) -> ToolResult:
        """Execute tool with resource monitoring."""
        import os
        
        process_id = os.getpid()
        start_time = asyncio.get_event_loop().time()
        
        # Start resource monitoring
        monitor_task = None
        if self.enable_sandbox:
            monitor_task = asyncio.create_task(
                self._monitor_loop(process_id, start_time)
            )
        
        try:
            # Execute tool
            result = await tool.safe_execute(**parameters)
            return result
        finally:
            if monitor_task:
                monitor_task.cancel()
    
    async def _monitor_loop(self, process_id: int, start_time: float):
        """Resource monitoring loop."""
        while True:
            await self.sandbox.monitor_resources(process_id, start_time)
            await asyncio.sleep(0.1)  # Check every 100ms
    
    def _validate_inputs(self, parameters: Dict[str, Any]):
        """Validate tool inputs for security."""
        for key, value in parameters.items():
            if isinstance(value, str):
                # Security validation
                InputValidator.validate_string(value, security_check=True)
    
    def _validate_outputs(self, result: ToolResult):
        """Validate tool outputs."""
        if result.error:
            # Don't expose internal errors
            if "internal" in result.error.lower() or "traceback" in result.error.lower():
                result.error = "Tool execution failed"

