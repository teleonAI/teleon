"""
Audit Logging Module

Tracks every action performed by AI agents for compliance and governance.
Supports both local storage and remote submission to the Teleon platform.
"""

import json
import uuid
import os
import asyncio
import threading
import atexit
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Callable
from enum import Enum

from pydantic import BaseModel, ConfigDict

# Optional httpx import for remote logging
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class ActionType(str, Enum):
    """Types of actions that can be audited"""
    REQUEST = "request"
    RESPONSE = "response"
    DEPLOYMENT = "deployment"
    CONFIGURATION = "configuration"
    DATA_ACCESS = "data_access"
    ERROR = "error"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"


class AuditLog(BaseModel):
    """Individual audit log entry"""
    id: str
    timestamp: datetime
    agent_id: str
    agent_name: str
    action_type: ActionType
    action: str
    user_id: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    duration_ms: Optional[int] = None
    status: str = "success"  # success, error, warning
    metadata: Dict[str, Any] = {}

    model_config = ConfigDict(use_enum_values=True)


class AuditLogger:
    """
    Audit Logger for tracking all agent actions.

    Supports both local storage and remote submission to the Teleon platform.
    When remote logging is enabled, logs are batched and submitted asynchronously
    for optimal performance.

    Usage:
        # Basic usage (local only)
        logger = AuditLogger(agent_id="my-agent", agent_name="Customer Support Bot")

        # With remote logging to platform (auto-detects from env vars)
        logger = AuditLogger(
            agent_id="my-agent",
            agent_name="Customer Support Bot",
            enable_remote_logging=True,  # Uses TELEON_API_URL and TELEON_API_KEY env vars
        )

        # Log a request
        logger.log_request(
            action="Process customer query",
            input_data={"query": "How do I reset my password?"},
            user_id="user-123"
        )

        # Log a response
        logger.log_response(
            action="Generated response",
            output_data={"response": "Click forgot password..."},
            duration_ms=250
        )

        # Important: Call close() to flush remaining logs before shutdown
        await logger.close()
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        storage_backend: Optional[Any] = None,
        enable_pii_detection: bool = True,
        # Remote logging configuration
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        enable_remote_logging: bool = True,
        batch_size: int = 50,
        flush_interval: float = 10.0,
    ):
        """
        Initialize the AuditLogger.

        Args:
            agent_id: Unique identifier for the agent
            agent_name: Human-readable name of the agent
            storage_backend: Optional custom storage backend for local persistence
            enable_pii_detection: Enable automatic PII redaction (default: True)
            api_url: Platform API URL (defaults to TELEON_API_URL env var)
            api_key: API key for authentication (defaults to TELEON_API_KEY env var)
            enable_remote_logging: Enable submission to platform (default: True)
            batch_size: Number of logs to batch before sending (default: 50)
            flush_interval: Seconds between automatic flushes (default: 10.0)
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.storage_backend = storage_backend
        self.enable_pii_detection = enable_pii_detection
        self.logs: List[AuditLog] = []

        # Remote logging configuration (from env vars or params)
        self.api_url = api_url or os.environ.get("TELEON_API_URL")
        self.api_key = api_key or os.environ.get("TELEON_API_KEY")
        
        # Detect if this is an agent context (service account token vs user JWT)
        # Service account tokens are JWTs that don't look like user tokens
        # We'll use the agent endpoint if the key looks like a service account token
        self._is_agent_context = self._detect_agent_context()

        # Only enable remote logging if we have the required credentials and httpx
        self.enable_remote_logging = (
            enable_remote_logging
            and bool(self.api_url)
            and bool(self.api_key)
            and HTTPX_AVAILABLE
        )

        # Batching configuration
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._pending_logs: List[AuditLog] = []
        self._pending_lock = threading.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._shutdown = False

        # Register cleanup on exit
        atexit.register(self._sync_flush_on_exit)

        # Start background flush task if remote logging is enabled
        if self.enable_remote_logging:
            self._try_start_flush_task()
    
    def _detect_agent_context(self) -> bool:
        """
        Detect if this AuditLogger is running in an agent context.
        
        Agents use service account tokens (JWT) for auth and are deployed
        with specific environment variables. User contexts use user JWT tokens.
        
        Returns:
            True if agent context, False if user context
        """
        if not self.api_key:
            return False
        
        # Check for agent-specific environment variables (set during deployment)
        # These are only present in deployed agent containers
        agent_env_indicators = [
            "TELEON_DEPLOYMENT_ID",
            "TELEON_AGENT_ID", 
            "ECS_CONTAINER_METADATA_URI",  # AWS ECS indicator
            "KUBERNETES_SERVICE_HOST",  # Kubernetes indicator
        ]
        
        for env_var in agent_env_indicators:
            if os.environ.get(env_var):
                return True
        
        # If the API key is a UUID (legacy deployment_id), it's an agent context
        import re
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        if re.match(uuid_pattern, self.api_key, re.IGNORECASE):
            return True
        
        # If we're running in a containerized environment and have TELEON_API_KEY,
        # it's likely a deployed agent (service account token)
        # Check for container indicators
        container_indicators = [
            os.environ.get("ECS_CONTAINER_METADATA_URI"),
            os.environ.get("KUBERNETES_SERVICE_HOST"),
            os.environ.get("DOCKER_CONTAINER"),
        ]
        
        if any(container_indicators) and self.api_key.startswith("eyJ"):
            # In container + JWT token = likely service account token (agent)
            return True
        
        # Default: assume user context (safer - user endpoint has more validation)
        return False

    def log(
        self,
        action_type: ActionType,
        action: str,
        user_id: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[int] = None,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditLog:
        """Log any action"""
        # Redact PII if enabled
        if self.enable_pii_detection:
            input_data = self._redact_pii(input_data) if input_data else None
            output_data = self._redact_pii(output_data) if output_data else None

        log = AuditLog(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            action_type=action_type,
            action=action,
            user_id=user_id,
            input_data=input_data,
            output_data=output_data,
            duration_ms=duration_ms,
            status=status,
            metadata=metadata or {},
        )

        # Store locally
        self.logs.append(log)
        if self.storage_backend:
            self._store_log(log)

        # Queue for remote submission (NON-BLOCKING)
        # CRITICAL: This must never block agent execution
        if self.enable_remote_logging:
            with self._pending_lock:
                self._pending_logs.append(log)
                # Trigger flush if batch is full (fire-and-forget, non-blocking)
                if len(self._pending_logs) >= self.batch_size:
                    self._try_flush_async()  # Schedules flush but returns immediately

        return log

    def log_request(
        self,
        action: str,
        input_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditLog:
        """Log an incoming request"""
        return self.log(
            action_type=ActionType.REQUEST,
            action=action,
            input_data=input_data,
            user_id=user_id,
            metadata=metadata,
        )

    def log_response(
        self,
        action: str,
        output_data: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[int] = None,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditLog:
        """Log an outgoing response"""
        return self.log(
            action_type=ActionType.RESPONSE,
            action=action,
            output_data=output_data,
            duration_ms=duration_ms,
            status=status,
            metadata=metadata,
        )

    def log_error(
        self,
        action: str,
        error: Exception,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditLog:
        """Log an error"""
        error_metadata = metadata or {}
        error_metadata.update({
            "error_type": type(error).__name__,
            "error_message": str(error),
        })

        return self.log(
            action_type=ActionType.ERROR,
            action=action,
            input_data=input_data,
            status="error",
            metadata=error_metadata,
        )

    def log_data_access(
        self,
        action: str,
        resource: str,
        operation: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditLog:
        """Log data access operations"""
        access_metadata = metadata or {}
        access_metadata.update({
            "resource": resource,
            "operation": operation,
        })

        return self.log(
            action_type=ActionType.DATA_ACCESS,
            action=action,
            user_id=user_id,
            metadata=access_metadata,
        )

    def log_configuration_change(
        self,
        action: str,
        changes: Dict[str, Any],
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditLog:
        """Log configuration changes"""
        config_metadata = metadata or {}
        config_metadata.update({"changes": changes})

        return self.log(
            action_type=ActionType.CONFIGURATION,
            action=action,
            user_id=user_id,
            metadata=config_metadata,
        )

    def get_logs(
        self,
        action_type: Optional[ActionType] = None,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditLog]:
        """Retrieve audit logs with filters"""
        filtered_logs = self.logs

        if action_type:
            filtered_logs = [log for log in filtered_logs if log.action_type == action_type]
        if user_id:
            filtered_logs = [log for log in filtered_logs if log.user_id == user_id]
        if status:
            filtered_logs = [log for log in filtered_logs if log.status == status]

        return filtered_logs[-limit:]

    def export_logs(self, format: str = "json") -> str:
        """Export logs in specified format"""
        if format == "json":
            return json.dumps([log.dict() for log in self.logs], indent=2, default=str)
        elif format == "csv":
            # Simple CSV export
            lines = ["id,timestamp,agent_id,action_type,action,user_id,status"]
            for log in self.logs:
                lines.append(
                    f"{log.id},{log.timestamp},{log.agent_id},{log.action_type},"
                    f"{log.action},{log.user_id or ''},{log.status}"
                )
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")

    async def flush(self):
        """Manually flush pending logs to the platform"""
        await self._flush_to_remote()

    async def close(self):
        """Flush remaining logs and cleanup. Call this before shutdown."""
        self._shutdown = True

        # Cancel the background flush task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush_to_remote()

    def _try_start_flush_task(self):
        """Try to start the background flush task"""
        try:
            loop = asyncio.get_running_loop()
            self._flush_task = loop.create_task(self._flush_periodically())
        except RuntimeError:
            # No running event loop - flush will happen on batch size only
            pass

    async def _flush_periodically(self):
        """Background task to periodically flush logs"""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.flush_interval)
                if not self._shutdown:
                    await self._flush_to_remote()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue
                print(f"Warning: Error in periodic flush: {e}")

    def _try_flush_async(self):
        """
        Try to trigger an async flush (NON-BLOCKING).
        
        CRITICAL: This method must never block agent execution.
        It schedules a flush task but returns immediately.
        """
        try:
            loop = asyncio.get_running_loop()
            # Fire-and-forget - create_task() schedules but doesn't block
            # _flush_to_remote() runs in background, zero latency impact
            loop.create_task(self._flush_to_remote())
        except RuntimeError:
            # No running event loop - will flush on periodic task or close()
            # This is fine - background flush will handle it
            pass

    async def _flush_to_remote(self):
        """Send pending logs to platform API"""
        if not self.enable_remote_logging or not self.api_url or not HTTPX_AVAILABLE:
            return

        # Get logs to send
        with self._pending_lock:
            if not self._pending_logs:
                return
            logs_to_send = self._pending_logs.copy()
            self._pending_logs = []

        try:
            # Prepare log entries
            log_entries = [
                {
                    "agent_id": log.agent_id,
                    "agent_name": log.agent_name,
                    "action_type": log.action_type if isinstance(log.action_type, str) else log.action_type.value,
                    "action": log.action,
                    "status": log.status,
                    "input_data": log.input_data,
                    "output_data": log.output_data,
                    "duration_ms": log.duration_ms,
                    "metadata": log.metadata,
                }
                for log in logs_to_send
            ]
            
            # Use agent endpoint if in agent context, otherwise user endpoint
            if self._is_agent_context:
                # Agent endpoint: expects list directly, uses service account token
                endpoint = f"{self.api_url}/api/v1/governance/audit-logs/batch/agent"
                payload = log_entries  # Send list directly
            else:
                # User endpoint: expects wrapped in "logs" key, uses user JWT
                endpoint = f"{self.api_url}/api/v1/governance/audit-logs/batch"
                payload = {"logs": log_entries}  # Wrap in dict

            # SECURITY: Enforce HTTPS (SOC 2 requirement)
            if not self.api_url.startswith("https://"):
                self.logger.warning(
                    "Audit log submission requires HTTPS. Disabling remote logging.",
                    extra={"api_url": self.api_url}
                )
                return
            
            # SECURITY: Create client with security settings
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(5.0, connect=2.0),  # Reduced timeout
                verify=True,  # Verify SSL certificates
                follow_redirects=False,  # Don't follow redirects (security)
            ) as client:
                response = await client.post(
                    endpoint,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "User-Agent": "Teleon-Agent-AuditLogger/1.0",  # Identify client
                    },
                )
                response.raise_for_status()
                
                # SECURITY: Log successful submission for audit trail
                self.logger.debug(
                    f"Successfully submitted {len(logs_to_send)} audit log(s)",
                    extra={"endpoint": endpoint, "status_code": response.status_code}
                )

        except httpx.HTTPStatusError as e:
            # SECURITY: Handle HTTP errors without leaking sensitive info
            status_code = e.response.status_code
            if status_code == 401:
                # Authentication failed - disable remote logging to prevent spam
                self.logger.error(
                    "Authentication failed for audit log submission. Disabling remote logging.",
                    extra={"status_code": status_code, "endpoint": endpoint}
                )
                self.enable_remote_logging = False
            elif status_code == 429:
                # Rate limited - back off
                self.logger.warning(
                    "Rate limited for audit log submission. Will retry later.",
                    extra={"status_code": status_code}
                )
                # Re-queue with backoff
                with self._pending_lock:
                    if len(self._pending_logs) < 1000:
                        self._pending_logs = logs_to_send + self._pending_logs
            else:
                # Other HTTP errors - re-queue
                self.logger.warning(
                    f"HTTP error submitting audit logs: {status_code}",
                    extra={"status_code": status_code, "endpoint": endpoint}
                )
                with self._pending_lock:
                    if len(self._pending_logs) < 1000:
                        self._pending_logs = logs_to_send + self._pending_logs
        except httpx.TimeoutException:
            # SECURITY: Timeout - re-queue but don't spam
            self.logger.warning("Timeout submitting audit logs. Will retry later.")
            with self._pending_lock:
                if len(self._pending_logs) < 1000:
                    self._pending_logs = logs_to_send + self._pending_logs
        except Exception as e:
            # SECURITY: Generic error handling - don't leak sensitive info
            self.logger.warning(
                "Failed to submit audit logs to platform",
                exc_info=True,
                extra={"error_type": type(e).__name__, "endpoint": endpoint}
            )
            # Re-queue failed logs (with limit to prevent memory issues)
            with self._pending_lock:
                if len(self._pending_logs) < 1000:
                    # Add back to front of queue
                    self._pending_logs = logs_to_send + self._pending_logs

    def _sync_flush_on_exit(self):
        """Synchronous flush for atexit handler"""
        if not self._pending_logs or not self.enable_remote_logging:
            return

        try:
            # Try to run async flush synchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._flush_to_remote())
            finally:
                loop.close()
        except Exception as e:
            print(f"Warning: Failed to flush audit logs on exit: {e}")

    def _redact_pii(self, data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Redact personally identifiable information"""
        if not data:
            return data

        # Simple PII detection (can be enhanced with ML models)
        pii_patterns = [
            "email", "phone", "ssn", "credit_card", "password",
            "address", "zip_code", "postal_code"
        ]

        redacted = data.copy()
        for key in redacted:
            if isinstance(redacted[key], str):
                for pattern in pii_patterns:
                    if pattern in key.lower():
                        redacted[key] = "[REDACTED]"
                        break

        return redacted

    def _store_log(self, log: AuditLog):
        """Store log in backend (Redis, PostgreSQL, etc.)"""
        if self.storage_backend:
            # Store in configured backend
            try:
                self.storage_backend.store(log.dict())
            except Exception as e:
                print(f"Failed to store audit log: {e}")


def create_audit_logger(
    agent_id: str,
    agent_name: str,
    storage_backend: Optional[Any] = None,
    enable_pii_detection: bool = True,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    enable_remote_logging: bool = True,
    batch_size: int = 50,
    flush_interval: float = 10.0,
) -> AuditLogger:
    """
    Factory function to create an audit logger.

    Args:
        agent_id: Unique identifier for the agent
        agent_name: Human-readable name of the agent
        storage_backend: Optional custom storage backend
        enable_pii_detection: Enable automatic PII redaction (default: True)
        api_url: Platform API URL (defaults to TELEON_API_URL env var)
        api_key: API key for authentication (defaults to TELEON_API_KEY env var)
        enable_remote_logging: Enable submission to platform (default: True)
        batch_size: Number of logs to batch before sending (default: 50)
        flush_interval: Seconds between automatic flushes (default: 10.0)

    Returns:
        Configured AuditLogger instance
    """
    return AuditLogger(
        agent_id=agent_id,
        agent_name=agent_name,
        storage_backend=storage_backend,
        enable_pii_detection=enable_pii_detection,
        api_url=api_url,
        api_key=api_key,
        enable_remote_logging=enable_remote_logging,
        batch_size=batch_size,
        flush_interval=flush_interval,
    )
