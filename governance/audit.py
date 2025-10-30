"""
Audit Logging Module

Tracks every action performed by AI agents for compliance and governance.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, List
from enum import Enum

from pydantic import BaseModel


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
    
    class Config:
        use_enum_values = True


class AuditLogger:
    """
    Audit Logger for tracking all agent actions
    
    Usage:
        logger = AuditLogger(agent_id="my-agent", agent_name="Customer Support Bot")
        
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
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        storage_backend: Optional[Any] = None,
        enable_pii_detection: bool = True,
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.storage_backend = storage_backend
        self.enable_pii_detection = enable_pii_detection
        self.logs: List[AuditLog] = []
    
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
            timestamp=datetime.utcnow(),
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
        
        # Store log
        self.logs.append(log)
        if self.storage_backend:
            self._store_log(log)
        
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
) -> AuditLogger:
    """Factory function to create an audit logger"""
    return AuditLogger(
        agent_id=agent_id,
        agent_name=agent_name,
        storage_backend=storage_backend,
        enable_pii_detection=enable_pii_detection,
    )

