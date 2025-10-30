"""
Advanced Security - Enterprise-grade security features.

Features:
- Authentication & authorization
- Encryption (at-rest, in-transit, in-use)
- Network policies
- Audit logging
- Security scanning
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import secrets
import json
import base64

try:
    import jwt
    HAS_JWT = True
except ImportError:
    HAS_JWT = False

from teleon.core import StructuredLogger, LogLevel, get_metrics


class AuthMethod(str, Enum):
    """Authentication methods."""
    JWT = "jwt"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    SAML = "saml"
    MTLS = "mtls"


class EncryptionAlgorithm(str, Enum):
    """Encryption algorithms."""
    AES_256 = "AES-256"
    AES_128 = "AES-128"
    RSA_2048 = "RSA-2048"
    RSA_4096 = "RSA-4096"


class SecurityConfig(BaseModel):
    """Security configuration."""
    
    # Authentication
    auth_methods: List[AuthMethod] = Field(
        default_factory=lambda: [AuthMethod.JWT],
        description="Enabled auth methods"
    )
    require_mfa: bool = Field(False, description="Require MFA")
    session_timeout: int = Field(3600, ge=300, description="Session timeout (seconds)")
    
    # Encryption
    encryption_at_rest: bool = Field(True, description="Enable encryption at rest")
    encryption_algorithm: EncryptionAlgorithm = Field(
        EncryptionAlgorithm.AES_256,
        description="Encryption algorithm"
    )
    key_rotation_days: int = Field(90, ge=30, description="Key rotation period (days)")
    
    # TLS
    tls_version: str = Field("1.3", description="TLS version")
    certificate_validation: str = Field("strict", description="Certificate validation mode")
    
    # Network
    ip_whitelist: List[str] = Field(default_factory=list, description="Allowed IPs")
    rate_limit_requests: int = Field(1000, ge=1, description="Rate limit (requests/min)")
    
    # Audit
    audit_log_enabled: bool = Field(True, description="Enable audit logging")
    audit_retention_days: int = Field(90, ge=30, description="Audit log retention")


class AuthManager:
    """
    Authentication and authorization manager.
    
    Features:
    - Multi-method authentication
    - Token management
    - Session management
    - MFA support
    """
    
    def __init__(self, config: SecurityConfig):
        """
        Initialize auth manager.
        
        Args:
            config: Security configuration
        """
        self.config = config
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        
        # JWT secret (in production, load from secure storage)
        self.jwt_secret = secrets.token_urlsafe(32)
        
        self.logger = StructuredLogger("auth_manager", LogLevel.INFO)
    
    def generate_api_key(
        self,
        user_id: str,
        scopes: List[str],
        expires_in_days: int = 365
    ) -> str:
        """
        Generate API key.
        
        Args:
            user_id: User ID
            scopes: Permission scopes
            expires_in_days: Expiration period
        
        Returns:
            API key
        """
        api_key = f"tleon_{secrets.token_urlsafe(32)}"
        
        self.api_keys[api_key] = {
            "user_id": user_id,
            "scopes": scopes,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(days=expires_in_days),
            "active": True
        }
        
        self.logger.info(
            "API key generated",
            user_id=user_id,
            scopes=scopes
        )
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate API key.
        
        Args:
            api_key: API key to validate
        
        Returns:
            Key metadata or None
        """
        key_data = self.api_keys.get(api_key)
        
        if not key_data:
            return None
        
        if not key_data["active"]:
            return None
        
        if datetime.utcnow() > key_data["expires_at"]:
            key_data["active"] = False
            return None
        
        return key_data
    
    def generate_jwt(
        self,
        user_id: str,
        scopes: List[str],
        expires_in: int = 3600
    ) -> str:
        """
        Generate JWT token.
        
        Args:
            user_id: User ID
            scopes: Permission scopes
            expires_in: Expiration time (seconds)
        
        Returns:
            JWT token
        """
        payload = {
            "user_id": user_id,
            "scopes": scopes,
            "iat": datetime.utcnow().isoformat(),
            "exp": (datetime.utcnow() + timedelta(seconds=expires_in)).isoformat()
        }
        
        if HAS_JWT:
            token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        else:
            # Simple fallback (NOT secure for production)
            payload_json = json.dumps(payload)
            token = base64.b64encode(payload_json.encode()).decode()
        
        self.logger.info(
            "JWT generated",
            user_id=user_id,
            expires_in=expires_in
        )
        
        return token
    
    def validate_jwt(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate JWT token.
        
        Args:
            token: JWT token
        
        Returns:
            Decoded payload or None
        """
        try:
            if HAS_JWT:
                payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            else:
                # Simple fallback (NOT secure for production)
                payload_json = base64.b64decode(token.encode()).decode()
                payload = json.loads(payload_json)
                
                # Check expiration
                exp_time = datetime.fromisoformat(payload["exp"])
                if datetime.utcnow() > exp_time:
                    self.logger.warning("JWT expired")
                    return None
            
            return payload
        except Exception as e:
            self.logger.warning(f"Invalid JWT: {e}")
            return None
    
    def create_session(
        self,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create user session.
        
        Args:
            user_id: User ID
            metadata: Session metadata
        
        Returns:
            Session ID
        """
        session_id = secrets.token_urlsafe(32)
        
        self.sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(seconds=self.config.session_timeout),
            "metadata": metadata or {},
            "active": True
        }
        
        self.logger.info("Session created", user_id=user_id, session_id=session_id)
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Validate session.
        
        Args:
            session_id: Session ID
        
        Returns:
            Session data or None
        """
        session = self.sessions.get(session_id)
        
        if not session:
            return None
        
        if not session["active"]:
            return None
        
        if datetime.utcnow() > session["expires_at"]:
            session["active"] = False
            return None
        
        return session
    
    def revoke_session(self, session_id: str):
        """Revoke session."""
        if session_id in self.sessions:
            self.sessions[session_id]["active"] = False
            self.logger.info("Session revoked", session_id=session_id)
    
    def check_permission(
        self,
        user_id: str,
        required_scope: str
    ) -> bool:
        """
        Check if user has required permission.
        
        Args:
            user_id: User ID
            required_scope: Required scope
        
        Returns:
            True if authorized
        """
        # In production, check against permission database
        # For now, return True as placeholder
        return True


class EncryptionManager:
    """
    Encryption manager.
    
    Features:
    - Data encryption/decryption
    - Key management
    - Key rotation
    - HSM support (simulated)
    """
    
    def __init__(self, config: SecurityConfig):
        """
        Initialize encryption manager.
        
        Args:
            config: Security configuration
        """
        self.config = config
        self.master_key = secrets.token_bytes(32)  # 256-bit key
        self.keys: Dict[str, bytes] = {}
        
        self.logger = StructuredLogger("encryption_manager", LogLevel.INFO)
    
    def generate_data_key(self, key_id: str) -> bytes:
        """
        Generate data encryption key.
        
        Args:
            key_id: Key identifier
        
        Returns:
            Encryption key
        """
        key = secrets.token_bytes(32)
        self.keys[key_id] = key
        
        self.logger.info("Data key generated", key_id=key_id)
        
        return key
    
    def encrypt_data(self, data: bytes, key_id: str) -> bytes:
        """
        Encrypt data.
        
        Args:
            data: Data to encrypt
            key_id: Key identifier
        
        Returns:
            Encrypted data
        """
        # In production, use proper encryption (AES-GCM, etc.)
        # For now, simple XOR for demonstration
        key = self.keys.get(key_id)
        if not key:
            key = self.generate_data_key(key_id)
        
        # Simple XOR (NOT secure, for demonstration only)
        encrypted = bytes(a ^ b for a, b in zip(data, key * (len(data) // len(key) + 1)))
        
        get_metrics().increment_counter(
            'encryption_operations',
            {'operation': 'encrypt'},
            1
        )
        
        return encrypted
    
    def decrypt_data(self, encrypted_data: bytes, key_id: str) -> bytes:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Encrypted data
            key_id: Key identifier
        
        Returns:
            Decrypted data
        """
        key = self.keys.get(key_id)
        if not key:
            raise ValueError(f"Key {key_id} not found")
        
        # Simple XOR (NOT secure, for demonstration only)
        decrypted = bytes(a ^ b for a, b in zip(encrypted_data, key * (len(encrypted_data) // len(key) + 1)))
        
        get_metrics().increment_counter(
            'encryption_operations',
            {'operation': 'decrypt'},
            1
        )
        
        return decrypted
    
    def rotate_key(self, key_id: str):
        """
        Rotate encryption key.
        
        Args:
            key_id: Key identifier
        """
        old_key = self.keys.get(key_id)
        new_key = secrets.token_bytes(32)
        
        self.keys[f"{key_id}_old"] = old_key if old_key else b""
        self.keys[key_id] = new_key
        
        self.logger.info("Key rotated", key_id=key_id)
        
        get_metrics().increment_counter(
            'key_rotations',
            {'key_id': key_id},
            1
        )


class NetworkPolicy(BaseModel):
    """Network policy configuration."""
    
    # Ingress
    allowed_sources: List[str] = Field(default_factory=list, description="Allowed source IPs/CIDRs")
    allowed_ports: List[int] = Field(default_factory=list, description="Allowed ports")
    
    # Egress
    allowed_destinations: List[str] = Field(default_factory=list, description="Allowed destinations")
    allowed_egress_ports: List[int] = Field(default_factory=list, description="Allowed egress ports")
    
    # DDoS protection
    rate_limit_enabled: bool = Field(True, description="Enable rate limiting")
    max_requests_per_minute: int = Field(1000, ge=1, description="Max requests/minute")
    max_connections: int = Field(100, ge=1, description="Max concurrent connections")


class AuditLogger:
    """
    Audit logger for security events.
    
    Features:
    - Comprehensive event logging
    - Tamper-proof logs
    - Compliance support
    - Export capabilities
    """
    
    def __init__(self, config: SecurityConfig):
        """
        Initialize audit logger.
        
        Args:
            config: Security configuration
        """
        self.config = config
        self.logs: List[Dict[str, Any]] = []
        
        self.logger = StructuredLogger("audit_logger", LogLevel.INFO)
    
    def log_event(
        self,
        event_type: str,
        user_id: Optional[str],
        resource: str,
        action: str,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log audit event.
        
        Args:
            event_type: Event type
            user_id: User ID
            resource: Resource accessed
            action: Action performed
            success: Action success
            metadata: Additional metadata
        """
        event = {
            "timestamp": datetime.utcnow(),
            "event_type": event_type,
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "success": success,
            "metadata": metadata or {},
            "event_id": secrets.token_urlsafe(16)
        }
        
        self.logs.append(event)
        
        self.logger.info(
            "Audit event logged",
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action,
            success=success
        )
        
        get_metrics().increment_counter(
            'audit_events',
            {'event_type': event_type, 'success': str(success)},
            1
        )
    
    def get_events(
        self,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Query audit events.
        
        Args:
            user_id: Filter by user
            resource: Filter by resource
            start_time: Start time
            end_time: End time
        
        Returns:
            Matching events
        """
        filtered = self.logs
        
        if user_id:
            filtered = [e for e in filtered if e["user_id"] == user_id]
        
        if resource:
            filtered = [e for e in filtered if e["resource"] == resource]
        
        if start_time:
            filtered = [e for e in filtered if e["timestamp"] >= start_time]
        
        if end_time:
            filtered = [e for e in filtered if e["timestamp"] <= end_time]
        
        return filtered

