"""
Sentinel Violation Persistence - Dedicated persistence layer for Sentinel violations.

Handles submission of Sentinel violations to the platform database,
separate from general audit logging.
"""

import os
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from teleon.core import StructuredLogger, LogLevel

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class SentinelViolationPersistence:
    """
    Dedicated persistence layer for Sentinel violations.
    
    Submits violations to the platform API for storage in the database.
    Keeps violations separate from general audit logs.
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_name: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize Sentinel violation persistence.
        
        Args:
            agent_id: Agent ID
            agent_name: Agent name
            api_url: Platform API URL (defaults to TELEON_API_URL env var)
            api_key: Platform API key (defaults to TELEON_API_KEY env var)
        """
        self.agent_id = agent_id
        self.agent_name = agent_name or agent_id
        self.api_url = api_url or os.getenv('TELEON_API_URL')
        self.api_key = api_key or os.getenv('TELEON_API_KEY')
        self.logger = StructuredLogger("sentinel.persistence", LogLevel.INFO)
        self._pending_violations: List[Dict[str, Any]] = []
        self._pending_lock = asyncio.Lock()
        self._enabled = bool(self.api_url and self.api_key)
        
        if not self._enabled:
            self.logger.warning(
                "Sentinel violation persistence disabled - API URL or key not configured"
            )
    
    async def submit_violation(
        self,
        violation_type: str,
        action_taken: str,
        details: Dict[str, Any],
        validation_type: str = "input"  # "input" or "output"
    ) -> bool:
        """
        Submit a Sentinel violation to the platform.
        
        Args:
            violation_type: Type of violation (e.g., "pii_detection", "toxicity")
            action_taken: Action taken (e.g., "block", "flag", "redact")
            details: Violation details
            validation_type: Whether this was input or output validation
        
        Returns:
            True if submitted successfully, False otherwise
        """
        if not self._enabled:
            return False
        
        # Prepare violation payload
        violation_payload = {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "action_type": "agent_request" if validation_type == "input" else "agent_response",
            "action": f"Sentinel violation: {violation_type}",
            "status": "warning",
            "extra_metadata": {
                "violation_type": violation_type,
                "action_taken": action_taken,
                "violation_details": details,
                "source": "sentinel",
                "validation_type": validation_type
            }
        }
        
        # Add to pending queue
        async with self._pending_lock:
            self._pending_violations.append(violation_payload)
        
        # Submit immediately (could be batched in the future)
        try:
            await self._flush_violations()
            return True
        except Exception as e:
            self.logger.warning(f"Failed to submit Sentinel violation: {e}")
            return False
    
    async def submit_pii_detection(
        self,
        pii_type: str,
        detected_value: Optional[str] = None,
        context: Optional[str] = None,
        redacted: bool = False
    ) -> bool:
        """
        Submit a PII detection to the platform.
        
        Args:
            pii_type: Type of PII (e.g., "email", "phone", "ssn")
            detected_value: The detected PII value (may be redacted/hashed)
            context: Context where PII was found
            redacted: Whether the PII was redacted
        
        Returns:
            True if submitted successfully, False otherwise
        """
        if not self._enabled:
            return False
        
        # PII detections are stored separately in PIIDetection table
        # For now, we'll create an AuditLog entry with PII metadata
        # The platform can extract this to create PIIDetection records
        
        pii_payload = {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "action_type": "agent_request",
            "action": f"PII detected: {pii_type}",
            "status": "warning",
            "extra_metadata": {
                "violation_type": "pii_detection",
                "pii_type": pii_type,
                "detected_value": detected_value,
                "context": context,
                "redacted": redacted,
                "source": "sentinel"
            }
        }
        
        async with self._pending_lock:
            self._pending_violations.append(pii_payload)
        
        try:
            await self._flush_violations()
            return True
        except Exception as e:
            self.logger.warning(f"Failed to submit PII detection: {e}")
            return False
    
    async def _flush_violations(self) -> None:
        """Flush pending violations to the platform API."""
        if not HTTPX_AVAILABLE:
            self.logger.warning("httpx not available, cannot submit violations")
            return
        
        async with self._pending_lock:
            if not self._pending_violations:
                return
            
            violations_to_send = self._pending_violations.copy()
            self._pending_violations = []
        
        try:
            async with httpx.AsyncClient() as client:
                # Submit violations as AuditLog entries via batch endpoint
                # The platform API will store them with proper metadata
                batch_payload = {"logs": violations_to_send}
                response = await client.post(
                    f"{self.api_url}/api/v1/governance/audit-logs/batch",
                    json=batch_payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                
                self.logger.debug(
                    f"Submitted {len(violations_to_send)} Sentinel violation(s) to platform"
                )
        except httpx.HTTPError as e:
            # Re-queue failed violations (with limit)
            self.logger.warning(f"Failed to submit violations to platform: {e}")
            async with self._pending_lock:
                if len(self._pending_violations) < 100:
                    self._pending_violations = violations_to_send + self._pending_violations
        except Exception as e:
            self.logger.error(f"Unexpected error submitting violations: {e}", exc_info=True)
    
    async def flush(self) -> None:
        """Manually flush all pending violations."""
        await self._flush_violations()
    
    async def close(self) -> None:
        """Flush remaining violations and cleanup."""
        await self._flush_violations()

