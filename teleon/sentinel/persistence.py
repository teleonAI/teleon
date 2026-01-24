"""
Sentinel Violation Persistence - Dedicated persistence layer for Sentinel violations.

Handles submission of Sentinel violations to the platform database,
separate from general audit logging.

Production-ready with:
- Non-blocking submission (zero latency impact)
- Background batching (efficient HTTP calls)
- Circuit breaker (stops trying if API is down)
- Retry logic with exponential backoff
"""

import os
import asyncio
import time
import re
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
        api_key: Optional[str] = None,
        batch_size: int = 10,
        flush_interval: float = 5.0,
        max_queue_size: int = 1000
    ):
        """
        Initialize Sentinel violation persistence.
        
        Args:
            agent_id: Agent ID
            agent_name: Agent name
            api_url: Platform API URL (defaults to TELEON_API_URL env var)
            api_key: Platform API key (defaults to TELEON_API_KEY env var)
            batch_size: Number of violations to batch before flushing
            flush_interval: Seconds between automatic flushes
            max_queue_size: Maximum violations to queue before dropping
        """
        self.agent_id = agent_id
        self.agent_name = agent_name or agent_id
        # Check both TELEON_API_URL and TELEON_PLATFORM_URL (for compatibility)
        self.api_url = api_url or os.getenv('TELEON_API_URL') or os.getenv('TELEON_PLATFORM_URL')
        self.api_key = api_key or os.getenv('TELEON_API_KEY')
        self.logger = StructuredLogger("sentinel.persistence", LogLevel.INFO)
        self._pending_violations: List[Dict[str, Any]] = []
        self._pending_lock = asyncio.Lock()
        self._enabled = bool(self.api_url and self.api_key)
        
        # Batching configuration
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_queue_size = max_queue_size
        
        # Circuit breaker state
        self._circuit_open = False
        self._circuit_open_until = 0.0
        self._circuit_failure_count = 0
        self._circuit_failure_threshold = 5
        self._circuit_reset_timeout = 60.0  # 1 minute
        
        # Background flush task
        self._flush_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        if not self._enabled:
            self.logger.warning(
                "Sentinel violation persistence disabled - API URL or key not configured"
            )
        else:
            # Start background flush task
            self._start_background_flush()
    
    def _start_background_flush(self) -> None:
        """Start background flush task (non-blocking)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._flush_task = loop.create_task(self._background_flush_loop())
            else:
                # If no running loop, schedule for when loop starts
                asyncio.ensure_future(self._background_flush_loop())
        except RuntimeError:
            # No event loop available, will start when one is available
            pass
    
    async def _background_flush_loop(self) -> None:
        """Background task that flushes violations periodically."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.flush_interval)
                if not self._shutdown and self._pending_violations:
                    await self._flush_violations()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(f"Error in background flush loop: {e}")
    
    async def submit_violation(
        self,
        violation_type: str,
        action_taken: str,
        details: Dict[str, Any],
        validation_type: str = "input"  # "input" or "output"
    ) -> bool:
        """
        Submit a Sentinel violation to the platform (NON-BLOCKING).
        
        This method returns immediately after queuing the violation.
        Actual submission happens in the background.
        
        Args:
            violation_type: Type of violation (e.g., "pii_detection", "toxicity")
            action_taken: Action taken (e.g., "block", "flag", "redact")
            details: Violation details
            validation_type: Whether this was input or output validation
        
        Returns:
            True if queued successfully, False otherwise
        """
        if not self._enabled:
            return False
        
        # Check circuit breaker
        if self._circuit_open:
            if time.time() < self._circuit_open_until:
                return False  # Circuit is open, don't queue
            else:
                # Try to close circuit
                self._circuit_open = False
                self._circuit_failure_count = 0
        
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
        
        # Add to pending queue (non-blocking)
        try:
            async with self._pending_lock:
                if len(self._pending_violations) >= self.max_queue_size:
                    # Queue is full, drop oldest violation
                    self._pending_violations.pop(0)
                    self.logger.warning(
                        f"Violation queue full, dropping oldest violation. "
                        f"Consider increasing max_queue_size or fixing API connectivity."
                    )
                
                self._pending_violations.append(violation_payload)
                queue_size = len(self._pending_violations)
            
            # Trigger immediate flush if batch size reached (fire-and-forget)
            if queue_size >= self.batch_size:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Schedule flush but don't await
                        loop.create_task(self._flush_violations())
                    else:
                        # If no loop, violations will be flushed by background task
                        pass
                except RuntimeError:
                    pass
            
            return True
        except Exception as e:
            self.logger.warning(f"Failed to queue Sentinel violation: {e}")
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
        
        # Use same non-blocking submission as violations
        return await self.submit_violation(
            violation_type="pii_detection",
            action_taken="redact" if redacted else "flag",
            details={
                "pii_type": pii_type,
                "detected_value": detected_value,
                "context": context,
                "redacted": redacted
            },
            validation_type="input"  # PII detection is typically on input
        )
    
    async def _flush_violations(self) -> None:
        """Flush pending violations to the platform API (with retry logic)."""
        if not HTTPX_AVAILABLE:
            self.logger.warning("httpx not available, cannot submit violations")
            return
        
        # Check circuit breaker
        if self._circuit_open:
            if time.time() < self._circuit_open_until:
                return  # Circuit is open, don't try
            else:
                # Try to close circuit
                self._circuit_open = False
                self._circuit_failure_count = 0
        
        # Get violations to send
        async with self._pending_lock:
            if not self._pending_violations:
                return
            
            violations_to_send = self._pending_violations.copy()
            self._pending_violations = []
        
        # Retry logic with exponential backoff
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    # Submit violations as AuditLog entries via agent batch endpoint
                    # This endpoint supports both JWT service account tokens and deployment_id (legacy)
                    batch_payload = {"logs": violations_to_send}
                    endpoint = f"{self.api_url}/api/v1/governance/audit-logs/batch/agent"
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    }
                    
                    response = await client.post(
                        endpoint,
                        json=batch_payload,
                        headers=headers,
                        timeout=5.0,  # Reduced from 30s to 5s
                    )
                    response.raise_for_status()
                    
                    # Success - reset circuit breaker
                    self._circuit_failure_count = 0
                    self._circuit_open = False
                    
                    self.logger.debug(
                        f"Submitted {len(violations_to_send)} Sentinel violation(s) to platform"
                    )
                    return  # Success, exit retry loop
                    
            except httpx.HTTPError as e:
                # HTTP error - might be transient
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(
                        f"Failed to submit violations (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Final attempt failed
                    self._circuit_failure_count += 1
                    if self._circuit_failure_count >= self._circuit_failure_threshold:
                        # Open circuit breaker
                        self._circuit_open = True
                        self._circuit_open_until = time.time() + self._circuit_reset_timeout
                        self.logger.error(
                            f"Circuit breaker opened after {self._circuit_failure_count} failures. "
                            f"Will retry after {self._circuit_reset_timeout}s"
                        )
                    
                    # Re-queue failed violations (with limit)
                    self.logger.warning(f"Failed to submit violations to platform after {max_retries} attempts: {e}")
                    async with self._pending_lock:
                        if len(self._pending_violations) < self.max_queue_size:
                            # Add back to front of queue
                            self._pending_violations = violations_to_send + self._pending_violations
                    return
                    
            except Exception as e:
                # Unexpected error
                self.logger.error(f"Unexpected error submitting violations: {e}", exc_info=True)
                # Re-queue
                async with self._pending_lock:
                    if len(self._pending_violations) < self.max_queue_size:
                        self._pending_violations = violations_to_send + self._pending_violations
                return
    
    async def flush(self) -> None:
        """Manually flush all pending violations."""
        await self._flush_violations()
    
    async def close(self) -> None:
        """Flush remaining violations and cleanup."""
        self._shutdown = True
        
        # Cancel background flush task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self._flush_violations()

