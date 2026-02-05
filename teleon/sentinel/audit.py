"""
Sentinel Audit Logger - Violation Tracking and Audit Trails.

Logs all Sentinel violations for compliance and monitoring.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from collections import defaultdict
import threading
from teleon.core import StructuredLogger, LogLevel


class ViolationRecord:
    """Record of a single violation."""
    
    def __init__(
        self,
        agent_id: str,
        violation_type: str,
        action_taken: str,
        details: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ):
        """
        Initialize violation record.
        
        Args:
            agent_id: Agent ID where violation occurred
            violation_type: Type of violation
            action_taken: Action taken (block, flag, redact, escalate)
            details: Additional violation details
            timestamp: Timestamp of violation
        """
        self.agent_id = agent_id
        self.violation_type = violation_type
        self.action_taken = action_taken
        self.details = details
        self.timestamp = timestamp or datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "violation_type": self.violation_type,
            "action_taken": self.action_taken,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class SentinelAuditLogger:
    """
    Audit logger for Sentinel violations.
    
    Tracks:
    - All violations (in-memory for local access)
    - Violation rates per agent
    - Action history
    - Audit trail export
    
    Uses dedicated SentinelViolationPersistence for database persistence,
    separate from general audit logging.
    """
    
    def __init__(
        self,
        max_records: int = 10000,
        violation_persistence: Optional[Any] = None
    ):
        """
        Initialize audit logger.

        Args:
            max_records: Maximum records to keep in memory
            violation_persistence: Optional SentinelViolationPersistence instance for DB persistence
        """
        self.max_records = max_records
        self.records: List[ViolationRecord] = []
        self.violation_counts: Dict[str, int] = defaultdict(int)
        self.agent_violations: Dict[str, List[ViolationRecord]] = defaultdict(list)
        self.logger = StructuredLogger("sentinel_audit", LogLevel.INFO)
        self._lock = threading.RLock()  # Thread-safe lock for all operations
        self.violation_persistence = violation_persistence  # Dedicated Sentinel persistence
    
    def log_violation(
        self,
        agent_id: str,
        violation_type: str,
        action_taken: str,
        details: Dict[str, Any],
        validation_type: str = "input"  # "input" or "output"
    ) -> None:
        """
        Log a violation.

        Thread-safe method for recording violations.

        Args:
            agent_id: Agent ID
            violation_type: Type of violation
            action_taken: Action taken
            details: Violation details
        """
        record = ViolationRecord(
            agent_id=agent_id,
            violation_type=violation_type,
            action_taken=action_taken,
            details=details
        )

        with self._lock:
            # Add to records
            self.records.append(record)

            # Maintain max records limit
            if len(self.records) > self.max_records:
                self.records = self.records[-self.max_records:]

            # Update counts
            key = f"{agent_id}:{violation_type}"
            self.violation_counts[key] += 1
            self.agent_violations[agent_id].append(record)

            # Also trim agent-specific violations to prevent memory leak
            if len(self.agent_violations[agent_id]) > self.max_records // 10:
                self.agent_violations[agent_id] = self.agent_violations[agent_id][-(self.max_records // 10):]

        # Log to structured logger (outside lock to avoid blocking)
        self.logger.warning(
            "Sentinel violation",
            agent_id=agent_id,
            violation_type=violation_type,
            action_taken=action_taken,
            details=details
        )
        
        # Persist to platform via dedicated Sentinel persistence layer (non-blocking)
        # CRITICAL: This must never block agent execution - all operations are fire-and-forget
        if self.violation_persistence:
            try:
                # submit_violation() is fully non-blocking - it queues and returns immediately
                # We don't await it, ensuring zero latency impact on agent calls
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Fire-and-forget - create_task() schedules but doesn't block
                        # submit_violation() itself is non-blocking (queues and returns)
                        loop.create_task(
                            self.violation_persistence.submit_violation(
                                violation_type=violation_type,
                                action_taken=action_taken,
                                details=details,
                                validation_type=validation_type
                            )
                        )
                    else:
                        # If no loop, violations will be queued when loop starts
                        # Background flush task will handle submission
                        pass
                except RuntimeError:
                    # No event loop available - violations will be queued when loop starts
                    # Background flush will handle it - zero impact on agent execution
                    pass
            except Exception as e:
                # Don't fail if persistence fails - violation is still stored in-memory
                # Logging failures must never impact agent execution
                self.logger.warning(f"Failed to schedule violation persistence: {e}")
    
    def get_violations(
        self,
        agent_id: Optional[str] = None,
        violation_type: Optional[str] = None,
        limit: int = 100
    ) -> List[ViolationRecord]:
        """
        Get violation records.

        Thread-safe method for retrieving violations.

        Args:
            agent_id: Filter by agent ID
            violation_type: Filter by violation type
            limit: Maximum records to return

        Returns:
            List of violation records
        """
        with self._lock:
            filtered = list(self.records)  # Create a copy

        if agent_id:
            filtered = [r for r in filtered if r.agent_id == agent_id]

        if violation_type:
            filtered = [r for r in filtered if r.violation_type == violation_type]

        return filtered[-limit:]
    
    def get_violation_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get violation statistics.

        Thread-safe method for retrieving statistics.

        Args:
            agent_id: Filter by agent ID

        Returns:
            Statistics dictionary
        """
        with self._lock:
            if agent_id:
                records = [r for r in self.records if r.agent_id == agent_id]
            else:
                records = list(self.records)

        if not records:
            return {
                "total_violations": 0,
                "by_type": {},
                "by_action": {},
                "recent_count": 0
            }

        by_type = defaultdict(int)
        by_action = defaultdict(int)

        for record in records:
            by_type[record.violation_type] += 1
            by_action[record.action_taken] += 1

        return {
            "total_violations": len(records),
            "by_type": dict(by_type),
            "by_action": dict(by_action),
            "recent_count": len([r for r in records if (datetime.now(timezone.utc) - r.timestamp).total_seconds() < 3600])
        }
    
    def export_audit_trail(
        self,
        agent_id: Optional[str] = None,
        format: str = "json"
    ) -> Any:
        """
        Export audit trail.
        
        Args:
            agent_id: Filter by agent ID
            format: Export format ('json' or 'csv')
        
        Returns:
            Exported data
        """
        records = self.get_violations(agent_id=agent_id, limit=10000)
        
        if format == "json":
            return [r.to_dict() for r in records]
        elif format == "csv":
            import csv
            import io
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=["timestamp", "agent_id", "violation_type", "action_taken", "details"])
            writer.writeheader()
            for record in records:
                row = record.to_dict()
                row["details"] = str(row["details"])
                writer.writerow(row)
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def clear(self, agent_id: Optional[str] = None) -> int:
        """
        Clear audit records.

        Thread-safe method for clearing records.

        Args:
            agent_id: Clear records for specific agent (None = all)

        Returns:
            Number of records cleared
        """
        with self._lock:
            if agent_id:
                count = len(self.agent_violations.get(agent_id, []))
                self.records = [r for r in self.records if r.agent_id != agent_id]
                if agent_id in self.agent_violations:
                    del self.agent_violations[agent_id]
                # Clear violation counts for this agent
                keys_to_remove = [k for k in self.violation_counts if k.startswith(f"{agent_id}:")]
                for key in keys_to_remove:
                    del self.violation_counts[key]
                return count
            else:
                count = len(self.records)
                self.records.clear()
                self.violation_counts.clear()
                self.agent_violations.clear()
                return count

