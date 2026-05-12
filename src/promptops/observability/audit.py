"""Audit trail - logs every prompt change with full context.

Provides a complete history of who changed what, when, and why
for compliance and governance requirements.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AuditAction(Enum):
    """Types of auditable actions."""

    CREATED = "created"
    UPDATED = "updated"
    DEPLOYED = "deployed"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"
    DELETED = "deleted"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPERIMENT_STARTED = "experiment_started"
    EXPERIMENT_STOPPED = "experiment_stopped"
    CONFIG_CHANGED = "config_changed"


@dataclass
class AuditEntry:
    """A single audit log entry."""

    action: AuditAction
    prompt_name: str
    version: Optional[str] = None
    environment: Optional[str] = None
    actor: str = "system"
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    previous_state: Optional[Dict[str, Any]] = None
    new_state: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "action": self.action.value,
            "prompt_name": self.prompt_name,
            "version": self.version,
            "environment": self.environment,
            "actor": self.actor,
            "timestamp": self.timestamp,
            "details": self.details,
            "reason": self.reason,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
        }

    def summary_line(self) -> str:
        """One-line summary for display."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        version_str = f"@{self.version}" if self.version else ""
        env_str = f" [{self.environment}]" if self.environment else ""
        return f"[{ts}] {self.action.value}: {self.prompt_name}{version_str}{env_str} by {self.actor}"


@dataclass
class AuditQuery:
    """Query parameters for filtering audit entries."""

    prompt_name: Optional[str] = None
    action: Optional[AuditAction] = None
    actor: Optional[str] = None
    environment: Optional[str] = None
    since: Optional[float] = None
    until: Optional[float] = None
    limit: int = 100


class AuditTrail:
    """Manages the audit trail for all prompt operations.

    Usage:
        audit = AuditTrail()
        audit.log(AuditAction.DEPLOYED, "summarize", version="1.2.0", actor="gaurav@substrai.dev")
        entries = audit.query(AuditQuery(prompt_name="summarize", limit=10))
    """

    def __init__(self, max_entries: int = 10000):
        self._entries: List[AuditEntry] = []
        self._max_entries = max_entries

    def log(
        self,
        action: AuditAction,
        prompt_name: str,
        version: Optional[str] = None,
        environment: Optional[str] = None,
        actor: str = "system",
        reason: str = "",
        details: Optional[Dict[str, Any]] = None,
        previous_state: Optional[Dict[str, Any]] = None,
        new_state: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Log an audit entry.

        Args:
            action: The action performed
            prompt_name: Prompt affected
            version: Version involved
            environment: Target environment
            actor: Who performed the action
            reason: Why the action was taken
            details: Additional context
            previous_state: State before the change
            new_state: State after the change

        Returns:
            The created AuditEntry
        """
        entry = AuditEntry(
            action=action,
            prompt_name=prompt_name,
            version=version,
            environment=environment,
            actor=actor,
            timestamp=time.time(),
            details=details or {},
            reason=reason,
            previous_state=previous_state,
            new_state=new_state,
        )
        self._entries.append(entry)

        # Trim if over limit
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        return entry

    def query(self, query: AuditQuery) -> List[AuditEntry]:
        """Query audit entries with filters.

        Args:
            query: AuditQuery with filter parameters

        Returns:
            Matching entries (newest first)
        """
        results = []
        for entry in reversed(self._entries):
            if query.prompt_name and entry.prompt_name != query.prompt_name:
                continue
            if query.action and entry.action != query.action:
                continue
            if query.actor and entry.actor != query.actor:
                continue
            if query.environment and entry.environment != query.environment:
                continue
            if query.since and entry.timestamp < query.since:
                continue
            if query.until and entry.timestamp > query.until:
                continue
            results.append(entry)
            if len(results) >= query.limit:
                break
        return results

    def get_history(self, prompt_name: str, limit: int = 50) -> List[AuditEntry]:
        """Get full history for a prompt."""
        return self.query(AuditQuery(prompt_name=prompt_name, limit=limit))

    def get_recent(self, limit: int = 20) -> List[AuditEntry]:
        """Get most recent entries across all prompts."""
        return self.query(AuditQuery(limit=limit))

    def get_by_actor(self, actor: str, limit: int = 50) -> List[AuditEntry]:
        """Get entries by a specific actor."""
        return self.query(AuditQuery(actor=actor, limit=limit))

    def generate_report(
        self,
        prompt_name: Optional[str] = None,
        period_days: int = 30,
    ) -> Dict[str, Any]:
        """Generate a compliance report.

        Args:
            prompt_name: Filter by prompt (None = all)
            period_days: Report period in days

        Returns:
            Report dictionary suitable for JSON/PDF export
        """
        since = time.time() - (period_days * 86400)
        entries = self.query(AuditQuery(prompt_name=prompt_name, since=since, limit=10000))

        # Aggregate by action
        action_counts: Dict[str, int] = {}
        actors: set = set()
        prompts_affected: set = set()

        for entry in entries:
            action_counts[entry.action.value] = action_counts.get(entry.action.value, 0) + 1
            actors.add(entry.actor)
            prompts_affected.add(entry.prompt_name)

        return {
            "report_type": "compliance_audit",
            "generated_at": time.time(),
            "period_days": period_days,
            "prompt_filter": prompt_name,
            "total_entries": len(entries),
            "action_breakdown": action_counts,
            "unique_actors": len(actors),
            "actors": sorted(actors),
            "prompts_affected": sorted(prompts_affected),
            "entries": [e.to_dict() for e in entries[:100]],  # Cap at 100 for report
        }

    @property
    def total_entries(self) -> int:
        return len(self._entries)
