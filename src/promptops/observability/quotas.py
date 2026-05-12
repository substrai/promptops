"""Usage quotas - per-team/per-user rate limits and budget caps.

Enforces usage limits to prevent cost overruns and ensure
fair resource allocation across teams.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class QuotaStatus(Enum):
    """Status of a quota check."""

    ALLOWED = "allowed"
    WARNING = "warning"  # approaching limit
    BLOCKED = "blocked"  # limit exceeded
    DOWNGRADED = "downgraded"  # switched to cheaper model


class QuotaPeriod(Enum):
    """Quota reset periods."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

    @property
    def seconds(self) -> int:
        return {
            QuotaPeriod.HOURLY: 3600,
            QuotaPeriod.DAILY: 86400,
            QuotaPeriod.WEEKLY: 604800,
            QuotaPeriod.MONTHLY: 2592000,
        }[self]


@dataclass
class UsageQuota:
    """A usage quota definition."""

    name: str
    entity_type: str  # "user", "team", "prompt", "global"
    entity_id: str  # user_id, team_name, prompt_name, or "*"
    period: QuotaPeriod = QuotaPeriod.DAILY
    max_invocations: Optional[int] = None
    max_cost: Optional[float] = None
    max_tokens: Optional[int] = None
    warning_threshold: float = 0.8  # warn at 80% usage
    on_exceed: str = "block"  # "block", "downgrade", "alert"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "period": self.period.value,
            "max_invocations": self.max_invocations,
            "max_cost": self.max_cost,
            "max_tokens": self.max_tokens,
            "warning_threshold": self.warning_threshold,
            "on_exceed": self.on_exceed,
        }


@dataclass
class QuotaCheckResult:
    """Result of checking a quota."""

    status: QuotaStatus
    quota_name: str
    current_usage: float
    limit: float
    usage_percent: float
    message: str
    remaining: float = 0.0

    @property
    def is_allowed(self) -> bool:
        return self.status in (QuotaStatus.ALLOWED, QuotaStatus.WARNING)


@dataclass
class UsageRecord:
    """Tracks usage for a specific entity in a period."""

    entity_key: str  # "{entity_type}:{entity_id}"
    period_start: float
    invocations: int = 0
    cost: float = 0.0
    tokens: int = 0


class QuotaManager:
    """Manages usage quotas and enforcement.

    Usage:
        manager = QuotaManager()
        manager.add_quota(UsageQuota(
            name="team-budget",
            entity_type="team",
            entity_id="ml-team",
            max_cost=100.0,
            period=QuotaPeriod.DAILY,
        ))
        result = manager.check_and_record("team", "ml-team", cost=0.05, tokens=500)
    """

    def __init__(self):
        self._quotas: List[UsageQuota] = []
        self._usage: Dict[str, UsageRecord] = {}

    def add_quota(self, quota: UsageQuota) -> None:
        """Add a usage quota."""
        self._quotas.append(quota)

    def remove_quota(self, name: str) -> bool:
        """Remove a quota by name."""
        before = len(self._quotas)
        self._quotas = [q for q in self._quotas if q.name != name]
        return len(self._quotas) < before

    def check(
        self,
        entity_type: str,
        entity_id: str,
    ) -> List[QuotaCheckResult]:
        """Check quotas for an entity without recording usage.

        Args:
            entity_type: "user", "team", "prompt"
            entity_id: The entity identifier

        Returns:
            List of QuotaCheckResults for applicable quotas
        """
        results = []
        for quota in self._quotas:
            if quota.entity_type != entity_type:
                continue
            if quota.entity_id != "*" and quota.entity_id != entity_id:
                continue
            result = self._evaluate_quota(quota, entity_type, entity_id)
            results.append(result)
        return results

    def check_and_record(
        self,
        entity_type: str,
        entity_id: str,
        cost: float = 0.0,
        tokens: int = 0,
        invocations: int = 1,
    ) -> List[QuotaCheckResult]:
        """Check quotas and record usage.

        Args:
            entity_type: "user", "team", "prompt"
            entity_id: The entity identifier
            cost: Cost of this invocation
            tokens: Tokens used
            invocations: Number of invocations (usually 1)

        Returns:
            List of QuotaCheckResults
        """
        # Record usage first
        key = f"{entity_type}:{entity_id}"
        now = time.time()

        if key not in self._usage:
            self._usage[key] = UsageRecord(entity_key=key, period_start=now)

        record = self._usage[key]
        record.invocations += invocations
        record.cost += cost
        record.tokens += tokens

        # Check quotas
        return self.check(entity_type, entity_id)

    def get_usage(self, entity_type: str, entity_id: str) -> Optional[UsageRecord]:
        """Get current usage for an entity."""
        key = f"{entity_type}:{entity_id}"
        return self._usage.get(key)

    def reset_usage(self, entity_type: str, entity_id: str) -> None:
        """Reset usage counters for an entity."""
        key = f"{entity_type}:{entity_id}"
        if key in self._usage:
            self._usage[key] = UsageRecord(entity_key=key, period_start=time.time())

    def get_all_quotas(self) -> List[UsageQuota]:
        """Get all configured quotas."""
        return self._quotas

    def get_usage_report(self) -> Dict[str, Any]:
        """Get usage report for all entities."""
        report = {}
        for key, record in self._usage.items():
            report[key] = {
                "invocations": record.invocations,
                "cost": round(record.cost, 6),
                "tokens": record.tokens,
                "period_start": record.period_start,
            }
        return report

    def _evaluate_quota(
        self, quota: UsageQuota, entity_type: str, entity_id: str
    ) -> QuotaCheckResult:
        """Evaluate a single quota against current usage."""
        key = f"{entity_type}:{entity_id}"
        record = self._usage.get(key)

        if not record:
            return QuotaCheckResult(
                status=QuotaStatus.ALLOWED,
                quota_name=quota.name,
                current_usage=0,
                limit=quota.max_cost or quota.max_invocations or quota.max_tokens or 0,
                usage_percent=0.0,
                message="No usage recorded",
                remaining=quota.max_cost or quota.max_invocations or quota.max_tokens or 0,
            )

        # Check period reset
        now = time.time()
        if now - record.period_start > quota.period.seconds:
            # Period expired, reset
            self._usage[key] = UsageRecord(entity_key=key, period_start=now)
            record = self._usage[key]

        # Determine which limit to check
        if quota.max_cost is not None:
            current = record.cost
            limit = quota.max_cost
            metric = "cost"
        elif quota.max_invocations is not None:
            current = record.invocations
            limit = quota.max_invocations
            metric = "invocations"
        elif quota.max_tokens is not None:
            current = record.tokens
            limit = quota.max_tokens
            metric = "tokens"
        else:
            return QuotaCheckResult(
                status=QuotaStatus.ALLOWED,
                quota_name=quota.name,
                current_usage=0,
                limit=0,
                usage_percent=0.0,
                message="No limits configured",
                remaining=0,
            )

        usage_percent = current / limit if limit > 0 else 0.0
        remaining = max(limit - current, 0)

        if usage_percent >= 1.0:
            status = QuotaStatus.BLOCKED
            message = f"Quota exceeded: {metric} {current:.4f} >= {limit:.4f}"
        elif usage_percent >= quota.warning_threshold:
            status = QuotaStatus.WARNING
            message = f"Approaching limit: {metric} at {usage_percent:.0%}"
        else:
            status = QuotaStatus.ALLOWED
            message = f"Within limits: {metric} at {usage_percent:.0%}"

        return QuotaCheckResult(
            status=status,
            quota_name=quota.name,
            current_usage=current,
            limit=limit,
            usage_percent=round(usage_percent, 4),
            message=message,
            remaining=remaining,
        )
