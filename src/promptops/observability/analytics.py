"""Invocation analytics - tracks per-prompt metrics.

Records and aggregates latency, cost, token usage, quality scores,
error rates, and invocation counts per prompt/version/environment.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class InvocationRecord:
    """A single invocation record."""

    prompt_name: str
    version: str
    environment: str
    model: str
    timestamp: float
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost: float
    quality_score: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    user_id: Optional[str] = None
    team: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsSummary:
    """Aggregated analytics for a time period."""

    prompt_name: str
    period: str  # "1h", "24h", "7d", "30d"
    total_invocations: int = 0
    successful_invocations: int = 0
    failed_invocations: int = 0
    total_cost: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_quality_score: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    unique_users: int = 0
    error_rate: float = 0.0
    cost_per_invocation: float = 0.0
    versions_used: List[str] = field(default_factory=list)
    models_used: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_invocations == 0:
            return 0.0
        return self.successful_invocations / self.total_invocations

    def summary(self) -> str:
        lines = [
            f"Analytics: {self.prompt_name} (last {self.period})",
            f"  Invocations: {self.total_invocations} ({self.success_rate:.0%} success)",
            f"  Cost: ${self.total_cost:.4f} (${self.cost_per_invocation:.6f}/call)",
            f"  Latency: avg={self.avg_latency_ms:.0f}ms, p95={self.p95_latency_ms:.0f}ms",
            f"  Quality: {self.avg_quality_score:.2f}",
            f"  Tokens: {self.total_input_tokens} in / {self.total_output_tokens} out",
            f"  Users: {self.unique_users}",
            f"  Versions: {', '.join(self.versions_used)}",
        ]
        return "\n".join(lines)


class InvocationAnalytics:
    """Tracks and aggregates prompt invocation metrics.

    Usage:
        analytics = InvocationAnalytics()
        analytics.record(InvocationRecord(...))
        summary = analytics.get_summary("summarize", period="24h")
    """

    def __init__(self, retention_hours: int = 720):  # 30 days default
        self._records: List[InvocationRecord] = []
        self._retention_seconds = retention_hours * 3600

    def record(self, invocation: InvocationRecord) -> None:
        """Record an invocation."""
        self._records.append(invocation)
        self._cleanup_old_records()

    def record_from_dict(self, data: Dict[str, Any]) -> None:
        """Record from a dictionary (e.g., from PromptClient result)."""
        record = InvocationRecord(
            prompt_name=data.get("prompt_name", ""),
            version=data.get("version", ""),
            environment=data.get("environment", ""),
            model=data.get("model", ""),
            timestamp=data.get("timestamp", time.time()),
            latency_ms=data.get("latency_ms", 0),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cost=data.get("cost", 0),
            quality_score=data.get("quality_score"),
            success=data.get("success", True),
            error=data.get("error"),
            user_id=data.get("user_id"),
            team=data.get("team"),
        )
        self.record(record)

    def get_summary(
        self,
        prompt_name: Optional[str] = None,
        period: str = "24h",
        environment: Optional[str] = None,
    ) -> AnalyticsSummary:
        """Get aggregated analytics summary.

        Args:
            prompt_name: Filter by prompt (None = all)
            period: Time period ("1h", "24h", "7d", "30d")
            environment: Filter by environment

        Returns:
            AnalyticsSummary with aggregated metrics
        """
        cutoff = time.time() - self._parse_period(period)
        records = self._filter_records(prompt_name, cutoff, environment)

        if not records:
            return AnalyticsSummary(
                prompt_name=prompt_name or "all",
                period=period,
            )

        latencies = [r.latency_ms for r in records]
        latencies.sort()
        quality_scores = [r.quality_score for r in records if r.quality_score is not None]
        users = set(r.user_id for r in records if r.user_id)
        versions = list(set(r.version for r in records))
        models = list(set(r.model for r in records))

        total = len(records)
        successful = sum(1 for r in records if r.success)
        total_cost = sum(r.cost for r in records)

        return AnalyticsSummary(
            prompt_name=prompt_name or "all",
            period=period,
            total_invocations=total,
            successful_invocations=successful,
            failed_invocations=total - successful,
            total_cost=round(total_cost, 6),
            avg_latency_ms=round(sum(latencies) / len(latencies), 2) if latencies else 0,
            p95_latency_ms=latencies[int(len(latencies) * 0.95)] if latencies else 0,
            p99_latency_ms=latencies[int(len(latencies) * 0.99)] if latencies else 0,
            avg_quality_score=round(sum(quality_scores) / len(quality_scores), 4) if quality_scores else 0,
            total_input_tokens=sum(r.input_tokens for r in records),
            total_output_tokens=sum(r.output_tokens for r in records),
            unique_users=len(users),
            error_rate=round((total - successful) / total, 4) if total > 0 else 0,
            cost_per_invocation=round(total_cost / total, 8) if total > 0 else 0,
            versions_used=versions,
            models_used=models,
        )

    def get_cost_by_prompt(self, period: str = "24h") -> Dict[str, float]:
        """Get cost breakdown by prompt."""
        cutoff = time.time() - self._parse_period(period)
        costs: Dict[str, float] = defaultdict(float)
        for r in self._records:
            if r.timestamp >= cutoff:
                costs[r.prompt_name] += r.cost
        return dict(sorted(costs.items(), key=lambda x: -x[1]))

    def get_cost_by_team(self, period: str = "24h") -> Dict[str, float]:
        """Get cost breakdown by team."""
        cutoff = time.time() - self._parse_period(period)
        costs: Dict[str, float] = defaultdict(float)
        for r in self._records:
            if r.timestamp >= cutoff and r.team:
                costs[r.team] += r.cost
        return dict(sorted(costs.items(), key=lambda x: -x[1]))

    def get_error_rate(self, prompt_name: str, period: str = "1h") -> float:
        """Get error rate for a prompt in a time period."""
        cutoff = time.time() - self._parse_period(period)
        records = [r for r in self._records if r.prompt_name == prompt_name and r.timestamp >= cutoff]
        if not records:
            return 0.0
        return sum(1 for r in records if not r.success) / len(records)

    def _filter_records(
        self,
        prompt_name: Optional[str],
        cutoff: float,
        environment: Optional[str],
    ) -> List[InvocationRecord]:
        """Filter records by criteria."""
        records = []
        for r in self._records:
            if r.timestamp < cutoff:
                continue
            if prompt_name and r.prompt_name != prompt_name:
                continue
            if environment and r.environment != environment:
                continue
            records.append(r)
        return records

    def _parse_period(self, period: str) -> float:
        """Parse period string to seconds."""
        units = {"h": 3600, "d": 86400, "m": 60}
        num = int(period[:-1])
        unit = period[-1]
        return num * units.get(unit, 3600)

    def _cleanup_old_records(self) -> None:
        """Remove records older than retention period."""
        cutoff = time.time() - self._retention_seconds
        self._records = [r for r in self._records if r.timestamp >= cutoff]
