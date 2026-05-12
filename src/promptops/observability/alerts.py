"""Alert system - triggers notifications on quality drops, cost spikes, errors.

Supports configurable alert rules with severity levels and
notification channels (SNS, Slack, email).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertCondition(Enum):
    """Types of alert conditions."""

    ERROR_RATE_ABOVE = "error_rate_above"
    QUALITY_BELOW = "quality_below"
    COST_ABOVE = "cost_above"
    LATENCY_ABOVE = "latency_above"
    INVOCATIONS_ABOVE = "invocations_above"
    INVOCATIONS_BELOW = "invocations_below"


@dataclass
class AlertRule:
    """A rule that triggers an alert when conditions are met."""

    name: str
    condition: AlertCondition
    threshold: float
    severity: AlertSeverity = AlertSeverity.WARNING
    prompt_name: Optional[str] = None  # None = applies to all
    environment: Optional[str] = None
    cooldown_minutes: int = 15  # Don't re-alert within this window
    notify: List[str] = field(default_factory=list)  # email/channel list
    enabled: bool = True

    def evaluate(self, current_value: float) -> bool:
        """Check if the condition is met.

        Args:
            current_value: Current metric value

        Returns:
            True if alert should fire
        """
        if not self.enabled:
            return False

        if self.condition in (
            AlertCondition.ERROR_RATE_ABOVE,
            AlertCondition.COST_ABOVE,
            AlertCondition.LATENCY_ABOVE,
            AlertCondition.INVOCATIONS_ABOVE,
        ):
            return current_value > self.threshold
        elif self.condition in (
            AlertCondition.QUALITY_BELOW,
            AlertCondition.INVOCATIONS_BELOW,
        ):
            return current_value < self.threshold
        return False


@dataclass
class Alert:
    """A triggered alert."""

    rule_name: str
    severity: AlertSeverity
    prompt_name: str
    message: str
    current_value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False
    resolved: bool = False
    notify_targets: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "prompt_name": self.prompt_name,
            "message": self.message,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
        }

    def summary_line(self) -> str:
        icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(self.severity.value, "•")
        ts = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        return f"{icon} [{ts}] {self.rule_name}: {self.message}"


class AlertManager:
    """Manages alert rules and triggered alerts.

    Usage:
        manager = AlertManager()
        manager.add_rule(AlertRule(
            name="high-error-rate",
            condition=AlertCondition.ERROR_RATE_ABOVE,
            threshold=0.05,
            severity=AlertSeverity.CRITICAL,
        ))
        alerts = manager.check(prompt_name="summarize", metrics={"error_rate": 0.08})
    """

    def __init__(self):
        self._rules: List[AlertRule] = []
        self._alerts: List[Alert] = []
        self._last_fired: Dict[str, float] = {}  # rule_name -> timestamp
        self._handlers: List[Callable[[Alert], None]] = []

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self._rules.append(rule)

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name."""
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.name != rule_name]
        return len(self._rules) < before

    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add a notification handler (called when alerts fire)."""
        self._handlers.append(handler)

    def check(
        self,
        prompt_name: str,
        metrics: Dict[str, float],
        environment: Optional[str] = None,
    ) -> List[Alert]:
        """Check all rules against current metrics.

        Args:
            prompt_name: Prompt being checked
            metrics: Current metric values (error_rate, quality, cost, latency, invocations)
            environment: Current environment

        Returns:
            List of newly triggered alerts
        """
        triggered = []
        now = time.time()

        metric_mapping = {
            AlertCondition.ERROR_RATE_ABOVE: "error_rate",
            AlertCondition.QUALITY_BELOW: "quality",
            AlertCondition.COST_ABOVE: "cost",
            AlertCondition.LATENCY_ABOVE: "latency",
            AlertCondition.INVOCATIONS_ABOVE: "invocations",
            AlertCondition.INVOCATIONS_BELOW: "invocations",
        }

        for rule in self._rules:
            # Check if rule applies
            if rule.prompt_name and rule.prompt_name != prompt_name:
                continue
            if rule.environment and rule.environment != environment:
                continue

            # Check cooldown
            last = self._last_fired.get(rule.name, 0)
            if now - last < rule.cooldown_minutes * 60:
                continue

            # Get metric value
            metric_key = metric_mapping.get(rule.condition, "")
            current_value = metrics.get(metric_key, 0.0)

            # Evaluate
            if rule.evaluate(current_value):
                alert = Alert(
                    rule_name=rule.name,
                    severity=rule.severity,
                    prompt_name=prompt_name,
                    message=f"{rule.condition.value}: {current_value:.4f} (threshold: {rule.threshold})",
                    current_value=current_value,
                    threshold=rule.threshold,
                    timestamp=now,
                    notify_targets=rule.notify,
                )
                self._alerts.append(alert)
                self._last_fired[rule.name] = now
                triggered.append(alert)

                # Call handlers
                for handler in self._handlers:
                    try:
                        handler(alert)
                    except Exception:
                        pass

        return triggered

    def get_active_alerts(self) -> List[Alert]:
        """Get unresolved alerts."""
        return [a for a in self._alerts if not a.resolved]

    def get_alerts(
        self,
        prompt_name: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        limit: int = 50,
    ) -> List[Alert]:
        """Get alerts with optional filters."""
        results = []
        for alert in reversed(self._alerts):
            if prompt_name and alert.prompt_name != prompt_name:
                continue
            if severity and alert.severity != severity:
                continue
            results.append(alert)
            if len(results) >= limit:
                break
        return results

    def acknowledge(self, index: int) -> bool:
        """Acknowledge an alert by index."""
        active = self.get_active_alerts()
        if 0 <= index < len(active):
            active[index].acknowledged = True
            return True
        return False

    def resolve(self, rule_name: str) -> int:
        """Resolve all alerts for a rule."""
        count = 0
        for alert in self._alerts:
            if alert.rule_name == rule_name and not alert.resolved:
                alert.resolved = True
                count += 1
        return count

    @property
    def rules(self) -> List[AlertRule]:
        return self._rules
