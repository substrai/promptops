"""Quality drift detection - alerts when prompt quality degrades over time.

Monitors quality metrics over sliding windows and detects
statistically significant degradation.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DriftAlert:
    """Alert triggered by quality drift detection."""

    prompt_name: str
    metric: str
    baseline_value: float
    current_value: float
    drift_percent: float
    window_size: int
    timestamp: float = field(default_factory=time.time)
    severity: str = "warning"

    @property
    def is_significant(self) -> bool:
        return abs(self.drift_percent) > 10.0

    def summary(self) -> str:
        direction = "↓" if self.drift_percent < 0 else "↑"
        return (
            f"Drift detected: {self.prompt_name} {self.metric} "
            f"{direction}{abs(self.drift_percent):.1f}% "
            f"(baseline={self.baseline_value:.3f}, current={self.current_value:.3f})"
        )


@dataclass
class MetricWindow:
    """Sliding window of metric values."""

    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))

    def add(self, value: float, timestamp: Optional[float] = None) -> None:
        self.values.append(value)
        self.timestamps.append(timestamp or time.time())

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    @property
    def count(self) -> int:
        return len(self.values)

    def recent_mean(self, n: int = 50) -> float:
        """Mean of the most recent n values."""
        if not self.values:
            return 0.0
        recent = list(self.values)[-n:]
        return sum(recent) / len(recent)

    def baseline_mean(self, n: int = 200) -> float:
        """Mean of the baseline (older values)."""
        if len(self.values) <= n:
            return self.mean
        baseline = list(self.values)[:-n]
        if not baseline:
            return self.mean
        return sum(baseline) / len(baseline)


class QualityDriftDetector:
    """Detects quality degradation over time.

    Monitors quality metrics using sliding windows and alerts
    when current performance significantly deviates from baseline.

    Usage:
        detector = QualityDriftDetector(threshold_percent=10.0)
        detector.record("summarize", "quality_score", 0.92)
        # ... many more records ...
        alerts = detector.check("summarize")
    """

    def __init__(
        self,
        threshold_percent: float = 10.0,
        min_samples: int = 30,
        recent_window: int = 50,
        baseline_window: int = 200,
    ):
        """Initialize drift detector.

        Args:
            threshold_percent: Minimum drift % to trigger alert
            min_samples: Minimum samples before detection activates
            recent_window: Number of recent samples to compare
            baseline_window: Number of baseline samples
        """
        self.threshold_percent = threshold_percent
        self.min_samples = min_samples
        self.recent_window = recent_window
        self.baseline_window = baseline_window
        self._windows: Dict[str, Dict[str, MetricWindow]] = {}
        self._alerts: List[DriftAlert] = []

    def record(
        self,
        prompt_name: str,
        metric: str,
        value: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a metric value.

        Args:
            prompt_name: Prompt name
            metric: Metric name (e.g., "quality_score", "latency_ms")
            value: Metric value
            timestamp: Optional timestamp (defaults to now)
        """
        if prompt_name not in self._windows:
            self._windows[prompt_name] = {}
        if metric not in self._windows[prompt_name]:
            self._windows[prompt_name][metric] = MetricWindow()

        self._windows[prompt_name][metric].add(value, timestamp)

    def check(self, prompt_name: str) -> List[DriftAlert]:
        """Check for quality drift on a prompt.

        Args:
            prompt_name: Prompt to check

        Returns:
            List of DriftAlerts (empty if no drift detected)
        """
        alerts = []
        windows = self._windows.get(prompt_name, {})

        for metric, window in windows.items():
            if window.count < self.min_samples:
                continue

            baseline = window.baseline_mean(self.recent_window)
            current = window.recent_mean(self.recent_window)

            if baseline == 0:
                continue

            drift_percent = ((current - baseline) / baseline) * 100

            # For quality metrics, negative drift is bad
            # For cost/latency metrics, positive drift is bad
            is_quality_metric = "quality" in metric or "score" in metric
            is_degradation = (
                (is_quality_metric and drift_percent < -self.threshold_percent)
                or (not is_quality_metric and drift_percent > self.threshold_percent)
            )

            if is_degradation:
                severity = "critical" if abs(drift_percent) > 20 else "warning"
                alert = DriftAlert(
                    prompt_name=prompt_name,
                    metric=metric,
                    baseline_value=round(baseline, 4),
                    current_value=round(current, 4),
                    drift_percent=round(drift_percent, 2),
                    window_size=window.count,
                    severity=severity,
                )
                alerts.append(alert)
                self._alerts.append(alert)

        return alerts

    def check_all(self) -> List[DriftAlert]:
        """Check all monitored prompts for drift."""
        all_alerts = []
        for prompt_name in self._windows:
            alerts = self.check(prompt_name)
            all_alerts.extend(alerts)
        return all_alerts

    def get_baseline(self, prompt_name: str, metric: str) -> Optional[float]:
        """Get baseline value for a metric."""
        window = self._windows.get(prompt_name, {}).get(metric)
        if not window or window.count < self.min_samples:
            return None
        return window.baseline_mean(self.recent_window)

    def get_current(self, prompt_name: str, metric: str) -> Optional[float]:
        """Get current value for a metric."""
        window = self._windows.get(prompt_name, {}).get(metric)
        if not window or window.count == 0:
            return None
        return window.recent_mean(self.recent_window)

    @property
    def alert_history(self) -> List[DriftAlert]:
        return self._alerts

    def get_monitored_prompts(self) -> List[str]:
        """Get list of prompts being monitored."""
        return list(self._windows.keys())
