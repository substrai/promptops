"""Tests for the observability module."""

import time
import pytest
from promptops.observability.analytics import InvocationAnalytics, InvocationRecord
from promptops.observability.audit import AuditTrail, AuditAction, AuditQuery
from promptops.observability.alerts import (
    AlertManager, AlertRule, AlertCondition, AlertSeverity,
)
from promptops.observability.quotas import (
    QuotaManager, UsageQuota, QuotaPeriod, QuotaStatus,
)
from promptops.observability.drift import QualityDriftDetector


class TestInvocationAnalytics:
    def setup_method(self):
        self.analytics = InvocationAnalytics()

    def test_record_and_summarize(self):
        now = time.time()
        for i in range(10):
            self.analytics.record(InvocationRecord(
                prompt_name="summarize", version="1.0.0", environment="prod",
                model="bedrock/claude-3-haiku", timestamp=now - i,
                latency_ms=100 + i * 10, input_tokens=500, output_tokens=200,
                cost=0.001, quality_score=0.9, success=True,
            ))

        summary = self.analytics.get_summary("summarize", period="1h")
        assert summary.total_invocations == 10
        assert summary.successful_invocations == 10
        assert summary.total_cost > 0
        assert summary.avg_quality_score == 0.9

    def test_error_rate(self):
        now = time.time()
        for i in range(8):
            self.analytics.record(InvocationRecord(
                prompt_name="test", version="1.0.0", environment="prod",
                model="m", timestamp=now, latency_ms=100,
                input_tokens=100, output_tokens=50, cost=0.001, success=True,
            ))
        for i in range(2):
            self.analytics.record(InvocationRecord(
                prompt_name="test", version="1.0.0", environment="prod",
                model="m", timestamp=now, latency_ms=100,
                input_tokens=100, output_tokens=50, cost=0.001, success=False,
            ))

        rate = self.analytics.get_error_rate("test", period="1h")
        assert abs(rate - 0.2) < 0.01

    def test_cost_by_prompt(self):
        now = time.time()
        self.analytics.record(InvocationRecord(
            prompt_name="a", version="1.0.0", environment="prod",
            model="m", timestamp=now, latency_ms=100,
            input_tokens=100, output_tokens=50, cost=0.01, success=True,
        ))
        self.analytics.record(InvocationRecord(
            prompt_name="b", version="1.0.0", environment="prod",
            model="m", timestamp=now, latency_ms=100,
            input_tokens=100, output_tokens=50, cost=0.05, success=True,
        ))

        costs = self.analytics.get_cost_by_prompt("1h")
        assert "b" in costs
        assert costs["b"] > costs["a"]

    def test_filter_by_environment(self):
        now = time.time()
        self.analytics.record(InvocationRecord(
            prompt_name="test", version="1.0.0", environment="prod",
            model="m", timestamp=now, latency_ms=100,
            input_tokens=100, output_tokens=50, cost=0.001, success=True,
        ))
        self.analytics.record(InvocationRecord(
            prompt_name="test", version="1.0.0", environment="dev",
            model="m", timestamp=now, latency_ms=100,
            input_tokens=100, output_tokens=50, cost=0.001, success=True,
        ))

        summary = self.analytics.get_summary("test", period="1h", environment="prod")
        assert summary.total_invocations == 1


class TestAuditTrail:
    def setup_method(self):
        self.audit = AuditTrail()

    def test_log_entry(self):
        entry = self.audit.log(
            AuditAction.DEPLOYED, "summarize",
            version="1.2.0", environment="prod",
            actor="gaurav@substrai.dev", reason="Phase 2 release",
        )
        assert entry.action == AuditAction.DEPLOYED
        assert entry.prompt_name == "summarize"

    def test_query_by_prompt(self):
        self.audit.log(AuditAction.CREATED, "summarize", actor="user1")
        self.audit.log(AuditAction.CREATED, "classify", actor="user2")
        self.audit.log(AuditAction.DEPLOYED, "summarize", actor="user1")

        results = self.audit.query(AuditQuery(prompt_name="summarize"))
        assert len(results) == 2

    def test_query_by_actor(self):
        self.audit.log(AuditAction.CREATED, "a", actor="alice")
        self.audit.log(AuditAction.CREATED, "b", actor="bob")
        self.audit.log(AuditAction.DEPLOYED, "a", actor="alice")

        results = self.audit.get_by_actor("alice")
        assert len(results) == 2

    def test_generate_report(self):
        self.audit.log(AuditAction.CREATED, "summarize", actor="user1")
        self.audit.log(AuditAction.DEPLOYED, "summarize", environment="prod", actor="user1")
        self.audit.log(AuditAction.ROLLED_BACK, "summarize", environment="prod", actor="user2")

        report = self.audit.generate_report("summarize", period_days=30)
        assert report["total_entries"] == 3
        assert "created" in report["action_breakdown"]
        assert len(report["actors"]) == 2

    def test_entry_summary_line(self):
        entry = self.audit.log(AuditAction.PROMOTED, "summarize", version="1.0.0", environment="prod")
        line = entry.summary_line()
        assert "promoted" in line
        assert "summarize" in line


class TestAlertManager:
    def setup_method(self):
        self.manager = AlertManager()
        self.manager.add_rule(AlertRule(
            name="high-error-rate",
            condition=AlertCondition.ERROR_RATE_ABOVE,
            threshold=0.05,
            severity=AlertSeverity.CRITICAL,
        ))
        self.manager.add_rule(AlertRule(
            name="quality-drop",
            condition=AlertCondition.QUALITY_BELOW,
            threshold=0.80,
            severity=AlertSeverity.WARNING,
        ))

    def test_alert_fires(self):
        alerts = self.manager.check("summarize", {"error_rate": 0.10, "quality": 0.95})
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_alert_not_fired(self):
        alerts = self.manager.check("summarize", {"error_rate": 0.02, "quality": 0.95})
        assert len(alerts) == 0

    def test_quality_alert(self):
        alerts = self.manager.check("summarize", {"quality": 0.70})
        assert len(alerts) == 1
        assert alerts[0].rule_name == "quality-drop"

    def test_cooldown_prevents_duplicate(self):
        self.manager.check("summarize", {"error_rate": 0.10})
        alerts2 = self.manager.check("summarize", {"error_rate": 0.10})
        assert len(alerts2) == 0  # cooldown active

    def test_handler_called(self):
        received = []
        self.manager.add_handler(lambda alert: received.append(alert))
        self.manager.check("summarize", {"error_rate": 0.10, "quality": 0.95})
        assert len(received) == 1

    def test_resolve_alerts(self):
        self.manager.check("summarize", {"error_rate": 0.10, "quality": 0.95})
        assert len(self.manager.get_active_alerts()) == 1
        self.manager.resolve("high-error-rate")
        assert len(self.manager.get_active_alerts()) == 0

    def test_remove_rule(self):
        assert self.manager.remove_rule("high-error-rate")
        alerts = self.manager.check("summarize", {"error_rate": 0.10, "quality": 0.95})
        assert len(alerts) == 0


class TestQuotaManager:
    def setup_method(self):
        self.manager = QuotaManager()
        self.manager.add_quota(UsageQuota(
            name="team-daily-cost",
            entity_type="team",
            entity_id="ml-team",
            period=QuotaPeriod.DAILY,
            max_cost=1.0,
        ))
        self.manager.add_quota(UsageQuota(
            name="user-invocations",
            entity_type="user",
            entity_id="user-1",
            period=QuotaPeriod.HOURLY,
            max_invocations=100,
        ))

    def test_within_quota(self):
        results = self.manager.check_and_record("team", "ml-team", cost=0.05)
        assert all(r.is_allowed for r in results)

    def test_quota_exceeded(self):
        # Use up the budget
        for _ in range(20):
            self.manager.check_and_record("team", "ml-team", cost=0.05)

        results = self.manager.check_and_record("team", "ml-team", cost=0.05)
        blocked = [r for r in results if r.status == QuotaStatus.BLOCKED]
        assert len(blocked) == 1

    def test_warning_threshold(self):
        # Use 85% of budget
        for _ in range(17):
            self.manager.check_and_record("team", "ml-team", cost=0.05)

        results = self.manager.check("team", "ml-team")
        warnings = [r for r in results if r.status == QuotaStatus.WARNING]
        assert len(warnings) == 1

    def test_invocation_quota(self):
        for _ in range(100):
            self.manager.check_and_record("user", "user-1", invocations=1)

        results = self.manager.check("user", "user-1")
        blocked = [r for r in results if r.status == QuotaStatus.BLOCKED]
        assert len(blocked) == 1

    def test_reset_usage(self):
        self.manager.check_and_record("team", "ml-team", cost=0.50)
        self.manager.reset_usage("team", "ml-team")
        results = self.manager.check("team", "ml-team")
        assert all(r.status == QuotaStatus.ALLOWED for r in results)

    def test_usage_report(self):
        self.manager.check_and_record("team", "ml-team", cost=0.10, tokens=500)
        report = self.manager.get_usage_report()
        assert "team:ml-team" in report
        assert report["team:ml-team"]["cost"] == 0.10


class TestQualityDriftDetector:
    def setup_method(self):
        self.detector = QualityDriftDetector(
            threshold_percent=10.0, min_samples=20, recent_window=10
        )

    def test_no_drift_stable_quality(self):
        for i in range(50):
            self.detector.record("summarize", "quality_score", 0.90)

        alerts = self.detector.check("summarize")
        assert len(alerts) == 0

    def test_drift_detected(self):
        # Baseline: high quality
        for i in range(40):
            self.detector.record("summarize", "quality_score", 0.90)
        # Recent: quality dropped
        for i in range(15):
            self.detector.record("summarize", "quality_score", 0.70)

        alerts = self.detector.check("summarize")
        assert len(alerts) == 1
        assert alerts[0].drift_percent < 0
        assert "summarize" in alerts[0].summary()

    def test_insufficient_samples(self):
        for i in range(5):
            self.detector.record("summarize", "quality_score", 0.50)

        alerts = self.detector.check("summarize")
        assert len(alerts) == 0  # not enough data

    def test_latency_drift(self):
        # Baseline: low latency
        for i in range(40):
            self.detector.record("summarize", "latency_ms", 100)
        # Recent: latency increased
        for i in range(15):
            self.detector.record("summarize", "latency_ms", 200)

        alerts = self.detector.check("summarize")
        assert len(alerts) == 1
        assert alerts[0].drift_percent > 0

    def test_get_baseline_and_current(self):
        for i in range(50):
            self.detector.record("summarize", "quality_score", 0.90)

        baseline = self.detector.get_baseline("summarize", "quality_score")
        current = self.detector.get_current("summarize", "quality_score")
        assert baseline is not None
        assert current is not None
        assert abs(baseline - 0.90) < 0.01

    def test_check_all(self):
        for i in range(50):
            self.detector.record("prompt-a", "quality_score", 0.90)
            self.detector.record("prompt-b", "quality_score", 0.85)

        alerts = self.detector.check_all()
        assert isinstance(alerts, list)

    def test_monitored_prompts(self):
        self.detector.record("a", "quality", 0.9)
        self.detector.record("b", "quality", 0.8)
        prompts = self.detector.get_monitored_prompts()
        assert "a" in prompts
        assert "b" in prompts
