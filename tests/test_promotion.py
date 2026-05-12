"""Tests for the promotion manager."""

import pytest
from promptops.promotion.manager import PromotionManager, PromotionResult
from promptops.promotion.environments import Environment, EnvironmentConfig
from promptops.promotion.rollback import RollbackManager


class TestPromotionManager:
    def setup_method(self):
        self.manager = PromotionManager()

    def test_promote_dev_to_staging(self):
        result = self.manager.promote(
            "summarize", "1.0.0",
            from_env="dev", to_env="staging",
            test_results={"pass_rate": 0.96},
        )
        assert result.success
        assert "quality_gate" in result.checks_passed

    def test_promote_fails_quality_gate(self):
        result = self.manager.promote(
            "summarize", "1.0.0",
            from_env="dev", to_env="staging",
            test_results={"pass_rate": 0.50},
        )
        assert not result.success
        assert "quality_gate" in result.checks_failed

    def test_promote_to_prod_requires_approval(self):
        result = self.manager.promote(
            "summarize", "1.0.0",
            from_env="staging", to_env="prod",
            test_results={"pass_rate": 0.98},
        )
        assert not result.success
        assert result.requires_approval
        assert result.approval_id is not None

    def test_approve_promotion(self):
        # Request promotion
        result = self.manager.promote(
            "summarize", "1.0.0",
            from_env="staging", to_env="prod",
            test_results={"pass_rate": 0.98},
        )
        assert result.requires_approval

        # Approve it
        approved = self.manager.approve(result.approval_id, "admin@company.com")
        assert approved.success

    def test_reject_promotion(self):
        result = self.manager.promote(
            "summarize", "1.0.0",
            from_env="staging", to_env="prod",
            test_results={"pass_rate": 0.98},
        )
        rejected = self.manager.reject(result.approval_id, "admin@company.com", "Not ready")
        assert rejected

    def test_invalid_promotion_path(self):
        result = self.manager.promote(
            "summarize", "1.0.0",
            from_env="prod", to_env="dev",
        )
        assert not result.success
        assert "promotion_path" in result.checks_failed

    def test_force_skip_quality_gate(self):
        result = self.manager.promote(
            "summarize", "1.0.0",
            from_env="dev", to_env="staging",
            test_results={"pass_rate": 0.50},
            force=True,
        )
        assert result.success

    def test_deployment_state_tracked(self):
        self.manager.promote(
            "summarize", "1.0.0",
            from_env="dev", to_env="staging",
            test_results={"pass_rate": 0.96},
        )
        state = self.manager.get_deployment_state("summarize", "staging")
        assert state is not None
        assert state.active_version == "1.0.0"

    def test_pending_approvals(self):
        self.manager.promote(
            "summarize", "1.0.0",
            from_env="staging", to_env="prod",
            test_results={"pass_rate": 0.98},
        )
        pending = self.manager.get_pending_approvals()
        assert len(pending) == 1
        assert pending[0]["prompt_name"] == "summarize"


class TestRollbackManager:
    def setup_method(self):
        self.manager = RollbackManager()

    def test_rollback_to_previous(self):
        self.manager.record_deployment("summarize", "1.0.0", "prod")
        self.manager.record_deployment("summarize", "1.1.0", "prod")

        record = self.manager.rollback("summarize", "prod", reason="quality drop")
        assert record.from_version == "1.1.0"
        assert record.to_version == "1.0.0"

    def test_rollback_to_specific_version(self):
        self.manager.record_deployment("summarize", "1.0.0", "prod")
        self.manager.record_deployment("summarize", "1.1.0", "prod")
        self.manager.record_deployment("summarize", "1.2.0", "prod")

        record = self.manager.rollback("summarize", "prod", to_version="1.0.0")
        assert record.to_version == "1.0.0"

    def test_rollback_no_previous_raises(self):
        self.manager.record_deployment("summarize", "1.0.0", "prod")
        with pytest.raises(ValueError):
            self.manager.rollback("summarize", "prod")

    def test_auto_rollback_triggered(self):
        self.manager.record_deployment("summarize", "1.0.0", "prod")
        self.manager.record_deployment("summarize", "1.1.0", "prod")

        record = self.manager.auto_rollback("summarize", "prod", quality_score=0.60, threshold=0.85)
        assert record is not None
        assert record.automatic
        assert record.to_version == "1.0.0"

    def test_auto_rollback_not_triggered(self):
        self.manager.record_deployment("summarize", "1.0.0", "prod")
        self.manager.record_deployment("summarize", "1.1.0", "prod")

        record = self.manager.auto_rollback("summarize", "prod", quality_score=0.92, threshold=0.85)
        assert record is None

    def test_rollback_log(self):
        self.manager.record_deployment("summarize", "1.0.0", "prod")
        self.manager.record_deployment("summarize", "1.1.0", "prod")
        self.manager.rollback("summarize", "prod", reason="test")

        log = self.manager.get_rollback_log("summarize")
        assert len(log) == 1
        assert log[0].reason == "test"

    def test_get_current_version(self):
        self.manager.record_deployment("summarize", "1.0.0", "prod")
        assert self.manager.get_current_version("summarize", "prod") == "1.0.0"


class TestEnvironment:
    def test_next_environment(self):
        assert Environment.DEV.next == Environment.STAGING
        assert Environment.STAGING.next == Environment.PROD
        assert Environment.PROD.next is None

    def test_previous_environment(self):
        assert Environment.PROD.previous == Environment.STAGING
        assert Environment.STAGING.previous == Environment.DEV
        assert Environment.DEV.previous is None

    def test_from_string(self):
        assert Environment.from_string("dev") == Environment.DEV
        assert Environment.from_string("PROD") == Environment.PROD

    def test_invalid_environment(self):
        with pytest.raises(ValueError):
            Environment.from_string("invalid")
