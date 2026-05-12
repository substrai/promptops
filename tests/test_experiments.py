"""Tests for the A/B testing / experiments module."""

import pytest
from promptops.experiments.experiment import (
    Experiment,
    ExperimentConfig,
    ExperimentVariant,
    ExperimentStatus,
    SuccessCriterion,
)
from promptops.experiments.router import TrafficRouter
from promptops.experiments.analyzer import ExperimentAnalyzer


EXPERIMENT_YAML = """
experiment:
  name: "summarize-v2-test"
  prompt: summarize
  duration_hours: 72

  variants:
    - name: control
      version: "1.2.0"
      traffic: 70
    - name: treatment
      version: "2.0.0-rc1"
      traffic: 30

  metrics:
    primary: quality_score
    secondary: [latency_p95, cost_per_request]

  success_criteria:
    - metric: quality_score
      condition: "treatment > control"
      confidence: 0.95
    - metric: cost_per_request
      condition: "treatment <= control * 1.2"
      confidence: 0.90

  on_success: promote_treatment
  on_failure: keep_control
"""


class TestExperimentConfig:
    def test_parse_yaml(self):
        config = ExperimentConfig.from_yaml(EXPERIMENT_YAML)
        assert config.name == "summarize-v2-test"
        assert config.prompt_name == "summarize"
        assert len(config.variants) == 2
        assert config.variants[0].traffic_percent == 70
        assert config.variants[1].traffic_percent == 30

    def test_validate_valid_config(self):
        config = ExperimentConfig.from_yaml(EXPERIMENT_YAML)
        errors = config.validate()
        assert errors == []

    def test_validate_traffic_not_100(self):
        config = ExperimentConfig(
            name="test",
            prompt_name="test",
            variants=[
                ExperimentVariant(name="a", version="1.0.0", traffic_percent=60),
                ExperimentVariant(name="b", version="2.0.0", traffic_percent=60),
            ],
        )
        errors = config.validate()
        assert any("100%" in e for e in errors)

    def test_validate_single_variant(self):
        config = ExperimentConfig(
            name="test",
            prompt_name="test",
            variants=[ExperimentVariant(name="a", version="1.0.0", traffic_percent=100)],
        )
        errors = config.validate()
        assert any("at least 2" in e for e in errors)


class TestExperiment:
    def test_start_experiment(self):
        config = ExperimentConfig.from_yaml(EXPERIMENT_YAML)
        exp = Experiment(config=config)
        exp.start()
        assert exp.status == ExperimentStatus.RUNNING
        assert exp.started_at is not None

    def test_record_invocation(self):
        config = ExperimentConfig.from_yaml(EXPERIMENT_YAML)
        exp = Experiment(config=config)
        exp.start()
        exp.record_invocation("control", {"quality_score": 0.85, "cost_per_request": 0.002})
        exp.record_invocation("control", {"quality_score": 0.90, "cost_per_request": 0.001})
        assert exp.invocation_counts["control"] == 2

    def test_get_variant_metrics(self):
        config = ExperimentConfig.from_yaml(EXPERIMENT_YAML)
        exp = Experiment(config=config)
        exp.start()
        exp.record_invocation("control", {"quality_score": 0.80})
        exp.record_invocation("control", {"quality_score": 0.90})
        metrics = exp.get_variant_metrics("control")
        assert abs(metrics["quality_score"] - 0.85) < 0.001  # average

    def test_stop_experiment(self):
        config = ExperimentConfig.from_yaml(EXPERIMENT_YAML)
        exp = Experiment(config=config)
        exp.start()
        exp.stop(winner="treatment")
        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.winner == "treatment"

    def test_pause_resume(self):
        config = ExperimentConfig.from_yaml(EXPERIMENT_YAML)
        exp = Experiment(config=config)
        exp.start()
        exp.pause()
        assert exp.status == ExperimentStatus.PAUSED
        exp.resume()
        assert exp.status == ExperimentStatus.RUNNING


class TestTrafficRouter:
    def setup_method(self):
        config = ExperimentConfig.from_yaml(EXPERIMENT_YAML)
        self.experiment = Experiment(config=config)
        self.experiment.start()
        self.router = TrafficRouter()
        self.router.register_experiment(self.experiment)

    def test_routes_to_variant(self):
        variant = self.router.route("summarize", user_id="user-1")
        assert variant is not None
        assert variant.name in ["control", "treatment"]

    def test_consistent_routing(self):
        # Same user should always get same variant
        results = set()
        for _ in range(10):
            variant = self.router.route("summarize", user_id="user-fixed")
            results.add(variant.name)
        assert len(results) == 1  # always same variant

    def test_traffic_distribution(self):
        # With many random users, should approximate 70/30 split
        control_count = 0
        total = 1000
        for i in range(total):
            variant = self.router.route("summarize", user_id=f"user-{i}")
            if variant.name == "control":
                control_count += 1
        # Should be roughly 70% (allow 10% tolerance)
        assert 600 < control_count < 800

    def test_feature_flag_override(self):
        self.router.set_feature_flag("summarize", "vip-user", "treatment")
        variant = self.router.route("summarize", user_id="vip-user")
        assert variant.name == "treatment"

    def test_no_experiment_returns_none(self):
        variant = self.router.route("nonexistent", user_id="user-1")
        assert variant is None

    def test_unregister_experiment(self):
        self.router.unregister_experiment("summarize")
        variant = self.router.route("summarize", user_id="user-1")
        assert variant is None


class TestExperimentAnalyzer:
    def test_analyze_treatment_wins(self):
        config = ExperimentConfig.from_yaml(EXPERIMENT_YAML)
        exp = Experiment(config=config)
        exp.start()

        # Simulate invocations where treatment is better
        for i in range(50):
            exp.record_invocation("control", {"quality_score": 0.80, "cost_per_request": 0.003})
        for i in range(30):
            exp.record_invocation("treatment", {"quality_score": 0.90, "cost_per_request": 0.002})

        analyzer = ExperimentAnalyzer()
        result = analyzer.analyze(exp)
        assert result.winner == "treatment"
        assert result.all_criteria_met

    def test_analyze_control_wins(self):
        config = ExperimentConfig.from_yaml(EXPERIMENT_YAML)
        exp = Experiment(config=config)
        exp.start()

        # Simulate invocations where control is better
        for i in range(50):
            exp.record_invocation("control", {"quality_score": 0.90, "cost_per_request": 0.002})
        for i in range(30):
            exp.record_invocation("treatment", {"quality_score": 0.70, "cost_per_request": 0.005})

        analyzer = ExperimentAnalyzer()
        result = analyzer.analyze(exp)
        assert result.winner == "control"
        assert not result.all_criteria_met

    def test_analyze_insufficient_data(self):
        config = ExperimentConfig.from_yaml(EXPERIMENT_YAML)
        exp = Experiment(config=config)
        exp.start()

        # Only a few invocations
        exp.record_invocation("control", {"quality_score": 0.85})
        exp.record_invocation("treatment", {"quality_score": 0.90})

        analyzer = ExperimentAnalyzer()
        result = analyzer.analyze(exp)
        assert result.winner is None
        assert "Insufficient" in result.recommendation

    def test_should_stop_early(self):
        config = ExperimentConfig.from_yaml(EXPERIMENT_YAML)
        exp = Experiment(config=config)
        exp.start()

        # Treatment is terrible
        for i in range(40):
            exp.record_invocation("control", {"quality_score": 0.90})
        for i in range(20):
            exp.record_invocation("treatment", {"quality_score": 0.30})

        analyzer = ExperimentAnalyzer()
        reason = analyzer.should_stop_early(exp)
        assert reason is not None
        assert "worse" in reason.lower()


class TestSuccessCriterion:
    def test_treatment_greater(self):
        criterion = SuccessCriterion(metric="quality", condition="treatment > control")
        assert criterion.evaluate(0.80, 0.90)
        assert not criterion.evaluate(0.90, 0.80)

    def test_cost_within_budget(self):
        criterion = SuccessCriterion(metric="cost", condition="treatment <= control * 1.2")
        assert criterion.evaluate(0.003, 0.003)  # equal
        assert criterion.evaluate(0.003, 0.0035)  # within 20%
        assert not criterion.evaluate(0.003, 0.005)  # exceeds 20%
