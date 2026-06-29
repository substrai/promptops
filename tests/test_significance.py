"""Tests for prompt A/B test statistical significance calculator."""

from __future__ import annotations

import pytest

from promptops.experiments.significance import (
    SampleSizeEstimate,
    SignificanceCalculator,
    SignificanceResult,
)


class TestSignificanceCalculatorInit:
    """Test calculator initialization."""

    def test_default_confidence(self):
        calc = SignificanceCalculator()
        assert calc.confidence_level == 0.95
        assert abs(calc.alpha - 0.05) < 0.001

    def test_custom_confidence(self):
        calc = SignificanceCalculator(confidence_level=0.99)
        assert calc.confidence_level == 0.99
        assert abs(calc.alpha - 0.01) < 0.001

    def test_invalid_confidence_too_low(self):
        with pytest.raises(ValueError):
            SignificanceCalculator(confidence_level=0.5)

    def test_invalid_confidence_too_high(self):
        with pytest.raises(ValueError):
            SignificanceCalculator(confidence_level=1.0)


class TestSignificanceTest:
    """Test the main significance test."""

    def test_clearly_significant_result(self):
        calc = SignificanceCalculator(confidence_level=0.95)
        result = calc.test(
            control_successes=50, control_total=100,
            treatment_successes=75, treatment_total=100,
        )
        assert result.is_significant is True
        assert result.p_value < 0.05
        assert result.lift_percent > 0

    def test_clearly_not_significant(self):
        calc = SignificanceCalculator(confidence_level=0.95)
        result = calc.test(
            control_successes=50, control_total=100,
            treatment_successes=52, treatment_total=100,
        )
        assert result.is_significant is False
        assert result.p_value > 0.05

    def test_identical_rates(self):
        calc = SignificanceCalculator()
        result = calc.test(
            control_successes=80, control_total=100,
            treatment_successes=80, treatment_total=100,
        )
        assert result.is_significant is False
        assert result.lift_percent == 0.0

    def test_treatment_worse_than_control(self):
        calc = SignificanceCalculator()
        result = calc.test(
            control_successes=90, control_total=100,
            treatment_successes=60, treatment_total=100,
        )
        assert result.lift_percent < 0
        assert result.lift_absolute < 0

    def test_small_sample_not_adequate(self):
        calc = SignificanceCalculator(min_observations=30)
        result = calc.test(
            control_successes=5, control_total=10,
            treatment_successes=8, treatment_total=10,
        )
        assert result.sample_size_adequate is False
        # Even if p-value is low, not significant without adequate sample
        assert result.is_significant is False

    def test_rates_calculated_correctly(self):
        calc = SignificanceCalculator()
        result = calc.test(
            control_successes=80, control_total=200,
            treatment_successes=100, treatment_total=200,
        )
        assert abs(result.control_rate - 0.4) < 0.001
        assert abs(result.treatment_rate - 0.5) < 0.001

    def test_invalid_inputs_negative_successes(self):
        calc = SignificanceCalculator()
        with pytest.raises(ValueError):
            calc.test(control_successes=-1, control_total=100,
                      treatment_successes=50, treatment_total=100)

    def test_invalid_inputs_successes_exceed_total(self):
        calc = SignificanceCalculator()
        with pytest.raises(ValueError):
            calc.test(control_successes=101, control_total=100,
                      treatment_successes=50, treatment_total=100)

    def test_invalid_inputs_zero_total(self):
        calc = SignificanceCalculator()
        with pytest.raises(ValueError):
            calc.test(control_successes=0, control_total=0,
                      treatment_successes=50, treatment_total=100)


class TestConfidenceInterval:
    """Test confidence interval calculation."""

    def test_ci_contains_zero_when_not_significant(self):
        calc = SignificanceCalculator()
        result = calc.test(
            control_successes=50, control_total=100,
            treatment_successes=52, treatment_total=100,
        )
        # CI should span zero when not significant
        assert result.confidence_interval_lower < 0
        assert result.confidence_interval_upper > 0

    def test_ci_positive_when_treatment_clearly_better(self):
        calc = SignificanceCalculator()
        result = calc.test(
            control_successes=40, control_total=200,
            treatment_successes=80, treatment_total=200,
        )
        # Both bounds should be positive
        assert result.confidence_interval_lower > 0

    def test_ci_width_decreases_with_more_data(self):
        calc = SignificanceCalculator()
        result_small = calc.test(
            control_successes=50, control_total=100,
            treatment_successes=60, treatment_total=100,
        )
        result_large = calc.test(
            control_successes=500, control_total=1000,
            treatment_successes=600, treatment_total=1000,
        )
        width_small = result_small.confidence_interval_upper - result_small.confidence_interval_lower
        width_large = result_large.confidence_interval_upper - result_large.confidence_interval_lower
        assert width_large < width_small


class TestSampleSizeEstimation:
    """Test minimum sample size calculation."""

    def test_basic_sample_size(self):
        calc = SignificanceCalculator(confidence_level=0.95)
        estimate = calc.estimate_sample_size(
            baseline_rate=0.5,
            minimum_detectable_effect=0.05,
        )
        assert estimate.min_samples_per_variant > 100
        assert estimate.total_samples_needed == estimate.min_samples_per_variant * 2

    def test_smaller_effect_needs_more_samples(self):
        calc = SignificanceCalculator()
        small_effect = calc.estimate_sample_size(baseline_rate=0.5, minimum_detectable_effect=0.02)
        large_effect = calc.estimate_sample_size(baseline_rate=0.5, minimum_detectable_effect=0.10)
        assert small_effect.min_samples_per_variant > large_effect.min_samples_per_variant

    def test_higher_confidence_needs_more_samples(self):
        calc_95 = SignificanceCalculator(confidence_level=0.95)
        calc_99 = SignificanceCalculator(confidence_level=0.99)
        est_95 = calc_95.estimate_sample_size(baseline_rate=0.5, minimum_detectable_effect=0.05)
        est_99 = calc_99.estimate_sample_size(baseline_rate=0.5, minimum_detectable_effect=0.05)
        assert est_99.min_samples_per_variant > est_95.min_samples_per_variant

    def test_duration_estimation(self):
        calc = SignificanceCalculator()
        estimate = calc.estimate_sample_size(
            baseline_rate=0.5,
            minimum_detectable_effect=0.05,
            daily_traffic=500,
        )
        assert estimate.estimated_duration_days is not None
        assert estimate.estimated_duration_days > 0

    def test_invalid_baseline_rate(self):
        calc = SignificanceCalculator()
        with pytest.raises(ValueError):
            calc.estimate_sample_size(baseline_rate=0.0, minimum_detectable_effect=0.05)

    def test_invalid_effect_size(self):
        calc = SignificanceCalculator()
        with pytest.raises(ValueError):
            calc.estimate_sample_size(baseline_rate=0.5, minimum_detectable_effect=0)


class TestRecommendations:
    """Test recommendation generation."""

    def test_significant_positive_recommends_treatment(self):
        calc = SignificanceCalculator()
        result = calc.test(
            control_successes=50, control_total=200,
            treatment_successes=80, treatment_total=200,
        )
        assert "promot" in result.recommendation.lower() or "better" in result.recommendation.lower()

    def test_significant_negative_recommends_control(self):
        calc = SignificanceCalculator()
        result = calc.test(
            control_successes=80, control_total=200,
            treatment_successes=50, treatment_total=200,
        )
        assert "control" in result.recommendation.lower() or "worse" in result.recommendation.lower()

    def test_insufficient_data_recommends_continue(self):
        calc = SignificanceCalculator(min_observations=100)
        result = calc.test(
            control_successes=8, control_total=10,
            treatment_successes=9, treatment_total=10,
        )
        assert "insufficient" in result.recommendation.lower() or "continue" in result.recommendation.lower()
