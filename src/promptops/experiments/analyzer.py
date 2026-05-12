"""Experiment analyzer - determines winners and generates reports.

Analyzes experiment results, evaluates success criteria,
and determines the winning variant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from promptops.experiments.experiment import Experiment, ExperimentStatus, SuccessCriterion


@dataclass
class MetricComparison:
    """Comparison of a metric between variants."""

    metric_name: str
    control_value: float
    treatment_value: float
    difference: float
    difference_percent: float
    is_significant: bool
    criterion_met: bool


@dataclass
class ExperimentResult:
    """Final result of an experiment analysis."""

    experiment_name: str
    prompt_name: str
    winner: Optional[str]
    confidence: float
    comparisons: List[MetricComparison] = field(default_factory=list)
    recommendation: str = ""
    all_criteria_met: bool = False
    total_invocations: int = 0

    def summary(self) -> str:
        """Generate human-readable result summary."""
        lines = [
            f"Experiment Result: {self.experiment_name}",
            f"Prompt: {self.prompt_name}",
            f"Total invocations: {self.total_invocations}",
            f"Winner: {self.winner or 'No clear winner'}",
            f"Confidence: {self.confidence:.0%}",
            f"All criteria met: {'Yes' if self.all_criteria_met else 'No'}",
            "",
            "Metric Comparisons:",
        ]

        for comp in self.comparisons:
            direction = "↑" if comp.difference > 0 else "↓"
            lines.append(
                f"  {comp.metric_name}: control={comp.control_value:.4f}, "
                f"treatment={comp.treatment_value:.4f} "
                f"({direction}{abs(comp.difference_percent):.1f}%) "
                f"{'✓' if comp.criterion_met else '✗'}"
            )

        lines.append(f"\nRecommendation: {self.recommendation}")
        return "\n".join(lines)


class ExperimentAnalyzer:
    """Analyzes experiment results and determines winners.

    Usage:
        analyzer = ExperimentAnalyzer()
        result = analyzer.analyze(experiment)
        print(result.winner)
    """

    def analyze(self, experiment: Experiment) -> ExperimentResult:
        """Analyze an experiment and determine the winner.

        Args:
            experiment: The experiment to analyze

        Returns:
            ExperimentResult with winner and comparisons
        """
        config = experiment.config
        variants = config.variants

        if len(variants) < 2:
            return ExperimentResult(
                experiment_name=config.name,
                prompt_name=config.prompt_name,
                winner=None,
                confidence=0.0,
                recommendation="Insufficient variants for comparison",
            )

        # Identify control and treatment
        control = variants[0]
        treatment = variants[1]

        control_metrics = experiment.get_variant_metrics(control.name)
        treatment_metrics = experiment.get_variant_metrics(treatment.name)

        total_invocations = sum(experiment.invocation_counts.values())

        # Compare metrics
        comparisons = []
        all_criteria_met = True

        for criterion in config.success_criteria:
            control_value = control_metrics.get(criterion.metric, 0.0)
            treatment_value = treatment_metrics.get(criterion.metric, 0.0)

            difference = treatment_value - control_value
            diff_percent = (
                (difference / control_value * 100) if control_value != 0 else 0.0
            )

            # Check if criterion is met
            criterion_met = criterion.evaluate(control_value, treatment_value)
            if not criterion_met:
                all_criteria_met = False

            # Simple significance check (need enough samples)
            min_samples = 30
            control_count = experiment.invocation_counts.get(control.name, 0)
            treatment_count = experiment.invocation_counts.get(treatment.name, 0)
            is_significant = (
                control_count >= min_samples and treatment_count >= min_samples
            )

            comparisons.append(
                MetricComparison(
                    metric_name=criterion.metric,
                    control_value=control_value,
                    treatment_value=treatment_value,
                    difference=difference,
                    difference_percent=diff_percent,
                    is_significant=is_significant,
                    criterion_met=criterion_met,
                )
            )

        # Also compare primary metric if not in criteria
        if config.primary_metric not in [c.metric for c in config.success_criteria]:
            control_primary = control_metrics.get(config.primary_metric, 0.0)
            treatment_primary = treatment_metrics.get(config.primary_metric, 0.0)
            diff = treatment_primary - control_primary
            diff_pct = (diff / control_primary * 100) if control_primary != 0 else 0.0

            comparisons.append(
                MetricComparison(
                    metric_name=config.primary_metric,
                    control_value=control_primary,
                    treatment_value=treatment_primary,
                    difference=diff,
                    difference_percent=diff_pct,
                    is_significant=total_invocations >= 60,
                    criterion_met=diff >= 0,
                )
            )

        # Determine winner
        winner = None
        confidence = 0.0

        if all_criteria_met and total_invocations >= 60:
            winner = treatment.name
            confidence = 0.95
            recommendation = f"Promote '{treatment.name}' (v{treatment.version}) to replace control"
        elif not all_criteria_met and total_invocations >= 60:
            winner = control.name
            confidence = 0.90
            recommendation = f"Keep '{control.name}' (v{control.version}). Treatment did not meet criteria."
        else:
            recommendation = (
                f"Insufficient data ({total_invocations} invocations). "
                f"Need at least 60 for significance."
            )

        return ExperimentResult(
            experiment_name=config.name,
            prompt_name=config.prompt_name,
            winner=winner,
            confidence=confidence,
            comparisons=comparisons,
            recommendation=recommendation,
            all_criteria_met=all_criteria_met,
            total_invocations=total_invocations,
        )

    def should_stop_early(self, experiment: Experiment) -> Optional[str]:
        """Check if experiment should be stopped early.

        Returns reason to stop, or None to continue.
        """
        # Stop if expired
        if experiment.is_expired:
            return "Duration exceeded"

        # Stop if one variant is clearly worse (guardrail)
        total = sum(experiment.invocation_counts.values())
        if total < 30:
            return None

        # Check for extreme quality differences
        variants = experiment.config.variants
        if len(variants) >= 2:
            control_metrics = experiment.get_variant_metrics(variants[0].name)
            treatment_metrics = experiment.get_variant_metrics(variants[1].name)

            primary = experiment.config.primary_metric
            control_val = control_metrics.get(primary, 0.5)
            treatment_val = treatment_metrics.get(primary, 0.5)

            # Stop if treatment is >30% worse
            if control_val > 0 and treatment_val < control_val * 0.7:
                return f"Treatment significantly worse: {treatment_val:.3f} vs {control_val:.3f}"

        return None
