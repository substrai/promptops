"""Experiment definition and lifecycle management.

Defines A/B test experiments with variants, traffic allocation,
metrics, and success criteria.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import yaml


class ExperimentStatus(Enum):
    """Experiment lifecycle states."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentVariant:
    """A single variant in an A/B experiment."""

    name: str
    version: str
    traffic_percent: int
    description: str = ""

    def validate(self) -> List[str]:
        """Validate variant configuration."""
        errors = []
        if not self.name:
            errors.append("Variant must have a name")
        if not self.version:
            errors.append(f"Variant '{self.name}' must have a version")
        if self.traffic_percent < 0 or self.traffic_percent > 100:
            errors.append(
                f"Variant '{self.name}' traffic must be 0-100, got {self.traffic_percent}"
            )
        return errors


@dataclass
class SuccessCriterion:
    """A single success criterion for an experiment."""

    metric: str
    condition: str  # e.g., "treatment > control", "treatment <= control * 1.2"
    confidence: float = 0.95

    def evaluate(self, control_value: float, treatment_value: float) -> bool:
        """Evaluate if the criterion is met.

        Args:
            control_value: Metric value for control
            treatment_value: Metric value for treatment

        Returns:
            True if criterion is met
        """
        # Simple evaluation of conditions
        context = {"control": control_value, "treatment": treatment_value}
        try:
            return bool(eval(self.condition, {"__builtins__": {}}, context))
        except Exception:
            return False


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    name: str
    prompt_name: str
    duration_hours: int = 72
    variants: List[ExperimentVariant] = field(default_factory=list)
    primary_metric: str = "quality_score"
    secondary_metrics: List[str] = field(default_factory=list)
    success_criteria: List[SuccessCriterion] = field(default_factory=list)
    on_success: str = "promote_treatment"  # promote_treatment | keep_control | manual
    on_failure: str = "keep_control"

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "ExperimentConfig":
        """Parse experiment config from YAML."""
        data = yaml.safe_load(yaml_content)
        if not data:
            raise ValueError("Empty experiment config")

        exp_data = data.get("experiment", data)

        variants = []
        for v in exp_data.get("variants", []):
            variants.append(
                ExperimentVariant(
                    name=v.get("name", ""),
                    version=v.get("version", ""),
                    traffic_percent=v.get("traffic", 50),
                    description=v.get("description", ""),
                )
            )

        criteria = []
        for c in exp_data.get("success_criteria", []):
            criteria.append(
                SuccessCriterion(
                    metric=c.get("metric", ""),
                    condition=c.get("condition", ""),
                    confidence=c.get("confidence", 0.95),
                )
            )

        metrics_data = exp_data.get("metrics", {})

        return cls(
            name=exp_data.get("name", ""),
            prompt_name=exp_data.get("prompt", ""),
            duration_hours=exp_data.get("duration_hours", 72),
            variants=variants,
            primary_metric=metrics_data.get("primary", "quality_score"),
            secondary_metrics=metrics_data.get("secondary", []),
            success_criteria=criteria,
            on_success=exp_data.get("on_success", "promote_treatment"),
            on_failure=exp_data.get("on_failure", "keep_control"),
        )

    def validate(self) -> List[str]:
        """Validate the experiment configuration."""
        errors = []

        if not self.name:
            errors.append("Experiment must have a name")
        if not self.prompt_name:
            errors.append("Experiment must specify a prompt")
        if not self.variants:
            errors.append("Experiment must have at least 2 variants")
        elif len(self.variants) < 2:
            errors.append("Experiment must have at least 2 variants")

        # Validate traffic sums to 100
        total_traffic = sum(v.traffic_percent for v in self.variants)
        if total_traffic != 100:
            errors.append(f"Variant traffic must sum to 100%, got {total_traffic}%")

        # Validate each variant
        for variant in self.variants:
            errors.extend(variant.validate())

        return errors


@dataclass
class Experiment:
    """A running A/B experiment."""

    config: ExperimentConfig
    status: ExperimentStatus = ExperimentStatus.DRAFT
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    winner: Optional[str] = None
    invocation_counts: Dict[str, int] = field(default_factory=dict)
    metric_values: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def is_active(self) -> bool:
        return self.status == ExperimentStatus.RUNNING

    @property
    def elapsed_hours(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.ended_at or time.time()
        return (end - self.started_at) / 3600

    @property
    def is_expired(self) -> bool:
        return self.elapsed_hours >= self.config.duration_hours

    def start(self) -> None:
        """Start the experiment."""
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid experiment config: {'; '.join(errors)}")
        self.status = ExperimentStatus.RUNNING
        self.started_at = time.time()
        # Initialize counters
        for variant in self.config.variants:
            self.invocation_counts[variant.name] = 0
            self.metric_values[variant.name] = {}

    def stop(self, winner: Optional[str] = None) -> None:
        """Stop the experiment."""
        self.status = ExperimentStatus.COMPLETED
        self.ended_at = time.time()
        self.winner = winner

    def pause(self) -> None:
        """Pause the experiment."""
        self.status = ExperimentStatus.PAUSED

    def resume(self) -> None:
        """Resume a paused experiment."""
        self.status = ExperimentStatus.RUNNING

    def record_invocation(self, variant_name: str, metrics: Dict[str, float] = None) -> None:
        """Record an invocation for a variant.

        Args:
            variant_name: Which variant was used
            metrics: Metric values for this invocation
        """
        if variant_name in self.invocation_counts:
            self.invocation_counts[variant_name] += 1

        if metrics and variant_name in self.metric_values:
            for metric_name, value in metrics.items():
                if metric_name not in self.metric_values[variant_name]:
                    self.metric_values[variant_name][metric_name] = []
                self.metric_values[variant_name][metric_name].append(value)

    def get_variant_metrics(self, variant_name: str) -> Dict[str, float]:
        """Get average metrics for a variant."""
        result = {}
        variant_metrics = self.metric_values.get(variant_name, {})
        for metric_name, values in variant_metrics.items():
            if values:
                result[metric_name] = sum(values) / len(values)
        return result

    def summary(self) -> str:
        """Generate experiment summary."""
        lines = [
            f"Experiment: {self.name}",
            f"Status: {self.status.value}",
            f"Duration: {self.elapsed_hours:.1f}h / {self.config.duration_hours}h",
            "",
            "Variants:",
        ]
        for variant in self.config.variants:
            count = self.invocation_counts.get(variant.name, 0)
            metrics = self.get_variant_metrics(variant.name)
            metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
            lines.append(
                f"  {variant.name} ({variant.traffic_percent}%): "
                f"{count} invocations"
                + (f" | {metrics_str}" if metrics_str else "")
            )

        if self.winner:
            lines.append(f"\nWinner: {self.winner}")

        return "\n".join(lines)
