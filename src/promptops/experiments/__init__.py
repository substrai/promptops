"""A/B testing and experimentation framework for PromptOps.

Enables running experiments to compare prompt variants with
traffic splitting, metric collection, and auto-promotion.
"""

from promptops.experiments.experiment import (
    Experiment,
    ExperimentVariant,
    ExperimentConfig,
    ExperimentStatus,
)
from promptops.experiments.router import TrafficRouter
from promptops.experiments.analyzer import ExperimentAnalyzer, ExperimentResult

__all__ = [
    "Experiment",
    "ExperimentVariant",
    "ExperimentConfig",
    "ExperimentStatus",
    "TrafficRouter",
    "ExperimentAnalyzer",
    "ExperimentResult",
]
