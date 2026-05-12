"""Traffic router for A/B experiments.

Routes incoming requests to experiment variants based on
configured traffic percentages.
"""

from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Optional

from promptops.experiments.experiment import Experiment, ExperimentVariant


class TrafficRouter:
    """Routes traffic to experiment variants.

    Supports:
    - Percentage-based traffic splitting
    - Consistent routing (same user always gets same variant)
    - Feature flag overrides

    Usage:
        router = TrafficRouter()
        router.register_experiment(experiment)
        variant = router.route("summarize", user_id="user-123")
    """

    def __init__(self):
        self._experiments: Dict[str, Experiment] = {}
        self._feature_flags: Dict[str, Dict[str, str]] = {}  # prompt -> {user -> variant}

    def register_experiment(self, experiment: Experiment) -> None:
        """Register an active experiment for routing.

        Args:
            experiment: The experiment to register
        """
        self._experiments[experiment.config.prompt_name] = experiment

    def unregister_experiment(self, prompt_name: str) -> None:
        """Remove an experiment from routing."""
        self._experiments.pop(prompt_name, None)

    def set_feature_flag(
        self, prompt_name: str, user_id: str, variant_name: str
    ) -> None:
        """Override routing for a specific user (feature flag).

        Args:
            prompt_name: Prompt name
            user_id: User identifier
            variant_name: Variant to always serve to this user
        """
        if prompt_name not in self._feature_flags:
            self._feature_flags[prompt_name] = {}
        self._feature_flags[prompt_name][user_id] = variant_name

    def clear_feature_flag(self, prompt_name: str, user_id: str) -> None:
        """Remove a feature flag override."""
        if prompt_name in self._feature_flags:
            self._feature_flags[prompt_name].pop(user_id, None)

    def route(
        self,
        prompt_name: str,
        user_id: Optional[str] = None,
    ) -> Optional[ExperimentVariant]:
        """Route a request to an experiment variant.

        Args:
            prompt_name: Prompt being invoked
            user_id: Optional user ID for consistent routing

        Returns:
            ExperimentVariant if experiment is active, None otherwise
        """
        experiment = self._experiments.get(prompt_name)
        if not experiment or not experiment.is_active:
            return None

        # Check feature flag override
        if user_id and prompt_name in self._feature_flags:
            override = self._feature_flags[prompt_name].get(user_id)
            if override:
                for variant in experiment.config.variants:
                    if variant.name == override:
                        return variant

        # Consistent routing based on user_id hash
        if user_id:
            bucket = self._hash_to_bucket(user_id, prompt_name)
        else:
            bucket = random.randint(0, 99)

        # Route based on traffic percentages
        cumulative = 0
        for variant in experiment.config.variants:
            cumulative += variant.traffic_percent
            if bucket < cumulative:
                return variant

        # Fallback to last variant
        return experiment.config.variants[-1] if experiment.config.variants else None

    def route_with_version(
        self,
        prompt_name: str,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Route and return the version string directly.

        Args:
            prompt_name: Prompt being invoked
            user_id: Optional user ID

        Returns:
            Version string if experiment active, None otherwise
        """
        variant = self.route(prompt_name, user_id)
        return variant.version if variant else None

    def get_active_experiments(self) -> List[str]:
        """Get list of prompts with active experiments."""
        return [
            name for name, exp in self._experiments.items() if exp.is_active
        ]

    def _hash_to_bucket(self, user_id: str, prompt_name: str) -> int:
        """Hash user_id + prompt to a consistent 0-99 bucket."""
        key = f"{user_id}:{prompt_name}"
        hash_bytes = hashlib.md5(key.encode()).digest()
        return int.from_bytes(hash_bytes[:2], "big") % 100
