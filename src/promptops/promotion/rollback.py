"""Rollback manager for instant version rollback.

Provides one-command rollback to any previous version with
full audit trail.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from promptops.core.version import PromptVersion
from promptops.promotion.environments import DeploymentState, Environment


@dataclass
class RollbackRecord:
    """Record of a rollback operation."""

    prompt_name: str
    environment: str
    from_version: str
    to_version: str
    reason: str
    rolled_back_at: float
    rolled_back_by: str = "system"
    automatic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "prompt_name": self.prompt_name,
            "environment": self.environment,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "reason": self.reason,
            "rolled_back_at": self.rolled_back_at,
            "rolled_back_by": self.rolled_back_by,
            "automatic": self.automatic,
        }


class RollbackManager:
    """Manages prompt version rollbacks.

    Supports:
    - Instant rollback to previous version
    - Rollback to any specific version
    - Automatic rollback on quality degradation
    - Full audit trail of all rollbacks

    Usage:
        manager = RollbackManager()
        result = manager.rollback("summarize", env="prod", to_version="1.1.0", reason="quality drop")
    """

    def __init__(self):
        self._version_history: Dict[str, Dict[str, List[str]]] = {}
        self._rollback_log: List[RollbackRecord] = []
        self._deployments: Dict[str, Dict[str, DeploymentState]] = {}

    def record_deployment(
        self, prompt_name: str, version: str, environment: str
    ) -> None:
        """Record a deployment for rollback tracking.

        Args:
            prompt_name: Prompt name
            version: Version deployed
            environment: Target environment
        """
        if prompt_name not in self._version_history:
            self._version_history[prompt_name] = {}
        if environment not in self._version_history[prompt_name]:
            self._version_history[prompt_name][environment] = []

        history = self._version_history[prompt_name][environment]
        if not history or history[-1] != version:
            history.append(version)

        # Update deployment state
        if prompt_name not in self._deployments:
            self._deployments[prompt_name] = {}

        previous = self._deployments[prompt_name].get(environment)
        self._deployments[prompt_name][environment] = DeploymentState(
            prompt_name=prompt_name,
            environment=Environment.from_string(environment),
            active_version=version,
            previous_version=previous.active_version if previous else None,
            deployed_at=str(time.time()),
        )

    def rollback(
        self,
        prompt_name: str,
        environment: str,
        to_version: Optional[str] = None,
        reason: str = "manual rollback",
        rolled_back_by: str = "user",
    ) -> RollbackRecord:
        """Rollback a prompt to a previous version.

        Args:
            prompt_name: Prompt name
            environment: Environment to rollback in
            to_version: Target version (None = previous version)
            reason: Reason for rollback
            rolled_back_by: Who initiated the rollback

        Returns:
            RollbackRecord with details

        Raises:
            ValueError: If no previous version available
        """
        current_state = self._deployments.get(prompt_name, {}).get(environment)
        if not current_state:
            raise ValueError(
                f"No deployment found for '{prompt_name}' in '{environment}'"
            )

        current_version = current_state.active_version

        # Determine target version
        if to_version:
            target = to_version
        elif current_state.previous_version:
            target = current_state.previous_version
        else:
            # Look in history
            history = self._version_history.get(prompt_name, {}).get(environment, [])
            if len(history) < 2:
                raise ValueError(
                    f"No previous version available for '{prompt_name}' in '{environment}'"
                )
            target = history[-2]

        # Execute rollback
        record = RollbackRecord(
            prompt_name=prompt_name,
            environment=environment,
            from_version=current_version,
            to_version=target,
            reason=reason,
            rolled_back_at=time.time(),
            rolled_back_by=rolled_back_by,
            automatic=rolled_back_by == "system",
        )

        # Update deployment state
        self._deployments[prompt_name][environment] = DeploymentState(
            prompt_name=prompt_name,
            environment=Environment.from_string(environment),
            active_version=target,
            previous_version=current_version,
            deployed_at=str(time.time()),
            status="active",
        )

        self._rollback_log.append(record)
        return record

    def auto_rollback(
        self,
        prompt_name: str,
        environment: str,
        quality_score: float,
        threshold: float = 0.85,
    ) -> Optional[RollbackRecord]:
        """Automatically rollback if quality drops below threshold.

        Args:
            prompt_name: Prompt name
            environment: Environment
            quality_score: Current quality score
            threshold: Minimum acceptable quality

        Returns:
            RollbackRecord if rollback triggered, None otherwise
        """
        if quality_score >= threshold:
            return None

        try:
            return self.rollback(
                prompt_name=prompt_name,
                environment=environment,
                reason=f"Auto-rollback: quality {quality_score:.2%} < threshold {threshold:.2%}",
                rolled_back_by="system",
            )
        except ValueError:
            return None

    def get_history(
        self, prompt_name: str, environment: str
    ) -> List[str]:
        """Get version history for a prompt in an environment."""
        return self._version_history.get(prompt_name, {}).get(environment, [])

    def get_rollback_log(
        self, prompt_name: Optional[str] = None
    ) -> List[RollbackRecord]:
        """Get rollback log, optionally filtered by prompt."""
        if prompt_name:
            return [r for r in self._rollback_log if r.prompt_name == prompt_name]
        return self._rollback_log

    def get_current_version(self, prompt_name: str, environment: str) -> Optional[str]:
        """Get the currently deployed version."""
        state = self._deployments.get(prompt_name, {}).get(environment)
        return state.active_version if state else None
