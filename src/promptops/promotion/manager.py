"""Promotion manager - handles environment promotion lifecycle.

Manages the flow: dev → staging → prod with quality gates,
approval workflows, and canary deployments.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from promptops.core.version import PromptVersion
from promptops.promotion.environments import (
    CanaryConfig,
    DeploymentState,
    Environment,
    EnvironmentConfig,
)


@dataclass
class PromotionResult:
    """Result of a promotion attempt."""

    success: bool
    prompt_name: str
    version: str
    from_env: Environment
    to_env: Environment
    message: str
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    requires_approval: bool = False
    approval_id: Optional[str] = None
    canary_active: bool = False

    @property
    def blocked(self) -> bool:
        """Check if promotion was blocked."""
        return not self.success and not self.requires_approval

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "✓" if self.success else ("⏳" if self.requires_approval else "✗")
        lines = [
            f"{status} Promote {self.prompt_name}@{self.version}: "
            f"{self.from_env.value} → {self.to_env.value}",
            f"  Status: {self.message}",
        ]
        if self.checks_passed:
            lines.append(f"  Passed: {', '.join(self.checks_passed)}")
        if self.checks_failed:
            lines.append(f"  Failed: {', '.join(self.checks_failed)}")
        if self.requires_approval:
            lines.append(f"  Awaiting approval (ID: {self.approval_id})")
        if self.canary_active:
            lines.append("  Canary deployment active")
        return "\n".join(lines)


class PromotionManager:
    """Manages prompt promotion through environments.

    Usage:
        manager = PromotionManager(configs)
        result = manager.promote("summarize", "1.2.0", from_env="dev", to_env="staging")
    """

    def __init__(
        self,
        env_configs: Optional[Dict[str, EnvironmentConfig]] = None,
    ):
        """Initialize the promotion manager.

        Args:
            env_configs: Environment configurations (defaults used if None)
        """
        self.env_configs = env_configs or EnvironmentConfig.defaults()
        self._deployments: Dict[str, Dict[str, DeploymentState]] = {}
        self._approval_queue: List[Dict[str, Any]] = []
        self._promotion_history: List[PromotionResult] = []

    def promote(
        self,
        prompt_name: str,
        version: str,
        from_env: str,
        to_env: str,
        test_results: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> PromotionResult:
        """Promote a prompt version from one environment to another.

        Args:
            prompt_name: Name of the prompt
            version: Version to promote
            from_env: Source environment
            to_env: Target environment
            test_results: Optional test results (pass_rate, etc.)
            force: Skip quality gates (not recommended for prod)

        Returns:
            PromotionResult with success/failure details
        """
        source = Environment.from_string(from_env)
        target = Environment.from_string(to_env)
        target_config = self.env_configs.get(to_env)

        if not target_config:
            return PromotionResult(
                success=False,
                prompt_name=prompt_name,
                version=version,
                from_env=source,
                to_env=target,
                message=f"Unknown target environment: {to_env}",
            )

        checks_passed = []
        checks_failed = []

        # Check 1: Validate promotion path
        if not self._is_valid_promotion_path(source, target):
            return PromotionResult(
                success=False,
                prompt_name=prompt_name,
                version=version,
                from_env=source,
                to_env=target,
                message=f"Invalid promotion path: {from_env} → {to_env}. "
                f"Must follow: dev → staging → prod",
                checks_failed=["promotion_path"],
            )
        checks_passed.append("promotion_path")

        # Check 2: Quality gate
        if target_config.run_tests and not force:
            pass_rate = (test_results or {}).get("pass_rate", 0.0)
            if pass_rate < target_config.quality_gate:
                checks_failed.append("quality_gate")
                return PromotionResult(
                    success=False,
                    prompt_name=prompt_name,
                    version=version,
                    from_env=source,
                    to_env=target,
                    message=f"Quality gate failed: {pass_rate:.0%} < {target_config.quality_gate:.0%}",
                    checks_passed=checks_passed,
                    checks_failed=checks_failed,
                )
            checks_passed.append("quality_gate")

        # Check 3: Approval required
        if target_config.approval_required and not force:
            approval_id = f"approval-{prompt_name}-{version}-{int(time.time())}"
            self._approval_queue.append({
                "id": approval_id,
                "prompt_name": prompt_name,
                "version": version,
                "to_env": to_env,
                "approvers": target_config.approvers,
                "requested_at": time.time(),
                "status": "pending",
            })
            result = PromotionResult(
                success=False,
                prompt_name=prompt_name,
                version=version,
                from_env=source,
                to_env=target,
                message="Awaiting approval",
                checks_passed=checks_passed,
                requires_approval=True,
                approval_id=approval_id,
            )
            self._promotion_history.append(result)
            return result

        checks_passed.append("approval")

        # Check 4: Canary deployment
        canary_active = False
        if target_config.canary and target_config.canary.enabled:
            canary_active = True
            self._start_canary(prompt_name, version, target, target_config.canary)
            checks_passed.append("canary_started")

        # Deploy
        self._deploy(prompt_name, version, target)

        result = PromotionResult(
            success=True,
            prompt_name=prompt_name,
            version=version,
            from_env=source,
            to_env=target,
            message="Promoted successfully" + (" (canary active)" if canary_active else ""),
            checks_passed=checks_passed,
            canary_active=canary_active,
        )
        self._promotion_history.append(result)
        return result

    def approve(self, approval_id: str, approver: str) -> PromotionResult:
        """Approve a pending promotion.

        Args:
            approval_id: The approval ID from the pending promotion
            approver: Email/ID of the approver

        Returns:
            PromotionResult after approval
        """
        for item in self._approval_queue:
            if item["id"] == approval_id and item["status"] == "pending":
                item["status"] = "approved"
                item["approved_by"] = approver
                item["approved_at"] = time.time()

                # Execute the promotion
                target = Environment.from_string(item["to_env"])
                self._deploy(item["prompt_name"], item["version"], target)

                return PromotionResult(
                    success=True,
                    prompt_name=item["prompt_name"],
                    version=item["version"],
                    from_env=target.previous or Environment.DEV,
                    to_env=target,
                    message=f"Approved by {approver} and deployed",
                    checks_passed=["approval"],
                )

        return PromotionResult(
            success=False,
            prompt_name="unknown",
            version="unknown",
            from_env=Environment.DEV,
            to_env=Environment.PROD,
            message=f"Approval ID not found or already processed: {approval_id}",
        )

    def reject(self, approval_id: str, rejector: str, reason: str = "") -> bool:
        """Reject a pending promotion."""
        for item in self._approval_queue:
            if item["id"] == approval_id and item["status"] == "pending":
                item["status"] = "rejected"
                item["rejected_by"] = rejector
                item["rejection_reason"] = reason
                return True
        return False

    def get_deployment_state(
        self, prompt_name: str, environment: str
    ) -> Optional[DeploymentState]:
        """Get current deployment state for a prompt in an environment."""
        env_deployments = self._deployments.get(prompt_name, {})
        return env_deployments.get(environment)

    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get all pending approval requests."""
        return [a for a in self._approval_queue if a["status"] == "pending"]

    @property
    def history(self) -> List[PromotionResult]:
        """Get promotion history."""
        return self._promotion_history

    def _is_valid_promotion_path(self, source: Environment, target: Environment) -> bool:
        """Validate that the promotion path is valid."""
        valid_paths = {
            (Environment.DEV, Environment.STAGING),
            (Environment.STAGING, Environment.PROD),
            (Environment.DEV, Environment.PROD),  # Allow skip for hotfixes
        }
        return (source, target) in valid_paths

    def _deploy(self, prompt_name: str, version: str, environment: Environment) -> None:
        """Record a deployment."""
        if prompt_name not in self._deployments:
            self._deployments[prompt_name] = {}

        previous = self._deployments[prompt_name].get(environment.value)
        previous_version = previous.active_version if previous else None

        self._deployments[prompt_name][environment.value] = DeploymentState(
            prompt_name=prompt_name,
            environment=environment,
            active_version=version,
            previous_version=previous_version,
            deployed_at=str(time.time()),
            status="active",
        )

    def _start_canary(
        self,
        prompt_name: str,
        version: str,
        environment: Environment,
        canary_config: CanaryConfig,
    ) -> None:
        """Start a canary deployment."""
        if prompt_name not in self._deployments:
            self._deployments[prompt_name] = {}

        current = self._deployments[prompt_name].get(environment.value)
        if current:
            current.canary_version = version
            current.canary_traffic_percent = canary_config.initial_traffic_percent
            current.status = "canary"
