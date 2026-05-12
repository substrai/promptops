"""Environment configuration and management.

Defines the environment hierarchy and promotion rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Environment(Enum):
    """Standard deployment environments."""

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"

    @property
    def next(self) -> Optional["Environment"]:
        """Get the next environment in the promotion chain."""
        order = [Environment.DEV, Environment.STAGING, Environment.PROD]
        idx = order.index(self)
        return order[idx + 1] if idx < len(order) - 1 else None

    @property
    def previous(self) -> Optional["Environment"]:
        """Get the previous environment."""
        order = [Environment.DEV, Environment.STAGING, Environment.PROD]
        idx = order.index(self)
        return order[idx - 1] if idx > 0 else None

    @classmethod
    def from_string(cls, value: str) -> "Environment":
        """Parse environment from string."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Unknown environment: '{value}'. Must be one of: dev, staging, prod"
            )


@dataclass
class EnvironmentConfig:
    """Configuration for a single environment."""

    name: Environment
    auto_deploy: bool = False
    approval_required: bool = False
    approvers: List[str] = field(default_factory=list)
    run_tests: bool = True
    quality_gate: float = 0.95
    model_override: Optional[str] = None
    canary: Optional["CanaryConfig"] = None
    max_rollback_versions: int = 5

    @classmethod
    def from_dict(cls, env_name: str, config: Dict[str, Any]) -> "EnvironmentConfig":
        """Parse from dictionary."""
        env = Environment.from_string(env_name)
        canary_data = config.get("canary")
        canary = CanaryConfig.from_dict(canary_data) if canary_data else None

        return cls(
            name=env,
            auto_deploy=config.get("auto_deploy", False),
            approval_required=config.get("approval_required", False),
            approvers=config.get("approvers", []),
            run_tests=config.get("run_tests", True),
            quality_gate=config.get("quality_gate", 0.95),
            model_override=config.get("model_override"),
            canary=canary,
            max_rollback_versions=config.get("max_rollback_versions", 5),
        )

    @classmethod
    def defaults(cls) -> Dict[str, "EnvironmentConfig"]:
        """Get default environment configurations."""
        return {
            "dev": cls(
                name=Environment.DEV,
                auto_deploy=True,
                run_tests=False,
                quality_gate=0.8,
            ),
            "staging": cls(
                name=Environment.STAGING,
                run_tests=True,
                quality_gate=0.90,
            ),
            "prod": cls(
                name=Environment.PROD,
                approval_required=True,
                run_tests=True,
                quality_gate=0.95,
                canary=CanaryConfig(),
            ),
        }


@dataclass
class CanaryConfig:
    """Configuration for canary deployments."""

    enabled: bool = True
    initial_traffic_percent: int = 5
    increment_percent: int = 25
    interval_minutes: int = 30
    rollback_on_error_rate: float = 0.05  # 5% error rate triggers rollback
    rollback_on_quality_drop: float = 0.10  # 10% quality drop triggers rollback

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanaryConfig":
        """Parse from dictionary."""
        if not data:
            return cls(enabled=False)
        return cls(
            enabled=data.get("enabled", True),
            initial_traffic_percent=data.get("initial_traffic", 5),
            increment_percent=data.get("increment", 25),
            interval_minutes=data.get("interval_minutes", 30),
            rollback_on_error_rate=data.get("rollback_on_error_rate", 0.05),
            rollback_on_quality_drop=data.get("rollback_on_quality_drop", 0.10),
        )


@dataclass
class DeploymentState:
    """Current deployment state for a prompt in an environment."""

    prompt_name: str
    environment: Environment
    active_version: str
    previous_version: Optional[str] = None
    canary_version: Optional[str] = None
    canary_traffic_percent: int = 0
    deployed_at: Optional[str] = None
    deployed_by: str = "system"
    status: str = "active"  # active, canary, rolling_back, pending_approval

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "prompt_name": self.prompt_name,
            "environment": self.environment.value,
            "active_version": self.active_version,
            "previous_version": self.previous_version,
            "canary_version": self.canary_version,
            "canary_traffic_percent": self.canary_traffic_percent,
            "deployed_at": self.deployed_at,
            "deployed_by": self.deployed_by,
            "status": self.status,
        }
