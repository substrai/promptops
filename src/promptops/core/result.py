"""Invocation result model for PromptOps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from promptops.core.version import PromptVersion


@dataclass
class InvocationResult:
    """Result of a prompt invocation.

    Contains the output, metadata about the invocation (cost, latency,
    tokens used), and the version that was resolved.
    """

    output: Any
    prompt_name: str
    version: PromptVersion
    model: str
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    environment: str = "dev"
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: list = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if the invocation was successful."""
        return len(self.errors) == 0

    @property
    def total_cost_display(self) -> str:
        """Format cost for display."""
        return f"${self.cost:.6f}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "output": self.output,
            "prompt_name": self.prompt_name,
            "version": str(self.version),
            "model": self.model,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "environment": self.environment,
            "cached": self.cached,
            "success": self.success,
            "metadata": self.metadata,
        }
