"""Fallback chain for model resilience.

If the primary model fails or times out, automatically
falls back to the next model in the chain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class FallbackResult:
    """Result of a fallback chain execution."""

    success: bool
    model_used: str
    attempt_number: int
    total_attempts: int
    output: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    fallback_triggered: bool = False
    attempt_log: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        status = "✓" if self.success else "✗"
        fb = " (fallback)" if self.fallback_triggered else ""
        return (
            f"{status} Model: {self.model_used}{fb} "
            f"(attempt {self.attempt_number}/{self.total_attempts}, "
            f"{self.latency_ms:.0f}ms)"
        )


class FallbackChain:
    """Executes a prompt with automatic model fallback.

    If the primary model fails, tries the next model in the chain
    until one succeeds or all fail.

    Usage:
        chain = FallbackChain(
            models=["bedrock/claude-3-sonnet", "bedrock/claude-3-haiku", "bedrock/amazon-titan-text-express"],
            max_retries_per_model=1,
            timeout_ms=5000,
        )
        result = chain.execute(invoke_fn, prompt="Hello")
    """

    def __init__(
        self,
        models: List[str],
        max_retries_per_model: int = 1,
        timeout_ms: int = 10000,
        on_fallback: Optional[Callable[[str, str, str], None]] = None,
    ):
        """Initialize fallback chain.

        Args:
            models: Ordered list of models to try (primary first)
            max_retries_per_model: Max retries before moving to next model
            timeout_ms: Timeout per attempt in milliseconds
            on_fallback: Optional callback(from_model, to_model, error)
        """
        if not models:
            raise ValueError("Fallback chain must have at least one model")
        self.models = models
        self.max_retries = max_retries_per_model
        self.timeout_ms = timeout_ms
        self.on_fallback = on_fallback
        self._stats: Dict[str, Dict[str, int]] = {
            m: {"attempts": 0, "successes": 0, "failures": 0} for m in models
        }

    def execute(
        self,
        invoke_fn: Callable[[str, str], Any],
        prompt: str,
    ) -> FallbackResult:
        """Execute with fallback chain.

        Args:
            invoke_fn: Function(model, prompt) -> response that may raise
            prompt: The rendered prompt to send

        Returns:
            FallbackResult with the outcome
        """
        attempt_log = []
        total_start = time.time()

        for model_idx, model in enumerate(self.models):
            for retry in range(self.max_retries + 1):
                attempt_start = time.time()
                attempt_num = sum(
                    len(a) for a in [[]] * model_idx
                ) + retry + 1

                try:
                    self._stats[model]["attempts"] += 1
                    output = invoke_fn(model, prompt)
                    latency = (time.time() - attempt_start) * 1000

                    self._stats[model]["successes"] += 1

                    attempt_log.append({
                        "model": model,
                        "retry": retry,
                        "success": True,
                        "latency_ms": latency,
                    })

                    return FallbackResult(
                        success=True,
                        model_used=model,
                        attempt_number=model_idx + 1,
                        total_attempts=len(self.models),
                        output=output,
                        latency_ms=(time.time() - total_start) * 1000,
                        fallback_triggered=model_idx > 0,
                        attempt_log=attempt_log,
                    )

                except Exception as e:
                    latency = (time.time() - attempt_start) * 1000
                    self._stats[model]["failures"] += 1

                    attempt_log.append({
                        "model": model,
                        "retry": retry,
                        "success": False,
                        "error": str(e),
                        "latency_ms": latency,
                    })

            # Moving to next model - trigger callback
            if model_idx < len(self.models) - 1 and self.on_fallback:
                next_model = self.models[model_idx + 1]
                self.on_fallback(model, next_model, str(attempt_log[-1].get("error", "")))

        # All models failed
        total_latency = (time.time() - total_start) * 1000
        return FallbackResult(
            success=False,
            model_used=self.models[-1],
            attempt_number=len(self.models),
            total_attempts=len(self.models),
            error="All models in fallback chain failed",
            latency_ms=total_latency,
            fallback_triggered=True,
            attempt_log=attempt_log,
        )

    @property
    def stats(self) -> Dict[str, Dict[str, int]]:
        """Get execution statistics per model."""
        return self._stats

    def success_rate(self, model: str) -> float:
        """Get success rate for a specific model."""
        stats = self._stats.get(model, {"attempts": 0, "successes": 0})
        if stats["attempts"] == 0:
            return 0.0
        return stats["successes"] / stats["attempts"]
