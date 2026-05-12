"""Cost-aware model router.

Routes prompt invocations to the optimal model based on
cost, quality, and latency requirements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from promptops.models.pricing import CostCalculator, ModelPricing


class RoutingStrategy(Enum):
    """Model routing strategies."""

    COST_OPTIMIZED = "cost-optimized"  # Cheapest model meeting quality threshold
    QUALITY_FIRST = "quality-first"  # Best quality model within budget
    BALANCED = "balanced"  # Balance cost and quality
    FIXED = "fixed"  # Always use specified model


@dataclass
class RoutingDecision:
    """Result of a model routing decision."""

    selected_model: str
    strategy: RoutingStrategy
    estimated_cost: float
    reason: str
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    quality_score: Optional[float] = None

    def summary(self) -> str:
        return (
            f"Selected: {self.selected_model} "
            f"(${self.estimated_cost:.6f}, strategy={self.strategy.value})"
        )


@dataclass
class ModelQualityProfile:
    """Quality profile for a model on a specific prompt type."""

    model: str
    quality_score: float  # 0.0 to 1.0
    avg_latency_ms: float = 500.0
    reliability: float = 0.99  # uptime/success rate


class ModelRouter:
    """Routes requests to optimal model based on strategy.

    Usage:
        router = ModelRouter(strategy=RoutingStrategy.COST_OPTIMIZED)
        decision = router.route(
            input_tokens=500,
            output_tokens=200,
            candidates=["bedrock/claude-3-haiku", "bedrock/claude-3-sonnet"],
            quality_threshold=0.85,
        )
    """

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.COST_OPTIMIZED,
        quality_profiles: Optional[Dict[str, ModelQualityProfile]] = None,
        max_cost_per_request: Optional[float] = None,
        quality_threshold: float = 0.7,
    ):
        self.strategy = strategy
        self.calculator = CostCalculator()
        self.max_cost = max_cost_per_request
        self.quality_threshold = quality_threshold
        self._quality_profiles = quality_profiles or self._default_profiles()

    def route(
        self,
        input_tokens: int,
        output_tokens: int,
        candidates: Optional[List[str]] = None,
        quality_threshold: Optional[float] = None,
        max_cost: Optional[float] = None,
    ) -> RoutingDecision:
        """Route to the optimal model.

        Args:
            input_tokens: Estimated input token count
            output_tokens: Estimated output token count
            candidates: List of candidate models (all if None)
            quality_threshold: Minimum quality score (overrides default)
            max_cost: Maximum cost per request (overrides default)

        Returns:
            RoutingDecision with selected model and reasoning
        """
        threshold = quality_threshold or self.quality_threshold
        budget = max_cost or self.max_cost
        models = candidates or list(self._quality_profiles.keys())

        # Calculate costs and filter
        options = []
        for model in models:
            cost = self.calculator.estimate(model, input_tokens, output_tokens)
            profile = self._quality_profiles.get(model)
            quality = profile.quality_score if profile else 0.5

            # Apply filters
            if budget and cost > budget:
                continue
            if quality < threshold:
                continue

            options.append({
                "model": model,
                "cost": cost,
                "quality": quality,
                "latency": profile.avg_latency_ms if profile else 500.0,
            })

        if not options:
            # Fallback: pick cheapest regardless of quality
            costs = self.calculator.compare_costs(input_tokens, output_tokens, models)
            if costs:
                cheapest = next(iter(costs))
                return RoutingDecision(
                    selected_model=cheapest,
                    strategy=self.strategy,
                    estimated_cost=costs[cheapest],
                    reason="No model met all constraints; using cheapest available",
                    alternatives=[],
                )
            return RoutingDecision(
                selected_model=models[0] if models else "bedrock/claude-3-haiku",
                strategy=self.strategy,
                estimated_cost=0.0,
                reason="No pricing data available; using default",
                alternatives=[],
            )

        # Apply strategy
        if self.strategy == RoutingStrategy.COST_OPTIMIZED:
            options.sort(key=lambda x: x["cost"])
            reason = "Cheapest model meeting quality threshold"
        elif self.strategy == RoutingStrategy.QUALITY_FIRST:
            options.sort(key=lambda x: -x["quality"])
            reason = "Highest quality model within budget"
        elif self.strategy == RoutingStrategy.BALANCED:
            # Score = quality / normalized_cost
            max_cost_val = max(o["cost"] for o in options) or 1
            for o in options:
                o["score"] = o["quality"] / (o["cost"] / max_cost_val + 0.01)
            options.sort(key=lambda x: -x.get("score", 0))
            reason = "Best quality-to-cost ratio"
        else:  # FIXED
            reason = "Fixed model selection"

        selected = options[0]
        alternatives = options[1:4]  # top 3 alternatives

        return RoutingDecision(
            selected_model=selected["model"],
            strategy=self.strategy,
            estimated_cost=selected["cost"],
            reason=reason,
            quality_score=selected["quality"],
            alternatives=[
                {"model": a["model"], "cost": a["cost"], "quality": a["quality"]}
                for a in alternatives
            ],
        )

    def update_quality_profile(
        self, model: str, quality_score: float, latency_ms: float = 500.0
    ) -> None:
        """Update quality profile for a model (from experiment results)."""
        self._quality_profiles[model] = ModelQualityProfile(
            model=model,
            quality_score=quality_score,
            avg_latency_ms=latency_ms,
        )

    def _default_profiles(self) -> Dict[str, ModelQualityProfile]:
        """Default quality profiles based on general benchmarks."""
        return {
            "bedrock/claude-3-opus": ModelQualityProfile("bedrock/claude-3-opus", 0.95, 2000),
            "bedrock/claude-3-5-sonnet": ModelQualityProfile("bedrock/claude-3-5-sonnet", 0.92, 1200),
            "bedrock/claude-3-sonnet": ModelQualityProfile("bedrock/claude-3-sonnet", 0.88, 1000),
            "bedrock/claude-3-haiku": ModelQualityProfile("bedrock/claude-3-haiku", 0.80, 400),
            "bedrock/amazon-titan-text-express": ModelQualityProfile("bedrock/amazon-titan-text-express", 0.72, 300),
            "bedrock/amazon-titan-text-lite": ModelQualityProfile("bedrock/amazon-titan-text-lite", 0.65, 200),
            "bedrock/llama-3-70b": ModelQualityProfile("bedrock/llama-3-70b", 0.85, 800),
            "bedrock/llama-3-8b": ModelQualityProfile("bedrock/llama-3-8b", 0.72, 300),
            "openai/gpt-4o": ModelQualityProfile("openai/gpt-4o", 0.93, 1500),
            "openai/gpt-4o-mini": ModelQualityProfile("openai/gpt-4o-mini", 0.82, 400),
            "openai/gpt-4-turbo": ModelQualityProfile("openai/gpt-4-turbo", 0.91, 2000),
            "anthropic/claude-3-5-sonnet": ModelQualityProfile("anthropic/claude-3-5-sonnet", 0.92, 1200),
            "anthropic/claude-3-haiku": ModelQualityProfile("anthropic/claude-3-haiku", 0.80, 400),
        }
