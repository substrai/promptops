"""Model pricing and cost calculation.

Maintains pricing tables for LLM providers and calculates
costs for prompt invocations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ModelPricing:
    """Pricing for a single model (per 1K tokens)."""

    provider: str
    model_id: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    context_window: int = 128000
    max_output_tokens: int = 4096
    supports_streaming: bool = True

    @property
    def full_name(self) -> str:
        return f"{self.provider}/{self.model_id}"

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token counts."""
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k
        return round(input_cost + output_cost, 8)


# Built-in pricing table (updated periodically)
PRICING_TABLE: Dict[str, ModelPricing] = {
    "bedrock/claude-3-haiku": ModelPricing(
        provider="bedrock", model_id="claude-3-haiku",
        input_cost_per_1k=0.00025, output_cost_per_1k=0.00125,
        context_window=200000, max_output_tokens=4096,
    ),
    "bedrock/claude-3-sonnet": ModelPricing(
        provider="bedrock", model_id="claude-3-sonnet",
        input_cost_per_1k=0.003, output_cost_per_1k=0.015,
        context_window=200000, max_output_tokens=4096,
    ),
    "bedrock/claude-3-5-sonnet": ModelPricing(
        provider="bedrock", model_id="claude-3-5-sonnet",
        input_cost_per_1k=0.003, output_cost_per_1k=0.015,
        context_window=200000, max_output_tokens=8192,
    ),
    "bedrock/claude-3-opus": ModelPricing(
        provider="bedrock", model_id="claude-3-opus",
        input_cost_per_1k=0.015, output_cost_per_1k=0.075,
        context_window=200000, max_output_tokens=4096,
    ),
    "bedrock/amazon-titan-text-lite": ModelPricing(
        provider="bedrock", model_id="amazon-titan-text-lite",
        input_cost_per_1k=0.00015, output_cost_per_1k=0.00015,
        context_window=4096, max_output_tokens=4096,
    ),
    "bedrock/amazon-titan-text-express": ModelPricing(
        provider="bedrock", model_id="amazon-titan-text-express",
        input_cost_per_1k=0.0008, output_cost_per_1k=0.0016,
        context_window=8192, max_output_tokens=8192,
    ),
    "bedrock/llama-3-8b": ModelPricing(
        provider="bedrock", model_id="llama-3-8b",
        input_cost_per_1k=0.0003, output_cost_per_1k=0.0006,
        context_window=8192, max_output_tokens=2048,
    ),
    "bedrock/llama-3-70b": ModelPricing(
        provider="bedrock", model_id="llama-3-70b",
        input_cost_per_1k=0.00265, output_cost_per_1k=0.0035,
        context_window=8192, max_output_tokens=2048,
    ),
    "openai/gpt-4o": ModelPricing(
        provider="openai", model_id="gpt-4o",
        input_cost_per_1k=0.005, output_cost_per_1k=0.015,
        context_window=128000, max_output_tokens=4096,
    ),
    "openai/gpt-4o-mini": ModelPricing(
        provider="openai", model_id="gpt-4o-mini",
        input_cost_per_1k=0.00015, output_cost_per_1k=0.0006,
        context_window=128000, max_output_tokens=16384,
    ),
    "openai/gpt-4-turbo": ModelPricing(
        provider="openai", model_id="gpt-4-turbo",
        input_cost_per_1k=0.01, output_cost_per_1k=0.03,
        context_window=128000, max_output_tokens=4096,
    ),
    "anthropic/claude-3-haiku": ModelPricing(
        provider="anthropic", model_id="claude-3-haiku",
        input_cost_per_1k=0.00025, output_cost_per_1k=0.00125,
        context_window=200000, max_output_tokens=4096,
    ),
    "anthropic/claude-3-5-sonnet": ModelPricing(
        provider="anthropic", model_id="claude-3-5-sonnet",
        input_cost_per_1k=0.003, output_cost_per_1k=0.015,
        context_window=200000, max_output_tokens=8192,
    ),
}


class CostCalculator:
    """Calculates and compares costs across models.

    Usage:
        calc = CostCalculator()
        cost = calc.estimate("bedrock/claude-3-haiku", input_tokens=500, output_tokens=200)
        cheapest = calc.find_cheapest(input_tokens=500, output_tokens=200)
    """

    def __init__(self, custom_pricing: Optional[Dict[str, ModelPricing]] = None):
        self._pricing = dict(PRICING_TABLE)
        if custom_pricing:
            self._pricing.update(custom_pricing)

    def estimate(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a specific model.

        Args:
            model: Model identifier (e.g., "bedrock/claude-3-haiku")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        pricing = self._pricing.get(model)
        if not pricing:
            # Default fallback pricing
            return (input_tokens / 1000) * 0.001 + (output_tokens / 1000) * 0.002
        return pricing.calculate_cost(input_tokens, output_tokens)

    def compare_costs(
        self, input_tokens: int, output_tokens: int, models: Optional[list] = None
    ) -> Dict[str, float]:
        """Compare costs across multiple models.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            models: Optional list of models to compare (all if None)

        Returns:
            Dict of model -> cost, sorted cheapest first
        """
        target_models = models or list(self._pricing.keys())
        costs = {}
        for model in target_models:
            if model in self._pricing:
                costs[model] = self.estimate(model, input_tokens, output_tokens)

        return dict(sorted(costs.items(), key=lambda x: x[1]))

    def find_cheapest(
        self,
        input_tokens: int,
        output_tokens: int,
        models: Optional[list] = None,
        max_cost: Optional[float] = None,
    ) -> Optional[str]:
        """Find the cheapest model that fits constraints.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            models: Optional list of candidate models
            max_cost: Optional maximum cost threshold

        Returns:
            Model identifier of cheapest option, or None
        """
        costs = self.compare_costs(input_tokens, output_tokens, models)
        for model, cost in costs.items():
            if max_cost is None or cost <= max_cost:
                return model
        return None

    def get_pricing(self, model: str) -> Optional[ModelPricing]:
        """Get pricing info for a model."""
        return self._pricing.get(model)

    def list_models(self, provider: Optional[str] = None) -> list:
        """List available models, optionally filtered by provider."""
        if provider:
            return [m for m in self._pricing.keys() if m.startswith(f"{provider}/")]
        return list(self._pricing.keys())
