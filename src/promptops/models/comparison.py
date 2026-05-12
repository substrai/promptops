"""Model comparison - run same inputs across models and compare results.

Enables side-by-side comparison of quality, cost, and latency
across different models for the same prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from promptops.models.pricing import CostCalculator


@dataclass
class ModelScore:
    """Score for a single model on a comparison run."""

    model: str
    quality_score: float
    cost: float
    latency_ms: float
    output_sample: str = ""
    token_count: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    @property
    def value_score(self) -> float:
        """Quality per dollar (higher = better value)."""
        if self.cost == 0:
            return float('inf') if self.quality_score > 0 else 0.0
        return self.quality_score / self.cost


@dataclass
class ComparisonResult:
    """Result of comparing multiple models."""

    prompt_name: str
    models_compared: List[str]
    scores: List[ModelScore]
    best_quality: Optional[str] = None
    best_value: Optional[str] = None
    cheapest: Optional[str] = None

    @property
    def ranked_by_quality(self) -> List[ModelScore]:
        return sorted(self.scores, key=lambda x: -x.quality_score)

    @property
    def ranked_by_cost(self) -> List[ModelScore]:
        return sorted(self.scores, key=lambda x: x.cost)

    @property
    def ranked_by_value(self) -> List[ModelScore]:
        return sorted(self.scores, key=lambda x: -x.value_score)

    def summary(self) -> str:
        lines = [
            f"Model Comparison: {self.prompt_name}",
            f"Models tested: {len(self.models_compared)}",
            "",
            f"{'Model':<35} {'Quality':<10} {'Cost':<12} {'Latency':<10} {'Value':<10}",
            "-" * 77,
        ]

        for score in self.ranked_by_quality:
            value = f"{score.value_score:.0f}" if score.value_score != float('inf') else "∞"
            lines.append(
                f"{score.model:<35} {score.quality_score:<10.3f} "
                f"${score.cost:<11.6f} {score.latency_ms:<10.0f} {value:<10}"
            )

        lines.append("")
        lines.append(f"Best quality: {self.best_quality}")
        lines.append(f"Best value:   {self.best_value}")
        lines.append(f"Cheapest:     {self.cheapest}")

        return "\n".join(lines)


class ModelComparison:
    """Compares models side-by-side on the same inputs.

    Usage:
        comparison = ModelComparison(
            models=["bedrock/claude-3-haiku", "bedrock/claude-3-sonnet", "openai/gpt-4o-mini"],
        )
        result = comparison.run(
            prompt_name="summarize",
            inputs=[{"document": "...", "max_words": 100}],
            quality_fn=lambda output, inputs: score_quality(output),
        )
    """

    def __init__(
        self,
        models: List[str],
        calculator: Optional[CostCalculator] = None,
    ):
        self.models = models
        self.calculator = calculator or CostCalculator()

    def run(
        self,
        prompt_name: str,
        inputs: List[Dict[str, Any]],
        invoke_fn: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        quality_fn: Optional[Callable[[Any, Dict[str, Any]], float]] = None,
    ) -> ComparisonResult:
        """Run comparison across all models.

        Args:
            prompt_name: Name of the prompt being compared
            inputs: List of input sets to test
            invoke_fn: Function(model, inputs) -> output (optional, uses estimation if None)
            quality_fn: Function(output, inputs) -> quality_score (0-1)

        Returns:
            ComparisonResult with scores for each model
        """
        scores = []

        for model in self.models:
            model_scores: List[float] = []
            total_cost = 0.0
            total_latency = 0.0
            total_tokens = 0
            errors: List[str] = []
            sample_output = ""

            for input_set in inputs:
                # Estimate tokens from input
                input_text = " ".join(str(v) for v in input_set.values())
                input_tokens = len(input_text) // 4
                output_tokens = 500  # estimated average

                # Calculate cost
                cost = self.calculator.estimate(model, input_tokens, output_tokens)
                total_cost += cost
                total_tokens += input_tokens + output_tokens

                # Invoke if function provided
                if invoke_fn:
                    try:
                        import time
                        start = time.time()
                        output = invoke_fn(model, input_set)
                        latency = (time.time() - start) * 1000
                        total_latency += latency

                        if quality_fn:
                            quality = quality_fn(output, input_set)
                            model_scores.append(quality)

                        if not sample_output:
                            sample_output = str(output)[:200]
                    except Exception as e:
                        errors.append(str(e))
                else:
                    # Estimation mode - use default quality profiles
                    total_latency += 500  # estimated
                    model_scores.append(self._estimated_quality(model))

            avg_quality = sum(model_scores) / len(model_scores) if model_scores else 0.0
            avg_latency = total_latency / len(inputs) if inputs else 0.0

            scores.append(ModelScore(
                model=model,
                quality_score=avg_quality,
                cost=total_cost,
                latency_ms=avg_latency,
                output_sample=sample_output,
                token_count=total_tokens,
                errors=errors,
            ))

        # Determine winners
        successful = [s for s in scores if s.success]
        best_quality = max(successful, key=lambda x: x.quality_score).model if successful else None
        best_value = max(successful, key=lambda x: x.value_score).model if successful else None
        cheapest = min(successful, key=lambda x: x.cost).model if successful else None

        return ComparisonResult(
            prompt_name=prompt_name,
            models_compared=self.models,
            scores=scores,
            best_quality=best_quality,
            best_value=best_value,
            cheapest=cheapest,
        )

    def _estimated_quality(self, model: str) -> float:
        """Default quality estimates for models."""
        estimates = {
            "bedrock/claude-3-opus": 0.95,
            "bedrock/claude-3-5-sonnet": 0.92,
            "bedrock/claude-3-sonnet": 0.88,
            "bedrock/claude-3-haiku": 0.80,
            "bedrock/amazon-titan-text-express": 0.72,
            "bedrock/amazon-titan-text-lite": 0.65,
            "bedrock/llama-3-70b": 0.85,
            "bedrock/llama-3-8b": 0.72,
            "openai/gpt-4o": 0.93,
            "openai/gpt-4o-mini": 0.82,
            "openai/gpt-4-turbo": 0.91,
            "anthropic/claude-3-5-sonnet": 0.92,
            "anthropic/claude-3-haiku": 0.80,
        }
        return estimates.get(model, 0.75)
