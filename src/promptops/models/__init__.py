"""Multi-model targeting and optimization for PromptOps.

Enables same logical prompt with optimized variants per model,
cost-aware routing, fallback chains, and token optimization.
"""

from promptops.models.router import ModelRouter, RoutingStrategy, RoutingDecision
from promptops.models.fallback import FallbackChain, FallbackResult
from promptops.models.optimizer import TokenOptimizer, OptimizationSuggestion
from promptops.models.comparison import ModelComparison, ComparisonResult
from promptops.models.pricing import ModelPricing, CostCalculator

__all__ = [
    "ModelRouter",
    "RoutingStrategy",
    "RoutingDecision",
    "FallbackChain",
    "FallbackResult",
    "TokenOptimizer",
    "OptimizationSuggestion",
    "ModelComparison",
    "ComparisonResult",
    "ModelPricing",
    "CostCalculator",
]
