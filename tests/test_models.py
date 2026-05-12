"""Tests for the multi-model targeting module."""

import pytest
from promptops.models.pricing import CostCalculator, ModelPricing, PRICING_TABLE
from promptops.models.router import ModelRouter, RoutingStrategy, ModelQualityProfile
from promptops.models.fallback import FallbackChain
from promptops.models.optimizer import TokenOptimizer
from promptops.models.comparison import ModelComparison


class TestCostCalculator:
    def setup_method(self):
        self.calc = CostCalculator()

    def test_estimate_known_model(self):
        cost = self.calc.estimate("bedrock/claude-3-haiku", input_tokens=1000, output_tokens=500)
        assert cost > 0
        assert cost < 0.01  # should be very cheap

    def test_estimate_expensive_model(self):
        cheap = self.calc.estimate("bedrock/claude-3-haiku", 1000, 500)
        expensive = self.calc.estimate("bedrock/claude-3-opus", 1000, 500)
        assert expensive > cheap

    def test_compare_costs(self):
        costs = self.calc.compare_costs(1000, 500)
        assert len(costs) > 5
        # Should be sorted cheapest first
        values = list(costs.values())
        assert values == sorted(values)

    def test_find_cheapest(self):
        cheapest = self.calc.find_cheapest(1000, 500)
        assert cheapest is not None
        assert "titan" in cheapest or "mini" in cheapest or "haiku" in cheapest

    def test_find_cheapest_with_budget(self):
        result = self.calc.find_cheapest(1000, 500, max_cost=0.0001)
        # Very tight budget - might not find anything or find cheapest
        # Just verify it doesn't crash
        assert result is None or isinstance(result, str)

    def test_list_models(self):
        all_models = self.calc.list_models()
        assert len(all_models) > 10

    def test_list_models_by_provider(self):
        bedrock = self.calc.list_models(provider="bedrock")
        openai = self.calc.list_models(provider="openai")
        assert all("bedrock/" in m for m in bedrock)
        assert all("openai/" in m for m in openai)


class TestModelRouter:
    def test_cost_optimized_routing(self):
        router = ModelRouter(strategy=RoutingStrategy.COST_OPTIMIZED)
        decision = router.route(
            input_tokens=500,
            output_tokens=200,
            candidates=["bedrock/claude-3-haiku", "bedrock/claude-3-sonnet", "bedrock/claude-3-opus"],
            quality_threshold=0.7,
        )
        assert decision.selected_model == "bedrock/claude-3-haiku"
        assert decision.estimated_cost > 0

    def test_quality_first_routing(self):
        router = ModelRouter(strategy=RoutingStrategy.QUALITY_FIRST)
        decision = router.route(
            input_tokens=500,
            output_tokens=200,
            candidates=["bedrock/claude-3-haiku", "bedrock/claude-3-sonnet", "bedrock/claude-3-opus"],
        )
        assert decision.selected_model == "bedrock/claude-3-opus"

    def test_balanced_routing(self):
        router = ModelRouter(strategy=RoutingStrategy.BALANCED)
        decision = router.route(
            input_tokens=500,
            output_tokens=200,
            candidates=["bedrock/claude-3-haiku", "bedrock/claude-3-sonnet", "bedrock/claude-3-opus"],
        )
        # Balanced should pick something in the middle or best value
        assert decision.selected_model in [
            "bedrock/claude-3-haiku", "bedrock/claude-3-sonnet", "bedrock/claude-3-opus"
        ]

    def test_quality_threshold_filters(self):
        router = ModelRouter(strategy=RoutingStrategy.COST_OPTIMIZED)
        decision = router.route(
            input_tokens=500,
            output_tokens=200,
            candidates=["bedrock/claude-3-haiku", "bedrock/claude-3-sonnet"],
            quality_threshold=0.85,
        )
        # Haiku is 0.80, should be filtered out
        assert decision.selected_model == "bedrock/claude-3-sonnet"

    def test_budget_constraint(self):
        router = ModelRouter(strategy=RoutingStrategy.QUALITY_FIRST)
        decision = router.route(
            input_tokens=500,
            output_tokens=200,
            candidates=["bedrock/claude-3-haiku", "bedrock/claude-3-opus"],
            max_cost=0.001,
        )
        # Opus is too expensive
        assert decision.selected_model == "bedrock/claude-3-haiku"

    def test_alternatives_provided(self):
        router = ModelRouter(strategy=RoutingStrategy.COST_OPTIMIZED)
        decision = router.route(
            input_tokens=500,
            output_tokens=200,
            quality_threshold=0.5,
        )
        assert len(decision.alternatives) > 0

    def test_update_quality_profile(self):
        router = ModelRouter(strategy=RoutingStrategy.COST_OPTIMIZED)
        router.update_quality_profile("bedrock/claude-3-haiku", quality_score=0.95)
        decision = router.route(
            input_tokens=500,
            output_tokens=200,
            candidates=["bedrock/claude-3-haiku", "bedrock/claude-3-sonnet"],
            quality_threshold=0.90,
        )
        # Now haiku meets the threshold and is cheaper
        assert decision.selected_model == "bedrock/claude-3-haiku"


class TestFallbackChain:
    def test_primary_succeeds(self):
        chain = FallbackChain(models=["model-a", "model-b", "model-c"])

        def invoke(model, prompt):
            return f"Response from {model}"

        result = chain.execute(invoke, "Hello")
        assert result.success
        assert result.model_used == "model-a"
        assert not result.fallback_triggered

    def test_fallback_to_second(self):
        chain = FallbackChain(models=["model-a", "model-b", "model-c"])
        call_count = {"n": 0}

        def invoke(model, prompt):
            call_count["n"] += 1
            if model == "model-a":
                raise RuntimeError("Model A failed")
            return f"Response from {model}"

        result = chain.execute(invoke, "Hello")
        assert result.success
        assert result.model_used == "model-b"
        assert result.fallback_triggered

    def test_all_fail(self):
        chain = FallbackChain(models=["model-a", "model-b"])

        def invoke(model, prompt):
            raise RuntimeError(f"{model} failed")

        result = chain.execute(invoke, "Hello")
        assert not result.success
        assert result.fallback_triggered
        assert "All models" in result.error

    def test_retry_before_fallback(self):
        chain = FallbackChain(models=["model-a", "model-b"], max_retries_per_model=2)
        attempts = []

        def invoke(model, prompt):
            attempts.append(model)
            if model == "model-a":
                raise RuntimeError("fail")
            return "ok"

        result = chain.execute(invoke, "Hello")
        assert result.success
        assert result.model_used == "model-b"
        # model-a should have been tried 3 times (1 + 2 retries)
        assert attempts.count("model-a") == 3

    def test_stats_tracking(self):
        chain = FallbackChain(models=["model-a", "model-b"])

        def invoke(model, prompt):
            if model == "model-a":
                raise RuntimeError("fail")
            return "ok"

        chain.execute(invoke, "test1")
        chain.execute(invoke, "test2")

        assert chain.success_rate("model-a") == 0.0
        assert chain.success_rate("model-b") == 1.0

    def test_on_fallback_callback(self):
        callbacks = []
        chain = FallbackChain(
            models=["model-a", "model-b"],
            on_fallback=lambda f, t, e: callbacks.append((f, t)),
        )

        def invoke(model, prompt):
            if model == "model-a":
                raise RuntimeError("fail")
            return "ok"

        chain.execute(invoke, "test")
        assert len(callbacks) == 1
        assert callbacks[0] == ("model-a", "model-b")

    def test_empty_chain_raises(self):
        with pytest.raises(ValueError):
            FallbackChain(models=[])


class TestTokenOptimizer:
    def test_detect_verbose_phrasing(self):
        optimizer = TokenOptimizer()
        template = "In order to summarize this document, I want you to please read it carefully."
        report = optimizer.analyze(template, "test-prompt")
        assert len(report.suggestions) > 0
        categories = [s.category for s in report.suggestions]
        assert "verbosity" in categories

    def test_detect_excessive_whitespace(self):
        optimizer = TokenOptimizer()
        template = "Line 1\n\n\n\n\nLine 2\n\n\n\nLine 3"
        report = optimizer.analyze(template, "test-prompt")
        whitespace_suggestions = [s for s in report.suggestions if s.category == "whitespace"]
        assert len(whitespace_suggestions) > 0

    def test_detect_polite_language(self):
        optimizer = TokenOptimizer()
        template = "Please summarize. Please be concise. Please format as JSON. Please include key points."
        report = optimizer.analyze(template, "test-prompt")
        assert any("Please" in s.description or "please" in s.description for s in report.suggestions)

    def test_clean_prompt_minimal_suggestions(self):
        optimizer = TokenOptimizer()
        template = "Summarize this document in 100 words.\n\nDocument: {document}\n\nRespond as JSON."
        report = optimizer.analyze(template, "test-prompt")
        # Clean prompt should have few or no suggestions
        high_severity = [s for s in report.suggestions if s.severity == "high"]
        assert len(high_severity) == 0

    def test_savings_calculation(self):
        optimizer = TokenOptimizer()
        template = "In order to do this, I want you to please please please summarize."
        report = optimizer.analyze(template, "test-prompt")
        assert report.total_savings > 0
        assert report.savings_percent > 0
        assert report.total_cost_savings_per_1k > 0

    def test_report_summary(self):
        optimizer = TokenOptimizer()
        template = "I want you to please summarize in order to get the key points."
        report = optimizer.analyze(template, "verbose-prompt")
        summary = report.summary()
        assert "verbose-prompt" in summary
        assert "tokens" in summary.lower()


class TestModelComparison:
    def test_comparison_estimation_mode(self):
        comparison = ModelComparison(
            models=["bedrock/claude-3-haiku", "bedrock/claude-3-sonnet", "openai/gpt-4o-mini"]
        )
        result = comparison.run(
            prompt_name="summarize",
            inputs=[{"document": "Hello world " * 50, "max_words": 100}],
        )
        assert len(result.scores) == 3
        assert result.best_quality is not None
        assert result.cheapest is not None
        assert result.best_value is not None

    def test_comparison_with_invoke(self):
        comparison = ModelComparison(
            models=["model-a", "model-b"]
        )

        def invoke(model, inputs):
            return f"Response from {model}"

        def quality(output, inputs):
            return 0.9 if "model-a" in output else 0.7

        result = comparison.run(
            prompt_name="test",
            inputs=[{"text": "hello"}],
            invoke_fn=invoke,
            quality_fn=quality,
        )
        assert result.best_quality == "model-a"

    def test_comparison_summary(self):
        comparison = ModelComparison(
            models=["bedrock/claude-3-haiku", "bedrock/claude-3-sonnet"]
        )
        result = comparison.run(
            prompt_name="test",
            inputs=[{"text": "hello world"}],
        )
        summary = result.summary()
        assert "Model Comparison" in summary
        assert "claude-3-haiku" in summary

    def test_ranked_by_value(self):
        comparison = ModelComparison(
            models=["bedrock/claude-3-haiku", "bedrock/claude-3-opus"]
        )
        result = comparison.run(
            prompt_name="test",
            inputs=[{"text": "hello"}],
        )
        ranked = result.ranked_by_value
        # Haiku should be better value (good quality, very cheap)
        assert ranked[0].model == "bedrock/claude-3-haiku"
