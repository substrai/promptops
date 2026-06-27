"""Tests for token usage profiler with optimization suggestions."""

from __future__ import annotations

import pytest

from promptops.scoring.token_profiler import (
    OptimizationType,
    ProfileReport,
    Severity,
    TokenBreakdown,
    TokenProfiler,
    TokenSuggestion,
)


class TestTokenEstimation:
    """Test token count estimation."""

    def test_empty_string(self):
        profiler = TokenProfiler()
        assert profiler.estimate_tokens("") == 0

    def test_short_text(self):
        profiler = TokenProfiler()
        # "hello" = 5 chars / 4 chars_per_token = 1
        tokens = profiler.estimate_tokens("hello")
        assert tokens >= 1

    def test_longer_text(self):
        profiler = TokenProfiler()
        text = "This is a longer sentence with multiple words."
        tokens = profiler.estimate_tokens(text)
        assert tokens > 5
        assert tokens < 20

    def test_custom_chars_per_token(self):
        profiler = TokenProfiler(chars_per_token=3.0)
        text = "twelve chars"  # 12 chars / 3 = 4 tokens
        tokens = profiler.estimate_tokens(text)
        assert tokens == 4


class TestRedundantPatternDetection:
    """Test detection of redundant instruction patterns."""

    def test_detect_please_make_sure(self):
        profiler = TokenProfiler()
        template = "Please make sure to respond in JSON format."
        report = profiler.profile(template)
        types = [s.type for s in report.suggestions]
        assert OptimizationType.REDUNDANT_INSTRUCTION in types

    def test_detect_it_is_important(self):
        profiler = TokenProfiler()
        template = "It is important that you follow these rules."
        report = profiler.profile(template)
        redundant = [s for s in report.suggestions if s.type == OptimizationType.REDUNDANT_INSTRUCTION]
        assert len(redundant) > 0

    def test_detect_remember_to_always(self):
        profiler = TokenProfiler()
        template = "Remember to always validate the output."
        report = profiler.profile(template)
        redundant = [s for s in report.suggestions if s.type == OptimizationType.REDUNDANT_INSTRUCTION]
        assert len(redundant) > 0

    def test_no_detection_on_clean_prompt(self):
        profiler = TokenProfiler()
        template = "Summarize the document. Return JSON with 'summary' key."
        report = profiler.profile(template)
        redundant = [s for s in report.suggestions if s.type == OptimizationType.REDUNDANT_INSTRUCTION]
        assert len(redundant) == 0


class TestVerbosePhrasingDetection:
    """Test detection of verbose phrasing."""

    def test_detect_in_order_to(self):
        profiler = TokenProfiler()
        template = "In order to achieve the best results, use concise prompts."
        report = profiler.profile(template)
        verbose = [s for s in report.suggestions if s.type == OptimizationType.VERBOSE_PHRASING]
        assert len(verbose) > 0
        assert verbose[0].suggested_text == "to"

    def test_detect_due_to_the_fact_that(self):
        profiler = TokenProfiler()
        template = "Due to the fact that the input is long, summarize it."
        report = profiler.profile(template)
        verbose = [s for s in report.suggestions if s.type == OptimizationType.VERBOSE_PHRASING]
        assert len(verbose) > 0

    def test_detect_in_the_event_that(self):
        profiler = TokenProfiler()
        template = "In the event that the response is too long, truncate it."
        report = profiler.profile(template)
        verbose = [s for s in report.suggestions if s.type == OptimizationType.VERBOSE_PHRASING]
        assert any("if" in s.suggested_text for s in verbose)


class TestWhitespaceDetection:
    """Test whitespace waste detection."""

    def test_detect_multiple_blank_lines(self):
        profiler = TokenProfiler()
        template = "Line 1\n\n\n\n\nLine 2"
        report = profiler.profile(template)
        whitespace = [s for s in report.suggestions if s.type == OptimizationType.WHITESPACE_WASTE]
        assert len(whitespace) > 0

    def test_no_issue_with_single_blank_line(self):
        profiler = TokenProfiler()
        template = "Line 1\n\nLine 2"
        report = profiler.profile(template)
        whitespace = [s for s in report.suggestions if s.type == OptimizationType.WHITESPACE_WASTE]
        # Single blank line should not trigger
        assert len(whitespace) == 0


class TestExampleReduction:
    """Test excessive example detection."""

    def test_detect_too_many_examples(self):
        profiler = TokenProfiler()
        template = (
            "Example 1: input -> output\n"
            "Example 2: input -> output\n"
            "For instance: input -> output\n"
            "Here is an example: input -> output\n"
            "Sample: input -> output\n"
        )
        report = profiler.profile(template)
        examples = [s for s in report.suggestions if s.type == OptimizationType.EXAMPLE_REDUCTION]
        assert len(examples) > 0

    def test_accept_reasonable_examples(self):
        profiler = TokenProfiler()
        template = (
            "Example 1: good\n"
            "Example 2: also good\n"
        )
        report = profiler.profile(template)
        examples = [s for s in report.suggestions if s.type == OptimizationType.EXAMPLE_REDUCTION]
        assert len(examples) == 0


class TestProfileReport:
    """Test the complete profile report."""

    def test_report_has_total_tokens(self):
        profiler = TokenProfiler()
        report = profiler.profile("Summarize this document in 100 words.")
        assert report.total_tokens > 0

    def test_report_has_breakdown(self):
        profiler = TokenProfiler()
        report = profiler.profile("Summarize: {document}")
        assert isinstance(report.breakdown, TokenBreakdown)

    def test_report_cost_estimate(self):
        profiler = TokenProfiler()
        report = profiler.profile("A" * 400)  # ~100 tokens
        assert report.cost_estimate_usd > 0
        assert report.optimized_cost_estimate_usd <= report.cost_estimate_usd

    def test_optimization_score_perfect(self):
        profiler = TokenProfiler()
        template = "Summarize: {text}"
        report = profiler.profile(template)
        # Short clean prompt should score high
        assert report.optimization_score >= 80.0

    def test_optimization_score_poor(self):
        profiler = TokenProfiler()
        template = (
            "Please make sure to remember to always "
            "in order to achieve the best results "
            "due to the fact that it is important that you "
            "keep in mind that your task is to summarize."
        )
        report = profiler.profile(template)
        # Wasteful prompt should score lower
        assert report.optimization_score < 90.0

    def test_savings_percent_calculated(self):
        profiler = TokenProfiler()
        template = "Please make sure to always respond in JSON format."
        report = profiler.profile(template)
        if report.suggestions:
            assert report.estimated_savings_percent > 0.0

    def test_suggestions_sorted_by_savings(self):
        profiler = TokenProfiler()
        template = (
            "In order to make sure that you must always ensure that "
            "it is important that you please note that the output."
        )
        report = profiler.profile(template)
        if len(report.suggestions) > 1:
            savings = [s.estimated_savings for s in report.suggestions]
            assert savings == sorted(savings, reverse=True)
