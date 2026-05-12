"""Token optimizer - analyzes prompts for token waste and suggests compression.

Identifies patterns that waste tokens and provides actionable
suggestions to reduce cost without impacting quality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OptimizationSuggestion:
    """A single optimization suggestion."""

    category: str  # "whitespace", "redundancy", "verbosity", "structure"
    description: str
    estimated_token_savings: int
    estimated_cost_savings: float = 0.0  # per 1K invocations
    severity: str = "info"  # "info", "warning", "high"
    original_snippet: str = ""
    suggested_snippet: str = ""

    @property
    def priority_score(self) -> float:
        """Higher = more impactful."""
        severity_weights = {"high": 3.0, "warning": 2.0, "info": 1.0}
        return self.estimated_token_savings * severity_weights.get(self.severity, 1.0)


@dataclass
class OptimizationReport:
    """Full optimization report for a prompt."""

    prompt_name: str
    original_tokens: int
    optimized_tokens: int
    suggestions: List[OptimizationSuggestion] = field(default_factory=list)

    @property
    def total_savings(self) -> int:
        return sum(s.estimated_token_savings for s in self.suggestions)

    @property
    def savings_percent(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return self.total_savings / self.original_tokens

    @property
    def total_cost_savings_per_1k(self) -> float:
        return sum(s.estimated_cost_savings for s in self.suggestions)

    def summary(self) -> str:
        lines = [
            f"Token Optimization Report: {self.prompt_name}",
            f"Original: {self.original_tokens} tokens",
            f"Potential savings: {self.total_savings} tokens ({self.savings_percent:.0%})",
            f"Cost savings per 1K invocations: ${self.total_cost_savings_per_1k:.4f}",
            "",
            "Suggestions:",
        ]
        for i, s in enumerate(sorted(self.suggestions, key=lambda x: -x.priority_score), 1):
            lines.append(f"  {i}. [{s.severity.upper()}] {s.description}")
            lines.append(f"     Saves ~{s.estimated_token_savings} tokens")
            if s.original_snippet:
                lines.append(f"     Before: \"{s.original_snippet[:60]}...\"")
            if s.suggested_snippet:
                lines.append(f"     After:  \"{s.suggested_snippet[:60]}...\"")
        return "\n".join(lines)


class TokenOptimizer:
    """Analyzes prompts and suggests token optimizations.

    Usage:
        optimizer = TokenOptimizer()
        report = optimizer.analyze("You are a helpful assistant. Please summarize...")
    """

    def __init__(self, cost_per_1k_input_tokens: float = 0.00025):
        """Initialize optimizer.

        Args:
            cost_per_1k_input_tokens: Cost per 1K input tokens for savings calculation
        """
        self.cost_per_token = cost_per_1k_input_tokens / 1000

    def analyze(self, template: str, prompt_name: str = "prompt") -> OptimizationReport:
        """Analyze a prompt template for optimization opportunities.

        Args:
            template: The prompt template text
            prompt_name: Name for the report

        Returns:
            OptimizationReport with suggestions
        """
        original_tokens = len(template) // 4
        suggestions = []

        suggestions.extend(self._check_excessive_whitespace(template))
        suggestions.extend(self._check_redundant_instructions(template))
        suggestions.extend(self._check_verbose_phrasing(template))
        suggestions.extend(self._check_unnecessary_examples(template))
        suggestions.extend(self._check_repeated_context(template))

        # Calculate cost savings
        for s in suggestions:
            s.estimated_cost_savings = round(
                s.estimated_token_savings * self.cost_per_token * 1000, 6
            )

        optimized_tokens = original_tokens - sum(s.estimated_token_savings for s in suggestions)

        return OptimizationReport(
            prompt_name=prompt_name,
            original_tokens=original_tokens,
            optimized_tokens=max(optimized_tokens, 0),
            suggestions=suggestions,
        )

    def _check_excessive_whitespace(self, template: str) -> List[OptimizationSuggestion]:
        """Check for excessive whitespace that wastes tokens."""
        suggestions = []

        # Multiple blank lines
        multi_blank = re.findall(r'\n{3,}', template)
        if multi_blank:
            wasted = sum(len(m) - 2 for m in multi_blank) // 4
            suggestions.append(OptimizationSuggestion(
                category="whitespace",
                description="Multiple consecutive blank lines found. Use single blank lines.",
                estimated_token_savings=wasted,
                severity="info",
            ))

        # Trailing spaces
        trailing = re.findall(r' +\n', template)
        if len(trailing) > 3:
            wasted = sum(len(m) - 1 for m in trailing) // 4
            suggestions.append(OptimizationSuggestion(
                category="whitespace",
                description=f"Trailing spaces on {len(trailing)} lines.",
                estimated_token_savings=wasted,
                severity="info",
            ))

        # Excessive indentation
        deep_indent = re.findall(r'^[ ]{8,}', template, re.MULTILINE)
        if deep_indent:
            wasted = sum(len(m) - 4 for m in deep_indent) // 4
            suggestions.append(OptimizationSuggestion(
                category="whitespace",
                description="Deep indentation (8+ spaces). Consider reducing.",
                estimated_token_savings=wasted,
                severity="info",
            ))

        return suggestions

    def _check_redundant_instructions(self, template: str) -> List[OptimizationSuggestion]:
        """Check for redundant or unnecessary instructions."""
        suggestions = []
        lower = template.lower()

        # "Please" and overly polite language
        please_count = lower.count("please ")
        if please_count > 2:
            suggestions.append(OptimizationSuggestion(
                category="verbosity",
                description=f"'Please' used {please_count} times. LLMs don't need politeness.",
                estimated_token_savings=please_count,
                severity="info",
                original_snippet="Please summarize the following...",
                suggested_snippet="Summarize the following...",
            ))

        # "I want you to" / "I need you to"
        want_patterns = re.findall(r'i (?:want|need|would like) you to', lower)
        if want_patterns:
            suggestions.append(OptimizationSuggestion(
                category="verbosity",
                description=f"Indirect instructions ('I want you to...') found {len(want_patterns)} times. Use direct imperatives.",
                estimated_token_savings=len(want_patterns) * 4,
                severity="warning",
                original_snippet="I want you to summarize this document",
                suggested_snippet="Summarize this document",
            ))

        # "You are a helpful assistant" boilerplate
        if "you are a helpful" in lower or "you are an ai" in lower:
            suggestions.append(OptimizationSuggestion(
                category="redundancy",
                description="Generic role description ('You are a helpful assistant'). Use specific role instead.",
                estimated_token_savings=8,
                severity="warning",
                original_snippet="You are a helpful AI assistant that...",
                suggested_snippet="Role: [specific role]. Task: ...",
            ))

        return suggestions

    def _check_verbose_phrasing(self, template: str) -> List[OptimizationSuggestion]:
        """Check for verbose phrasing that can be compressed."""
        suggestions = []
        lower = template.lower()

        verbose_patterns = {
            "in order to": ("to", 2),
            "due to the fact that": ("because", 4),
            "at this point in time": ("now", 4),
            "in the event that": ("if", 3),
            "for the purpose of": ("to", 3),
            "with regard to": ("about", 2),
            "in addition to": ("also", 2),
            "it is important to note that": ("note:", 5),
            "make sure to": ("ensure", 2),
            "take into account": ("consider", 2),
        }

        for verbose, (concise, savings) in verbose_patterns.items():
            count = lower.count(verbose)
            if count > 0:
                suggestions.append(OptimizationSuggestion(
                    category="verbosity",
                    description=f"Verbose phrase '{verbose}' used {count} time(s). Replace with '{concise}'.",
                    estimated_token_savings=savings * count,
                    severity="info",
                    original_snippet=verbose,
                    suggested_snippet=concise,
                ))

        return suggestions

    def _check_unnecessary_examples(self, template: str) -> List[OptimizationSuggestion]:
        """Check for excessive examples that could be reduced."""
        suggestions = []

        # Count example blocks
        example_markers = re.findall(
            r'(?:example|for instance|e\.g\.|such as)[:\s]', template.lower()
        )
        if len(example_markers) > 3:
            suggestions.append(OptimizationSuggestion(
                category="structure",
                description=f"Found {len(example_markers)} example sections. Consider reducing to 2-3 key examples.",
                estimated_token_savings=(len(example_markers) - 3) * 20,
                severity="warning",
            ))

        return suggestions

    def _check_repeated_context(self, template: str) -> List[OptimizationSuggestion]:
        """Check for repeated phrases or context."""
        suggestions = []

        # Find repeated phrases (4+ words)
        words = template.lower().split()
        phrases_4 = [" ".join(words[i:i+4]) for i in range(len(words) - 3)]
        phrase_counts: Dict[str, int] = {}
        for phrase in phrases_4:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        repeated = {p: c for p, c in phrase_counts.items() if c > 2}
        if repeated:
            total_waste = sum((c - 1) * 4 for c in repeated.values())
            top_repeated = sorted(repeated.items(), key=lambda x: -x[1])[:3]
            examples = ", ".join(f"'{p}' ({c}x)" for p, c in top_repeated)
            suggestions.append(OptimizationSuggestion(
                category="redundancy",
                description=f"Repeated phrases detected: {examples}",
                estimated_token_savings=total_waste // 4,
                severity="info" if total_waste < 20 else "warning",
            ))

        return suggestions
