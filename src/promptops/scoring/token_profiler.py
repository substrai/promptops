"""Token usage profiler with optimization suggestions.

Analyzes prompt templates for token waste, identifies redundant instructions,
and suggests compression strategies to reduce costs without sacrificing quality.

Usage:
    from promptops.scoring.token_profiler import TokenProfiler

    profiler = TokenProfiler()
    report = profiler.profile(template="Your prompt template here...")
    print(f"Estimated tokens: {report.total_tokens}")
    for suggestion in report.suggestions:
        print(f"  [{suggestion.severity}] {suggestion.message}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Severity(str, Enum):
    """Severity level for optimization suggestions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OptimizationType(str, Enum):
    """Type of token optimization."""

    REDUNDANT_INSTRUCTION = "redundant_instruction"
    VERBOSE_PHRASING = "verbose_phrasing"
    UNNECESSARY_CONTEXT = "unnecessary_context"
    REPETITIVE_PATTERN = "repetitive_pattern"
    FORMAT_OVERHEAD = "format_overhead"
    INSTRUCTION_COMPRESSION = "instruction_compression"
    EXAMPLE_REDUCTION = "example_reduction"
    WHITESPACE_WASTE = "whitespace_waste"


@dataclass
class TokenSuggestion:
    """A single optimization suggestion."""

    type: OptimizationType
    severity: Severity
    message: str
    estimated_savings: int  # tokens saved
    line_range: Optional[Tuple[int, int]] = None
    original_text: Optional[str] = None
    suggested_text: Optional[str] = None


@dataclass
class TokenBreakdown:
    """Breakdown of token usage by category."""

    instructions: int = 0
    context: int = 0
    examples: int = 0
    formatting: int = 0
    variables: int = 0
    whitespace: int = 0


@dataclass
class ProfileReport:
    """Complete token profiling report."""

    template: str
    total_tokens: int
    breakdown: TokenBreakdown
    suggestions: List[TokenSuggestion]
    estimated_savings_tokens: int
    estimated_savings_percent: float
    cost_estimate_usd: float  # At Claude 3 Haiku input rate
    optimized_cost_estimate_usd: float

    @property
    def optimization_score(self) -> float:
        """Score from 0-100 where 100 is perfectly optimized."""
        if self.total_tokens == 0:
            return 100.0
        waste_ratio = self.estimated_savings_tokens / self.total_tokens
        return max(0.0, min(100.0, (1.0 - waste_ratio) * 100.0))


# Approximate tokens per character (English text average)
CHARS_PER_TOKEN = 4.0

# Pricing (Claude 3 Haiku input per token)
HAIKU_INPUT_PRICE_PER_TOKEN = 0.00000025

# Redundant instruction patterns
REDUNDANT_PATTERNS: List[Tuple[str, str, int]] = [
    (r"please\s+make\s+sure\s+to", "Remove politeness prefix — models don't need 'please make sure to'", 4),
    (r"it\s+is\s+important\s+that\s+you", "Remove emphasis filler — 'it is important that you'", 5),
    (r"remember\s+to\s+always", "Remove 'remember to always' — unnecessary emphasis", 3),
    (r"you\s+must\s+always\s+ensure\s+that", "Remove verbose mandate — 'you must always ensure that'", 5),
    (r"please\s+note\s+that", "Remove 'please note that' — adds no value", 3),
    (r"keep\s+in\s+mind\s+that", "Remove 'keep in mind that' — filler phrase", 4),
    (r"i\s+want\s+you\s+to", "Remove 'I want you to' — implicit in instructions", 4),
    (r"your\s+task\s+is\s+to", "Consider removing 'Your task is to' — use direct imperatives", 4),
    (r"make\s+sure\s+that", "Remove 'make sure that' — use direct instruction", 3),
]

# Verbose phrasing patterns with shorter alternatives
VERBOSE_PATTERNS: List[Tuple[str, str, str, int]] = [
    (r"in\s+order\s+to", "in order to", "to", 2),
    (r"due\s+to\s+the\s+fact\s+that", "due to the fact that", "because", 4),
    (r"in\s+the\s+event\s+that", "in the event that", "if", 3),
    (r"at\s+this\s+point\s+in\s+time", "at this point in time", "now", 4),
    (r"in\s+a\s+manner\s+that", "in a manner that", "that", 3),
    (r"for\s+the\s+purpose\s+of", "for the purpose of", "to", 3),
    (r"with\s+regard\s+to", "with regard to", "about", 2),
    (r"in\s+spite\s+of\s+the\s+fact\s+that", "in spite of the fact that", "although", 5),
    (r"it\s+should\s+be\s+noted\s+that", "it should be noted that", "", 5),
    (r"as\s+a\s+matter\s+of\s+fact", "as a matter of fact", "actually", 4),
]


class TokenProfiler:
    """Analyzes prompt templates for token efficiency.

    Detects wasteful patterns, redundant instructions, verbose phrasing,
    and suggests optimizations with estimated token savings.
    """

    def __init__(self, chars_per_token: float = CHARS_PER_TOKEN):
        """Initialize the profiler.

        Args:
            chars_per_token: Average characters per token for estimation.
        """
        self._chars_per_token = chars_per_token

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string.

        Uses character-based estimation. For exact counts,
        integrate with tiktoken or model-specific tokenizers.

        Args:
            text: Text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0
        return max(1, int(len(text) / self._chars_per_token))

    def profile(self, template: str) -> ProfileReport:
        """Profile a prompt template for token efficiency.

        Args:
            template: The prompt template to analyze.

        Returns:
            ProfileReport with breakdown and suggestions.
        """
        total_tokens = self.estimate_tokens(template)
        breakdown = self._compute_breakdown(template)
        suggestions = self._analyze(template)

        total_savings = sum(s.estimated_savings for s in suggestions)
        savings_percent = (total_savings / total_tokens * 100) if total_tokens > 0 else 0.0

        cost_estimate = total_tokens * HAIKU_INPUT_PRICE_PER_TOKEN
        optimized_cost = (total_tokens - total_savings) * HAIKU_INPUT_PRICE_PER_TOKEN

        return ProfileReport(
            template=template,
            total_tokens=total_tokens,
            breakdown=breakdown,
            suggestions=suggestions,
            estimated_savings_tokens=total_savings,
            estimated_savings_percent=savings_percent,
            cost_estimate_usd=cost_estimate,
            optimized_cost_estimate_usd=optimized_cost,
        )

    def _compute_breakdown(self, template: str) -> TokenBreakdown:
        """Compute token usage breakdown by category."""
        lines = template.split("\n")
        breakdown = TokenBreakdown()

        for line in lines:
            tokens = self.estimate_tokens(line)
            stripped = line.strip()

            if not stripped:
                breakdown.whitespace += 1
            elif stripped.startswith("#") or stripped.startswith("---"):
                breakdown.formatting += tokens
            elif "{" in stripped and "}" in stripped:
                breakdown.variables += tokens
            elif re.match(r"^(example|e\.g\.|for instance|sample)", stripped.lower()):
                breakdown.examples += tokens
            elif re.match(r"^(context|background|given)", stripped.lower()):
                breakdown.context += tokens
            else:
                breakdown.instructions += tokens

        return breakdown

    def _analyze(self, template: str) -> List[TokenSuggestion]:
        """Analyze template for optimization opportunities."""
        suggestions: List[TokenSuggestion] = []

        # Check redundant instruction patterns
        suggestions.extend(self._check_redundant_patterns(template))

        # Check verbose phrasing
        suggestions.extend(self._check_verbose_phrasing(template))

        # Check repetitive content
        suggestions.extend(self._check_repetition(template))

        # Check whitespace waste
        suggestions.extend(self._check_whitespace(template))

        # Check excessive examples
        suggestions.extend(self._check_examples(template))

        # Sort by estimated savings (highest first)
        suggestions.sort(key=lambda s: s.estimated_savings, reverse=True)

        return suggestions

    def _check_redundant_patterns(self, template: str) -> List[TokenSuggestion]:
        """Check for redundant instruction patterns."""
        suggestions: List[TokenSuggestion] = []
        template_lower = template.lower()

        for pattern, message, savings in REDUNDANT_PATTERNS:
            matches = list(re.finditer(pattern, template_lower))
            if matches:
                for match in matches:
                    suggestions.append(TokenSuggestion(
                        type=OptimizationType.REDUNDANT_INSTRUCTION,
                        severity=Severity.MEDIUM,
                        message=message,
                        estimated_savings=savings,
                        original_text=match.group(),
                    ))

        return suggestions

    def _check_verbose_phrasing(self, template: str) -> List[TokenSuggestion]:
        """Check for verbose phrases with shorter alternatives."""
        suggestions: List[TokenSuggestion] = []
        template_lower = template.lower()

        for pattern, original, replacement, savings in VERBOSE_PATTERNS:
            if re.search(pattern, template_lower):
                suggestions.append(TokenSuggestion(
                    type=OptimizationType.VERBOSE_PHRASING,
                    severity=Severity.LOW,
                    message=f"Replace '{original}' with '{replacement}' (saves ~{savings} tokens)",
                    estimated_savings=savings,
                    original_text=original,
                    suggested_text=replacement,
                ))

        return suggestions

    def _check_repetition(self, template: str) -> List[TokenSuggestion]:
        """Check for repetitive content in the template."""
        suggestions: List[TokenSuggestion] = []
        sentences = re.split(r'[.!?]\s+', template)

        # Check for semantically similar sentences
        seen_phrases: Dict[str, int] = {}
        for sentence in sentences:
            # Normalize for comparison
            normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
            words = normalized.split()
            if len(words) < 4:
                continue

            # Check 4-gram overlap
            for i in range(len(words) - 3):
                gram = " ".join(words[i:i + 4])
                seen_phrases[gram] = seen_phrases.get(gram, 0) + 1

        repeated = {k: v for k, v in seen_phrases.items() if v > 1}
        if repeated:
            total_repeated_tokens = sum(
                self.estimate_tokens(phrase) * (count - 1)
                for phrase, count in repeated.items()
            )
            if total_repeated_tokens > 5:
                suggestions.append(TokenSuggestion(
                    type=OptimizationType.REPETITIVE_PATTERN,
                    severity=Severity.MEDIUM,
                    message=f"Found {len(repeated)} repeated phrases — consolidate to reduce tokens",
                    estimated_savings=min(total_repeated_tokens, 20),
                ))

        return suggestions

    def _check_whitespace(self, template: str) -> List[TokenSuggestion]:
        """Check for excessive whitespace."""
        suggestions: List[TokenSuggestion] = []

        # Multiple consecutive blank lines
        multi_blank = re.findall(r'\n{3,}', template)
        if multi_blank:
            waste = sum(len(m) - 2 for m in multi_blank)
            suggestions.append(TokenSuggestion(
                type=OptimizationType.WHITESPACE_WASTE,
                severity=Severity.LOW,
                message=f"Reduce consecutive blank lines ({waste} extra newlines)",
                estimated_savings=max(1, waste // 4),
            ))

        # Trailing whitespace
        trailing = re.findall(r' +\n', template)
        if len(trailing) > 3:
            suggestions.append(TokenSuggestion(
                type=OptimizationType.WHITESPACE_WASTE,
                severity=Severity.LOW,
                message=f"Remove trailing whitespace on {len(trailing)} lines",
                estimated_savings=max(1, len(trailing) // 4),
            ))

        return suggestions

    def _check_examples(self, template: str) -> List[TokenSuggestion]:
        """Check for excessive examples that could be reduced."""
        suggestions: List[TokenSuggestion] = []

        # Count example blocks
        example_markers = re.findall(
            r'(example|e\.g\.|for instance|sample|here is an example)',
            template.lower(),
        )

        if len(example_markers) > 3:
            suggestions.append(TokenSuggestion(
                type=OptimizationType.EXAMPLE_REDUCTION,
                severity=Severity.MEDIUM,
                message=(
                    f"Found {len(example_markers)} example sections. "
                    "Consider reducing to 2-3 representative examples."
                ),
                estimated_savings=self.estimate_tokens(template) // 10,
            ))

        return suggestions
