"""Prompt quality scoring with configurable rubrics.

Score prompts on clarity, specificity, safety, and cost-efficiency (0-100).
Configurable rubric weights, per-dimension scores, and overall grade.

Features:
- Multi-dimensional prompt quality assessment
- Configurable rubric weights per scoring dimension
- Per-dimension scores with detailed feedback
- Overall composite grade (A-F scale)
- Extensible scoring dimensions
- Prompt improvement suggestions
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class Grade(Enum):
    """Letter grade for overall quality."""

    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class ScoringDimension(Enum):
    """Standard scoring dimensions for prompt quality."""

    CLARITY = "clarity"
    SPECIFICITY = "specificity"
    SAFETY = "safety"
    COST_EFFICIENCY = "cost_efficiency"
    COMPLETENESS = "completeness"
    STRUCTURE = "structure"


@dataclass
class DimensionScore:
    """Score for a single quality dimension.

    Attributes:
        dimension: Which dimension was scored.
        score: Numeric score from 0 to 100.
        weight: Weight of this dimension in overall score.
        feedback: Specific feedback for this dimension.
        suggestions: Improvement suggestions.
        penalties: List of penalty reasons applied.
    """

    dimension: str
    score: float
    weight: float = 1.0
    feedback: str = ""
    suggestions: List[str] = field(default_factory=list)
    penalties: List[str] = field(default_factory=list)


@dataclass
class QualityScore:
    """Complete quality scoring result for a prompt.

    Attributes:
        overall_score: Weighted composite score (0-100).
        grade: Letter grade based on overall score.
        dimension_scores: Individual dimension scores.
        prompt_length: Length of the analyzed prompt.
        word_count: Word count of the prompt.
        improvement_suggestions: Overall improvement suggestions.
        metadata: Additional scoring metadata.
    """

    overall_score: float = 0.0
    grade: Grade = Grade.F
    dimension_scores: List[DimensionScore] = field(default_factory=list)
    prompt_length: int = 0
    word_count: int = 0
    improvement_suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether the prompt meets minimum quality threshold (>= 60)."""
        return self.overall_score >= 60.0

    def get_dimension_score(self, dimension: str) -> Optional[DimensionScore]:
        """Get score for a specific dimension."""
        for ds in self.dimension_scores:
            if ds.dimension == dimension:
                return ds
        return None


@dataclass
class ScoringRubric:
    """Configurable rubric for prompt quality scoring.

    Attributes:
        weights: Weight for each scoring dimension (0.0 to 1.0).
        min_prompt_length: Minimum prompt length for full score.
        max_prompt_length: Maximum ideal prompt length.
        require_examples: Whether examples are expected in prompt.
        require_constraints: Whether constraints should be specified.
        require_output_format: Whether output format should be defined.
        custom_scorers: Additional custom scoring functions.
        penalty_vague_words: Words that indicate vagueness.
        bonus_precision_words: Words that indicate precision.
    """

    weights: Dict[str, float] = field(default_factory=lambda: {
        ScoringDimension.CLARITY.value: 0.25,
        ScoringDimension.SPECIFICITY.value: 0.25,
        ScoringDimension.SAFETY.value: 0.20,
        ScoringDimension.COST_EFFICIENCY.value: 0.15,
        ScoringDimension.COMPLETENESS.value: 0.15,
    })
    min_prompt_length: int = 20
    max_prompt_length: int = 2000
    require_examples: bool = False
    require_constraints: bool = False
    require_output_format: bool = False
    custom_scorers: Dict[str, Callable[[str], DimensionScore]] = field(default_factory=dict)
    penalty_vague_words: List[str] = field(default_factory=lambda: [
        "something", "stuff", "things", "whatever", "somehow",
        "maybe", "probably", "kind of", "sort of", "etc",
    ])
    bonus_precision_words: List[str] = field(default_factory=lambda: [
        "exactly", "specifically", "must", "required", "ensure",
        "format", "output", "constraint", "limit", "maximum",
    ])


class PromptQualityScorer:
    """Scores prompts on multiple quality dimensions.

    Analyzes prompts for clarity, specificity, safety, cost-efficiency,
    and completeness using configurable rubrics.

    Args:
        rubric: Scoring rubric configuration. Uses defaults if not provided.

    Example:
        scorer = PromptQualityScorer()
        result = scorer.score("Summarize the following document in 3 bullet points...")
        print(f"Score: {result.overall_score}/100 ({result.grade.value})")
    """

    def __init__(self, rubric: Optional[ScoringRubric] = None):
        self.rubric = rubric or ScoringRubric()

    def score(self, prompt: str) -> QualityScore:
        """Score a prompt across all configured dimensions.

        Args:
            prompt: The prompt text to evaluate.

        Returns:
            QualityScore with per-dimension and overall scores.
        """
        result = QualityScore(
            prompt_length=len(prompt),
            word_count=len(prompt.split()),
        )

        # Score each dimension
        dimension_scores = []

        if ScoringDimension.CLARITY.value in self.rubric.weights:
            dimension_scores.append(self._score_clarity(prompt))

        if ScoringDimension.SPECIFICITY.value in self.rubric.weights:
            dimension_scores.append(self._score_specificity(prompt))

        if ScoringDimension.SAFETY.value in self.rubric.weights:
            dimension_scores.append(self._score_safety(prompt))

        if ScoringDimension.COST_EFFICIENCY.value in self.rubric.weights:
            dimension_scores.append(self._score_cost_efficiency(prompt))

        if ScoringDimension.COMPLETENESS.value in self.rubric.weights:
            dimension_scores.append(self._score_completeness(prompt))

        # Run custom scorers
        for dim_name, scorer_fn in self.rubric.custom_scorers.items():
            custom_score = scorer_fn(prompt)
            custom_score.weight = self.rubric.weights.get(dim_name, 0.1)
            dimension_scores.append(custom_score)

        result.dimension_scores = dimension_scores

        # Calculate weighted overall score
        total_weight = sum(ds.weight for ds in dimension_scores)
        if total_weight > 0:
            result.overall_score = sum(
                ds.score * ds.weight for ds in dimension_scores
            ) / total_weight
        else:
            result.overall_score = 0.0

        # Assign grade
        result.grade = self._score_to_grade(result.overall_score)

        # Gather improvement suggestions
        for ds in dimension_scores:
            result.improvement_suggestions.extend(ds.suggestions)

        return result

    def _score_clarity(self, prompt: str) -> DimensionScore:
        """Score prompt clarity - how easy is it to understand?"""
        score = 100.0
        penalties = []
        suggestions = []

        # Penalize very short prompts
        if len(prompt) < self.rubric.min_prompt_length:
            penalty = min(40, (self.rubric.min_prompt_length - len(prompt)) * 2)
            score -= penalty
            penalties.append(f"Too short ({len(prompt)} chars)")
            suggestions.append("Add more detail to your prompt")

        # Penalize run-on sentences (very long without punctuation)
        sentences = re.split(r'[.!?\n]', prompt)
        long_sentences = [s for s in sentences if len(s.strip()) > 200]
        if long_sentences:
            score -= len(long_sentences) * 10
            penalties.append(f"{len(long_sentences)} run-on sentence(s)")
            suggestions.append("Break long sentences into shorter, clearer ones")

        # Check for ambiguous pronouns without clear referents
        ambiguous_patterns = [r'\bit\b', r'\bthis\b', r'\bthat\b', r'\bthey\b']
        ambiguous_count = 0
        for pattern in ambiguous_patterns:
            ambiguous_count += len(re.findall(pattern, prompt, re.IGNORECASE))
        if ambiguous_count > 3:
            score -= min(20, ambiguous_count * 3)
            penalties.append(f"Many ambiguous references ({ambiguous_count})")
            suggestions.append("Replace ambiguous pronouns with specific nouns")

        # Bonus for clear structure (numbered lists, headers, etc.)
        if re.search(r'\d+\.\s|\-\s|\*\s', prompt):
            score = min(100, score + 10)

        weight = self.rubric.weights.get(ScoringDimension.CLARITY.value, 0.25)
        return DimensionScore(
            dimension=ScoringDimension.CLARITY.value,
            score=max(0, min(100, score)),
            weight=weight,
            feedback=f"Clarity score based on structure and readability",
            suggestions=suggestions,
            penalties=penalties,
        )

    def _score_specificity(self, prompt: str) -> DimensionScore:
        """Score prompt specificity - how precise are the instructions?"""
        score = 60.0  # Start at middle
        penalties = []
        suggestions = []

        # Penalize vague words
        prompt_lower = prompt.lower()
        vague_count = sum(1 for word in self.rubric.penalty_vague_words if word in prompt_lower)
        if vague_count > 0:
            score -= vague_count * 8
            penalties.append(f"Vague language ({vague_count} instances)")
            suggestions.append("Replace vague words with specific instructions")

        # Bonus for precision words
        precision_count = sum(1 for word in self.rubric.bonus_precision_words if word in prompt_lower)
        score += precision_count * 5

        # Bonus for numbers/quantities
        numbers = re.findall(r'\b\d+\b', prompt)
        score += min(20, len(numbers) * 5)

        # Bonus for quoted examples
        quotes = [m for m in re.findall(r'"[^"]+"', prompt)]
        score += min(15, len(quotes) * 5)

        # Bonus for explicit output format specification
        format_indicators = ["json", "markdown", "bullet", "table", "list", "csv", "format:"]
        if any(indicator in prompt_lower for indicator in format_indicators):
            score += 10

        weight = self.rubric.weights.get(ScoringDimension.SPECIFICITY.value, 0.25)
        return DimensionScore(
            dimension=ScoringDimension.SPECIFICITY.value,
            score=max(0, min(100, score)),
            weight=weight,
            feedback=f"Specificity based on precision of instructions",
            suggestions=suggestions,
            penalties=penalties,
        )

    def _score_safety(self, prompt: str) -> DimensionScore:
        """Score prompt safety - are there potential injection or harmful patterns?"""
        score = 100.0
        penalties = []
        suggestions = []

        prompt_lower = prompt.lower()

        # Check for potential injection patterns
        injection_patterns = [
            r"ignore\s+(previous|all|above)",
            r"disregard\s+(instructions|rules)",
            r"you\s+are\s+now",
            r"pretend\s+to\s+be",
            r"act\s+as\s+if",
            r"new\s+instructions?\s*:",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, prompt_lower):
                score -= 30
                penalties.append(f"Potential injection pattern detected")
                suggestions.append("Remove language that could be misinterpreted as injection")
                break

        # Check for requests to bypass safety
        bypass_keywords = ["bypass", "override", "hack", "exploit", "jailbreak", "unrestricted"]
        bypass_count = sum(1 for kw in bypass_keywords if kw in prompt_lower)
        if bypass_count > 0:
            score -= bypass_count * 20
            penalties.append(f"Safety bypass keywords ({bypass_count})")
            suggestions.append("Remove terms that suggest bypassing safety measures")

        # Check for PII patterns
        pii_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{16}\b",  # Credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        ]
        for pattern in pii_patterns:
            if re.search(pattern, prompt):
                score -= 15
                penalties.append("PII detected in prompt")
                suggestions.append("Remove personal identifiable information from prompts")
                break

        weight = self.rubric.weights.get(ScoringDimension.SAFETY.value, 0.20)
        return DimensionScore(
            dimension=ScoringDimension.SAFETY.value,
            score=max(0, min(100, score)),
            weight=weight,
            feedback=f"Safety assessment for injection and harmful patterns",
            suggestions=suggestions,
            penalties=penalties,
        )

    def _score_cost_efficiency(self, prompt: str) -> DimensionScore:
        """Score cost efficiency - is the prompt optimized for token usage?"""
        score = 80.0
        penalties = []
        suggestions = []

        word_count = len(prompt.split())

        # Penalize extremely long prompts
        if word_count > 500:
            excess = word_count - 500
            score -= min(40, excess * 0.1)
            penalties.append(f"Very long prompt ({word_count} words)")
            suggestions.append("Consider condensing the prompt to reduce token usage")

        # Penalize excessive repetition
        words = prompt.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                score -= 20
                penalties.append(f"High repetition (unique ratio: {unique_ratio:.2f})")
                suggestions.append("Reduce repetitive language")

        # Penalize unnecessary filler
        filler_phrases = [
            "please note that", "it is important to", "i would like you to",
            "as an ai language model", "in order to", "for the purpose of",
        ]
        filler_count = sum(1 for phrase in filler_phrases if phrase in prompt.lower())
        if filler_count > 0:
            score -= filler_count * 5
            penalties.append(f"Filler phrases ({filler_count})")
            suggestions.append("Remove filler phrases for more efficient prompts")

        # Bonus for concise prompts that are still meaningful
        if 20 <= word_count <= 200:
            score = min(100, score + 10)

        weight = self.rubric.weights.get(ScoringDimension.COST_EFFICIENCY.value, 0.15)
        return DimensionScore(
            dimension=ScoringDimension.COST_EFFICIENCY.value,
            score=max(0, min(100, score)),
            weight=weight,
            feedback=f"Cost efficiency based on token optimization",
            suggestions=suggestions,
            penalties=penalties,
        )

    def _score_completeness(self, prompt: str) -> DimensionScore:
        """Score completeness - does the prompt include all necessary components?"""
        score = 50.0  # Start at middle
        penalties = []
        suggestions = []

        prompt_lower = prompt.lower()

        # Check for task description
        task_indicators = ["summarize", "translate", "generate", "analyze", "classify",
                          "extract", "compare", "explain", "write", "create", "list"]
        has_task = any(indicator in prompt_lower for indicator in task_indicators)
        if has_task:
            score += 15
        else:
            penalties.append("No clear task verb")
            suggestions.append("Start with a clear action verb (summarize, analyze, etc.)")

        # Check for context/input
        if len(prompt) > 50:
            score += 10

        # Check for output format specification
        format_keywords = ["format", "output", "respond", "return", "provide"]
        if any(kw in prompt_lower for kw in format_keywords):
            score += 10

        # Check for constraints
        constraint_keywords = ["must", "should", "limit", "maximum", "minimum",
                              "only", "do not", "avoid", "constraint"]
        constraint_count = sum(1 for kw in constraint_keywords if kw in prompt_lower)
        score += min(15, constraint_count * 5)

        # Check for examples
        if self.rubric.require_examples:
            if "example" in prompt_lower or "e.g." in prompt_lower or '"""' in prompt:
                score += 10
            else:
                penalties.append("No examples provided")
                suggestions.append("Include examples to clarify expected output")

        weight = self.rubric.weights.get(ScoringDimension.COMPLETENESS.value, 0.15)
        return DimensionScore(
            dimension=ScoringDimension.COMPLETENESS.value,
            score=max(0, min(100, score)),
            weight=weight,
            feedback=f"Completeness of prompt components",
            suggestions=suggestions,
            penalties=penalties,
        )

    def _score_to_grade(self, score: float) -> Grade:
        """Convert numeric score to letter grade."""
        if score >= 95:
            return Grade.A_PLUS
        elif score >= 80:
            return Grade.A
        elif score >= 70:
            return Grade.B
        elif score >= 60:
            return Grade.C
        elif score >= 50:
            return Grade.D
        else:
            return Grade.F
