"""Built-in evaluators for prompt quality assessment.

Evaluators measure the quality of prompt outputs using various
metrics: semantic similarity, factual accuracy, format compliance,
and custom scoring.
"""

from __future__ import annotations

import re
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationScore:
    """Result of an evaluation."""

    evaluator: str
    score: float  # 0.0 to 1.0
    passed: bool
    details: Dict[str, Any]
    message: str


class BaseEvaluator(ABC):
    """Base class for all evaluators."""

    name: str = "base"

    @abstractmethod
    def evaluate(
        self,
        output: Any,
        inputs: Dict[str, Any],
        reference: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> EvaluationScore:
        """Evaluate a prompt output.

        Args:
            output: The prompt output to evaluate
            inputs: Original inputs
            reference: Optional reference/expected output
            config: Optional evaluator configuration

        Returns:
            EvaluationScore with score and details
        """
        pass


class SemanticSimilarityEvaluator(BaseEvaluator):
    """Evaluates semantic similarity between output and reference.

    Uses a simple word overlap metric (Jaccard similarity) as a baseline.
    In production, this would use embedding-based similarity.
    """

    name = "semantic_similarity"

    def evaluate(
        self,
        output: Any,
        inputs: Dict[str, Any],
        reference: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> EvaluationScore:
        config = config or {}
        threshold = config.get("threshold", 0.7)

        if not reference:
            return EvaluationScore(
                evaluator=self.name,
                score=0.0,
                passed=False,
                details={"error": "No reference provided"},
                message="Cannot evaluate without reference",
            )

        output_text = str(output) if not isinstance(output, str) else output

        # Jaccard similarity on word sets (baseline metric)
        output_words = set(self._tokenize(output_text))
        reference_words = set(self._tokenize(reference))

        if not output_words and not reference_words:
            score = 1.0
        elif not output_words or not reference_words:
            score = 0.0
        else:
            intersection = output_words & reference_words
            union = output_words | reference_words
            score = len(intersection) / len(union)

        passed = score >= threshold

        return EvaluationScore(
            evaluator=self.name,
            score=round(score, 4),
            passed=passed,
            details={
                "threshold": threshold,
                "overlap_words": len(output_words & reference_words),
                "output_words": len(output_words),
                "reference_words": len(reference_words),
            },
            message=f"Similarity {score:.2%} {'≥' if passed else '<'} {threshold:.0%}",
        )

    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        return re.findall(r'\b\w+\b', text.lower())


class SchemaComplianceEvaluator(BaseEvaluator):
    """Evaluates whether output conforms to expected JSON schema."""

    name = "schema_compliance"

    def evaluate(
        self,
        output: Any,
        inputs: Dict[str, Any],
        reference: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> EvaluationScore:
        config = config or {}
        expected_fields = config.get("expected_fields", [])

        # Try to parse as JSON if string
        parsed = output
        if isinstance(output, str):
            try:
                parsed = json.loads(output)
            except (json.JSONDecodeError, ValueError):
                return EvaluationScore(
                    evaluator=self.name,
                    score=0.0,
                    passed=False,
                    details={"error": "Output is not valid JSON"},
                    message="Output is not valid JSON",
                )

        if not isinstance(parsed, dict):
            return EvaluationScore(
                evaluator=self.name,
                score=0.0,
                passed=False,
                details={"error": f"Expected dict, got {type(parsed).__name__}"},
                message="Output is not a dictionary",
            )

        # Check expected fields
        if expected_fields:
            present = [f for f in expected_fields if f in parsed]
            missing = [f for f in expected_fields if f not in parsed]
            score = len(present) / len(expected_fields)
        else:
            # Just check it's a valid dict with content
            score = 1.0 if len(parsed) > 0 else 0.5
            present = list(parsed.keys())
            missing = []

        passed = score >= 0.9  # 90% of fields must be present

        return EvaluationScore(
            evaluator=self.name,
            score=round(score, 4),
            passed=passed,
            details={
                "present_fields": present,
                "missing_fields": missing,
                "total_fields": len(parsed),
            },
            message=f"Schema compliance: {len(present)}/{len(expected_fields or present)} fields present",
        )


class FormatComplianceEvaluator(BaseEvaluator):
    """Evaluates format compliance (length, structure, etc.)."""

    name = "format_compliance"

    def evaluate(
        self,
        output: Any,
        inputs: Dict[str, Any],
        reference: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> EvaluationScore:
        config = config or {}
        max_words = config.get("max_words")
        min_words = config.get("min_words")
        max_chars = config.get("max_chars")
        required_format = config.get("format")  # "json", "markdown", "plain"

        output_text = str(output)
        word_count = len(output_text.split())
        char_count = len(output_text)

        checks_passed = 0
        checks_total = 0
        details: Dict[str, Any] = {"word_count": word_count, "char_count": char_count}

        if max_words is not None:
            checks_total += 1
            if word_count <= max_words:
                checks_passed += 1
            details["max_words_check"] = word_count <= max_words

        if min_words is not None:
            checks_total += 1
            if word_count >= min_words:
                checks_passed += 1
            details["min_words_check"] = word_count >= min_words

        if max_chars is not None:
            checks_total += 1
            if char_count <= max_chars:
                checks_passed += 1
            details["max_chars_check"] = char_count <= max_chars

        if required_format == "json":
            checks_total += 1
            try:
                json.loads(output_text)
                checks_passed += 1
                details["json_valid"] = True
            except (json.JSONDecodeError, ValueError):
                details["json_valid"] = False

        score = checks_passed / checks_total if checks_total > 0 else 1.0
        passed = score >= 1.0

        return EvaluationScore(
            evaluator=self.name,
            score=round(score, 4),
            passed=passed,
            details=details,
            message=f"Format compliance: {checks_passed}/{checks_total} checks passed",
        )


class InjectionDetectionEvaluator(BaseEvaluator):
    """Detects potential prompt injection in outputs."""

    name = "injection_detection"

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions",
        r"system\s+prompt",
        r"you\s+are\s+(now|a)\s+",
        r"disregard\s+(all|any|previous)",
        r"forget\s+(everything|all|your)",
        r"new\s+instructions?:",
        r"override\s+(your|the)\s+",
        r"act\s+as\s+(if|a)\s+",
    ]

    def evaluate(
        self,
        output: Any,
        inputs: Dict[str, Any],
        reference: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> EvaluationScore:
        output_text = str(output).lower()
        input_text = " ".join(str(v) for v in inputs.values()).lower()

        # Check output for injection patterns
        output_matches = []
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, output_text):
                output_matches.append(pattern)

        # Check if input contained injection attempts that leaked to output
        input_matches = []
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, input_text):
                input_matches.append(pattern)

        # Score: 1.0 = clean, 0.0 = injection detected
        if output_matches:
            score = 0.0
        elif input_matches and any(p in output_text for p in ["system", "instruction", "prompt"]):
            score = 0.3  # suspicious
        else:
            score = 1.0

        passed = score >= 0.8

        return EvaluationScore(
            evaluator=self.name,
            score=score,
            passed=passed,
            details={
                "output_injection_patterns": output_matches,
                "input_injection_patterns": input_matches,
            },
            message="Clean" if passed else f"Injection detected: {len(output_matches)} patterns matched",
        )


class CostEvaluator(BaseEvaluator):
    """Evaluates cost against budget thresholds."""

    name = "cost_estimation"

    def evaluate(
        self,
        output: Any,
        inputs: Dict[str, Any],
        reference: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> EvaluationScore:
        config = config or {}
        max_cost = config.get("max_cost_per_request", 0.01)
        model = config.get("model", "bedrock/claude-3-haiku")

        # Estimate based on input + output tokens
        input_text = " ".join(str(v) for v in inputs.values())
        output_text = str(output)

        input_tokens = len(input_text) // 4
        output_tokens = len(output_text) // 4

        # Pricing per 1K tokens
        pricing = {
            "bedrock/claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "bedrock/claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "bedrock/amazon-titan-text-lite": {"input": 0.00015, "output": 0.00015},
            "openai/gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        }

        model_pricing = pricing.get(model, {"input": 0.001, "output": 0.002})
        estimated_cost = (
            (input_tokens / 1000) * model_pricing["input"]
            + (output_tokens / 1000) * model_pricing["output"]
        )

        passed = estimated_cost <= max_cost
        score = 1.0 - min(estimated_cost / max_cost, 1.0) if max_cost > 0 else 0.0

        return EvaluationScore(
            evaluator=self.name,
            score=round(max(score, 0.0), 4),
            passed=passed,
            details={
                "estimated_cost": round(estimated_cost, 6),
                "max_cost": max_cost,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "model": model,
            },
            message=f"Cost ${estimated_cost:.6f} {'≤' if passed else '>'} ${max_cost:.6f}",
        )


# Registry of built-in evaluators
EVALUATOR_REGISTRY: Dict[str, type] = {
    "semantic_similarity": SemanticSimilarityEvaluator,
    "schema_compliance": SchemaComplianceEvaluator,
    "format_compliance": FormatComplianceEvaluator,
    "injection_detection": InjectionDetectionEvaluator,
    "cost_estimation": CostEvaluator,
}


def get_evaluator(name: str) -> BaseEvaluator:
    """Get an evaluator by name.

    Args:
        name: Evaluator name

    Returns:
        Evaluator instance

    Raises:
        KeyError: If evaluator not found
    """
    if name not in EVALUATOR_REGISTRY:
        available = list(EVALUATOR_REGISTRY.keys())
        raise KeyError(f"Evaluator '{name}' not found. Available: {available}")
    return EVALUATOR_REGISTRY[name]()


def register_evaluator(name: str, evaluator_class: type) -> None:
    """Register a custom evaluator.

    Args:
        name: Evaluator name
        evaluator_class: Class that extends BaseEvaluator
    """
    EVALUATOR_REGISTRY[name] = evaluator_class
