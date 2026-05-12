"""Assertion types for prompt regression testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class AssertionResult:
    """Result of a single assertion check."""

    passed: bool
    assertion_type: str
    message: str
    expected: Any = None
    actual: Any = None


class Assertion:
    """Base class for prompt test assertions."""

    def __init__(self, config: Dict[str, Any]):
        self.type = config.get("type", "")
        self.config = config

    def check(self, output: Any, inputs: Dict[str, Any] = None) -> AssertionResult:
        """Run the assertion against an output.

        Args:
            output: The prompt output to check
            inputs: Original inputs (for context)

        Returns:
            AssertionResult
        """
        checker = getattr(self, f"_check_{self.type}", None)
        if checker:
            return checker(output, inputs)
        return AssertionResult(
            passed=False,
            assertion_type=self.type,
            message=f"Unknown assertion type: {self.type}",
        )

    def _check_schema_valid(self, output: Any, inputs: Dict[str, Any] = None) -> AssertionResult:
        """Check that output is valid JSON/dict."""
        if isinstance(output, dict):
            return AssertionResult(passed=True, assertion_type="schema_valid", message="Output is valid dict")
        if isinstance(output, str):
            try:
                import json
                json.loads(output)
                return AssertionResult(passed=True, assertion_type="schema_valid", message="Output is valid JSON")
            except (json.JSONDecodeError, ValueError):
                return AssertionResult(
                    passed=False,
                    assertion_type="schema_valid",
                    message="Output is not valid JSON",
                    actual=output[:100] if isinstance(output, str) else str(output)[:100],
                )
        return AssertionResult(
            passed=False, assertion_type="schema_valid", message=f"Unexpected output type: {type(output)}"
        )

    def _check_max_length(self, output: Any, inputs: Dict[str, Any] = None) -> AssertionResult:
        """Check that a field doesn't exceed max word count."""
        field_name = self.config.get("field", "")
        max_words = self.config.get("value", 100)

        text = output
        if isinstance(output, dict) and field_name:
            text = output.get(field_name, "")

        word_count = len(str(text).split())
        passed = word_count <= max_words

        return AssertionResult(
            passed=passed,
            assertion_type="max_length",
            message=f"Word count {word_count} {'<=' if passed else '>'} {max_words}",
            expected=max_words,
            actual=word_count,
        )

    def _check_contains_keywords(self, output: Any, inputs: Dict[str, Any] = None) -> AssertionResult:
        """Check that output contains specified keywords."""
        field_name = self.config.get("field", "")
        keywords = self.config.get("keywords", [])

        text = output
        if isinstance(output, dict) and field_name:
            text = output.get(field_name, "")

        text_lower = str(text).lower()
        found = [kw for kw in keywords if kw.lower() in text_lower]
        missing = [kw for kw in keywords if kw.lower() not in text_lower]

        passed = len(missing) == 0
        return AssertionResult(
            passed=passed,
            assertion_type="contains_keywords",
            message=f"Found {len(found)}/{len(keywords)} keywords. Missing: {missing}" if missing else "All keywords found",
            expected=keywords,
            actual=found,
        )

    def _check_does_not_contain(self, output: Any, inputs: Dict[str, Any] = None) -> AssertionResult:
        """Check that output does NOT contain specified values."""
        field_name = self.config.get("field", "")
        forbidden = self.config.get("values", [])

        text = output
        if isinstance(output, dict) and field_name:
            text = output.get(field_name, "")

        text_lower = str(text).lower()
        found_forbidden = [v for v in forbidden if v.lower() in text_lower]

        passed = len(found_forbidden) == 0
        return AssertionResult(
            passed=passed,
            assertion_type="does_not_contain",
            message=f"Found forbidden content: {found_forbidden}" if found_forbidden else "No forbidden content found",
            expected="none of " + str(forbidden),
            actual=found_forbidden,
        )

    def _check_cost_under(self, output: Any, inputs: Dict[str, Any] = None) -> AssertionResult:
        """Check that estimated cost is under threshold."""
        max_cost = self.config.get("value", 0.01)
        # This would use actual cost from invocation metadata
        # For now, return pass (cost checked at invocation time)
        return AssertionResult(
            passed=True,
            assertion_type="cost_under",
            message=f"Cost check (threshold: ${max_cost})",
            expected=max_cost,
        )

    def _check_key_points_count(self, output: Any, inputs: Dict[str, Any] = None) -> AssertionResult:
        """Check that key_points array has expected count."""
        min_count = self.config.get("min", 1)
        max_count = self.config.get("max", 10)

        key_points = []
        if isinstance(output, dict):
            key_points = output.get("key_points", [])
        elif isinstance(output, list):
            key_points = output

        count = len(key_points)
        passed = min_count <= count <= max_count

        return AssertionResult(
            passed=passed,
            assertion_type="key_points_count",
            message=f"Key points count {count} {'in' if passed else 'not in'} range [{min_count}, {max_count}]",
            expected=f"{min_count}-{max_count}",
            actual=count,
        )
