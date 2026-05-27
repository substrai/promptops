"""Prompt linting rules for detecting common anti-patterns.

Provides a configurable linting system that detects:
- Missing format instructions (no output format specified)
- Injection vulnerabilities (user input without sanitization boundaries)
- Unused template variables (defined but never referenced)
- Overly long prompts (exceeding token/character limits)
- Ambiguous instructions (vague or contradictory directives)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("promptops.linting")


class LintSeverity(Enum):
    """Severity levels for lint findings."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class LintResult:
    """A single lint finding.

    Attributes:
        rule_id: Unique identifier for the rule that triggered.
        severity: Severity level of the finding.
        message: Human-readable description of the issue.
        line: Line number where the issue was found (1-indexed, if applicable).
        column: Column number where the issue starts (if applicable).
        suggestion: Suggested fix for the issue.
        context: The relevant portion of the prompt text.
    """
    rule_id: str
    severity: LintSeverity
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    suggestion: Optional[str] = None
    context: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "message": self.message,
            "line": self.line,
            "column": self.column,
            "suggestion": self.suggestion,
            "context": self.context,
        }


@dataclass
class LintRule:
    """A configurable lint rule.

    Attributes:
        rule_id: Unique identifier for the rule.
        name: Human-readable name.
        description: What this rule checks for.
        severity: Default severity level.
        enabled: Whether this rule is active.
        check_fn: The function that performs the check.
        config: Rule-specific configuration parameters.
    """
    rule_id: str
    name: str
    description: str
    severity: LintSeverity = LintSeverity.WARNING
    enabled: bool = True
    check_fn: Optional[Callable[[str, Dict[str, Any]], List[LintResult]]] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LinterConfig:
    """Configuration for the prompt linter.

    Attributes:
        max_prompt_length: Maximum allowed prompt length in characters.
        max_prompt_tokens_estimate: Estimated max tokens (chars / 4).
        severity_overrides: Override severity for specific rules.
        disabled_rules: Set of rule IDs to disable.
        custom_rules: Additional custom rules to include.
        variable_pattern: Regex pattern for template variables.
    """
    max_prompt_length: int = 8000
    max_prompt_tokens_estimate: int = 2000
    severity_overrides: Dict[str, LintSeverity] = field(default_factory=dict)
    disabled_rules: Set[str] = field(default_factory=set)
    custom_rules: List[LintRule] = field(default_factory=list)
    variable_pattern: str = r"\{(\w+)\}"


class PromptLinter:
    """Lints prompts for common anti-patterns and best practice violations.

    Usage:
        linter = PromptLinter()
        results = linter.lint("Tell me about {topic}")

        for result in results:
            print(f"[{result.severity.value}] {result.rule_id}: {result.message}")

    With configuration:
        config = LinterConfig(
            max_prompt_length=4000,
            severity_overrides={"missing-format": LintSeverity.ERROR},
            disabled_rules={"overly-long"},
        )
        linter = PromptLinter(config)
    """

    def __init__(self, config: Optional[LinterConfig] = None):
        self._config = config or LinterConfig()
        self._rules: List[LintRule] = self._build_rules()

    @property
    def config(self) -> LinterConfig:
        """Get the linter configuration."""
        return self._config

    @property
    def rules(self) -> List[LintRule]:
        """Get all registered rules."""
        return self._rules

    def lint(self, prompt: str, variables: Optional[Dict[str, str]] = None) -> List[LintResult]:
        """Lint a prompt template and return all findings.

        Args:
            prompt: The prompt text or template to lint.
            variables: Optional dict of template variables and their values.

        Returns:
            List of LintResult findings, sorted by severity.
        """
        results: List[LintResult] = []
        context = {
            "variables": variables or {},
            "config": self._config,
        }

        for rule in self._rules:
            if not rule.enabled:
                continue
            if rule.rule_id in self._config.disabled_rules:
                continue

            try:
                if rule.check_fn:
                    findings = rule.check_fn(prompt, context)
                    # Apply severity overrides
                    for finding in findings:
                        if finding.rule_id in self._config.severity_overrides:
                            finding.severity = self._config.severity_overrides[finding.rule_id]
                    results.extend(findings)
            except Exception as e:
                logger.warning(f"Rule {rule.rule_id} failed: {e}")

        # Sort by severity (error > warning > info)
        severity_order = {LintSeverity.ERROR: 0, LintSeverity.WARNING: 1, LintSeverity.INFO: 2}
        results.sort(key=lambda r: severity_order.get(r.severity, 99))

        return results

    def lint_batch(self, prompts: List[str]) -> Dict[int, List[LintResult]]:
        """Lint multiple prompts and return results indexed by position.

        Args:
            prompts: List of prompt texts to lint.

        Returns:
            Dictionary mapping prompt index to list of findings.
        """
        return {i: self.lint(prompt) for i, prompt in enumerate(prompts)}

    def add_rule(self, rule: LintRule) -> None:
        """Add a custom lint rule."""
        self._rules.append(rule)

    def _build_rules(self) -> List[LintRule]:
        """Build the default set of lint rules."""
        rules = [
            LintRule(
                rule_id="missing-format",
                name="Missing Format Instructions",
                description="Prompt does not specify expected output format",
                severity=LintSeverity.WARNING,
                check_fn=self._check_missing_format,
            ),
            LintRule(
                rule_id="injection-risk",
                name="Injection Vulnerability",
                description="Prompt may be vulnerable to injection attacks",
                severity=LintSeverity.ERROR,
                check_fn=self._check_injection_risk,
            ),
            LintRule(
                rule_id="unused-variables",
                name="Unused Template Variables",
                description="Template variables defined but not used in prompt",
                severity=LintSeverity.WARNING,
                check_fn=self._check_unused_variables,
            ),
            LintRule(
                rule_id="overly-long",
                name="Overly Long Prompt",
                description="Prompt exceeds recommended length limits",
                severity=LintSeverity.WARNING,
                check_fn=self._check_overly_long,
            ),
            LintRule(
                rule_id="ambiguous-instructions",
                name="Ambiguous Instructions",
                description="Prompt contains vague or contradictory directives",
                severity=LintSeverity.INFO,
                check_fn=self._check_ambiguous_instructions,
            ),
            LintRule(
                rule_id="no-role-definition",
                name="Missing Role Definition",
                description="Prompt does not define a clear role or persona",
                severity=LintSeverity.INFO,
                check_fn=self._check_no_role,
            ),
            LintRule(
                rule_id="hardcoded-examples",
                name="Hardcoded Examples Without Separation",
                description="Examples embedded without clear delimiters",
                severity=LintSeverity.INFO,
                check_fn=self._check_hardcoded_examples,
            ),
        ]

        # Add custom rules from config
        rules.extend(self._config.custom_rules)
        return rules

    @staticmethod
    def _check_missing_format(prompt: str, context: Dict[str, Any]) -> List[LintResult]:
        """Check if the prompt specifies an output format."""
        format_indicators = [
            r"\b(json|xml|yaml|csv|markdown|html)\b",
            r"\b(format|structured|output as|respond with|return as)\b",
            r"\b(bullet points|numbered list|table)\b",
            r"```",
        ]

        for pattern in format_indicators:
            if re.search(pattern, prompt, re.IGNORECASE):
                return []

        return [
            LintResult(
                rule_id="missing-format",
                severity=LintSeverity.WARNING,
                message="Prompt does not specify expected output format. Consider adding format instructions (e.g., 'Respond in JSON format').",
                suggestion="Add explicit format instructions like 'Format your response as JSON' or 'Respond with bullet points'.",
            )
        ]

    @staticmethod
    def _check_injection_risk(prompt: str, context: Dict[str, Any]) -> List[LintResult]:
        """Check for potential injection vulnerabilities."""
        results: List[LintResult] = []
        variable_pattern = context["config"].variable_pattern

        # Find template variables
        variables = re.findall(variable_pattern, prompt)
        if not variables:
            return []

        # Check for user input variables without boundaries
        user_input_indicators = ["user_input", "query", "question", "message", "input", "text", "content"]
        risky_vars = [v for v in variables if any(ind in v.lower() for ind in user_input_indicators)]

        if risky_vars:
            # Check if there are boundary markers around the variable
            boundary_patterns = [
                r'["\'].*\{' + re.escape(risky_vars[0]) + r'\}.*["\']',
                r"<.*\{" + re.escape(risky_vars[0]) + r"\}.*>",
                r"```.*\{" + re.escape(risky_vars[0]) + r"\}.*```",
                r"---.*\{" + re.escape(risky_vars[0]) + r"\}.*---",
                r"\[INST\]",
                r"<\|.*\|>",
            ]

            has_boundary = any(
                re.search(pat, prompt, re.DOTALL | re.IGNORECASE)
                for pat in boundary_patterns
            )

            if not has_boundary:
                for var in risky_vars:
                    # Find line number
                    line_num = None
                    for i, line in enumerate(prompt.split("\n"), 1):
                        if f"{{{var}}}" in line:
                            line_num = i
                            break

                    results.append(
                        LintResult(
                            rule_id="injection-risk",
                            severity=LintSeverity.ERROR,
                            message=f"Variable '{{{var}}}' appears to contain user input without clear boundary markers. This may be vulnerable to prompt injection.",
                            line=line_num,
                            suggestion=f"Wrap user input in delimiters: <user_input>{{{var}}}</user_input> or use triple backticks.",
                            context=f"{{{var}}}",
                        )
                    )

        return results

    @staticmethod
    def _check_unused_variables(prompt: str, context: Dict[str, Any]) -> List[LintResult]:
        """Check for template variables provided but not used in the prompt."""
        results: List[LintResult] = []
        provided_vars = context.get("variables", {})
        variable_pattern = context["config"].variable_pattern

        if not provided_vars:
            return []

        # Find variables used in the prompt
        used_vars = set(re.findall(variable_pattern, prompt))

        # Find provided but unused variables
        for var_name in provided_vars:
            if var_name not in used_vars:
                results.append(
                    LintResult(
                        rule_id="unused-variables",
                        severity=LintSeverity.WARNING,
                        message=f"Variable '{var_name}' is provided but not referenced in the prompt template.",
                        suggestion=f"Either use '{{{var_name}}}' in the prompt or remove it from the variables.",
                    )
                )

        return results

    @staticmethod
    def _check_overly_long(prompt: str, context: Dict[str, Any]) -> List[LintResult]:
        """Check if the prompt exceeds recommended length limits."""
        results: List[LintResult] = []
        config = context["config"]

        char_count = len(prompt)
        estimated_tokens = char_count // 4  # Rough estimate

        if char_count > config.max_prompt_length:
            results.append(
                LintResult(
                    rule_id="overly-long",
                    severity=LintSeverity.WARNING,
                    message=f"Prompt is {char_count} characters ({estimated_tokens} estimated tokens), exceeding the recommended limit of {config.max_prompt_length} characters.",
                    suggestion="Consider breaking the prompt into smaller sections, using a system message for context, or summarizing verbose instructions.",
                )
            )

        # Check for excessive repetition (sign of bloat)
        lines = prompt.strip().split("\n")
        if len(lines) > 5:
            unique_lines = set(line.strip() for line in lines if line.strip())
            repetition_ratio = 1 - (len(unique_lines) / len([l for l in lines if l.strip()]))
            if repetition_ratio > 0.3:
                results.append(
                    LintResult(
                        rule_id="overly-long",
                        severity=LintSeverity.INFO,
                        message=f"Prompt has {repetition_ratio:.0%} repeated lines. This may indicate unnecessary verbosity.",
                        suggestion="Remove duplicate instructions or consolidate repeated patterns.",
                    )
                )

        return results

    @staticmethod
    def _check_ambiguous_instructions(prompt: str, context: Dict[str, Any]) -> List[LintResult]:
        """Check for vague or contradictory instructions."""
        results: List[LintResult] = []

        # Vague instruction patterns
        vague_patterns = [
            (r"\b(maybe|perhaps|possibly|might want to)\b", "Uses uncertain language"),
            (r"\b(do something|handle it|figure it out|whatever)\b", "Uses vague directives"),
            (r"\b(etc\.?|and so on|and more|stuff like that)\b", "Uses open-ended lists without specifics"),
        ]

        for pattern, description in vague_patterns:
            matches = list(re.finditer(pattern, prompt, re.IGNORECASE))
            for match in matches:
                line_num = prompt[:match.start()].count("\n") + 1
                results.append(
                    LintResult(
                        rule_id="ambiguous-instructions",
                        severity=LintSeverity.INFO,
                        message=f"{description}: '{match.group()}'",
                        line=line_num,
                        suggestion="Replace with specific, actionable instructions.",
                        context=match.group(),
                    )
                )

        # Check for contradictions
        contradiction_pairs = [
            (r"\bshort\b", r"\bdetailed\b"),
            (r"\bbrief\b", r"\bcomprehensive\b"),
            (r"\bsimple\b", r"\bcomplex\b"),
            (r"\bconcise\b", r"\bthorough\b"),
        ]

        for pattern_a, pattern_b in contradiction_pairs:
            if re.search(pattern_a, prompt, re.IGNORECASE) and re.search(pattern_b, prompt, re.IGNORECASE):
                results.append(
                    LintResult(
                        rule_id="ambiguous-instructions",
                        severity=LintSeverity.WARNING,
                        message=f"Potentially contradictory instructions: prompt asks for both '{pattern_a.strip(chr(92)).strip('b')}' and '{pattern_b.strip(chr(92)).strip('b')}' responses.",
                        suggestion="Clarify whether you want a brief or detailed response.",
                    )
                )

        return results

    @staticmethod
    def _check_no_role(prompt: str, context: Dict[str, Any]) -> List[LintResult]:
        """Check if the prompt defines a role or persona."""
        role_indicators = [
            r"\b(you are|act as|role|persona|assistant|expert|specialist)\b",
            r"\b(as a|pretend|imagine you|your job is)\b",
            r"^system:",
        ]

        for pattern in role_indicators:
            if re.search(pattern, prompt, re.IGNORECASE | re.MULTILINE):
                return []

        # Only flag if prompt is substantial enough to benefit from a role
        if len(prompt) > 100:
            return [
                LintResult(
                    rule_id="no-role-definition",
                    severity=LintSeverity.INFO,
                    message="Prompt does not define a clear role or persona for the model.",
                    suggestion="Consider adding a role definition like 'You are an expert in...' to improve response quality.",
                )
            ]

        return []

    @staticmethod
    def _check_hardcoded_examples(prompt: str, context: Dict[str, Any]) -> List[LintResult]:
        """Check for examples without clear delimiters."""
        results: List[LintResult] = []

        # Look for example-like patterns without delimiters
        example_indicators = [
            r"\b(for example|e\.g\.|example:)\b",
            r"\b(input:|output:|sample:)\b",
        ]

        has_examples = any(
            re.search(pat, prompt, re.IGNORECASE)
            for pat in example_indicators
        )

        if has_examples:
            # Check if examples are properly delimited
            delimiter_patterns = [
                r"```",
                r"---",
                r"<example>",
                r"\[EXAMPLE\]",
                r"#{2,}",
            ]

            has_delimiters = any(
                re.search(pat, prompt)
                for pat in delimiter_patterns
            )

            if not has_delimiters and len(prompt) > 200:
                results.append(
                    LintResult(
                        rule_id="hardcoded-examples",
                        severity=LintSeverity.INFO,
                        message="Prompt contains examples without clear delimiters. This may confuse the model about where examples end and instructions begin.",
                        suggestion="Wrap examples in delimiters like ``` or <example>...</example> tags.",
                    )
                )

        return results


def lint_prompt(
    prompt: str,
    variables: Optional[Dict[str, str]] = None,
    config: Optional[LinterConfig] = None,
) -> List[LintResult]:
    """Convenience function to lint a prompt with default settings.

    Args:
        prompt: The prompt text to lint.
        variables: Optional template variables.
        config: Optional linter configuration.

    Returns:
        List of LintResult findings.

    Example:
        results = lint_prompt("Tell me about {topic}")
        for r in results:
            print(f"[{r.severity.value}] {r.message}")
    """
    linter = PromptLinter(config)
    return linter.lint(prompt, variables)
