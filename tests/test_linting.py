"""Tests for prompt linting rules."""

import pytest

from promptops.linting.rules import (
    LinterConfig,
    LintResult,
    LintRule,
    LintSeverity,
    PromptLinter,
    lint_prompt,
)


class TestLintSeverity:
    """Tests for LintSeverity enum."""

    def test_severity_values(self):
        assert LintSeverity.ERROR.value == "error"
        assert LintSeverity.WARNING.value == "warning"
        assert LintSeverity.INFO.value == "info"


class TestLintResult:
    """Tests for LintResult dataclass."""

    def test_to_dict(self):
        result = LintResult(
            rule_id="test-rule",
            severity=LintSeverity.WARNING,
            message="Test message",
            line=5,
            suggestion="Fix it",
        )
        d = result.to_dict()
        assert d["rule_id"] == "test-rule"
        assert d["severity"] == "warning"
        assert d["message"] == "Test message"
        assert d["line"] == 5
        assert d["suggestion"] == "Fix it"

    def test_optional_fields(self):
        result = LintResult(
            rule_id="test",
            severity=LintSeverity.INFO,
            message="msg",
        )
        assert result.line is None
        assert result.column is None
        assert result.suggestion is None
        assert result.context is None


class TestMissingFormatRule:
    """Tests for the missing-format rule."""

    def test_no_format_instruction(self):
        """Prompt without format instructions should trigger warning."""
        results = lint_prompt("Tell me about Python programming")
        format_results = [r for r in results if r.rule_id == "missing-format"]
        assert len(format_results) == 1
        assert format_results[0].severity == LintSeverity.WARNING

    def test_json_format_specified(self):
        """Prompt with JSON format should not trigger."""
        results = lint_prompt("List the top 5 languages. Respond in JSON format.")
        format_results = [r for r in results if r.rule_id == "missing-format"]
        assert len(format_results) == 0

    def test_markdown_format_specified(self):
        """Prompt with markdown format should not trigger."""
        results = lint_prompt("Explain this concept using markdown with bullet points")
        format_results = [r for r in results if r.rule_id == "missing-format"]
        assert len(format_results) == 0

    def test_code_block_format(self):
        """Prompt with code blocks should not trigger."""
        results = lint_prompt("Write a function:\n```python\ndef hello():\n    pass\n```")
        format_results = [r for r in results if r.rule_id == "missing-format"]
        assert len(format_results) == 0

    def test_structured_output_keyword(self):
        """Prompt with 'structured' keyword should not trigger."""
        results = lint_prompt("Return a structured output with the following fields")
        format_results = [r for r in results if r.rule_id == "missing-format"]
        assert len(format_results) == 0


class TestInjectionRiskRule:
    """Tests for the injection-risk rule."""

    def test_unprotected_user_input(self):
        """User input variable without boundaries should trigger error."""
        prompt = "Answer the following question: {user_input}"
        results = lint_prompt(prompt)
        injection_results = [r for r in results if r.rule_id == "injection-risk"]
        assert len(injection_results) == 1
        assert injection_results[0].severity == LintSeverity.ERROR

    def test_protected_user_input(self):
        """User input with XML boundaries should not trigger."""
        prompt = "Answer the following:\n<user_input>{user_input}</user_input>"
        results = lint_prompt(prompt)
        injection_results = [r for r in results if r.rule_id == "injection-risk"]
        assert len(injection_results) == 0

    def test_non_user_variables_safe(self):
        """Non-user-input variables should not trigger injection warning."""
        prompt = "Translate {source_language} to {target_language}: {text}"
        # 'text' is a user input indicator but source_language/target_language are not
        results = lint_prompt(prompt)
        injection_results = [r for r in results if r.rule_id == "injection-risk"]
        # Only 'text' should trigger
        assert all("text" in r.context or "input" in r.context for r in injection_results)

    def test_query_variable_flagged(self):
        """Variable named 'query' should be flagged as potential user input."""
        prompt = "Search for: {query}"
        results = lint_prompt(prompt)
        injection_results = [r for r in results if r.rule_id == "injection-risk"]
        assert len(injection_results) == 1

    def test_no_variables_no_injection(self):
        """Prompt without variables should not trigger injection check."""
        prompt = "What is the capital of France?"
        results = lint_prompt(prompt)
        injection_results = [r for r in results if r.rule_id == "injection-risk"]
        assert len(injection_results) == 0


class TestUnusedVariablesRule:
    """Tests for the unused-variables rule."""

    def test_unused_variable_detected(self):
        """Provided variable not in prompt should trigger warning."""
        prompt = "Tell me about {topic}"
        variables = {"topic": "Python", "style": "formal"}
        results = lint_prompt(prompt, variables=variables)
        unused_results = [r for r in results if r.rule_id == "unused-variables"]
        assert len(unused_results) == 1
        assert "style" in unused_results[0].message

    def test_all_variables_used(self):
        """All provided variables used should not trigger."""
        prompt = "Write a {style} essay about {topic}"
        variables = {"topic": "AI", "style": "formal"}
        results = lint_prompt(prompt, variables=variables)
        unused_results = [r for r in results if r.rule_id == "unused-variables"]
        assert len(unused_results) == 0

    def test_no_variables_provided(self):
        """No variables provided should not trigger."""
        prompt = "Tell me about {topic}"
        results = lint_prompt(prompt)
        unused_results = [r for r in results if r.rule_id == "unused-variables"]
        assert len(unused_results) == 0

    def test_multiple_unused_variables(self):
        """Multiple unused variables should each generate a finding."""
        prompt = "Hello {name}"
        variables = {"name": "Alice", "age": "30", "city": "NYC"}
        results = lint_prompt(prompt, variables=variables)
        unused_results = [r for r in results if r.rule_id == "unused-variables"]
        assert len(unused_results) == 2


class TestOverlyLongRule:
    """Tests for the overly-long rule."""

    def test_short_prompt_ok(self):
        """Short prompt should not trigger."""
        results = lint_prompt("What is 2+2?")
        long_results = [r for r in results if r.rule_id == "overly-long"]
        assert len(long_results) == 0

    def test_long_prompt_triggers(self):
        """Prompt exceeding max length should trigger warning."""
        config = LinterConfig(max_prompt_length=100)
        prompt = "x" * 200
        results = lint_prompt(prompt, config=config)
        long_results = [r for r in results if r.rule_id == "overly-long"]
        assert len(long_results) >= 1
        assert long_results[0].severity == LintSeverity.WARNING

    def test_repetitive_content_flagged(self):
        """Highly repetitive prompt should be flagged."""
        lines = ["Do this task.\n"] * 20
        prompt = "".join(lines)
        results = lint_prompt(prompt)
        long_results = [r for r in results if r.rule_id == "overly-long"]
        repetition_results = [r for r in long_results if "repeated" in r.message]
        assert len(repetition_results) >= 1

    def test_custom_max_length(self):
        """Custom max length should be respected."""
        config = LinterConfig(max_prompt_length=50)
        prompt = "a" * 60
        results = lint_prompt(prompt, config=config)
        long_results = [r for r in results if r.rule_id == "overly-long"]
        assert len(long_results) >= 1


class TestAmbiguousInstructionsRule:
    """Tests for the ambiguous-instructions rule."""

    def test_vague_language_detected(self):
        """Vague language like 'maybe' should be flagged."""
        prompt = "Maybe you could help me with something. Do whatever seems right."
        results = lint_prompt(prompt)
        ambiguous_results = [r for r in results if r.rule_id == "ambiguous-instructions"]
        assert len(ambiguous_results) >= 1

    def test_contradictory_instructions(self):
        """Contradictory instructions should be flagged."""
        prompt = "Give me a short but comprehensive overview of quantum physics with detailed examples."
        results = lint_prompt(prompt)
        ambiguous_results = [r for r in results if r.rule_id == "ambiguous-instructions"]
        contradiction_results = [r for r in ambiguous_results if "contradictory" in r.message.lower()]
        assert len(contradiction_results) >= 1

    def test_clear_instructions_ok(self):
        """Clear, specific instructions should not trigger."""
        prompt = "You are a Python expert. List exactly 3 benefits of type hints. Format as JSON."
        results = lint_prompt(prompt)
        ambiguous_results = [r for r in results if r.rule_id == "ambiguous-instructions"]
        assert len(ambiguous_results) == 0


class TestPromptLinter:
    """Tests for the PromptLinter class."""

    def test_severity_override(self):
        """Test that severity overrides work."""
        config = LinterConfig(
            severity_overrides={"missing-format": LintSeverity.ERROR}
        )
        linter = PromptLinter(config)
        results = linter.lint("Tell me about Python")
        format_results = [r for r in results if r.rule_id == "missing-format"]
        assert format_results[0].severity == LintSeverity.ERROR

    def test_disabled_rules(self):
        """Test that disabled rules are skipped."""
        config = LinterConfig(disabled_rules={"missing-format", "no-role-definition"})
        linter = PromptLinter(config)
        results = linter.lint("Tell me about Python")
        assert all(r.rule_id != "missing-format" for r in results)

    def test_custom_rule(self):
        """Test adding a custom lint rule."""
        def check_no_please(prompt: str, context: dict) -> list:
            if "please" in prompt.lower():
                return [LintResult(
                    rule_id="no-please",
                    severity=LintSeverity.INFO,
                    message="Avoid politeness tokens in prompts.",
                )]
            return []

        custom_rule = LintRule(
            rule_id="no-please",
            name="No Please",
            description="Flags politeness tokens",
            check_fn=check_no_please,
        )

        config = LinterConfig(custom_rules=[custom_rule])
        linter = PromptLinter(config)
        results = linter.lint("Please tell me about Python")
        custom_results = [r for r in results if r.rule_id == "no-please"]
        assert len(custom_results) == 1

    def test_lint_batch(self):
        """Test batch linting of multiple prompts."""
        linter = PromptLinter()
        prompts = [
            "What is Python?",
            "Format as JSON: list items",
            "Tell me about {user_input}",
        ]
        results = linter.lint_batch(prompts)
        assert len(results) == 3
        assert 0 in results
        assert 1 in results
        assert 2 in results

    def test_results_sorted_by_severity(self):
        """Test that results are sorted by severity (errors first)."""
        prompt = "Maybe answer this {user_input} question"
        results = lint_prompt(prompt)
        if len(results) >= 2:
            severities = [r.severity for r in results]
            severity_order = {LintSeverity.ERROR: 0, LintSeverity.WARNING: 1, LintSeverity.INFO: 2}
            orders = [severity_order[s] for s in severities]
            assert orders == sorted(orders)

    def test_well_crafted_prompt_minimal_findings(self):
        """A well-crafted prompt should have minimal or no findings."""
        prompt = """You are an expert Python developer.

Given the following code, identify potential bugs and suggest fixes.
Format your response as JSON with the following structure:
{"bugs": [{"line": int, "issue": str, "fix": str}]}

Code:
```
{code}
```"""
        results = lint_prompt(prompt)
        # Should have no errors
        errors = [r for r in results if r.severity == LintSeverity.ERROR]
        assert len(errors) == 0
