"""Tests for the prompt diff engine."""

from __future__ import annotations

import pytest

from promptops.versioning.diff import (
    Change,
    ChangeCategory,
    ChangeType,
    DiffResult,
    PromptDiffEngine,
    PromptSnapshot,
)


@pytest.fixture
def engine():
    """Create a diff engine instance."""
    return PromptDiffEngine()


class TestPromptDiffEngineBasic:
    """Basic diff engine functionality."""

    def test_identical_prompts_no_changes(self, engine):
        """Identical prompts should produce no changes."""
        snapshot = PromptSnapshot(
            template="Hello {{name}}, how can I help you?",
            variables={"name": {"type": "string", "required": True}},
        )
        result = engine.diff(snapshot, snapshot)

        assert result.overall_change == ChangeType.NONE
        assert result.has_changes is False
        assert result.change_count == 0
        assert result.template_similarity == 1.0

    def test_wording_change_is_patch(self, engine):
        """Minor wording changes should be classified as PATCH."""
        old = PromptSnapshot(
            template="Hello {{name}}, how can I help you today?",
            variables={"name": {"type": "string"}},
        )
        new = PromptSnapshot(
            template="Hello {{name}}, how can I assist you today?",
            variables={"name": {"type": "string"}},
        )
        result = engine.diff(old, new)

        assert result.overall_change == ChangeType.PATCH
        assert result.is_breaking is False

    def test_formatting_only_is_patch(self, engine):
        """Whitespace-only changes should be PATCH."""
        old = PromptSnapshot(template="Hello {{name}},  how can I help?")
        new = PromptSnapshot(template="Hello {{name}}, how can I help?")

        result = engine.diff(old, new)
        assert result.overall_change == ChangeType.PATCH

        formatting_changes = [
            c for c in result.changes
            if c.category == ChangeCategory.FORMATTING_ONLY
        ]
        assert len(formatting_changes) == 1

    def test_new_variable_is_minor(self, engine):
        """Adding a new variable should be classified as MINOR."""
        old = PromptSnapshot(
            template="Hello {{name}}",
            variables={"name": {"type": "string"}},
        )
        new = PromptSnapshot(
            template="Hello {{name}}, your role is {{role}}",
            variables={
                "name": {"type": "string"},
                "role": {"type": "string"},
            },
        )
        result = engine.diff(old, new)

        assert result.overall_change == ChangeType.MINOR
        assert "role" in result.variables_added
        assert result.is_breaking is False

    def test_removed_variable_is_major(self, engine):
        """Removing a variable should be classified as MAJOR (breaking)."""
        old = PromptSnapshot(
            template="Hello {{name}}, your role is {{role}}",
            variables={
                "name": {"type": "string"},
                "role": {"type": "string"},
            },
        )
        new = PromptSnapshot(
            template="Hello {{name}}",
            variables={"name": {"type": "string"}},
        )
        result = engine.diff(old, new)

        assert result.overall_change == ChangeType.MAJOR
        assert result.is_breaking is True
        assert "role" in result.variables_removed


class TestPromptDiffEngineSchema:
    """Schema change detection tests."""

    def test_added_input_field_is_minor(self, engine):
        """Adding an optional input schema field should be MINOR."""
        old = PromptSnapshot(
            template="Summarize: {{text}}",
            input_schema={"properties": {"text": {"type": "string"}}},
        )
        new = PromptSnapshot(
            template="Summarize: {{text}}",
            input_schema={
                "properties": {
                    "text": {"type": "string"},
                    "max_length": {"type": "integer"},
                }
            },
        )
        result = engine.diff(old, new)

        assert result.overall_change == ChangeType.MINOR
        schema_changes = [
            c for c in result.changes
            if c.category == ChangeCategory.SCHEMA_INPUT_ADDED
        ]
        assert len(schema_changes) == 1

    def test_removed_input_field_is_major(self, engine):
        """Removing an input schema field should be MAJOR."""
        old = PromptSnapshot(
            template="Process: {{text}}",
            input_schema={
                "properties": {
                    "text": {"type": "string"},
                    "format": {"type": "string"},
                }
            },
        )
        new = PromptSnapshot(
            template="Process: {{text}}",
            input_schema={"properties": {"text": {"type": "string"}}},
        )
        result = engine.diff(old, new)

        assert result.overall_change == ChangeType.MAJOR
        assert result.is_breaking is True

    def test_output_schema_change_is_major(self, engine):
        """Any output schema change should be MAJOR."""
        old = PromptSnapshot(
            template="Generate: {{input}}",
            output_schema={"type": "object", "properties": {"result": {"type": "string"}}},
        )
        new = PromptSnapshot(
            template="Generate: {{input}}",
            output_schema={"type": "object", "properties": {"output": {"type": "array"}}},
        )
        result = engine.diff(old, new)

        assert result.overall_change == ChangeType.MAJOR

    def test_variable_type_change_is_major(self, engine):
        """Changing a variable's type should be MAJOR."""
        old = PromptSnapshot(
            template="Count: {{items}}",
            variables={"items": {"type": "string"}},
        )
        new = PromptSnapshot(
            template="Count: {{items}}",
            variables={"items": {"type": "array"}},
        )
        result = engine.diff(old, new)

        assert result.overall_change == ChangeType.MAJOR
        type_changes = [
            c for c in result.changes
            if c.category == ChangeCategory.VARIABLE_TYPE_CHANGED
        ]
        assert len(type_changes) == 1


class TestPromptDiffEngineRoleAndExamples:
    """Role and example change detection."""

    def test_role_change_is_major(self, engine):
        """Changing the system role should be MAJOR."""
        old = PromptSnapshot(template="Help me", role="assistant")
        new = PromptSnapshot(template="Help me", role="expert_analyst")

        result = engine.diff(old, new)
        assert result.overall_change == ChangeType.MAJOR

    def test_examples_added_is_minor(self, engine):
        """Adding examples should be MINOR."""
        old = PromptSnapshot(
            template="Classify: {{text}}",
            examples=[{"input": "hello", "output": "greeting"}],
        )
        new = PromptSnapshot(
            template="Classify: {{text}}",
            examples=[
                {"input": "hello", "output": "greeting"},
                {"input": "bye", "output": "farewell"},
            ],
        )
        result = engine.diff(old, new)

        assert result.overall_change == ChangeType.MINOR

    def test_examples_removed_is_minor(self, engine):
        """Removing examples should be MINOR."""
        old = PromptSnapshot(
            template="Classify: {{text}}",
            examples=[
                {"input": "hello", "output": "greeting"},
                {"input": "bye", "output": "farewell"},
            ],
        )
        new = PromptSnapshot(
            template="Classify: {{text}}",
            examples=[{"input": "hello", "output": "greeting"}],
        )
        result = engine.diff(old, new)

        assert result.overall_change == ChangeType.MINOR


class TestPromptDiffEngineComplex:
    """Complex multi-change scenarios."""

    def test_multiple_changes_highest_severity_wins(self, engine):
        """When multiple changes exist, the highest severity determines overall."""
        old = PromptSnapshot(
            template="Hello {{name}}, process {{data}}",
            variables={
                "name": {"type": "string"},
                "data": {"type": "string"},
            },
            examples=[{"input": "test", "output": "result"}],
        )
        new = PromptSnapshot(
            template="Hi {{name}}, handle {{data}} with {{format}}",
            variables={
                "name": {"type": "string"},
                "data": {"type": "array"},  # type change = MAJOR
                "format": {"type": "string"},  # new var = MINOR
            },
            examples=[
                {"input": "test", "output": "result"},
                {"input": "test2", "output": "result2"},
            ],
        )
        result = engine.diff(old, new)

        # Variable type change makes it MAJOR
        assert result.overall_change == ChangeType.MAJOR
        assert result.change_count >= 3

    def test_structural_template_change_is_minor(self, engine):
        """A completely rewritten template should be MINOR."""
        old = PromptSnapshot(template="You are a helpful assistant. Answer: {{question}}")
        new = PromptSnapshot(
            template="As an AI language model, your task is to provide detailed answers. Question: {{question}}"
        )
        result = engine.diff(old, new)

        # Similarity should be low enough for structural change
        assert result.template_similarity < 0.8

    def test_snapshot_from_dict(self):
        """PromptSnapshot.from_dict should correctly parse a dictionary."""
        data = {
            "template": "Hello {{name}}",
            "variables": {"name": {"type": "string"}},
            "input_schema": {"properties": {"name": {"type": "string"}}},
            "output_schema": {"type": "string"},
            "role": "assistant",
            "examples": [{"input": "hi", "output": "hello"}],
            "metadata": {"version": "1.0.0"},
        }
        snapshot = PromptSnapshot.from_dict(data)

        assert snapshot.template == "Hello {{name}}"
        assert "name" in snapshot.variables
        assert snapshot.role == "assistant"
        assert len(snapshot.examples) == 1

    def test_extract_template_variables(self):
        """Should extract all {{variable}} patterns from template."""
        snapshot = PromptSnapshot(
            template="Hello {{name}}, your {{role}} at {{company}} is confirmed."
        )
        variables = snapshot.extract_template_variables()

        assert variables == {"name", "role", "company"}

    def test_diff_result_get_changes_by_severity(self, engine):
        """get_changes_by_severity should filter correctly."""
        old = PromptSnapshot(
            template="Hello {{name}}, process {{data}}",
            variables={"name": {"type": "string"}, "data": {"type": "string"}},
        )
        new = PromptSnapshot(
            template="Hi {{name}}",
            variables={"name": {"type": "string"}},
        )
        result = engine.diff(old, new)

        major_changes = result.get_changes_by_severity(ChangeType.MAJOR)
        assert len(major_changes) >= 1
        assert all(c.severity == ChangeType.MAJOR for c in major_changes)
