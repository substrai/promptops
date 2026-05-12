"""Tests for breaking change detection."""

import pytest
from promptops.core.prompt import PromptDefinition
from promptops.testing.breaking_changes import BreakingChangeDetector, ChangeType


BASE_YAML = """
name: test-prompt
version: 1.0.0
model:
  default: bedrock/claude-3-haiku
input:
  schema:
    document:
      type: string
      required: true
    max_words:
      type: integer
      default: 100
    style:
      type: enum
      values: [executive, technical, casual]
      default: executive
output:
  schema:
    summary:
      type: string
    key_points:
      type: array
template: |
  Summarize in {max_words} words: {document}
settings:
  temperature: 0.3
"""


def test_no_changes():
    old = PromptDefinition.from_yaml(BASE_YAML)
    new = PromptDefinition.from_yaml(BASE_YAML)
    detector = BreakingChangeDetector()
    report = detector.detect(old, new)
    assert report.recommended_bump == ChangeType.NONE
    assert not report.has_breaking_changes


def test_template_change_is_patch():
    new_yaml = BASE_YAML.replace("Summarize in", "Please summarize in")
    old = PromptDefinition.from_yaml(BASE_YAML)
    new = PromptDefinition.from_yaml(new_yaml)
    detector = BreakingChangeDetector()
    report = detector.detect(old, new)
    assert report.recommended_bump == ChangeType.PATCH


def test_add_optional_field_is_minor():
    new_yaml = BASE_YAML.replace(
        "    style:",
        "    language:\n      type: string\n      default: english\n    style:"
    )
    old = PromptDefinition.from_yaml(BASE_YAML)
    new = PromptDefinition.from_yaml(new_yaml)
    detector = BreakingChangeDetector()
    report = detector.detect(old, new)
    assert report.recommended_bump == ChangeType.MINOR
    assert not report.has_breaking_changes


def test_remove_required_field_is_breaking():
    new_yaml = BASE_YAML.replace(
        "    document:\n      type: string\n      required: true\n", ""
    )
    old = PromptDefinition.from_yaml(BASE_YAML)
    new = PromptDefinition.from_yaml(new_yaml)
    detector = BreakingChangeDetector()
    report = detector.detect(old, new)
    assert report.has_breaking_changes
    assert report.recommended_bump == ChangeType.BREAKING


def test_change_field_type_is_breaking():
    new_yaml = BASE_YAML.replace("type: integer", "type: string")
    old = PromptDefinition.from_yaml(BASE_YAML)
    new = PromptDefinition.from_yaml(new_yaml)
    detector = BreakingChangeDetector()
    report = detector.detect(old, new)
    assert report.has_breaking_changes


def test_remove_enum_value_is_breaking():
    new_yaml = BASE_YAML.replace(
        "values: [executive, technical, casual]",
        "values: [executive, technical]"
    )
    old = PromptDefinition.from_yaml(BASE_YAML)
    new = PromptDefinition.from_yaml(new_yaml)
    detector = BreakingChangeDetector()
    report = detector.detect(old, new)
    assert report.has_breaking_changes


def test_add_enum_value_is_minor():
    new_yaml = BASE_YAML.replace(
        "values: [executive, technical, casual]",
        "values: [executive, technical, casual, academic]"
    )
    old = PromptDefinition.from_yaml(BASE_YAML)
    new = PromptDefinition.from_yaml(new_yaml)
    detector = BreakingChangeDetector()
    report = detector.detect(old, new)
    assert report.recommended_bump == ChangeType.MINOR
    assert not report.has_breaking_changes


def test_remove_output_field_is_breaking():
    new_yaml = BASE_YAML.replace(
        "    key_points:\n      type: array\n", ""
    )
    old = PromptDefinition.from_yaml(BASE_YAML)
    new = PromptDefinition.from_yaml(new_yaml)
    detector = BreakingChangeDetector()
    report = detector.detect(old, new)
    assert report.has_breaking_changes


def test_report_summary():
    new_yaml = BASE_YAML.replace("Summarize in", "Please summarize in")
    old = PromptDefinition.from_yaml(BASE_YAML)
    new = PromptDefinition.from_yaml(new_yaml)
    detector = BreakingChangeDetector()
    report = detector.detect(old, new)
    summary = report.summary()
    assert "test-prompt" in summary
    assert "Patch" in summary or "patch" in summary
