"""Tests for the prompt definition module."""

import pytest
from promptops.core.prompt import Prompt, PromptDefinition
from promptops.core.version import PromptVersion


SAMPLE_YAML = """
name: summarize
version: 1.2.0
description: "Summarize documents"

model:
  default: bedrock/claude-3-haiku
  fallback: bedrock/amazon-titan-text-lite

input:
  schema:
    document:
      type: string
      required: true
    max_words:
      type: integer
      default: 100

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
  max_tokens: 2000
"""


def test_parse_yaml():
    definition = PromptDefinition.from_yaml(SAMPLE_YAML)
    assert definition.name == "summarize"
    assert definition.version == PromptVersion(1, 2, 0)
    assert definition.default_model.provider == "bedrock/claude-3-haiku"


def test_render_template():
    definition = PromptDefinition.from_yaml(SAMPLE_YAML)
    rendered = definition.render({"document": "Hello world", "max_words": 50})
    assert "Hello world" in rendered
    assert "50" in rendered


def test_render_with_defaults():
    definition = PromptDefinition.from_yaml(SAMPLE_YAML)
    rendered = definition.render({"document": "Hello world"})
    assert "100" in rendered  # default max_words


def test_validate_inputs():
    definition = PromptDefinition.from_yaml(SAMPLE_YAML)
    prompt = Prompt(definition)
    errors = prompt.validate_inputs({"document": "test"})
    assert errors == []


def test_validate_missing_required():
    definition = PromptDefinition.from_yaml(SAMPLE_YAML)
    prompt = Prompt(definition)
    errors = prompt.validate_inputs({})
    assert len(errors) > 0
    assert any("required" in e for e in errors)


def test_estimate_cost():
    definition = PromptDefinition.from_yaml(SAMPLE_YAML)
    cost = definition.estimate_cost({"document": "Hello world", "max_words": 100})
    assert cost > 0
    assert cost < 1.0  # should be well under $1


def test_estimate_tokens():
    definition = PromptDefinition.from_yaml(SAMPLE_YAML)
    tokens = definition.estimate_tokens({"document": "Hello world", "max_words": 100})
    assert tokens > 0


def test_empty_yaml():
    with pytest.raises(ValueError):
        PromptDefinition.from_yaml("")


def test_missing_name():
    with pytest.raises(ValueError):
        PromptDefinition.from_yaml("version: 1.0.0\ntemplate: test")
