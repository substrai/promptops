"""Tests for the PromptClient."""

import tempfile
from pathlib import Path

import pytest
from promptops.core.client import PromptClient


SAMPLE_PROMPT = """
name: greet
version: 1.0.0
description: "Greeting prompt"

model:
  default: bedrock/claude-3-haiku

input:
  schema:
    name:
      type: string
      required: true
    language:
      type: enum
      values: [english, spanish, french]
      default: english

output:
  schema:
    greeting:
      type: string

template: |
  Say hello to {name} in {language}.

settings:
  temperature: 0.5
  max_tokens: 100
"""


@pytest.fixture
def prompts_dir(tmp_path):
    """Create a temporary prompts directory."""
    prompt_file = tmp_path / "greet.yaml"
    prompt_file.write_text(SAMPLE_PROMPT)
    return tmp_path


def test_client_invoke(prompts_dir):
    client = PromptClient(env="dev", prompts_dir=prompts_dir)
    result = client.invoke("greet", inputs={"name": "World"})
    assert result.success
    assert "World" in result.output
    assert result.cost > 0


def test_client_list_prompts(prompts_dir):
    client = PromptClient(env="dev", prompts_dir=prompts_dir)
    prompts = client.list_prompts()
    assert "greet" in prompts


def test_client_validate(prompts_dir):
    client = PromptClient(env="dev", prompts_dir=prompts_dir)
    errors = client.validate("greet", inputs={"name": "Test"})
    assert errors == []


def test_client_validate_invalid(prompts_dir):
    client = PromptClient(env="dev", prompts_dir=prompts_dir)
    errors = client.validate("greet", inputs={})
    assert len(errors) > 0


def test_client_estimate_cost(prompts_dir):
    client = PromptClient(env="dev", prompts_dir=prompts_dir)
    cost = client.estimate_cost("greet", inputs={"name": "Test"})
    assert cost > 0


def test_client_not_found(prompts_dir):
    client = PromptClient(env="dev", prompts_dir=prompts_dir)
    result = client.invoke("nonexistent", inputs={})
    assert not result.success
    assert len(result.errors) > 0
