"""Prompt definition parser and model.

Reads YAML prompt definitions and creates typed, versioned prompt objects.
This is the heart of the PromptOps framework.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from promptops.core.schema import InputSchema, OutputSchema
from promptops.core.version import PromptVersion


@dataclass
class ModelConfig:
    """Configuration for a model variant."""

    provider: str  # e.g., "bedrock/claude-3-haiku"
    template: Optional[str] = None  # override template for this model
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    stop_sequences: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, config: Dict[str, Any], provider: str = "") -> "ModelConfig":
        """Parse model config from dictionary."""
        settings = config.get("settings", {})
        return cls(
            provider=provider or config.get("provider", ""),
            template=config.get("template"),
            temperature=settings.get("temperature", 0.7),
            max_tokens=settings.get("max_tokens", 2000),
            top_p=settings.get("top_p", 1.0),
            stop_sequences=settings.get("stop_sequences", []),
        )


@dataclass
class PromptDefinition:
    """A parsed prompt definition from YAML.

    This represents the full definition of a prompt including its
    template, schemas, model configuration, and metadata.
    """

    name: str
    version: PromptVersion
    description: str
    template: str
    input_schema: InputSchema
    output_schema: OutputSchema
    default_model: ModelConfig
    fallback_model: Optional[ModelConfig] = None
    model_variants: Dict[str, ModelConfig] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_content: str, base_path: Optional[Path] = None) -> "PromptDefinition":
        """Parse a prompt definition from YAML content.

        Args:
            yaml_content: Raw YAML string
            base_path: Base path for resolving @file: references

        Returns:
            PromptDefinition instance

        Raises:
            ValueError: If the YAML is invalid or missing required fields
        """
        data = yaml.safe_load(yaml_content)
        if not data:
            raise ValueError("Empty prompt definition")

        # Required fields
        name = data.get("name")
        if not name:
            raise ValueError("Prompt definition must have a 'name' field")

        version_str = data.get("version", "0.1.0")
        version = PromptVersion.parse(str(version_str))

        description = data.get("description", "")

        # Template
        template = data.get("template", "")
        if template.startswith("@file:") and base_path:
            file_path = base_path / template[6:]
            if file_path.exists():
                template = file_path.read_text()

        # Schemas
        input_schema_data = data.get("input", {}).get("schema", {})
        input_schema = InputSchema.from_dict(input_schema_data)

        output_schema_data = data.get("output", {}).get("schema", {})
        output_schema = OutputSchema.from_dict(output_schema_data)

        # Model configuration
        model_data = data.get("model", {})
        default_provider = model_data.get("default", "bedrock/claude-3-haiku")
        default_model = ModelConfig(
            provider=default_provider,
            temperature=data.get("settings", {}).get("temperature", 0.7),
            max_tokens=data.get("settings", {}).get("max_tokens", 2000),
            stop_sequences=data.get("settings", {}).get("stop_sequences", []),
        )

        fallback_model = None
        if "fallback" in model_data:
            fallback_model = ModelConfig(provider=model_data["fallback"])

        # Model variants
        model_variants = {}
        for provider, variant_config in model_data.get("variants", {}).items():
            if isinstance(variant_config, dict):
                mc = ModelConfig.from_dict(variant_config, provider=provider)
                # Resolve @file: template references
                if mc.template and mc.template.startswith("@file:") and base_path:
                    file_path = base_path / mc.template[6:]
                    if file_path.exists():
                        mc = ModelConfig(
                            provider=mc.provider,
                            template=file_path.read_text(),
                            temperature=mc.temperature,
                            max_tokens=mc.max_tokens,
                            top_p=mc.top_p,
                            stop_sequences=mc.stop_sequences,
                        )
                model_variants[provider] = mc

        return cls(
            name=name,
            version=version,
            description=description,
            template=template,
            input_schema=input_schema,
            output_schema=output_schema,
            default_model=default_model,
            fallback_model=fallback_model,
            model_variants=model_variants,
            metadata=data.get("metadata", {}),
            settings=data.get("settings", {}),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "PromptDefinition":
        """Load a prompt definition from a YAML file.

        Args:
            path: Path to the YAML file

        Returns:
            PromptDefinition instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        content = path.read_text()
        return cls.from_yaml(content, base_path=path.parent)

    def render(self, inputs: Dict[str, Any]) -> str:
        """Render the prompt template with given inputs.

        Args:
            inputs: Dictionary of input values

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If inputs fail schema validation
        """
        # Apply defaults
        inputs_with_defaults = self.input_schema.apply_defaults(inputs)

        # Validate
        errors = self.input_schema.validate(inputs_with_defaults)
        if errors:
            raise ValueError(f"Input validation failed: {'; '.join(errors)}")

        # Render template using simple string formatting
        rendered = self.template
        for key, value in inputs_with_defaults.items():
            rendered = rendered.replace(f"{{{key}}}", str(value))

        return rendered

    def get_model_config(self, model: Optional[str] = None) -> ModelConfig:
        """Get model configuration for a specific model or default.

        Args:
            model: Optional model provider string

        Returns:
            ModelConfig for the requested or default model
        """
        if model and model in self.model_variants:
            return self.model_variants[model]
        return self.default_model

    def estimate_tokens(self, inputs: Dict[str, Any]) -> int:
        """Estimate token count for rendered prompt.

        Uses a rough approximation of 4 characters per token.

        Args:
            inputs: Input values to render

        Returns:
            Estimated token count
        """
        rendered = self.render(inputs)
        # Rough estimation: ~4 chars per token for English text
        return len(rendered) // 4

    def estimate_cost(self, inputs: Dict[str, Any], model: Optional[str] = None) -> float:
        """Estimate cost for a single invocation.

        Args:
            inputs: Input values
            model: Optional model override

        Returns:
            Estimated cost in USD
        """
        # Pricing per 1K tokens (approximate)
        pricing = {
            "bedrock/claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "bedrock/claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "bedrock/claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
            "bedrock/amazon-titan-text-lite": {"input": 0.00015, "output": 0.00015},
            "bedrock/amazon-titan-text": {"input": 0.0008, "output": 0.0008},
            "openai/gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "openai/gpt-4o": {"input": 0.005, "output": 0.015},
        }

        config = self.get_model_config(model)
        model_pricing = pricing.get(config.provider, {"input": 0.001, "output": 0.002})

        input_tokens = self.estimate_tokens(inputs)
        output_tokens = config.max_tokens // 2  # assume half of max

        cost = (input_tokens / 1000) * model_pricing["input"]
        cost += (output_tokens / 1000) * model_pricing["output"]

        return round(cost, 6)


class Prompt:
    """High-level prompt interface for runtime usage.

    This is the main class users interact with to invoke prompts.
    """

    def __init__(self, definition: PromptDefinition):
        self.definition = definition
        self.name = definition.name
        self.version = definition.version
        self.template = definition.template

    @classmethod
    def load(cls, path: str | Path) -> "Prompt":
        """Load a prompt from a YAML file."""
        definition = PromptDefinition.from_file(path)
        return cls(definition)

    def render(self, **kwargs) -> str:
        """Render the prompt with given inputs."""
        return self.definition.render(kwargs)

    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate inputs against the schema."""
        inputs_with_defaults = self.definition.input_schema.apply_defaults(inputs)
        return self.definition.input_schema.validate(inputs_with_defaults)

    def estimate_cost(self, inputs: Dict[str, Any]) -> float:
        """Estimate invocation cost."""
        return self.definition.estimate_cost(inputs)

    def __repr__(self) -> str:
        return f"Prompt(name='{self.name}', version='{self.version}')"
