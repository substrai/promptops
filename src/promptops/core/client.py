"""PromptOps client for runtime prompt invocation.

This is the primary interface for applications to invoke
versioned, managed prompts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from promptops.core.prompt import Prompt, PromptDefinition
from promptops.core.resolver import PromptResolver
from promptops.core.result import InvocationResult
from promptops.core.version import PromptVersion


class PromptClient:
    """Client for invoking PromptOps-managed prompts.

    Usage:
        client = PromptClient(env="prod", prompts_dir="./prompts")
        result = client.invoke("summarize", inputs={"document": "...", "max_words": 100})
    """

    def __init__(
        self,
        env: str = "dev",
        prompts_dir: Optional[str | Path] = None,
        registry_backend: str = "local",
    ):
        """Initialize the PromptOps client.

        Args:
            env: Environment name (dev, staging, prod)
            prompts_dir: Path to prompts directory (for local mode)
            registry_backend: Backend type ("local" or "dynamodb")
        """
        self.env = env
        self.prompts_dir = Path(prompts_dir) if prompts_dir else Path("./prompts")
        self.registry_backend = registry_backend
        self._resolver = PromptResolver(prompts_dir=self.prompts_dir)
        self._invocation_log: list = []

    def invoke(
        self,
        prompt: str,
        inputs: Dict[str, Any],
        version: str = "latest",
        model: Optional[str] = None,
    ) -> InvocationResult:
        """Invoke a prompt with given inputs.

        Resolves the prompt version, renders the template, and returns
        the result with metadata.

        Args:
            prompt: Prompt name
            inputs: Input values matching the prompt's input schema
            version: Version to use ("latest", "1.2.0", "~1.2")
            model: Optional model override

        Returns:
            InvocationResult with output and metadata
        """
        start_time = time.time()

        try:
            # Resolve prompt
            resolved = self._resolver.resolve(prompt, version=version, environment=self.env)
            definition = resolved.prompt.definition

            # Get model config
            model_config = definition.get_model_config(model)

            # Render template
            rendered = definition.render(inputs)

            # Estimate tokens and cost
            input_tokens = len(rendered) // 4
            output_tokens = model_config.max_tokens // 2
            cost = definition.estimate_cost(inputs, model)

            latency_ms = (time.time() - start_time) * 1000

            result = InvocationResult(
                output=rendered,  # In production, this would be the LLM response
                prompt_name=prompt,
                version=resolved.resolved_version,
                model=model_config.provider,
                latency_ms=round(latency_ms, 2),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost=cost,
                environment=self.env,
                metadata={
                    "rendered_template": rendered,
                    "model_config": {
                        "temperature": model_config.temperature,
                        "max_tokens": model_config.max_tokens,
                    },
                },
            )

            self._invocation_log.append(result.to_dict())
            return result

        except (KeyError, ValueError) as e:
            latency_ms = (time.time() - start_time) * 1000
            return InvocationResult(
                output=None,
                prompt_name=prompt,
                version=PromptVersion(0, 0, 0),
                model="unknown",
                latency_ms=round(latency_ms, 2),
                environment=self.env,
                errors=[str(e)],
            )

    def get(self, prompt: str, version: str = "latest") -> Prompt:
        """Get a prompt object without invoking it.

        Args:
            prompt: Prompt name
            version: Version to resolve

        Returns:
            Prompt object
        """
        resolved = self._resolver.resolve(prompt, version=version, environment=self.env)
        return resolved.prompt

    def list_prompts(self) -> list:
        """List all available prompts."""
        return self._resolver.list_prompts()

    def list_versions(self, prompt: str) -> list:
        """List all versions of a prompt."""
        return self._resolver.list_versions(prompt)

    def validate(self, prompt: str, inputs: Dict[str, Any], version: str = "latest") -> list:
        """Validate inputs against a prompt's schema without invoking.

        Args:
            prompt: Prompt name
            inputs: Input values to validate
            version: Version to validate against

        Returns:
            List of validation errors (empty if valid)
        """
        resolved = self._resolver.resolve(prompt, version=version, environment=self.env)
        definition = resolved.prompt.definition
        inputs_with_defaults = definition.input_schema.apply_defaults(inputs)
        return definition.input_schema.validate(inputs_with_defaults)

    def estimate_cost(self, prompt: str, inputs: Dict[str, Any], version: str = "latest") -> float:
        """Estimate cost without invoking.

        Args:
            prompt: Prompt name
            inputs: Input values
            version: Version to estimate for

        Returns:
            Estimated cost in USD
        """
        resolved = self._resolver.resolve(prompt, version=version, environment=self.env)
        return resolved.prompt.definition.estimate_cost(inputs)

    @property
    def invocation_history(self) -> list:
        """Get invocation history for this client session."""
        return self._invocation_log
