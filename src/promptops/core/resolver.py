"""Prompt resolver - resolves prompt names to specific versions.

Handles version resolution, environment-specific overrides,
and A/B experiment routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from promptops.core.prompt import Prompt, PromptDefinition
from promptops.core.version import PromptVersion, VersionRange


@dataclass
class ResolvedPrompt:
    """A resolved prompt ready for invocation."""

    prompt: Prompt
    resolved_version: PromptVersion
    environment: str
    experiment: Optional[str] = None


class PromptResolver:
    """Resolves prompt names to specific versioned definitions.

    The resolver maintains a registry of all known prompt versions
    and handles resolution based on environment, version ranges,
    and A/B experiments.
    """

    def __init__(self, prompts_dir: Optional[str | Path] = None):
        """Initialize the resolver.

        Args:
            prompts_dir: Directory containing prompt YAML files
        """
        self._registry: Dict[str, Dict[str, PromptDefinition]] = {}
        self._prompts_dir = Path(prompts_dir) if prompts_dir else None

        if self._prompts_dir and self._prompts_dir.exists():
            self._load_from_directory(self._prompts_dir)

    def _load_from_directory(self, directory: Path) -> None:
        """Load all prompt definitions from a directory."""
        for yaml_file in directory.glob("*.yaml"):
            try:
                definition = PromptDefinition.from_file(yaml_file)
                self.register(definition)
            except (ValueError, FileNotFoundError) as e:
                # Skip invalid files but log
                pass

        for yaml_file in directory.glob("*.yml"):
            try:
                definition = PromptDefinition.from_file(yaml_file)
                self.register(definition)
            except (ValueError, FileNotFoundError):
                pass

    def register(self, definition: PromptDefinition) -> None:
        """Register a prompt definition in the resolver.

        Args:
            definition: The prompt definition to register
        """
        if definition.name not in self._registry:
            self._registry[definition.name] = {}
        version_key = str(definition.version)
        self._registry[definition.name][version_key] = definition

    def resolve(
        self,
        name: str,
        version: str = "latest",
        environment: str = "prod",
    ) -> ResolvedPrompt:
        """Resolve a prompt name to a specific version.

        Args:
            name: Prompt name
            version: Version string ("latest", "1.2.0", "~1.2", etc.)
            environment: Target environment

        Returns:
            ResolvedPrompt with the matched definition

        Raises:
            KeyError: If prompt name not found
            ValueError: If no matching version found
        """
        if name not in self._registry:
            raise KeyError(f"Prompt '{name}' not found in registry")

        versions = self._registry[name]
        version_range = VersionRange(spec=version)

        # Find matching versions
        matching = []
        for ver_str, definition in versions.items():
            ver = PromptVersion.parse(ver_str)
            if version_range.matches(ver):
                matching.append((ver, definition))

        if not matching:
            available = list(versions.keys())
            raise ValueError(
                f"No version matching '{version}' for prompt '{name}'. "
                f"Available: {available}"
            )

        # Sort and pick latest matching
        matching.sort(key=lambda x: x[0], reverse=True)
        resolved_version, definition = matching[0]

        prompt = Prompt(definition)
        return ResolvedPrompt(
            prompt=prompt,
            resolved_version=resolved_version,
            environment=environment,
        )

    def list_prompts(self) -> List[str]:
        """List all registered prompt names."""
        return list(self._registry.keys())

    def list_versions(self, name: str) -> List[PromptVersion]:
        """List all versions for a prompt.

        Args:
            name: Prompt name

        Returns:
            Sorted list of versions (newest first)
        """
        if name not in self._registry:
            raise KeyError(f"Prompt '{name}' not found")

        versions = [
            PromptVersion.parse(v) for v in self._registry[name].keys()
        ]
        return sorted(versions, reverse=True)

    def get_latest(self, name: str) -> PromptDefinition:
        """Get the latest non-prerelease version of a prompt.

        Args:
            name: Prompt name

        Returns:
            Latest PromptDefinition
        """
        resolved = self.resolve(name, version="latest")
        return resolved.prompt.definition
