"""Golden dataset management for regression testing.

Golden datasets are curated sets of inputs with expected outputs
that serve as the ground truth for prompt quality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class GoldenCase:
    """A single golden test case with input and expected output."""

    name: str
    inputs: Dict[str, Any]
    expected_output: Optional[Any] = None
    reference_text: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_adversarial(self) -> bool:
        """Check if this is an adversarial test case."""
        return "adversarial" in self.tags or "injection" in self.tags


@dataclass
class GoldenDataset:
    """A collection of golden test cases for a prompt."""

    prompt_name: str
    cases: List[GoldenCase]
    version: str = "1.0.0"
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_content: str, base_path: Optional[Path] = None) -> "GoldenDataset":
        """Parse a golden dataset from YAML.

        Args:
            yaml_content: YAML string
            base_path: Base path for resolving @file: references

        Returns:
            GoldenDataset instance
        """
        data = yaml.safe_load(yaml_content)
        if not data:
            raise ValueError("Empty golden dataset")

        prompt_name = data.get("prompt", "")
        version = str(data.get("version", "1.0.0"))
        description = data.get("description", "")

        cases = []
        for case_data in data.get("cases", data.get("test_cases", [])):
            inputs = case_data.get("inputs", {})

            # Resolve @file: references in inputs
            if base_path:
                for key, value in inputs.items():
                    if isinstance(value, str) and value.startswith("@file:"):
                        file_path = base_path / value[6:]
                        if file_path.exists():
                            inputs[key] = file_path.read_text()

            # Resolve expected output
            expected = case_data.get("expected_output", case_data.get("expected"))
            reference = case_data.get("reference", case_data.get("reference_text"))
            if isinstance(reference, str) and reference.startswith("@file:") and base_path:
                file_path = base_path / reference[6:]
                if file_path.exists():
                    reference = file_path.read_text()

            cases.append(
                GoldenCase(
                    name=case_data.get("name", f"case_{len(cases)}"),
                    inputs=inputs,
                    expected_output=expected,
                    reference_text=reference,
                    tags=case_data.get("tags", []),
                    metadata=case_data.get("metadata", {}),
                )
            )

        return cls(
            prompt_name=prompt_name,
            cases=cases,
            version=version,
            description=description,
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "GoldenDataset":
        """Load a golden dataset from a file."""
        path = Path(path)
        content = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            return cls.from_yaml(content, base_path=path.parent)
        elif path.suffix == ".json":
            return cls.from_json(content, base_path=path.parent)
        raise ValueError(f"Unsupported file format: {path.suffix}")

    @classmethod
    def from_json(cls, json_content: str, base_path: Optional[Path] = None) -> "GoldenDataset":
        """Parse a golden dataset from JSON."""
        data = json.loads(json_content)
        # Convert to YAML-compatible format and reuse parser
        yaml_content = yaml.dump(data)
        return cls.from_yaml(yaml_content, base_path=base_path)

    def filter_by_tag(self, tag: str) -> "GoldenDataset":
        """Filter cases by tag."""
        filtered = [c for c in self.cases if tag in c.tags]
        return GoldenDataset(
            prompt_name=self.prompt_name,
            cases=filtered,
            version=self.version,
            description=f"{self.description} (filtered: {tag})",
        )

    @property
    def adversarial_cases(self) -> List[GoldenCase]:
        """Get only adversarial test cases."""
        return [c for c in self.cases if c.is_adversarial]

    @property
    def standard_cases(self) -> List[GoldenCase]:
        """Get non-adversarial test cases."""
        return [c for c in self.cases if not c.is_adversarial]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "prompt": self.prompt_name,
            "version": self.version,
            "description": self.description,
            "cases": [
                {
                    "name": c.name,
                    "inputs": c.inputs,
                    "expected_output": c.expected_output,
                    "reference_text": c.reference_text,
                    "tags": c.tags,
                }
                for c in self.cases
            ],
        }
