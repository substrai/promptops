"""Prompt diff engine for semantic change detection.

Analyzes differences between prompt versions and classifies changes
according to semantic versioning principles:

- PATCH: Wording/formatting changes that don't affect behavior
- MINOR: New optional variables, added examples, expanded instructions
- MAJOR: Breaking changes — removed variables, schema changes, role changes

The engine performs structural analysis rather than simple text diff,
understanding prompt components (template, variables, schema, metadata)
to provide meaningful change classification.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


class ChangeType(Enum):
    """Semantic version change classification."""

    PATCH = "patch"
    MINOR = "minor"
    MAJOR = "major"
    NONE = "none"


class ChangeCategory(Enum):
    """Categories of prompt changes."""

    TEMPLATE_WORDING = "template_wording"
    TEMPLATE_STRUCTURE = "template_structure"
    VARIABLE_ADDED = "variable_added"
    VARIABLE_REMOVED = "variable_removed"
    VARIABLE_RENAMED = "variable_renamed"
    VARIABLE_TYPE_CHANGED = "variable_type_changed"
    SCHEMA_INPUT_ADDED = "schema_input_added"
    SCHEMA_INPUT_REMOVED = "schema_input_removed"
    SCHEMA_OUTPUT_CHANGED = "schema_output_changed"
    ROLE_CHANGED = "role_changed"
    EXAMPLES_ADDED = "examples_added"
    EXAMPLES_REMOVED = "examples_removed"
    METADATA_CHANGED = "metadata_changed"
    FORMATTING_ONLY = "formatting_only"


# Classification rules: category -> change type
_CHANGE_SEVERITY: Dict[ChangeCategory, ChangeType] = {
    ChangeCategory.FORMATTING_ONLY: ChangeType.PATCH,
    ChangeCategory.TEMPLATE_WORDING: ChangeType.PATCH,
    ChangeCategory.METADATA_CHANGED: ChangeType.PATCH,
    ChangeCategory.EXAMPLES_ADDED: ChangeType.MINOR,
    ChangeCategory.VARIABLE_ADDED: ChangeType.MINOR,
    ChangeCategory.SCHEMA_INPUT_ADDED: ChangeType.MINOR,
    ChangeCategory.TEMPLATE_STRUCTURE: ChangeType.MINOR,
    ChangeCategory.VARIABLE_REMOVED: ChangeType.MAJOR,
    ChangeCategory.VARIABLE_RENAMED: ChangeType.MAJOR,
    ChangeCategory.VARIABLE_TYPE_CHANGED: ChangeType.MAJOR,
    ChangeCategory.SCHEMA_INPUT_REMOVED: ChangeType.MAJOR,
    ChangeCategory.SCHEMA_OUTPUT_CHANGED: ChangeType.MAJOR,
    ChangeCategory.ROLE_CHANGED: ChangeType.MAJOR,
    ChangeCategory.EXAMPLES_REMOVED: ChangeType.MINOR,
}


@dataclass
class Change:
    """A single detected change between prompt versions."""

    category: ChangeCategory
    description: str
    severity: ChangeType
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    path: str = ""

    def __post_init__(self):
        self.severity = _CHANGE_SEVERITY.get(self.category, ChangeType.PATCH)


@dataclass
class DiffResult:
    """Result of comparing two prompt versions.

    Contains all detected changes, the overall change classification,
    and a human-readable summary.
    """

    changes: List[Change] = field(default_factory=list)
    overall_change: ChangeType = ChangeType.NONE
    summary: str = ""
    old_variables: FrozenSet[str] = field(default_factory=frozenset)
    new_variables: FrozenSet[str] = field(default_factory=frozenset)
    variables_added: FrozenSet[str] = field(default_factory=frozenset)
    variables_removed: FrozenSet[str] = field(default_factory=frozenset)
    template_similarity: float = 1.0

    @property
    def is_breaking(self) -> bool:
        """Whether this diff contains breaking changes."""
        return self.overall_change == ChangeType.MAJOR

    @property
    def has_changes(self) -> bool:
        """Whether any changes were detected."""
        return self.overall_change != ChangeType.NONE

    @property
    def change_count(self) -> int:
        """Total number of individual changes."""
        return len(self.changes)

    def get_changes_by_severity(self, severity: ChangeType) -> List[Change]:
        """Get all changes of a specific severity level."""
        return [c for c in self.changes if c.severity == severity]


@dataclass
class PromptSnapshot:
    """A snapshot of a prompt version for comparison.

    Represents the structural components of a prompt that can be diffed.
    """

    template: str = ""
    variables: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    role: str = ""
    examples: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptSnapshot":
        """Create a snapshot from a dictionary representation."""
        return cls(
            template=data.get("template", ""),
            variables=data.get("variables", {}),
            input_schema=data.get("input_schema", {}),
            output_schema=data.get("output_schema", {}),
            role=data.get("role", ""),
            examples=data.get("examples", []),
            metadata=data.get("metadata", {}),
        )

    def extract_template_variables(self) -> Set[str]:
        """Extract variable names from the template using {{var}} pattern."""
        pattern = r"\{\{\s*(\w+)\s*\}\}"
        return set(re.findall(pattern, self.template))


class PromptDiffEngine:
    """Engine for computing semantic diffs between prompt versions.

    Analyzes structural differences between two prompt snapshots and
    classifies the overall change as PATCH, MINOR, or MAJOR.

    Example:
        engine = PromptDiffEngine()

        old = PromptSnapshot(
            template="Hello {{name}}, how can I help?",
            variables={"name": {"type": "string", "required": True}},
        )
        new = PromptSnapshot(
            template="Hello {{name}}, how can I assist you today?",
            variables={"name": {"type": "string", "required": True}},
        )

        result = engine.diff(old, new)
        assert result.overall_change == ChangeType.PATCH
    """

    # Threshold below which template changes are considered structural
    STRUCTURAL_SIMILARITY_THRESHOLD = 0.7

    def __init__(self, strict_mode: bool = False):
        """Initialize the diff engine.

        Args:
            strict_mode: If True, treat any variable change as MAJOR.
        """
        self.strict_mode = strict_mode

    def diff(self, old: PromptSnapshot, new: PromptSnapshot) -> DiffResult:
        """Compute the semantic diff between two prompt snapshots.

        Args:
            old: The previous prompt version.
            new: The updated prompt version.

        Returns:
            DiffResult with all changes and overall classification.
        """
        changes: List[Change] = []

        # 1. Analyze template changes
        template_changes, similarity = self._diff_template(old.template, new.template)
        changes.extend(template_changes)

        # 2. Analyze variable changes (from schema + template extraction)
        old_vars = set(old.variables.keys()) | old.extract_template_variables()
        new_vars = set(new.variables.keys()) | new.extract_template_variables()
        var_changes = self._diff_variables(old.variables, new.variables, old_vars, new_vars)
        changes.extend(var_changes)

        # 3. Analyze schema changes
        schema_changes = self._diff_schema(
            old.input_schema, new.input_schema,
            old.output_schema, new.output_schema,
        )
        changes.extend(schema_changes)

        # 4. Analyze role changes
        if old.role and new.role and old.role != new.role:
            changes.append(Change(
                category=ChangeCategory.ROLE_CHANGED,
                description=f"System role changed from '{old.role}' to '{new.role}'",
                severity=ChangeType.MAJOR,
                old_value=old.role,
                new_value=new.role,
                path="role",
            ))

        # 5. Analyze example changes
        example_changes = self._diff_examples(old.examples, new.examples)
        changes.extend(example_changes)

        # 6. Analyze metadata changes
        if old.metadata != new.metadata:
            changes.append(Change(
                category=ChangeCategory.METADATA_CHANGED,
                description="Prompt metadata was modified",
                severity=ChangeType.PATCH,
                path="metadata",
            ))

        # Determine overall change type (highest severity wins)
        overall = self._compute_overall_change(changes)

        # Build summary
        summary = self._build_summary(changes, overall)

        added = frozenset(new_vars - old_vars)
        removed = frozenset(old_vars - new_vars)

        return DiffResult(
            changes=changes,
            overall_change=overall,
            summary=summary,
            old_variables=frozenset(old_vars),
            new_variables=frozenset(new_vars),
            variables_added=added,
            variables_removed=removed,
            template_similarity=similarity,
        )

    def _diff_template(
        self, old_template: str, new_template: str
    ) -> Tuple[List[Change], float]:
        """Analyze template text changes.

        Returns changes and similarity ratio.
        """
        changes: List[Change] = []

        if old_template == new_template:
            return changes, 1.0

        # Compute similarity
        similarity = difflib.SequenceMatcher(
            None, old_template, new_template
        ).ratio()

        # Normalize whitespace for formatting-only check
        old_normalized = " ".join(old_template.split())
        new_normalized = " ".join(new_template.split())

        if old_normalized == new_normalized:
            changes.append(Change(
                category=ChangeCategory.FORMATTING_ONLY,
                description="Only whitespace/formatting changes detected",
                severity=ChangeType.PATCH,
                path="template",
            ))
        elif similarity >= self.STRUCTURAL_SIMILARITY_THRESHOLD:
            changes.append(Change(
                category=ChangeCategory.TEMPLATE_WORDING,
                description=f"Template wording changed (similarity: {similarity:.1%})",
                severity=ChangeType.PATCH,
                old_value=old_template[:100],
                new_value=new_template[:100],
                path="template",
            ))
        else:
            changes.append(Change(
                category=ChangeCategory.TEMPLATE_STRUCTURE,
                description=f"Template structure significantly changed (similarity: {similarity:.1%})",
                severity=ChangeType.MINOR,
                old_value=old_template[:100],
                new_value=new_template[:100],
                path="template",
            ))

        return changes, similarity

    def _diff_variables(
        self,
        old_vars: Dict[str, Dict[str, Any]],
        new_vars: Dict[str, Dict[str, Any]],
        old_var_names: Set[str],
        new_var_names: Set[str],
    ) -> List[Change]:
        """Analyze variable changes between versions."""
        changes: List[Change] = []

        added = new_var_names - old_var_names
        removed = old_var_names - new_var_names

        for var_name in added:
            changes.append(Change(
                category=ChangeCategory.VARIABLE_ADDED,
                description=f"New variable '{{{{{{var_name}}}}}}' added",
                severity=ChangeType.MINOR,
                new_value=var_name,
                path=f"variables.{var_name}",
            ))

        for var_name in removed:
            changes.append(Change(
                category=ChangeCategory.VARIABLE_REMOVED,
                description=f"Variable '{{{{{{var_name}}}}}}' removed (breaking)",
                severity=ChangeType.MAJOR,
                old_value=var_name,
                path=f"variables.{var_name}",
            ))

        # Check type changes for shared variables
        shared = old_var_names & new_var_names
        for var_name in shared:
            old_def = old_vars.get(var_name, {})
            new_def = new_vars.get(var_name, {})

            old_type = old_def.get("type", "string")
            new_type = new_def.get("type", "string")

            if old_type != new_type:
                changes.append(Change(
                    category=ChangeCategory.VARIABLE_TYPE_CHANGED,
                    description=f"Variable '{var_name}' type changed: {old_type} -> {new_type}",
                    severity=ChangeType.MAJOR,
                    old_value=old_type,
                    new_value=new_type,
                    path=f"variables.{var_name}.type",
                ))

        return changes

    def _diff_schema(
        self,
        old_input: Dict[str, Any],
        new_input: Dict[str, Any],
        old_output: Dict[str, Any],
        new_output: Dict[str, Any],
    ) -> List[Change]:
        """Analyze input/output schema changes."""
        changes: List[Change] = []

        # Input schema fields
        old_input_fields = set(old_input.get("properties", {}).keys())
        new_input_fields = set(new_input.get("properties", {}).keys())

        added_inputs = new_input_fields - old_input_fields
        removed_inputs = old_input_fields - new_input_fields

        for field_name in added_inputs:
            changes.append(Change(
                category=ChangeCategory.SCHEMA_INPUT_ADDED,
                description=f"New input schema field '{field_name}' added",
                severity=ChangeType.MINOR,
                new_value=field_name,
                path=f"input_schema.properties.{field_name}",
            ))

        for field_name in removed_inputs:
            changes.append(Change(
                category=ChangeCategory.SCHEMA_INPUT_REMOVED,
                description=f"Input schema field '{field_name}' removed (breaking)",
                severity=ChangeType.MAJOR,
                old_value=field_name,
                path=f"input_schema.properties.{field_name}",
            ))

        # Output schema changes (any change is breaking)
        if old_output and new_output and old_output != new_output:
            changes.append(Change(
                category=ChangeCategory.SCHEMA_OUTPUT_CHANGED,
                description="Output schema modified (breaking change)",
                severity=ChangeType.MAJOR,
                path="output_schema",
            ))

        return changes

    def _diff_examples(
        self, old_examples: List[Dict[str, str]], new_examples: List[Dict[str, str]]
    ) -> List[Change]:
        """Analyze example changes."""
        changes: List[Change] = []

        if len(new_examples) > len(old_examples):
            added_count = len(new_examples) - len(old_examples)
            changes.append(Change(
                category=ChangeCategory.EXAMPLES_ADDED,
                description=f"{added_count} example(s) added",
                severity=ChangeType.MINOR,
                path="examples",
            ))
        elif len(new_examples) < len(old_examples):
            removed_count = len(old_examples) - len(new_examples)
            changes.append(Change(
                category=ChangeCategory.EXAMPLES_REMOVED,
                description=f"{removed_count} example(s) removed",
                severity=ChangeType.MINOR,
                path="examples",
            ))

        return changes

    def _compute_overall_change(self, changes: List[Change]) -> ChangeType:
        """Compute the overall change type from individual changes."""
        if not changes:
            return ChangeType.NONE

        severity_order = {
            ChangeType.NONE: 0,
            ChangeType.PATCH: 1,
            ChangeType.MINOR: 2,
            ChangeType.MAJOR: 3,
        }

        max_severity = ChangeType.NONE
        for change in changes:
            if severity_order[change.severity] > severity_order[max_severity]:
                max_severity = change.severity

        return max_severity

    def _build_summary(self, changes: List[Change], overall: ChangeType) -> str:
        """Build a human-readable summary of changes."""
        if not changes:
            return "No changes detected."

        parts = [f"Detected {len(changes)} change(s) — classified as {overall.value.upper()}."]

        major_changes = [c for c in changes if c.severity == ChangeType.MAJOR]
        if major_changes:
            parts.append(f"Breaking changes: {', '.join(c.description for c in major_changes[:3])}")

        return " ".join(parts)
