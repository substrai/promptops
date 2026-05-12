"""Breaking change detection for prompt versions.

Detects when a prompt change would break consumers by analyzing
schema differences between versions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from promptops.core.prompt import PromptDefinition
from promptops.core.schema import InputSchema, OutputSchema
from promptops.core.version import PromptVersion


class ChangeType(Enum):
    """Types of changes between prompt versions."""

    BREAKING = "breaking"  # Major version bump required
    MINOR = "minor"  # Minor version bump (new optional fields)
    PATCH = "patch"  # Patch version bump (wording only)
    NONE = "none"  # No change


@dataclass
class SchemaChange:
    """A single schema change between versions."""

    field_name: str
    change_type: ChangeType
    description: str
    old_value: Any = None
    new_value: Any = None


@dataclass
class BreakingChangeReport:
    """Report of changes between two prompt versions."""

    prompt_name: str
    old_version: PromptVersion
    new_version: PromptVersion
    changes: List[SchemaChange] = field(default_factory=list)
    recommended_bump: ChangeType = ChangeType.NONE

    @property
    def has_breaking_changes(self) -> bool:
        return any(c.change_type == ChangeType.BREAKING for c in self.changes)

    @property
    def breaking_changes(self) -> List[SchemaChange]:
        return [c for c in self.changes if c.change_type == ChangeType.BREAKING]

    @property
    def minor_changes(self) -> List[SchemaChange]:
        return [c for c in self.changes if c.change_type == ChangeType.MINOR]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Change Report: {self.prompt_name} v{self.old_version} → v{self.new_version}",
            f"Recommended bump: {self.recommended_bump.value}",
            "",
        ]

        if self.breaking_changes:
            lines.append("⚠️  BREAKING CHANGES:")
            for c in self.breaking_changes:
                lines.append(f"  • {c.description}")

        if self.minor_changes:
            lines.append("\n📝 Minor changes:")
            for c in self.minor_changes:
                lines.append(f"  • {c.description}")

        patch_changes = [c for c in self.changes if c.change_type == ChangeType.PATCH]
        if patch_changes:
            lines.append(f"\n✏️  Patch changes: {len(patch_changes)} wording updates")

        return "\n".join(lines)


class BreakingChangeDetector:
    """Detects breaking changes between prompt versions.

    Breaking changes include:
    - Removing a required input field
    - Changing a field's type
    - Removing an output field that consumers depend on
    - Changing enum values (removing options)

    Minor changes include:
    - Adding a new optional input field
    - Adding a new output field
    - Adding enum values

    Patch changes include:
    - Template wording changes
    - Settings changes (temperature, max_tokens)
    - Metadata changes
    """

    def detect(
        self,
        old: PromptDefinition,
        new: PromptDefinition,
    ) -> BreakingChangeReport:
        """Detect changes between two prompt versions.

        Args:
            old: Previous prompt definition
            new: New prompt definition

        Returns:
            BreakingChangeReport with all detected changes
        """
        changes: List[SchemaChange] = []

        # Check input schema changes
        changes.extend(self._check_input_changes(old.input_schema, new.input_schema))

        # Check output schema changes
        changes.extend(self._check_output_changes(old.output_schema, new.output_schema))

        # Check template changes
        if old.template != new.template:
            changes.append(
                SchemaChange(
                    field_name="template",
                    change_type=ChangeType.PATCH,
                    description="Template wording changed",
                    old_value=old.template[:50] + "..." if len(old.template) > 50 else old.template,
                    new_value=new.template[:50] + "..." if len(new.template) > 50 else new.template,
                )
            )

        # Check model changes
        if old.default_model.provider != new.default_model.provider:
            changes.append(
                SchemaChange(
                    field_name="model.default",
                    change_type=ChangeType.MINOR,
                    description=f"Default model changed: {old.default_model.provider} → {new.default_model.provider}",
                    old_value=old.default_model.provider,
                    new_value=new.default_model.provider,
                )
            )

        # Determine recommended bump
        recommended = ChangeType.NONE
        for change in changes:
            if change.change_type == ChangeType.BREAKING:
                recommended = ChangeType.BREAKING
                break
            elif change.change_type == ChangeType.MINOR:
                recommended = ChangeType.MINOR
            elif change.change_type == ChangeType.PATCH and recommended == ChangeType.NONE:
                recommended = ChangeType.PATCH

        return BreakingChangeReport(
            prompt_name=new.name,
            old_version=old.version,
            new_version=new.version,
            changes=changes,
            recommended_bump=recommended,
        )

    def _check_input_changes(
        self, old_schema: InputSchema, new_schema: InputSchema
    ) -> List[SchemaChange]:
        """Check input schema for breaking changes."""
        changes = []
        old_fields = set(old_schema.fields.keys())
        new_fields = set(new_schema.fields.keys())

        # Removed fields = BREAKING
        for removed in old_fields - new_fields:
            old_field = old_schema.fields[removed]
            if old_field.required:
                changes.append(
                    SchemaChange(
                        field_name=f"input.{removed}",
                        change_type=ChangeType.BREAKING,
                        description=f"Required input field '{removed}' was removed",
                    )
                )
            else:
                changes.append(
                    SchemaChange(
                        field_name=f"input.{removed}",
                        change_type=ChangeType.MINOR,
                        description=f"Optional input field '{removed}' was removed",
                    )
                )

        # Added fields
        for added in new_fields - old_fields:
            new_field = new_schema.fields[added]
            if new_field.required and new_field.default is None:
                changes.append(
                    SchemaChange(
                        field_name=f"input.{added}",
                        change_type=ChangeType.BREAKING,
                        description=f"New required input field '{added}' added without default",
                    )
                )
            else:
                changes.append(
                    SchemaChange(
                        field_name=f"input.{added}",
                        change_type=ChangeType.MINOR,
                        description=f"New optional input field '{added}' added",
                    )
                )

        # Changed fields
        for field_name in old_fields & new_fields:
            old_field = old_schema.fields[field_name]
            new_field = new_schema.fields[field_name]

            # Type change = BREAKING
            if old_field.type != new_field.type:
                changes.append(
                    SchemaChange(
                        field_name=f"input.{field_name}",
                        change_type=ChangeType.BREAKING,
                        description=f"Field '{field_name}' type changed: {old_field.type} → {new_field.type}",
                        old_value=old_field.type,
                        new_value=new_field.type,
                    )
                )

            # Enum values removed = BREAKING
            if old_field.values and new_field.values:
                removed_values = set(old_field.values) - set(new_field.values)
                if removed_values:
                    changes.append(
                        SchemaChange(
                            field_name=f"input.{field_name}",
                            change_type=ChangeType.BREAKING,
                            description=f"Enum values removed from '{field_name}': {removed_values}",
                            old_value=old_field.values,
                            new_value=new_field.values,
                        )
                    )
                added_values = set(new_field.values) - set(old_field.values)
                if added_values:
                    changes.append(
                        SchemaChange(
                            field_name=f"input.{field_name}",
                            change_type=ChangeType.MINOR,
                            description=f"Enum values added to '{field_name}': {added_values}",
                        )
                    )

            # Required changed from optional to required = BREAKING
            if not old_field.required and new_field.required and new_field.default is None:
                changes.append(
                    SchemaChange(
                        field_name=f"input.{field_name}",
                        change_type=ChangeType.BREAKING,
                        description=f"Field '{field_name}' changed from optional to required",
                    )
                )

        return changes

    def _check_output_changes(
        self, old_schema: OutputSchema, new_schema: OutputSchema
    ) -> List[SchemaChange]:
        """Check output schema for breaking changes."""
        changes = []
        old_fields = set(old_schema.fields.keys())
        new_fields = set(new_schema.fields.keys())

        # Removed output fields = BREAKING (consumers may depend on them)
        for removed in old_fields - new_fields:
            changes.append(
                SchemaChange(
                    field_name=f"output.{removed}",
                    change_type=ChangeType.BREAKING,
                    description=f"Output field '{removed}' was removed",
                )
            )

        # Added output fields = MINOR
        for added in new_fields - old_fields:
            changes.append(
                SchemaChange(
                    field_name=f"output.{added}",
                    change_type=ChangeType.MINOR,
                    description=f"New output field '{added}' added",
                )
            )

        return changes
