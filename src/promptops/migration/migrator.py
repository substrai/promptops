"""Prompt migration CLI for version upgrades.

Detects breaking changes between prompt versions, generates migration
scripts, supports dry-run mode, rollback, and migration history tracking.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


class MigrationStatus(str, Enum):
    """Status of a migration."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class BreakingChangeType(str, Enum):
    """Types of breaking changes between prompt versions."""

    VARIABLE_REMOVED = "variable_removed"
    VARIABLE_RENAMED = "variable_renamed"
    VARIABLE_TYPE_CHANGED = "variable_type_changed"
    TEMPLATE_STRUCTURE_CHANGED = "template_structure_changed"
    SYSTEM_PROMPT_CHANGED = "system_prompt_changed"
    OUTPUT_FORMAT_CHANGED = "output_format_changed"
    MODEL_CONSTRAINT_CHANGED = "model_constraint_changed"


@dataclass
class BreakingChange:
    """A detected breaking change between versions."""

    change_type: BreakingChangeType
    description: str
    source_version: str
    target_version: str
    affected_field: str
    old_value: Any = None
    new_value: Any = None
    auto_fixable: bool = False
    fix_suggestion: Optional[str] = None


@dataclass
class MigrationStep:
    """A single step in a migration script."""

    order: int
    action: str
    description: str
    field_path: str
    old_value: Any = None
    new_value: Any = None
    transform: Optional[Callable[[Any], Any]] = None

    def execute(self, prompt_data: dict[str, Any]) -> dict[str, Any]:
        """Execute this migration step on prompt data."""
        keys = self.field_path.split(".")
        result = dict(prompt_data)

        if self.action == "rename":
            self._rename_field(result, keys, self.new_value)
        elif self.action == "remove":
            self._remove_field(result, keys)
        elif self.action == "add":
            self._set_field(result, keys, self.new_value)
        elif self.action == "transform":
            if self.transform:
                current = self._get_field(result, keys)
                self._set_field(result, keys, self.transform(current))
        elif self.action == "replace":
            self._set_field(result, keys, self.new_value)

        return result

    def _get_field(self, data: dict, keys: list[str]) -> Any:
        current = data
        for key in keys:
            current = current[key]
        return current

    def _set_field(self, data: dict, keys: list[str], value: Any) -> None:
        current = data
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value

    def _remove_field(self, data: dict, keys: list[str]) -> None:
        current = data
        for key in keys[:-1]:
            current = current[key]
        current.pop(keys[-1], None)

    def _rename_field(self, data: dict, keys: list[str], new_name: str) -> None:
        current = data
        for key in keys[:-1]:
            current = current[key]
        if keys[-1] in current:
            current[new_name] = current.pop(keys[-1])


@dataclass
class MigrationScript:
    """A complete migration script between two versions."""

    migration_id: str
    source_version: str
    target_version: str
    steps: list[MigrationStep] = field(default_factory=list)
    breaking_changes: list[BreakingChange] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    description: str = ""

    def execute(self, prompt_data: dict[str, Any]) -> dict[str, Any]:
        """Execute all migration steps in order."""
        result = dict(prompt_data)
        for step in sorted(self.steps, key=lambda s: s.order):
            result = step.execute(result)
        return result


@dataclass
class MigrationRecord:
    """Record of a completed migration in history."""

    migration_id: str
    source_version: str
    target_version: str
    status: MigrationStatus
    started_at: float
    completed_at: Optional[float] = None
    error: Optional[str] = None
    rollback_data: Optional[dict[str, Any]] = None
    checksum_before: Optional[str] = None
    checksum_after: Optional[str] = None


class PromptMigrator:
    """Manages prompt version migrations with dry-run and rollback support.

    Detects breaking changes between prompt versions, generates migration
    scripts, and tracks migration history.

    Example:
        >>> migrator = PromptMigrator()
        >>> changes = migrator.detect_breaking_changes(old_prompt, new_prompt)
        >>> script = migrator.generate_migration(old_prompt, new_prompt, "1.0", "2.0")
        >>> result = migrator.execute(prompt_data, script, dry_run=True)
    """

    def __init__(self, history_path: Optional[Path] = None) -> None:
        self._history: list[MigrationRecord] = []
        self._history_path = history_path

    @property
    def history(self) -> list[MigrationRecord]:
        """Migration history."""
        return list(self._history)

    def detect_breaking_changes(
        self,
        source_prompt: dict[str, Any],
        target_prompt: dict[str, Any],
        source_version: str = "unknown",
        target_version: str = "unknown",
    ) -> list[BreakingChange]:
        """Detect breaking changes between two prompt versions.

        Args:
            source_prompt: The original prompt configuration.
            target_prompt: The new prompt configuration.
            source_version: Version string of the source.
            target_version: Version string of the target.

        Returns:
            List of detected breaking changes.
        """
        changes: list[BreakingChange] = []

        # Check for removed variables
        source_vars = set(source_prompt.get("variables", {}).keys())
        target_vars = set(target_prompt.get("variables", {}).keys())

        for removed_var in source_vars - target_vars:
            changes.append(BreakingChange(
                change_type=BreakingChangeType.VARIABLE_REMOVED,
                description=f"Variable '{removed_var}' was removed",
                source_version=source_version,
                target_version=target_version,
                affected_field=f"variables.{removed_var}",
                old_value=source_prompt["variables"][removed_var],
            ))

        # Check for type changes in shared variables
        for var in source_vars & target_vars:
            old_type = type(source_prompt["variables"][var]).__name__
            new_type = type(target_prompt["variables"][var]).__name__
            if old_type != new_type:
                changes.append(BreakingChange(
                    change_type=BreakingChangeType.VARIABLE_TYPE_CHANGED,
                    description=f"Variable '{var}' type changed from {old_type} to {new_type}",
                    source_version=source_version,
                    target_version=target_version,
                    affected_field=f"variables.{var}",
                    old_value=old_type,
                    new_value=new_type,
                ))

        # Check for system prompt changes
        if source_prompt.get("system_prompt") != target_prompt.get("system_prompt"):
            if "system_prompt" in source_prompt and "system_prompt" in target_prompt:
                changes.append(BreakingChange(
                    change_type=BreakingChangeType.SYSTEM_PROMPT_CHANGED,
                    description="System prompt was modified",
                    source_version=source_version,
                    target_version=target_version,
                    affected_field="system_prompt",
                    auto_fixable=True,
                ))

        # Check for output format changes
        if source_prompt.get("output_format") != target_prompt.get("output_format"):
            if "output_format" in source_prompt and "output_format" in target_prompt:
                changes.append(BreakingChange(
                    change_type=BreakingChangeType.OUTPUT_FORMAT_CHANGED,
                    description="Output format was changed",
                    source_version=source_version,
                    target_version=target_version,
                    affected_field="output_format",
                    old_value=source_prompt.get("output_format"),
                    new_value=target_prompt.get("output_format"),
                ))

        # Check for model constraint changes
        if source_prompt.get("model") != target_prompt.get("model"):
            if "model" in source_prompt and "model" in target_prompt:
                changes.append(BreakingChange(
                    change_type=BreakingChangeType.MODEL_CONSTRAINT_CHANGED,
                    description="Model constraint was changed",
                    source_version=source_version,
                    target_version=target_version,
                    affected_field="model",
                    old_value=source_prompt.get("model"),
                    new_value=target_prompt.get("model"),
                ))

        return changes

    def generate_migration(
        self,
        source_prompt: dict[str, Any],
        target_prompt: dict[str, Any],
        source_version: str,
        target_version: str,
    ) -> MigrationScript:
        """Generate a migration script between two prompt versions.

        Args:
            source_prompt: The original prompt configuration.
            target_prompt: The new prompt configuration.
            source_version: Version string of the source.
            target_version: Version string of the target.

        Returns:
            A MigrationScript with steps to transform source to target.
        """
        changes = self.detect_breaking_changes(
            source_prompt, target_prompt, source_version, target_version
        )

        migration_id = self._generate_migration_id(source_version, target_version)
        steps: list[MigrationStep] = []
        order = 0

        # Generate steps for removed variables
        for change in changes:
            if change.change_type == BreakingChangeType.VARIABLE_REMOVED:
                order += 1
                steps.append(MigrationStep(
                    order=order,
                    action="remove",
                    description=f"Remove deprecated variable: {change.affected_field}",
                    field_path=change.affected_field,
                ))

        # Generate steps for new variables in target
        source_vars = set(source_prompt.get("variables", {}).keys())
        target_vars = set(target_prompt.get("variables", {}).keys())
        for new_var in target_vars - source_vars:
            order += 1
            steps.append(MigrationStep(
                order=order,
                action="add",
                description=f"Add new variable: {new_var}",
                field_path=f"variables.{new_var}",
                new_value=target_prompt["variables"][new_var],
            ))

        # Generate steps for changed fields
        for change in changes:
            if change.change_type in (
                BreakingChangeType.SYSTEM_PROMPT_CHANGED,
                BreakingChangeType.OUTPUT_FORMAT_CHANGED,
                BreakingChangeType.MODEL_CONSTRAINT_CHANGED,
            ):
                order += 1
                steps.append(MigrationStep(
                    order=order,
                    action="replace",
                    description=f"Update {change.affected_field}",
                    field_path=change.affected_field,
                    old_value=change.old_value,
                    new_value=target_prompt.get(change.affected_field, change.new_value),
                ))

        return MigrationScript(
            migration_id=migration_id,
            source_version=source_version,
            target_version=target_version,
            steps=steps,
            breaking_changes=changes,
            description=f"Migration from {source_version} to {target_version}",
        )

    def execute(
        self,
        prompt_data: dict[str, Any],
        script: MigrationScript,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Execute a migration script on prompt data.

        Args:
            prompt_data: The prompt data to migrate.
            script: The migration script to execute.
            dry_run: If True, simulate without modifying data.

        Returns:
            The migrated prompt data (or original if dry_run).
        """
        checksum_before = self._compute_checksum(prompt_data)

        record = MigrationRecord(
            migration_id=script.migration_id,
            source_version=script.source_version,
            target_version=script.target_version,
            status=MigrationStatus.IN_PROGRESS,
            started_at=time.time(),
            rollback_data=dict(prompt_data),
            checksum_before=checksum_before,
        )

        try:
            result = script.execute(prompt_data)

            if dry_run:
                record.status = MigrationStatus.PENDING
                record.completed_at = time.time()
                self._history.append(record)
                return result

            record.status = MigrationStatus.COMPLETED
            record.completed_at = time.time()
            record.checksum_after = self._compute_checksum(result)
            self._history.append(record)
            return result

        except Exception as e:
            record.status = MigrationStatus.FAILED
            record.completed_at = time.time()
            record.error = str(e)
            self._history.append(record)
            raise

    def rollback(self, migration_id: str) -> Optional[dict[str, Any]]:
        """Rollback a completed migration.

        Args:
            migration_id: The ID of the migration to rollback.

        Returns:
            The original prompt data before migration, or None if not found.
        """
        for record in reversed(self._history):
            if record.migration_id == migration_id:
                if record.rollback_data is not None:
                    record.status = MigrationStatus.ROLLED_BACK
                    return record.rollback_data
        return None

    def get_migration_status(self, migration_id: str) -> Optional[MigrationStatus]:
        """Get the status of a migration by ID."""
        for record in reversed(self._history):
            if record.migration_id == migration_id:
                return record.status
        return None

    def _generate_migration_id(self, source: str, target: str) -> str:
        """Generate a unique migration ID."""
        content = f"{source}->{target}-{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _compute_checksum(self, data: dict[str, Any]) -> str:
        """Compute a checksum for prompt data."""
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
