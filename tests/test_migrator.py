"""Tests for prompt migration CLI."""

import pytest

from promptops.migration.migrator import (
    BreakingChange,
    BreakingChangeType,
    MigrationScript,
    MigrationStatus,
    MigrationStep,
    PromptMigrator,
)


@pytest.fixture
def migrator() -> PromptMigrator:
    """Create a prompt migrator instance."""
    return PromptMigrator()


@pytest.fixture
def source_prompt() -> dict:
    """Create a source prompt configuration."""
    return {
        "version": "1.0",
        "system_prompt": "You are a helpful assistant.",
        "variables": {
            "user_name": "string",
            "context": "string",
            "max_tokens": 1000,
        },
        "output_format": "text",
        "model": "gpt-4",
    }


@pytest.fixture
def target_prompt() -> dict:
    """Create a target prompt configuration with breaking changes."""
    return {
        "version": "2.0",
        "system_prompt": "You are an expert AI assistant with tool access.",
        "variables": {
            "user_name": "string",
            "conversation_history": [],
            "max_tokens": "1000",  # Type changed from int to str
        },
        "output_format": "json",
        "model": "gpt-4o",
    }


class TestBreakingChangeDetection:
    """Test detection of breaking changes between versions."""

    def test_detects_removed_variables(
        self, migrator: PromptMigrator, source_prompt: dict, target_prompt: dict
    ) -> None:
        """Should detect variables that were removed."""
        changes = migrator.detect_breaking_changes(
            source_prompt, target_prompt, "1.0", "2.0"
        )
        removed = [c for c in changes if c.change_type == BreakingChangeType.VARIABLE_REMOVED]
        assert len(removed) == 1
        assert "context" in removed[0].affected_field

    def test_detects_type_changes(
        self, migrator: PromptMigrator, source_prompt: dict, target_prompt: dict
    ) -> None:
        """Should detect variable type changes."""
        changes = migrator.detect_breaking_changes(
            source_prompt, target_prompt, "1.0", "2.0"
        )
        type_changes = [c for c in changes if c.change_type == BreakingChangeType.VARIABLE_TYPE_CHANGED]
        assert len(type_changes) == 1
        assert "max_tokens" in type_changes[0].affected_field

    def test_detects_system_prompt_change(
        self, migrator: PromptMigrator, source_prompt: dict, target_prompt: dict
    ) -> None:
        """Should detect system prompt modifications."""
        changes = migrator.detect_breaking_changes(
            source_prompt, target_prompt, "1.0", "2.0"
        )
        sys_changes = [c for c in changes if c.change_type == BreakingChangeType.SYSTEM_PROMPT_CHANGED]
        assert len(sys_changes) == 1

    def test_detects_output_format_change(
        self, migrator: PromptMigrator, source_prompt: dict, target_prompt: dict
    ) -> None:
        """Should detect output format changes."""
        changes = migrator.detect_breaking_changes(
            source_prompt, target_prompt, "1.0", "2.0"
        )
        format_changes = [c for c in changes if c.change_type == BreakingChangeType.OUTPUT_FORMAT_CHANGED]
        assert len(format_changes) == 1
        assert format_changes[0].old_value == "text"
        assert format_changes[0].new_value == "json"

    def test_no_changes_for_identical_prompts(self, migrator: PromptMigrator, source_prompt: dict) -> None:
        """Should return empty list for identical prompts."""
        changes = migrator.detect_breaking_changes(source_prompt, source_prompt, "1.0", "1.0")
        assert len(changes) == 0


class TestMigrationGeneration:
    """Test migration script generation."""

    def test_generates_migration_script(
        self, migrator: PromptMigrator, source_prompt: dict, target_prompt: dict
    ) -> None:
        """Should generate a valid migration script."""
        script = migrator.generate_migration(source_prompt, target_prompt, "1.0", "2.0")
        assert script.source_version == "1.0"
        assert script.target_version == "2.0"
        assert len(script.steps) > 0
        assert len(script.breaking_changes) > 0

    def test_migration_script_has_remove_steps(
        self, migrator: PromptMigrator, source_prompt: dict, target_prompt: dict
    ) -> None:
        """Should include remove steps for deleted variables."""
        script = migrator.generate_migration(source_prompt, target_prompt, "1.0", "2.0")
        remove_steps = [s for s in script.steps if s.action == "remove"]
        assert len(remove_steps) >= 1

    def test_migration_script_has_add_steps(
        self, migrator: PromptMigrator, source_prompt: dict, target_prompt: dict
    ) -> None:
        """Should include add steps for new variables."""
        script = migrator.generate_migration(source_prompt, target_prompt, "1.0", "2.0")
        add_steps = [s for s in script.steps if s.action == "add"]
        assert len(add_steps) >= 1  # conversation_history is new


class TestMigrationExecution:
    """Test migration execution with dry-run and rollback."""

    def test_execute_migration(
        self, migrator: PromptMigrator, source_prompt: dict, target_prompt: dict
    ) -> None:
        """Should execute migration and transform data."""
        script = migrator.generate_migration(source_prompt, target_prompt, "1.0", "2.0")
        result = migrator.execute(source_prompt, script)

        # context should be removed
        assert "context" not in result.get("variables", {})
        # conversation_history should be added
        assert "conversation_history" in result.get("variables", {})

    def test_dry_run_does_not_modify_history_status(
        self, migrator: PromptMigrator, source_prompt: dict, target_prompt: dict
    ) -> None:
        """Should mark as pending in dry-run mode."""
        script = migrator.generate_migration(source_prompt, target_prompt, "1.0", "2.0")
        migrator.execute(source_prompt, script, dry_run=True)

        status = migrator.get_migration_status(script.migration_id)
        assert status == MigrationStatus.PENDING

    def test_rollback_restores_original(
        self, migrator: PromptMigrator, source_prompt: dict, target_prompt: dict
    ) -> None:
        """Should rollback to original data."""
        script = migrator.generate_migration(source_prompt, target_prompt, "1.0", "2.0")
        migrator.execute(source_prompt, script)

        rolled_back = migrator.rollback(script.migration_id)
        assert rolled_back is not None
        assert rolled_back == source_prompt

    def test_migration_history_tracking(
        self, migrator: PromptMigrator, source_prompt: dict, target_prompt: dict
    ) -> None:
        """Should track migration in history."""
        script = migrator.generate_migration(source_prompt, target_prompt, "1.0", "2.0")
        migrator.execute(source_prompt, script)

        assert len(migrator.history) == 1
        record = migrator.history[0]
        assert record.status == MigrationStatus.COMPLETED
        assert record.checksum_before is not None
        assert record.checksum_after is not None


class TestMigrationStep:
    """Test individual migration step execution."""

    def test_remove_step(self) -> None:
        """Should remove a field from data."""
        step = MigrationStep(order=1, action="remove", description="Remove field", field_path="variables.old_var")
        data = {"variables": {"old_var": "value", "keep": "yes"}}
        result = step.execute(data)
        assert "old_var" not in result["variables"]
        assert result["variables"]["keep"] == "yes"

    def test_add_step(self) -> None:
        """Should add a new field to data."""
        step = MigrationStep(order=1, action="add", description="Add field", field_path="variables.new_var", new_value="default")
        data = {"variables": {"existing": "value"}}
        result = step.execute(data)
        assert result["variables"]["new_var"] == "default"
