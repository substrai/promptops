"""Tests for golden dataset snapshot testing with diff reporting."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from promptops.testing.golden_snapshot import (
    DiffEntry,
    DiffType,
    GoldenSnapshot,
    GoldenSnapshotStore,
    GoldenSnapshotTester,
    Severity,
    SnapshotDiffReport,
    ToleranceConfig,
)


class TestGoldenSnapshot:
    """Tests for the GoldenSnapshot dataclass."""

    def test_snapshot_creates_checksum(self):
        """Test that checksum is computed on creation."""
        snapshot = GoldenSnapshot(
            test_id="test-1",
            input_data="hello",
            expected_output="world",
        )
        assert snapshot.checksum != ""
        assert len(snapshot.checksum) == 16

    def test_snapshot_checksum_deterministic(self):
        """Test that same content produces same checksum."""
        s1 = GoldenSnapshot(test_id="a", input_data="x", expected_output="y")
        s2 = GoldenSnapshot(test_id="b", input_data="z", expected_output="y")
        assert s1.checksum == s2.checksum  # Same expected_output

    def test_snapshot_different_outputs_different_checksums(self):
        """Test that different outputs produce different checksums."""
        s1 = GoldenSnapshot(test_id="a", input_data="x", expected_output="hello")
        s2 = GoldenSnapshot(test_id="a", input_data="x", expected_output="world")
        assert s1.checksum != s2.checksum


class TestGoldenSnapshotStore:
    """Tests for snapshot storage and retrieval."""

    def test_save_and_load_snapshot(self):
        """Test saving and loading a snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoldenSnapshotStore(snapshot_dir=tmpdir)
            snapshot = GoldenSnapshot(
                test_id="test-save",
                input_data={"prompt": "hello"},
                expected_output="response text",
                metadata={"model": "gpt-4"},
            )
            store.save_snapshot(snapshot)

            loaded = store.load_snapshot("test-save")
            assert loaded is not None
            assert loaded.test_id == "test-save"
            assert loaded.expected_output == "response text"
            assert loaded.metadata["model"] == "gpt-4"

    def test_load_nonexistent_returns_none(self):
        """Test loading a nonexistent snapshot returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoldenSnapshotStore(snapshot_dir=tmpdir)
            assert store.load_snapshot("nonexistent") is None

    def test_list_snapshots(self):
        """Test listing available snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoldenSnapshotStore(snapshot_dir=tmpdir)
            for i in range(3):
                store.save_snapshot(GoldenSnapshot(
                    test_id=f"test-{i}",
                    input_data="input",
                    expected_output=f"output-{i}",
                ))
            ids = store.list_snapshots()
            assert len(ids) == 3
            assert "test-0" in ids

    def test_delete_snapshot(self):
        """Test deleting a snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoldenSnapshotStore(snapshot_dir=tmpdir)
            store.save_snapshot(GoldenSnapshot(
                test_id="to-delete",
                input_data="x",
                expected_output="y",
            ))
            assert store.delete_snapshot("to-delete") is True
            assert store.load_snapshot("to-delete") is None
            assert store.delete_snapshot("to-delete") is False


class TestGoldenSnapshotTester:
    """Tests for the main snapshot testing engine."""

    def test_exact_match_passes(self):
        """Test that identical outputs produce a passing report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoldenSnapshotStore(snapshot_dir=tmpdir)
            tester = GoldenSnapshotTester(store=store)

            report = tester.compare("t1", "hello world", expected_output="hello world")
            assert report.passed is True
            assert report.overall_similarity == 1.0

    def test_different_outputs_fails(self):
        """Test that very different outputs fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoldenSnapshotStore(snapshot_dir=tmpdir)
            tolerance = ToleranceConfig(similarity_threshold=0.9)
            tester = GoldenSnapshotTester(store=store, tolerance=tolerance)

            report = tester.compare(
                "t2",
                "completely different text about cats",
                expected_output="the sky is blue and grass is green",
            )
            assert report.passed is False
            assert report.regression_detected is True

    def test_dict_comparison_detects_changes(self):
        """Test dictionary comparison detects field changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoldenSnapshotStore(snapshot_dir=tmpdir)
            tester = GoldenSnapshotTester(store=store)

            expected = {"name": "Alice", "age": "30", "city": "NYC"}
            actual = {"name": "Alice", "age": "31", "country": "USA"}

            report = tester.compare("dict-test", actual, expected_output=expected)
            assert report.removed_count >= 1  # city removed
            assert report.added_count >= 1  # country added
            assert report.changed_count >= 1  # age changed

    def test_list_comparison(self):
        """Test list comparison detects differences."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoldenSnapshotStore(snapshot_dir=tmpdir)
            tester = GoldenSnapshotTester(store=store)

            expected = ["apple", "banana", "cherry"]
            actual = ["apple", "blueberry", "cherry", "date"]

            report = tester.compare("list-test", actual, expected_output=expected)
            assert report.added_count == 1  # date
            assert report.changed_count >= 1  # banana -> blueberry

    def test_tolerance_ignore_whitespace(self):
        """Test that whitespace differences are ignored when configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoldenSnapshotStore(snapshot_dir=tmpdir)
            tolerance = ToleranceConfig(
                ignore_whitespace=True,
                similarity_threshold=0.9,
            )
            tester = GoldenSnapshotTester(store=store, tolerance=tolerance)

            report = tester.compare(
                "ws-test",
                "hello   world\n\nfoo",
                expected_output="hello world\nfoo",
            )
            assert report.passed is True

    def test_tolerance_ignore_case(self):
        """Test that case differences are ignored when configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoldenSnapshotStore(snapshot_dir=tmpdir)
            tolerance = ToleranceConfig(
                ignore_case=True,
                similarity_threshold=0.95,
            )
            tester = GoldenSnapshotTester(store=store, tolerance=tolerance)

            report = tester.compare(
                "case-test",
                "Hello World",
                expected_output="hello world",
            )
            assert report.passed is True

    def test_update_golden_creates_snapshot(self):
        """Test updating golden snapshot creates the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoldenSnapshotStore(snapshot_dir=tmpdir)
            tester = GoldenSnapshotTester(store=store)

            snapshot = tester.update_golden(
                test_id="new-golden",
                input_data="What is AI?",
                output="AI is artificial intelligence.",
                metadata={"model": "gpt-4", "version": "1.0"},
            )
            assert snapshot.test_id == "new-golden"

            # Now compare should work
            report = tester.compare("new-golden", "AI is artificial intelligence.")
            assert report.passed is True

    def test_run_suite(self):
        """Test running a full suite of snapshot tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoldenSnapshotStore(snapshot_dir=tmpdir)
            tester = GoldenSnapshotTester(store=store)

            # Create golden snapshots
            tester.update_golden("suite-1", "input1", "output1")
            tester.update_golden("suite-2", "input2", "output2")

            # Run suite
            reports = tester.run_suite([
                {"test_id": "suite-1", "actual_output": "output1"},
                {"test_id": "suite-2", "actual_output": "different output"},
            ])

            assert len(reports) == 2
            assert reports[0].passed is True
            assert reports[1].passed is False

    def test_missing_golden_snapshot(self):
        """Test behavior when golden snapshot doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoldenSnapshotStore(snapshot_dir=tmpdir)
            tester = GoldenSnapshotTester(store=store)

            report = tester.compare("missing-test", "some output")
            assert report.passed is False
            assert "No golden snapshot found" in report.summary

    def test_custom_comparator(self):
        """Test using a custom comparison function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoldenSnapshotStore(snapshot_dir=tmpdir)

            # Custom comparator that only checks length similarity
            def length_comparator(a: str, b: str) -> float:
                max_len = max(len(a), len(b))
                if max_len == 0:
                    return 1.0
                return 1.0 - abs(len(a) - len(b)) / max_len

            tester = GoldenSnapshotTester(
                store=store,
                custom_comparator=length_comparator,
            )

            # Same length but different content should pass
            report = tester.compare(
                "custom-test",
                "abcdefgh",
                expected_output="12345678",
            )
            assert report.overall_similarity == 1.0
