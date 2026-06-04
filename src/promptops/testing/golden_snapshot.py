"""Golden dataset snapshot testing with diff reporting.

Compare current LLM outputs against saved golden snapshots to detect
regressions and highlight changes with detailed diff reporting.

Features:
- Snapshot save/load for golden dataset management
- Diff reporting with added/removed/changed classification
- Configurable tolerance for fuzzy matching
- Semantic similarity option for soft comparisons
- Regression detection with severity classification
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


class DiffType(Enum):
    """Classification of differences between expected and actual output."""

    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


class Severity(Enum):
    """Severity levels for detected regressions."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class DiffEntry:
    """A single diff entry between golden and actual output.

    Attributes:
        diff_type: Type of change (added, removed, changed, unchanged).
        field: The field or key where the difference was found.
        expected: The expected (golden) value.
        actual: The actual (current) value.
        similarity: Similarity score between expected and actual (0.0 to 1.0).
        severity: Severity classification of the difference.
    """

    diff_type: DiffType
    field: str
    expected: Optional[str] = None
    actual: Optional[str] = None
    similarity: float = 0.0
    severity: Severity = Severity.MEDIUM


@dataclass
class SnapshotDiffReport:
    """Complete diff report comparing current output to golden snapshot.

    Attributes:
        test_id: Identifier for the test case.
        passed: Whether the test passed within tolerance.
        overall_similarity: Overall similarity score.
        diffs: List of individual diff entries.
        added_count: Number of added fields/lines.
        removed_count: Number of removed fields/lines.
        changed_count: Number of changed fields/lines.
        regression_detected: Whether a regression was detected.
        summary: Human-readable summary of changes.
    """

    test_id: str
    passed: bool
    overall_similarity: float
    diffs: List[DiffEntry] = field(default_factory=list)
    added_count: int = 0
    removed_count: int = 0
    changed_count: int = 0
    regression_detected: bool = False
    summary: str = ""


@dataclass
class GoldenSnapshot:
    """A golden snapshot entry for a single test case.

    Attributes:
        test_id: Unique identifier for this test case.
        input_data: The input that produced the output.
        expected_output: The golden (expected) output.
        metadata: Additional metadata (timestamp, model version, etc.).
        checksum: Hash of the expected output for integrity checks.
    """

    test_id: str
    input_data: Any
    expected_output: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute a checksum of the expected output."""
        content = json.dumps(self.expected_output, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ToleranceConfig:
    """Configuration for comparison tolerance.

    Attributes:
        exact_match: Whether to require exact string match.
        similarity_threshold: Minimum similarity score to pass (0.0 to 1.0).
        ignore_whitespace: Whether to ignore whitespace differences.
        ignore_case: Whether to ignore case differences.
        ignore_fields: Fields to exclude from comparison.
        numeric_tolerance: Tolerance for numeric comparisons.
        regression_threshold: Score drop that triggers regression alert.
    """

    exact_match: bool = False
    similarity_threshold: float = 0.8
    ignore_whitespace: bool = True
    ignore_case: bool = False
    ignore_fields: List[str] = field(default_factory=list)
    numeric_tolerance: float = 0.01
    regression_threshold: float = 0.1


class GoldenSnapshotStore:
    """Manages golden snapshot storage and retrieval.

    Handles saving, loading, and versioning of golden snapshots
    as JSON files on disk.

    Args:
        snapshot_dir: Directory to store snapshot files.
        auto_create: Whether to create the directory if it doesn't exist.
    """

    def __init__(
        self,
        snapshot_dir: Union[str, Path] = ".golden_snapshots",
        auto_create: bool = True,
    ):
        self.snapshot_dir = Path(snapshot_dir)
        if auto_create:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(self, snapshot: GoldenSnapshot) -> Path:
        """Save a golden snapshot to disk.

        Args:
            snapshot: The snapshot to save.

        Returns:
            Path to the saved snapshot file.
        """
        filepath = self.snapshot_dir / f"{snapshot.test_id}.json"
        data = {
            "test_id": snapshot.test_id,
            "input_data": snapshot.input_data,
            "expected_output": snapshot.expected_output,
            "metadata": snapshot.metadata,
            "checksum": snapshot.checksum,
        }
        filepath.write_text(json.dumps(data, indent=2, default=str))
        return filepath

    def load_snapshot(self, test_id: str) -> Optional[GoldenSnapshot]:
        """Load a golden snapshot from disk.

        Args:
            test_id: The test case identifier.

        Returns:
            GoldenSnapshot if found, None otherwise.
        """
        filepath = self.snapshot_dir / f"{test_id}.json"
        if not filepath.exists():
            return None

        data = json.loads(filepath.read_text())
        return GoldenSnapshot(
            test_id=data["test_id"],
            input_data=data["input_data"],
            expected_output=data["expected_output"],
            metadata=data.get("metadata", {}),
            checksum=data.get("checksum", ""),
        )

    def list_snapshots(self) -> List[str]:
        """List all available snapshot test IDs.

        Returns:
            List of test ID strings.
        """
        if not self.snapshot_dir.exists():
            return []
        return [
            f.stem for f in self.snapshot_dir.glob("*.json")
        ]

    def delete_snapshot(self, test_id: str) -> bool:
        """Delete a snapshot by test ID.

        Args:
            test_id: The test case identifier.

        Returns:
            True if deleted, False if not found.
        """
        filepath = self.snapshot_dir / f"{test_id}.json"
        if filepath.exists():
            filepath.unlink()
            return True
        return False


class GoldenSnapshotTester:
    """Main testing engine for golden snapshot comparisons.

    Compares current outputs against golden snapshots with configurable
    tolerance and produces detailed diff reports.

    Args:
        store: Snapshot storage backend.
        tolerance: Tolerance configuration for comparisons.
        custom_comparator: Optional custom comparison function.
    """

    def __init__(
        self,
        store: Optional[GoldenSnapshotStore] = None,
        tolerance: Optional[ToleranceConfig] = None,
        custom_comparator: Optional[Callable[[Any, Any], float]] = None,
    ):
        self.store = store or GoldenSnapshotStore()
        self.tolerance = tolerance or ToleranceConfig()
        self.custom_comparator = custom_comparator

    def compare(
        self,
        test_id: str,
        actual_output: Any,
        expected_output: Optional[Any] = None,
    ) -> SnapshotDiffReport:
        """Compare actual output against golden snapshot.

        Args:
            test_id: Test case identifier.
            actual_output: Current output to compare.
            expected_output: Override golden snapshot (if not loading from store).

        Returns:
            SnapshotDiffReport with detailed diff analysis.
        """
        # Load golden if not provided
        if expected_output is None:
            snapshot = self.store.load_snapshot(test_id)
            if snapshot is None:
                return SnapshotDiffReport(
                    test_id=test_id,
                    passed=False,
                    overall_similarity=0.0,
                    summary=f"No golden snapshot found for '{test_id}'",
                )
            expected_output = snapshot.expected_output

        # Perform comparison based on type
        if isinstance(expected_output, dict) and isinstance(actual_output, dict):
            return self._compare_dicts(test_id, expected_output, actual_output)
        elif isinstance(expected_output, str) and isinstance(actual_output, str):
            return self._compare_strings(test_id, expected_output, actual_output)
        elif isinstance(expected_output, list) and isinstance(actual_output, list):
            return self._compare_lists(test_id, expected_output, actual_output)
        else:
            return self._compare_generic(test_id, expected_output, actual_output)

    def _normalize(self, text: str) -> str:
        """Normalize text based on tolerance settings."""
        if self.tolerance.ignore_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        if self.tolerance.ignore_case:
            text = text.lower()
        return text

    def _text_similarity(self, a: str, b: str) -> float:
        """Compute text similarity using SequenceMatcher."""
        if self.custom_comparator:
            return self.custom_comparator(a, b)
        a_norm = self._normalize(a)
        b_norm = self._normalize(b)
        if a_norm == b_norm:
            return 1.0
        return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()

    def _compare_strings(
        self, test_id: str, expected: str, actual: str
    ) -> SnapshotDiffReport:
        """Compare two string outputs."""
        similarity = self._text_similarity(expected, actual)
        passed = similarity >= self.tolerance.similarity_threshold

        diffs: List[DiffEntry] = []
        if similarity < 1.0:
            # Generate line-by-line diff
            expected_lines = expected.splitlines()
            actual_lines = actual.splitlines()
            differ = difflib.unified_diff(
                expected_lines, actual_lines, lineterm=""
            )

            added = 0
            removed = 0
            changed = 0

            for line in differ:
                if line.startswith("+") and not line.startswith("+++"):
                    diffs.append(DiffEntry(
                        diff_type=DiffType.ADDED,
                        field="line",
                        actual=line[1:],
                        similarity=0.0,
                    ))
                    added += 1
                elif line.startswith("-") and not line.startswith("---"):
                    diffs.append(DiffEntry(
                        diff_type=DiffType.REMOVED,
                        field="line",
                        expected=line[1:],
                        similarity=0.0,
                    ))
                    removed += 1

            return SnapshotDiffReport(
                test_id=test_id,
                passed=passed,
                overall_similarity=similarity,
                diffs=diffs,
                added_count=added,
                removed_count=removed,
                changed_count=changed,
                regression_detected=not passed,
                summary=self._generate_summary(similarity, added, removed, changed),
            )

        return SnapshotDiffReport(
            test_id=test_id,
            passed=True,
            overall_similarity=1.0,
            summary="Output matches golden snapshot exactly.",
        )

    def _compare_dicts(
        self, test_id: str, expected: Dict, actual: Dict
    ) -> SnapshotDiffReport:
        """Compare two dictionary outputs field by field."""
        diffs: List[DiffEntry] = []
        added = 0
        removed = 0
        changed = 0
        total_fields = 0
        matching_fields = 0

        all_keys = set(expected.keys()) | set(actual.keys())

        for key in all_keys:
            if key in self.tolerance.ignore_fields:
                continue

            total_fields += 1

            if key in expected and key not in actual:
                diffs.append(DiffEntry(
                    diff_type=DiffType.REMOVED,
                    field=key,
                    expected=str(expected[key]),
                    severity=Severity.HIGH,
                ))
                removed += 1
            elif key not in expected and key in actual:
                diffs.append(DiffEntry(
                    diff_type=DiffType.ADDED,
                    field=key,
                    actual=str(actual[key]),
                    severity=Severity.LOW,
                ))
                added += 1
            else:
                exp_val = str(expected[key])
                act_val = str(actual[key])
                sim = self._text_similarity(exp_val, act_val)

                if sim < 1.0:
                    severity = (
                        Severity.CRITICAL if sim < 0.3
                        else Severity.HIGH if sim < 0.6
                        else Severity.MEDIUM
                    )
                    diffs.append(DiffEntry(
                        diff_type=DiffType.CHANGED,
                        field=key,
                        expected=exp_val,
                        actual=act_val,
                        similarity=sim,
                        severity=severity,
                    ))
                    changed += 1
                else:
                    matching_fields += 1

        overall_sim = matching_fields / total_fields if total_fields > 0 else 1.0
        passed = overall_sim >= self.tolerance.similarity_threshold

        return SnapshotDiffReport(
            test_id=test_id,
            passed=passed,
            overall_similarity=overall_sim,
            diffs=diffs,
            added_count=added,
            removed_count=removed,
            changed_count=changed,
            regression_detected=not passed,
            summary=self._generate_summary(overall_sim, added, removed, changed),
        )

    def _compare_lists(
        self, test_id: str, expected: List, actual: List
    ) -> SnapshotDiffReport:
        """Compare two list outputs."""
        diffs: List[DiffEntry] = []
        added = max(0, len(actual) - len(expected))
        removed = max(0, len(expected) - len(actual))
        changed = 0
        matching = 0

        for i in range(min(len(expected), len(actual))):
            exp_str = str(expected[i])
            act_str = str(actual[i])
            sim = self._text_similarity(exp_str, act_str)
            if sim < 1.0:
                diffs.append(DiffEntry(
                    diff_type=DiffType.CHANGED,
                    field=f"[{i}]",
                    expected=exp_str,
                    actual=act_str,
                    similarity=sim,
                ))
                changed += 1
            else:
                matching += 1

        total = max(len(expected), len(actual))
        overall_sim = matching / total if total > 0 else 1.0
        passed = overall_sim >= self.tolerance.similarity_threshold

        return SnapshotDiffReport(
            test_id=test_id,
            passed=passed,
            overall_similarity=overall_sim,
            diffs=diffs,
            added_count=added,
            removed_count=removed,
            changed_count=changed,
            regression_detected=not passed,
            summary=self._generate_summary(overall_sim, added, removed, changed),
        )

    def _compare_generic(
        self, test_id: str, expected: Any, actual: Any
    ) -> SnapshotDiffReport:
        """Compare two generic outputs."""
        exp_str = str(expected)
        act_str = str(actual)
        similarity = self._text_similarity(exp_str, act_str)
        passed = similarity >= self.tolerance.similarity_threshold

        diffs = []
        if similarity < 1.0:
            diffs.append(DiffEntry(
                diff_type=DiffType.CHANGED,
                field="value",
                expected=exp_str,
                actual=act_str,
                similarity=similarity,
            ))

        return SnapshotDiffReport(
            test_id=test_id,
            passed=passed,
            overall_similarity=similarity,
            diffs=diffs,
            changed_count=1 if similarity < 1.0 else 0,
            regression_detected=not passed,
            summary=f"Similarity: {similarity:.2%}",
        )

    def _generate_summary(
        self, similarity: float, added: int, removed: int, changed: int
    ) -> str:
        """Generate a human-readable summary of the diff."""
        parts = []
        if added:
            parts.append(f"{added} added")
        if removed:
            parts.append(f"{removed} removed")
        if changed:
            parts.append(f"{changed} changed")

        changes = ", ".join(parts) if parts else "no changes"
        return f"Similarity: {similarity:.2%} | Changes: {changes}"

    def update_golden(
        self,
        test_id: str,
        input_data: Any,
        output: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GoldenSnapshot:
        """Update or create a golden snapshot.

        Args:
            test_id: Test case identifier.
            input_data: The input that produced the output.
            output: The new golden output.
            metadata: Optional metadata to store.

        Returns:
            The saved GoldenSnapshot.
        """
        snapshot = GoldenSnapshot(
            test_id=test_id,
            input_data=input_data,
            expected_output=output,
            metadata=metadata or {},
        )
        self.store.save_snapshot(snapshot)
        return snapshot

    def run_suite(
        self,
        test_cases: List[Dict[str, Any]],
    ) -> List[SnapshotDiffReport]:
        """Run a suite of snapshot tests.

        Args:
            test_cases: List of dicts with 'test_id' and 'actual_output' keys.

        Returns:
            List of SnapshotDiffReport for each test case.
        """
        reports = []
        for case in test_cases:
            test_id = case["test_id"]
            actual = case["actual_output"]
            report = self.compare(test_id, actual)
            reports.append(report)
        return reports
