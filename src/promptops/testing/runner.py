"""Test runner for prompt regression testing.

Loads test definitions from YAML, runs prompts against test cases,
and reports results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from promptops.core.client import PromptClient
from promptops.testing.assertions import Assertion, AssertionResult


@dataclass
class TestCase:
    """A single test case for a prompt."""

    name: str
    inputs: Dict[str, Any]
    assertions: List[Assertion]


@dataclass
class TestResult:
    """Result of running a single test case."""

    test_name: str
    prompt_name: str
    passed: bool
    assertion_results: List[AssertionResult] = field(default_factory=list)
    error: Optional[str] = None
    latency_ms: float = 0.0

    @property
    def failed_assertions(self) -> List[AssertionResult]:
        return [r for r in self.assertion_results if not r.passed]


@dataclass
class TestSuite:
    """A collection of test cases for a prompt."""

    prompt_name: str
    test_cases: List[TestCase]
    pass_threshold: float = 0.95
    on_failure: str = "block_deploy"

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "TestSuite":
        """Parse a test suite from YAML content."""
        data = yaml.safe_load(yaml_content)
        if not data:
            raise ValueError("Empty test definition")

        prompt_name = data.get("prompt", "")
        pass_threshold = data.get("evaluation", {}).get("pass_threshold", 0.95)
        on_failure = data.get("evaluation", {}).get("on_failure", "block_deploy")

        test_cases = []
        for tc_data in data.get("test_cases", []):
            assertions = [
                Assertion(a_config) for a_config in tc_data.get("assertions", [])
            ]
            test_cases.append(
                TestCase(
                    name=tc_data.get("name", "unnamed"),
                    inputs=tc_data.get("inputs", {}),
                    assertions=assertions,
                )
            )

        return cls(
            prompt_name=prompt_name,
            test_cases=test_cases,
            pass_threshold=pass_threshold,
            on_failure=on_failure,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "TestSuite":
        """Load a test suite from a YAML file."""
        path = Path(path)
        content = path.read_text()
        return cls.from_yaml(content)


class TestRunner:
    """Runs prompt test suites and reports results.

    Usage:
        runner = TestRunner(prompts_dir="./prompts")
        results = runner.run_suite("./tests/summarize_tests.yaml")
        print(f"Passed: {results.pass_rate:.0%}")
    """

    def __init__(self, prompts_dir: str | Path = "./prompts", env: str = "dev"):
        """Initialize the test runner.

        Args:
            prompts_dir: Path to prompts directory
            env: Environment to test against
        """
        self.client = PromptClient(env=env, prompts_dir=prompts_dir)

    def run_suite(self, suite_path: str | Path) -> "SuiteResult":
        """Run all test cases in a test suite.

        Args:
            suite_path: Path to test suite YAML file

        Returns:
            SuiteResult with all test results
        """
        suite = TestSuite.from_file(suite_path)
        results = []

        for test_case in suite.test_cases:
            result = self._run_test_case(suite.prompt_name, test_case)
            results.append(result)

        return SuiteResult(
            prompt_name=suite.prompt_name,
            results=results,
            pass_threshold=suite.pass_threshold,
        )

    def run_all(self, tests_dir: str | Path = "./tests") -> List["SuiteResult"]:
        """Run all test suites in a directory.

        Args:
            tests_dir: Directory containing test YAML files

        Returns:
            List of SuiteResults
        """
        tests_dir = Path(tests_dir)
        suite_results = []

        for yaml_file in sorted(tests_dir.glob("*_tests.yaml")):
            try:
                result = self.run_suite(yaml_file)
                suite_results.append(result)
            except (ValueError, KeyError) as e:
                # Create a failed suite result
                suite_results.append(
                    SuiteResult(
                        prompt_name=yaml_file.stem,
                        results=[
                            TestResult(
                                test_name="suite_load",
                                prompt_name=yaml_file.stem,
                                passed=False,
                                error=str(e),
                            )
                        ],
                    )
                )

        return suite_results

    def _run_test_case(self, prompt_name: str, test_case: TestCase) -> TestResult:
        """Run a single test case."""
        try:
            # Invoke the prompt
            invocation = self.client.invoke(prompt_name, inputs=test_case.inputs)

            if not invocation.success:
                return TestResult(
                    test_name=test_case.name,
                    prompt_name=prompt_name,
                    passed=False,
                    error="; ".join(invocation.errors),
                    latency_ms=invocation.latency_ms,
                )

            # Run assertions
            assertion_results = []
            for assertion in test_case.assertions:
                result = assertion.check(invocation.output, test_case.inputs)
                assertion_results.append(result)

            all_passed = all(r.passed for r in assertion_results)

            return TestResult(
                test_name=test_case.name,
                prompt_name=prompt_name,
                passed=all_passed,
                assertion_results=assertion_results,
                latency_ms=invocation.latency_ms,
            )

        except Exception as e:
            return TestResult(
                test_name=test_case.name,
                prompt_name=prompt_name,
                passed=False,
                error=str(e),
            )


@dataclass
class SuiteResult:
    """Result of running a complete test suite."""

    prompt_name: str
    results: List[TestResult]
    pass_threshold: float = 0.95

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    @property
    def meets_threshold(self) -> bool:
        return self.pass_rate >= self.pass_threshold

    def summary(self) -> str:
        """Generate a human-readable summary."""
        status = "PASS" if self.meets_threshold else "FAIL"
        lines = [
            f"[{status}] {self.prompt_name}: {self.passed}/{self.total} tests passed ({self.pass_rate:.0%})",
            f"  Threshold: {self.pass_threshold:.0%}",
        ]
        for result in self.results:
            icon = "  ✓" if result.passed else "  ✗"
            lines.append(f"{icon} {result.test_name}")
            if not result.passed:
                if result.error:
                    lines.append(f"    Error: {result.error}")
                for ar in result.failed_assertions:
                    lines.append(f"    - {ar.assertion_type}: {ar.message}")
        return "\n".join(lines)
