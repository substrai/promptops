"""Testing framework for PromptOps prompts.

Includes:
- Test runner with golden datasets
- Built-in evaluators (semantic similarity, schema compliance, injection detection)
- Breaking change detection
- Custom evaluator plugin interface
"""

from promptops.testing.runner import TestRunner, TestResult, TestSuite, SuiteResult
from promptops.testing.assertions import Assertion, AssertionResult
from promptops.testing.evaluators import (
    BaseEvaluator,
    SemanticSimilarityEvaluator,
    SchemaComplianceEvaluator,
    FormatComplianceEvaluator,
    InjectionDetectionEvaluator,
    CostEvaluator,
    EvaluationScore,
    get_evaluator,
    register_evaluator,
)
from promptops.testing.golden import GoldenDataset, GoldenCase
from promptops.testing.breaking_changes import (
    BreakingChangeDetector,
    BreakingChangeReport,
    ChangeType,
    SchemaChange,
)

__all__ = [
    "TestRunner",
    "TestResult",
    "TestSuite",
    "SuiteResult",
    "Assertion",
    "AssertionResult",
    "BaseEvaluator",
    "SemanticSimilarityEvaluator",
    "SchemaComplianceEvaluator",
    "FormatComplianceEvaluator",
    "InjectionDetectionEvaluator",
    "CostEvaluator",
    "EvaluationScore",
    "get_evaluator",
    "register_evaluator",
    "GoldenDataset",
    "GoldenCase",
    "BreakingChangeDetector",
    "BreakingChangeReport",
    "ChangeType",
    "SchemaChange",
]
