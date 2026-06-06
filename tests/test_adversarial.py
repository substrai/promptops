"""Tests for adversarial prompt test generator."""

import pytest

from promptops.testing.adversarial import (
    AdversarialGenerator,
    AdversarialPrompt,
    AttackCategory,
    TestSuite,
)


@pytest.fixture
def generator():
    """Create a generator with fixed seed for reproducibility."""
    return AdversarialGenerator(seed=42)


@pytest.fixture
def injection_generator():
    """Create a generator focused on injection attacks."""
    return AdversarialGenerator(categories=[AttackCategory.INJECTION], seed=42)


class TestAdversarialPrompt:
    def test_prompt_structure(self):
        prompt = AdversarialPrompt(
            content="test content",
            category=AttackCategory.INJECTION,
            technique="basic",
            description="Test prompt",
            severity="high",
        )
        assert prompt.content == "test content"
        assert prompt.category == AttackCategory.INJECTION
        assert prompt.severity == "high"

    def test_default_severity(self):
        prompt = AdversarialPrompt(
            content="test",
            category=AttackCategory.JAILBREAK,
            technique="test",
            description="test",
        )
        assert prompt.severity == "medium"


class TestTestSuite:
    def test_add_prompt(self):
        suite = TestSuite(name="test")
        prompt = AdversarialPrompt(
            content="x",
            category=AttackCategory.INJECTION,
            technique="t",
            description="d",
        )
        suite.add(prompt)
        assert len(suite) == 1
        assert AttackCategory.INJECTION in suite.categories

    def test_filter_by_category(self):
        suite = TestSuite(name="test")
        suite.add(AdversarialPrompt("a", AttackCategory.INJECTION, "t", "d"))
        suite.add(AdversarialPrompt("b", AttackCategory.JAILBREAK, "t", "d"))
        suite.add(AdversarialPrompt("c", AttackCategory.INJECTION, "t", "d"))

        injections = suite.filter_by_category(AttackCategory.INJECTION)
        assert len(injections) == 2

    def test_filter_by_severity(self):
        suite = TestSuite(name="test")
        suite.add(AdversarialPrompt("a", AttackCategory.INJECTION, "t", "d", severity="high"))
        suite.add(AdversarialPrompt("b", AttackCategory.JAILBREAK, "t", "d", severity="low"))
        suite.add(AdversarialPrompt("c", AttackCategory.INJECTION, "t", "d", severity="high"))

        high_severity = suite.filter_by_severity("high")
        assert len(high_severity) == 2


class TestInjectionGeneration:
    def test_generates_injection_prompts(self, generator):
        prompts = generator.generate_injection(count=5)
        assert len(prompts) == 5
        assert all(p.category == AttackCategory.INJECTION for p in prompts)

    def test_injection_contains_payload(self, generator):
        prompts = generator.generate_injection(count=3)
        for prompt in prompts:
            assert len(prompt.content) > 0
            assert prompt.technique.startswith("injection_template")

    def test_injection_with_target(self, generator):
        prompts = generator.generate_injection(
            target_prompt="What is the weather?", count=3
        )
        # Some templates embed the target
        has_target = any("weather" in p.content.lower() for p in prompts)
        assert has_target or len(prompts) == 3


class TestJailbreakGeneration:
    def test_generates_jailbreak_prompts(self, generator):
        prompts = generator.generate_jailbreak(count=5)
        assert len(prompts) == 5
        assert all(p.category == AttackCategory.JAILBREAK for p in prompts)

    def test_jailbreak_techniques_varied(self, generator):
        prompts = generator.generate_jailbreak(count=5)
        techniques = {p.technique for p in prompts}
        assert len(techniques) == 5  # All unique techniques

    def test_jailbreak_severity_levels(self, generator):
        prompts = generator.generate_jailbreak(count=8)
        severities = {p.severity for p in prompts}
        assert len(severities) >= 2  # At least medium and high/critical


class TestEncodingBypass:
    def test_generates_encoding_prompts(self, generator):
        prompts = generator.generate_encoding_bypass(count=5)
        assert len(prompts) == 5
        assert all(p.category == AttackCategory.ENCODING_BYPASS for p in prompts)

    def test_encoding_metadata(self, generator):
        prompts = generator.generate_encoding_bypass(
            target_prompt="test payload", count=3
        )
        for prompt in prompts:
            assert "encoding" in prompt.metadata
            assert "original" in prompt.metadata

    def test_base64_encoding(self, generator):
        prompts = generator.generate_encoding_bypass(count=1)
        assert prompts[0].technique == "base64"
        assert "Decode" in prompts[0].content


class TestDelimiterAbuse:
    def test_generates_delimiter_prompts(self, generator):
        prompts = generator.generate_delimiter_abuse(count=5)
        assert len(prompts) == 5
        assert all(p.category == AttackCategory.DELIMITER_ABUSE for p in prompts)

    def test_delimiter_techniques(self, generator):
        prompts = generator.generate_delimiter_abuse(count=5)
        techniques = [p.technique for p in prompts]
        assert "markdown_headers" in techniques
        assert "xml_tags" in techniques


class TestFullSuiteGeneration:
    def test_generate_full_suite(self, generator):
        suite = generator.generate_suite(count_per_category=3)
        assert len(suite) == 3 * len(AttackCategory)
        assert len(suite.categories) == len(AttackCategory)

    def test_suite_with_specific_categories(self):
        gen = AdversarialGenerator(
            categories=[AttackCategory.INJECTION, AttackCategory.JAILBREAK]
        )
        suite = gen.generate_suite(count_per_category=3)
        assert len(suite) == 6
        assert suite.categories == {AttackCategory.INJECTION, AttackCategory.JAILBREAK}

    def test_custom_payloads(self):
        custom = {AttackCategory.INJECTION: ["Custom payload one", "Custom payload two"]}
        gen = AdversarialGenerator(custom_payloads=custom, seed=42)
        prompts = gen.generate_injection(count=7)
        contents = " ".join(p.content for p in prompts)
        assert "Custom payload" in contents

    def test_reproducibility_with_seed(self):
        gen1 = AdversarialGenerator(seed=123)
        gen2 = AdversarialGenerator(seed=123)
        suite1 = gen1.generate_suite(count_per_category=2)
        suite2 = gen2.generate_suite(count_per_category=2)
        assert len(suite1) == len(suite2)
        for p1, p2 in zip(suite1.prompts, suite2.prompts):
            assert p1.content == p2.content
