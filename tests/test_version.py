"""Tests for the versioning module."""

import pytest
from promptops.core.version import PromptVersion, VersionRange


def test_parse_version():
    v = PromptVersion.parse("1.2.3")
    assert v.major == 1
    assert v.minor == 2
    assert v.patch == 3
    assert v.prerelease is None


def test_parse_prerelease():
    v = PromptVersion.parse("2.0.0-rc1")
    assert v.major == 2
    assert v.prerelease == "rc1"
    assert v.is_prerelease()


def test_version_comparison():
    v1 = PromptVersion.parse("1.0.0")
    v2 = PromptVersion.parse("1.1.0")
    v3 = PromptVersion.parse("2.0.0")
    assert v1 < v2 < v3


def test_bump_patch():
    v = PromptVersion.parse("1.2.3")
    bumped = v.bump_patch()
    assert str(bumped) == "1.2.4"


def test_bump_minor():
    v = PromptVersion.parse("1.2.3")
    bumped = v.bump_minor()
    assert str(bumped) == "1.3.0"


def test_bump_major():
    v = PromptVersion.parse("1.2.3")
    bumped = v.bump_major()
    assert str(bumped) == "2.0.0"


def test_version_range_latest():
    vr = VersionRange.latest()
    assert vr.matches(PromptVersion.parse("1.0.0"))
    assert not vr.matches(PromptVersion.parse("1.0.0-rc1"))


def test_version_range_compatible():
    vr = VersionRange.compatible("1.2")
    assert vr.matches(PromptVersion.parse("1.2.0"))
    assert vr.matches(PromptVersion.parse("1.3.0"))
    assert not vr.matches(PromptVersion.parse("2.0.0"))


def test_version_range_exact():
    vr = VersionRange.exact("1.2.3")
    assert vr.matches(PromptVersion.parse("1.2.3"))
    assert not vr.matches(PromptVersion.parse("1.2.4"))


def test_invalid_version():
    with pytest.raises(ValueError):
        PromptVersion.parse("not-a-version")
