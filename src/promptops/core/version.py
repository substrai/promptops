"""Semantic versioning engine for prompts.

Prompts follow semantic versioning:
- PATCH: wording changes (no schema impact)
- MINOR: new optional variables added
- MAJOR: breaking schema changes (input/output schema modified)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, order=True)
class PromptVersion:
    """Represents a semantic version for a prompt.

    Examples:
        >>> v = PromptVersion(1, 2, 3)
        >>> str(v)
        '1.2.3'
        >>> v = PromptVersion.parse("2.0.0-rc1")
        >>> v.major
        2
    """

    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None

    _PATTERN = re.compile(
        r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
        r"(?:-(?P<prerelease>[0-9A-Za-z\-.]+))?$"
    )

    @classmethod
    def parse(cls, version_str: str) -> "PromptVersion":
        """Parse a version string into a PromptVersion.

        Args:
            version_str: A semver string like "1.2.3" or "2.0.0-rc1"

        Returns:
            PromptVersion instance

        Raises:
            ValueError: If the string is not a valid semver
        """
        match = cls._PATTERN.match(version_str.strip())
        if not match:
            raise ValueError(f"Invalid version string: '{version_str}'")
        return cls(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
            prerelease=match.group("prerelease"),
        )

    def bump_patch(self) -> "PromptVersion":
        """Bump patch version (wording changes)."""
        return PromptVersion(self.major, self.minor, self.patch + 1)

    def bump_minor(self) -> "PromptVersion":
        """Bump minor version (new optional variables)."""
        return PromptVersion(self.major, self.minor + 1, 0)

    def bump_major(self) -> "PromptVersion":
        """Bump major version (breaking schema changes)."""
        return PromptVersion(self.major + 1, 0, 0)

    def is_compatible_with(self, other: "PromptVersion") -> bool:
        """Check if this version is compatible with another (same major)."""
        return self.major == other.major

    def is_prerelease(self) -> bool:
        """Check if this is a pre-release version."""
        return self.prerelease is not None

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            return f"{base}-{self.prerelease}"
        return base

    def __repr__(self) -> str:
        return f"PromptVersion('{self}')"


@dataclass
class VersionRange:
    """Represents a version range for prompt resolution.

    Supports:
        - Exact: "1.2.3"
        - Compatible: "~1.2" (>=1.2.0, <2.0.0)
        - Latest: "latest"
        - Range: ">=1.0.0,<2.0.0"
    """

    spec: str

    _RANGE_PATTERN = re.compile(
        r"^(?P<op>>=|<=|>|<|=)?(?P<version>\d+\.\d+\.\d+(?:-[0-9A-Za-z\-.]+)?)$"
    )

    @classmethod
    def latest(cls) -> "VersionRange":
        """Create a 'latest' version range."""
        return cls(spec="latest")

    @classmethod
    def exact(cls, version: str) -> "VersionRange":
        """Create an exact version range."""
        PromptVersion.parse(version)  # validate
        return cls(spec=version)

    @classmethod
    def compatible(cls, version: str) -> "VersionRange":
        """Create a compatible version range (~major.minor)."""
        return cls(spec=f"~{version}")

    def matches(self, version: PromptVersion) -> bool:
        """Check if a version matches this range.

        Args:
            version: The version to check

        Returns:
            True if the version matches the range
        """
        if self.spec == "latest":
            return not version.is_prerelease()

        if self.spec.startswith("~"):
            # Compatible range: same major version
            base = self.spec[1:]
            parts = base.split(".")
            target_major = int(parts[0])
            target_minor = int(parts[1]) if len(parts) > 1 else 0
            return (
                version.major == target_major
                and version.minor >= target_minor
                and not version.is_prerelease()
            )

        # Check comma-separated constraints
        constraints = self.spec.split(",")
        for constraint in constraints:
            constraint = constraint.strip()
            match = self._RANGE_PATTERN.match(constraint)
            if not match:
                # Try exact match
                try:
                    exact = PromptVersion.parse(constraint)
                    if version != exact:
                        return False
                except ValueError:
                    return False
                continue

            op = match.group("op") or "="
            target = PromptVersion.parse(match.group("version"))

            if op == "=" and version != target:
                return False
            elif op == ">=" and version < target:
                return False
            elif op == "<=" and version > target:
                return False
            elif op == ">" and version <= target:
                return False
            elif op == "<" and version >= target:
                return False

        return True

    def __str__(self) -> str:
        return self.spec
