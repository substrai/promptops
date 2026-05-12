/**
 * Semantic versioning engine for prompts.
 */

export class PromptVersion {
  readonly major: number;
  readonly minor: number;
  readonly patch: number;
  readonly prerelease?: string;

  private static PATTERN = /^(?<major>0|[1-9]\d*)\.(?<minor>0|[1-9]\d*)\.(?<patch>0|[1-9]\d*)(?:-(?<prerelease>[0-9A-Za-z\-.]+))?$/;

  constructor(major: number, minor: number, patch: number, prerelease?: string) {
    this.major = major;
    this.minor = minor;
    this.patch = patch;
    this.prerelease = prerelease;
  }

  static parse(versionStr: string): PromptVersion {
    const match = versionStr.trim().match(PromptVersion.PATTERN);
    if (!match || !match.groups) {
      throw new Error(`Invalid version string: '${versionStr}'`);
    }
    return new PromptVersion(
      parseInt(match.groups.major),
      parseInt(match.groups.minor),
      parseInt(match.groups.patch),
      match.groups.prerelease || undefined
    );
  }

  bumpPatch(): PromptVersion {
    return new PromptVersion(this.major, this.minor, this.patch + 1);
  }

  bumpMinor(): PromptVersion {
    return new PromptVersion(this.major, this.minor + 1, 0);
  }

  bumpMajor(): PromptVersion {
    return new PromptVersion(this.major + 1, 0, 0);
  }

  isPrerelease(): boolean {
    return this.prerelease !== undefined;
  }

  isCompatibleWith(other: PromptVersion): boolean {
    return this.major === other.major;
  }

  compareTo(other: PromptVersion): number {
    if (this.major !== other.major) return this.major - other.major;
    if (this.minor !== other.minor) return this.minor - other.minor;
    return this.patch - other.patch;
  }

  toString(): string {
    const base = `${this.major}.${this.minor}.${this.patch}`;
    return this.prerelease ? `${base}-${this.prerelease}` : base;
  }
}

export class VersionRange {
  readonly spec: string;

  constructor(spec: string) {
    this.spec = spec;
  }

  static latest(): VersionRange {
    return new VersionRange("latest");
  }

  static exact(version: string): VersionRange {
    PromptVersion.parse(version);
    return new VersionRange(version);
  }

  static compatible(version: string): VersionRange {
    return new VersionRange(`~${version}`);
  }

  matches(version: PromptVersion): boolean {
    if (this.spec === "latest") {
      return !version.isPrerelease();
    }

    if (this.spec.startsWith("~")) {
      const base = this.spec.slice(1);
      const parts = base.split(".");
      const targetMajor = parseInt(parts[0]);
      const targetMinor = parts.length > 1 ? parseInt(parts[1]) : 0;
      return (
        version.major === targetMajor &&
        version.minor >= targetMinor &&
        !version.isPrerelease()
      );
    }

    try {
      const exact = PromptVersion.parse(this.spec);
      return (
        version.major === exact.major &&
        version.minor === exact.minor &&
        version.patch === exact.patch
      );
    } catch {
      return false;
    }
  }
}
