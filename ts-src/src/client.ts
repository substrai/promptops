/**
 * PromptOps client for runtime prompt invocation.
 */

import { PromptDefinition, Prompt } from "./prompt";
import { PromptVersion, VersionRange } from "./version";

export interface InvocationResult {
  output: any;
  promptName: string;
  version: string;
  model: string;
  latencyMs: number;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  cost: number;
  environment: string;
  success: boolean;
  errors: string[];
}

export class PromptClient {
  private env: string;
  private registry: Map<string, Map<string, PromptDefinition>>;

  constructor(options: { env?: string } = {}) {
    this.env = options.env || "dev";
    this.registry = new Map();
  }

  register(definition: PromptDefinition): void {
    if (!this.registry.has(definition.name)) {
      this.registry.set(definition.name, new Map());
    }
    this.registry.get(definition.name)!.set(definition.version.toString(), definition);
  }

  invoke(
    promptName: string,
    inputs: Record<string, any>,
    options: { version?: string; model?: string } = {}
  ): InvocationResult {
    const start = Date.now();
    const version = options.version || "latest";

    try {
      const definitions = this.registry.get(promptName);
      if (!definitions) {
        throw new Error(`Prompt '${promptName}' not found`);
      }

      const range = new VersionRange(version);
      let matched: PromptDefinition | undefined;
      let matchedVersion: PromptVersion | undefined;

      for (const [verStr, def] of definitions) {
        const ver = PromptVersion.parse(verStr);
        if (range.matches(ver)) {
          if (!matchedVersion || ver.compareTo(matchedVersion) > 0) {
            matched = def;
            matchedVersion = ver;
          }
        }
      }

      if (!matched || !matchedVersion) {
        throw new Error(`No version matching '${version}' for prompt '${promptName}'`);
      }

      const rendered = matched.render(inputs);
      const inputTokens = Math.ceil(rendered.length / 4);
      const outputTokens = Math.floor(matched.defaultModel.maxTokens / 2);
      const cost = matched.estimateCost(inputs);
      const latencyMs = Date.now() - start;

      return {
        output: rendered,
        promptName,
        version: matchedVersion.toString(),
        model: matched.defaultModel.provider,
        latencyMs,
        inputTokens,
        outputTokens,
        totalTokens: inputTokens + outputTokens,
        cost,
        environment: this.env,
        success: true,
        errors: [],
      };
    } catch (e: any) {
      return {
        output: null,
        promptName,
        version: "0.0.0",
        model: "unknown",
        latencyMs: Date.now() - start,
        inputTokens: 0,
        outputTokens: 0,
        totalTokens: 0,
        cost: 0,
        environment: this.env,
        success: false,
        errors: [e.message],
      };
    }
  }

  get(promptName: string, version: string = "latest"): Prompt {
    const definitions = this.registry.get(promptName);
    if (!definitions) {
      throw new Error(`Prompt '${promptName}' not found`);
    }

    const range = new VersionRange(version);
    let matched: PromptDefinition | undefined;

    for (const [verStr, def] of definitions) {
      const ver = PromptVersion.parse(verStr);
      if (range.matches(ver)) {
        matched = def;
      }
    }

    if (!matched) {
      throw new Error(`No version matching '${version}' for prompt '${promptName}'`);
    }

    return new Prompt(matched);
  }

  listPrompts(): string[] {
    return Array.from(this.registry.keys());
  }
}
