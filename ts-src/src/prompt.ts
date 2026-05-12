/**
 * Prompt definition parser and model.
 */

import { PromptVersion } from "./version";
import { InputSchema, OutputSchema } from "./schema";

export interface ModelConfig {
  provider: string;
  template?: string;
  temperature: number;
  maxTokens: number;
  topP: number;
  stopSequences: string[];
}

export interface PromptDefinitionData {
  name: string;
  version: string;
  description?: string;
  template: string;
  model?: {
    default?: string;
    fallback?: string;
    variants?: Record<string, any>;
  };
  input?: { schema?: Record<string, any> };
  output?: { schema?: Record<string, any> };
  settings?: Record<string, any>;
  metadata?: Record<string, any>;
}

export class PromptDefinition {
  readonly name: string;
  readonly version: PromptVersion;
  readonly description: string;
  readonly template: string;
  readonly inputSchema: InputSchema;
  readonly outputSchema: OutputSchema;
  readonly defaultModel: ModelConfig;
  readonly metadata: Record<string, any>;

  constructor(data: PromptDefinitionData) {
    this.name = data.name;
    this.version = PromptVersion.parse(data.version || "0.1.0");
    this.description = data.description || "";
    this.template = data.template || "";
    this.inputSchema = InputSchema.fromDict(data.input?.schema || {});
    this.outputSchema = OutputSchema.fromDict(data.output?.schema || {});
    this.defaultModel = {
      provider: data.model?.default || "bedrock/claude-3-haiku",
      temperature: data.settings?.temperature || 0.7,
      maxTokens: data.settings?.max_tokens || 2000,
      topP: data.settings?.top_p || 1.0,
      stopSequences: data.settings?.stop_sequences || [],
    };
    this.metadata = data.metadata || {};
  }

  render(inputs: Record<string, any>): string {
    const withDefaults = this.inputSchema.applyDefaults(inputs);
    const errors = this.inputSchema.validate(withDefaults);
    if (errors.length > 0) {
      throw new Error(`Input validation failed: ${errors.join("; ")}`);
    }
    let rendered = this.template;
    for (const [key, value] of Object.entries(withDefaults)) {
      rendered = rendered.replace(new RegExp(`\\{${key}\\}`, "g"), String(value));
    }
    return rendered;
  }

  estimateTokens(inputs: Record<string, any>): number {
    const rendered = this.render(inputs);
    return Math.ceil(rendered.length / 4);
  }

  estimateCost(inputs: Record<string, any>): number {
    const pricing: Record<string, { input: number; output: number }> = {
      "bedrock/claude-3-haiku": { input: 0.00025, output: 0.00125 },
      "bedrock/claude-3-sonnet": { input: 0.003, output: 0.015 },
      "bedrock/amazon-titan-text-lite": { input: 0.00015, output: 0.00015 },
      "openai/gpt-4o-mini": { input: 0.00015, output: 0.0006 },
    };
    const modelPricing = pricing[this.defaultModel.provider] || { input: 0.001, output: 0.002 };
    const inputTokens = this.estimateTokens(inputs);
    const outputTokens = Math.floor(this.defaultModel.maxTokens / 2);
    return (inputTokens / 1000) * modelPricing.input + (outputTokens / 1000) * modelPricing.output;
  }
}

export class Prompt {
  readonly definition: PromptDefinition;
  readonly name: string;
  readonly version: PromptVersion;
  readonly template: string;

  constructor(definition: PromptDefinition) {
    this.definition = definition;
    this.name = definition.name;
    this.version = definition.version;
    this.template = definition.template;
  }

  render(inputs: Record<string, any>): string {
    return this.definition.render(inputs);
  }

  validateInputs(inputs: Record<string, any>): string[] {
    const withDefaults = this.definition.inputSchema.applyDefaults(inputs);
    return this.definition.inputSchema.validate(withDefaults);
  }

  estimateCost(inputs: Record<string, any>): number {
    return this.definition.estimateCost(inputs);
  }
}
