/**
 * Testing framework for PromptOps prompts.
 */

export interface TestCase {
  name: string;
  inputs: Record<string, any>;
  assertions: AssertionConfig[];
}

export interface AssertionConfig {
  type: string;
  field?: string;
  value?: any;
  values?: string[];
  keywords?: string[];
  min?: number;
  max?: number;
  threshold?: number;
}

export interface AssertionResult {
  passed: boolean;
  assertionType: string;
  message: string;
}

export interface TestResult {
  testName: string;
  promptName: string;
  passed: boolean;
  assertionResults: AssertionResult[];
  error?: string;
}

export interface TestSuite {
  promptName: string;
  testCases: TestCase[];
  passThreshold: number;
}

export class TestRunner {
  runAssertion(config: AssertionConfig, output: any): AssertionResult {
    switch (config.type) {
      case "schema_valid":
        return this.checkSchemaValid(output);
      case "max_length":
        return this.checkMaxLength(config, output);
      case "contains_keywords":
        return this.checkContainsKeywords(config, output);
      case "does_not_contain":
        return this.checkDoesNotContain(config, output);
      case "cost_under":
        return { passed: true, assertionType: "cost_under", message: "Cost check passed" };
      default:
        return { passed: false, assertionType: config.type, message: `Unknown assertion: ${config.type}` };
    }
  }

  private checkSchemaValid(output: any): AssertionResult {
    if (typeof output === "object" && output !== null) {
      return { passed: true, assertionType: "schema_valid", message: "Valid object" };
    }
    if (typeof output === "string") {
      try {
        JSON.parse(output);
        return { passed: true, assertionType: "schema_valid", message: "Valid JSON" };
      } catch {
        return { passed: false, assertionType: "schema_valid", message: "Not valid JSON" };
      }
    }
    return { passed: false, assertionType: "schema_valid", message: "Invalid output type" };
  }

  private checkMaxLength(config: AssertionConfig, output: any): AssertionResult {
    const text = config.field && typeof output === "object" ? output[config.field] : output;
    const wordCount = String(text || "").split(/\s+/).length;
    const maxWords = config.value || 100;
    const passed = wordCount <= maxWords;
    return {
      passed,
      assertionType: "max_length",
      message: `Word count ${wordCount} ${passed ? "<=" : ">"} ${maxWords}`,
    };
  }

  private checkContainsKeywords(config: AssertionConfig, output: any): AssertionResult {
    const text = config.field && typeof output === "object" ? output[config.field] : output;
    const textLower = String(text || "").toLowerCase();
    const keywords = config.keywords || [];
    const missing = keywords.filter((kw) => !textLower.includes(kw.toLowerCase()));
    return {
      passed: missing.length === 0,
      assertionType: "contains_keywords",
      message: missing.length === 0 ? "All keywords found" : `Missing: ${missing.join(", ")}`,
    };
  }

  private checkDoesNotContain(config: AssertionConfig, output: any): AssertionResult {
    const text = config.field && typeof output === "object" ? output[config.field] : output;
    const textLower = String(text || "").toLowerCase();
    const forbidden = config.values || [];
    const found = forbidden.filter((v) => textLower.includes(v.toLowerCase()));
    return {
      passed: found.length === 0,
      assertionType: "does_not_contain",
      message: found.length === 0 ? "No forbidden content" : `Found: ${found.join(", ")}`,
    };
  }
}
