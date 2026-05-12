/**
 * Schema validation for prompt inputs and outputs.
 */

export interface FieldSchema {
  name: string;
  type: string;
  required?: boolean;
  default?: any;
  minValue?: number;
  maxValue?: number;
  minLength?: number;
  maxLength?: number;
  values?: string[];
  items?: string;
}

export class InputSchema {
  readonly fields: Map<string, FieldSchema>;

  constructor(fields: Map<string, FieldSchema>) {
    this.fields = fields;
  }

  static fromDict(schemaDict: Record<string, any>): InputSchema {
    const fields = new Map<string, FieldSchema>();
    for (const [name, config] of Object.entries(schemaDict)) {
      if (typeof config === "object" && config !== null) {
        fields.set(name, {
          name,
          type: config.type || "string",
          required: config.required !== false,
          default: config.default,
          minValue: config.min,
          maxValue: config.max,
          minLength: config.min_length,
          maxLength: config.max_length,
          values: config.values,
          items: config.items,
        });
      } else {
        fields.set(name, { name, type: String(config), required: true });
      }
    }
    return new InputSchema(fields);
  }

  validate(inputs: Record<string, any>): string[] {
    const errors: string[] = [];
    for (const [name, field] of this.fields) {
      const value = inputs[name] ?? field.default;
      if (value === undefined || value === null) {
        if (field.required && field.default === undefined) {
          errors.push(`Field '${name}' is required`);
        }
        continue;
      }
      if (field.type === "enum" && field.values && !field.values.includes(value)) {
        errors.push(`Field '${name}' must be one of [${field.values.join(", ")}]`);
      }
      if (typeof value === "number") {
        if (field.minValue !== undefined && value < field.minValue) {
          errors.push(`Field '${name}' must be >= ${field.minValue}`);
        }
        if (field.maxValue !== undefined && value > field.maxValue) {
          errors.push(`Field '${name}' must be <= ${field.maxValue}`);
        }
      }
      if (typeof value === "string") {
        if (field.maxLength !== undefined && value.length > field.maxLength) {
          errors.push(`Field '${name}' must be at most ${field.maxLength} chars`);
        }
      }
    }
    return errors;
  }

  applyDefaults(inputs: Record<string, any>): Record<string, any> {
    const result = { ...inputs };
    for (const [name, field] of this.fields) {
      if (!(name in result) && field.default !== undefined) {
        result[name] = field.default;
      }
    }
    return result;
  }
}

export class OutputSchema {
  readonly fields: Map<string, FieldSchema>;

  constructor(fields: Map<string, FieldSchema>) {
    this.fields = fields;
  }

  static fromDict(schemaDict: Record<string, any>): OutputSchema {
    const fields = new Map<string, FieldSchema>();
    for (const [name, config] of Object.entries(schemaDict)) {
      if (typeof config === "object" && config !== null) {
        fields.set(name, { name, type: config.type || "string", required: false });
      } else {
        fields.set(name, { name, type: String(config), required: false });
      }
    }
    return new OutputSchema(fields);
  }

  validate(output: Record<string, any>): string[] {
    const errors: string[] = [];
    for (const [name, field] of this.fields) {
      if (field.required && !(name in output)) {
        errors.push(`Missing required output field: '${name}'`);
      }
    }
    return errors;
  }
}
