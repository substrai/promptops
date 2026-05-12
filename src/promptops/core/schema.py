"""Schema validation for prompt inputs and outputs.

Validates that inputs match declared schemas and outputs conform
to expected structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FieldSchema:
    """Schema definition for a single field."""

    name: str
    type: str  # string, integer, float, boolean, array, enum, object
    required: bool = True
    default: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    values: Optional[List[str]] = None  # for enum type
    items: Optional[str] = None  # for array type

    def validate(self, value: Any) -> List[str]:
        """Validate a value against this field schema.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if value is None:
            if self.required and self.default is None:
                errors.append(f"Field '{self.name}' is required")
            return errors

        # Type checking
        type_map = {
            "string": str,
            "integer": int,
            "float": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        if self.type == "enum":
            if self.values and value not in self.values:
                errors.append(
                    f"Field '{self.name}' must be one of {self.values}, got '{value}'"
                )
        elif self.type in type_map:
            expected = type_map[self.type]
            if not isinstance(value, expected):
                errors.append(
                    f"Field '{self.name}' expected type {self.type}, got {type(value).__name__}"
                )

        # Range validation
        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                errors.append(
                    f"Field '{self.name}' must be >= {self.min_value}, got {value}"
                )
            if self.max_value is not None and value > self.max_value:
                errors.append(
                    f"Field '{self.name}' must be <= {self.max_value}, got {value}"
                )

        # Length validation
        if isinstance(value, str):
            if self.min_length is not None and len(value) < self.min_length:
                errors.append(
                    f"Field '{self.name}' must be at least {self.min_length} chars"
                )
            if self.max_length is not None and len(value) > self.max_length:
                errors.append(
                    f"Field '{self.name}' must be at most {self.max_length} chars"
                )

        return errors


@dataclass
class InputSchema:
    """Schema for prompt inputs."""

    fields: Dict[str, FieldSchema] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, schema_dict: Dict[str, Any]) -> "InputSchema":
        """Parse an input schema from a dictionary (YAML-parsed).

        Args:
            schema_dict: Dictionary with field names as keys

        Returns:
            InputSchema instance
        """
        fields = {}
        for name, config in schema_dict.items():
            if isinstance(config, dict):
                fields[name] = FieldSchema(
                    name=name,
                    type=config.get("type", "string"),
                    required=config.get("required", True),
                    default=config.get("default"),
                    min_value=config.get("min"),
                    max_value=config.get("max"),
                    min_length=config.get("min_length"),
                    max_length=config.get("max_length"),
                    values=config.get("values"),
                    items=config.get("items"),
                )
            else:
                # Simple type declaration: field_name: type
                fields[name] = FieldSchema(name=name, type=str(config))
        return cls(fields=fields)

    def validate(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate inputs against the schema.

        Args:
            inputs: Dictionary of input values

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        for name, field_schema in self.fields.items():
            value = inputs.get(name, field_schema.default)
            field_errors = field_schema.validate(value)
            errors.extend(field_errors)

        # Check for unknown fields
        known_fields = set(self.fields.keys())
        unknown_fields = set(inputs.keys()) - known_fields
        if unknown_fields:
            errors.append(f"Unknown input fields: {unknown_fields}")

        return errors

    def apply_defaults(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values to missing inputs.

        Args:
            inputs: Dictionary of input values

        Returns:
            Dictionary with defaults applied
        """
        result = dict(inputs)
        for name, field_schema in self.fields.items():
            if name not in result and field_schema.default is not None:
                result[name] = field_schema.default
        return result


@dataclass
class OutputSchema:
    """Schema for prompt outputs."""

    fields: Dict[str, FieldSchema] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, schema_dict: Dict[str, Any]) -> "OutputSchema":
        """Parse an output schema from a dictionary."""
        fields = {}
        for name, config in schema_dict.items():
            if isinstance(config, dict):
                fields[name] = FieldSchema(
                    name=name,
                    type=config.get("type", "string"),
                    required=config.get("required", False),
                    items=config.get("items"),
                )
            else:
                fields[name] = FieldSchema(name=name, type=str(config), required=False)
        return cls(fields=fields)

    def validate(self, output: Dict[str, Any]) -> List[str]:
        """Validate output against the schema."""
        errors = []
        for name, field_schema in self.fields.items():
            if field_schema.required and name not in output:
                errors.append(f"Missing required output field: '{name}'")
            elif name in output:
                field_errors = field_schema.validate(output[name])
                errors.extend(field_errors)
        return errors
