"""
PromptOps - Infrastructure-as-Code for Prompt Engineering

The first framework for managing prompts as versioned, tested, deployed infrastructure
with semantic versioning, environment promotion, regression testing, and multi-model targeting.

Usage:
    from promptops import PromptClient, Prompt, PromptVersion

    client = PromptClient(env="prod")
    result = client.invoke("summarize", inputs={"document": "...", "max_words": 100})
"""

__version__ = "1.4.0"

from promptops.core.prompt import Prompt, PromptDefinition
from promptops.core.version import PromptVersion, VersionRange
from promptops.core.resolver import PromptResolver
from promptops.core.schema import InputSchema, OutputSchema
from promptops.core.client import PromptClient
from promptops.core.result import InvocationResult

__all__ = [
    "Prompt",
    "PromptDefinition",
    "PromptVersion",
    "VersionRange",
    "PromptResolver",
    "PromptClient",
    "InputSchema",
    "OutputSchema",
    "InvocationResult",
]
