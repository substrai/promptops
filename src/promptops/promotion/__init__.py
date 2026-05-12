"""Environment promotion and deployment lifecycle for PromptOps.

Manages the promotion of prompts through environments:
dev → staging → prod with approval gates, quality checks, and rollback.
"""

from promptops.promotion.manager import PromotionManager, PromotionResult
from promptops.promotion.environments import Environment, EnvironmentConfig
from promptops.promotion.rollback import RollbackManager, RollbackRecord

__all__ = [
    "PromotionManager",
    "PromotionResult",
    "Environment",
    "EnvironmentConfig",
    "RollbackManager",
    "RollbackRecord",
]
