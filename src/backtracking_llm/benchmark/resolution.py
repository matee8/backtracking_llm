"""Provides logic for resolving operator class names to types."""

import logging
from typing import Type

from backtracking_llm import decision
from backtracking_llm.decision import Operator
from backtracking_llm.rl import operators

logger = logging.getLogger(__name__)


def resolve_operator_class(name: str) -> Type[Operator]:
    """Retrieves an Operator class by its name.

    Searches in `backtracking_llm.decision` and `backtracking_llm.rl.operators`.

    Args:
        name: The name of the class (e.g., 'ProbabilityThreshold').

    Returns:
        The class type corresponding to the name.

    Raises:
        ValueError: If the class is not found or is not a valid Operator.
    """
    if hasattr(decision, name):
        cls = getattr(decision, name)
    else:
        try:
            if hasattr(operators, name):
                cls = getattr(operators, name)
            else:
                raise AttributeError
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"'{name}' is not a valid Operator class name.") from e

    if not issubclass(cls, Operator):
        raise ValueError(f"'{name}' is not a valid Operator class name.")

    return cls
