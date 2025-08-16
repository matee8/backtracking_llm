"""Defines the decision functions.

This module defines the core logic for determining when to backtrack during text
generation process. Decision functions are callable objects that evaluate the
model's output at a given step and return an integer value.
"""

# pylint: disable=unused-argument

import logging
from typing import Protocol

from torch import Tensor

logger = logging.getLogger(__name__)


class Operator(Protocol):
    """A protocol for decision functions
    """

    def __call__(self, logits: Tensor, probabilities: Tensor, position: int,
                 token: str) -> int:
        """Determines whether to backtrack based on the latest generated token.

        Args:
            logits: The raw logits from the model for the curret token.
            probabilities: The probabilities for the current token.
            position: The position of the chosen token.
            token: The string representation of the chosen token.

        Returns:
            0, if backtracking should not occur, otherwise, the number of tokens
            that should be truncated.
        """
        ...


class Never:
    """A simple operator that never backtracks.

    Used for testing purposes.
    """

    def __call__(self, tokens: Tensor, probabilities: Tensor, position: int,
                 token: str) -> int:
        """Always returns 0, indicating no backtracking should occur."""
        return 0


class ProbabilityThreshold:
    """An operator that backtracks if a token's probability is too low.

    This decision function triggers a backtrack operation if the probability of
    the selected token falls below a pre-configured threshold.

    Attributes:
        min_prbability: The probability threshold below which backtracking is
            triggered.
        backtrack_count: The number of tokens to backtrack if the condition
            is met.
    """

    def __init__(self,
                 min_probability: float = 0.05,
                 backtrack_count: int = 1) -> None:
        """Initializes the ProbabilityThreshold operator.

        Args:
            min_probability: The probability threshold. Must be a value
                strictly between 0.0 and 1.0.
            backtrack_count: The number of tokens to remove when backtracking.
                Must be a positive integer.

        Raises:
            ValueError: If `min_probability` is not between 0.0 and 1.0, or if
                `backtrack_count` is not positive.
        """
        if not 0.0 < min_probability < 1.0:
            raise ValueError("`min_probability` must be between 0.0 and 1.0")

        if backtrack_count < 1:
            raise ValueError("`backtrack_count` must be positive")

        self.min_probability = min_probability
        self.backtrack_count = backtrack_count

    def __call__(self, tokens: Tensor, probabilities: Tensor, position: int,
                 token: str) -> int:
        """Determines if the token's probability is below the threshold.

        Returns:
            The configured `backtrack_count` if the token's probability is
            less than `min_probability`, otherwise 0.
        """
        if not 0 <= position < probabilities.shape[0]:
            logger.warning(
                "Chosen token position %d is out of bounds for "
                "probability tensor of size %d.", position,
                probabilities.shape[0])
            return 0

        if probabilities[position].item() < self.min_probability:
            return self.backtrack_count

        return 0
