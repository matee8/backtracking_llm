"""Defines the decision functions.

This module defines the core logic for determining when to backtrack during text
generation process. Decision functions are callable objects that evaluate the
model's output at a given step and return an integer value.
"""

# pylint: disable=unused-argument

from typing import Protocol

from torch import Tensor


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
