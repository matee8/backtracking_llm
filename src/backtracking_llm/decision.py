"""Defines the decision functions.

This module defines the core logic for determining when to backtrack during text
generation process. Decision functions are callable objects that evaluate the
model's output at a given step and return an integer value.
"""

# pylint: disable=unused-argument

import logging
from typing import Optional, Protocol

import torch
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

    Attributes:
        min_probability: The probability threshold below which backtracking is
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
        """Implements the Operator protocol by checking whether the last chosen
        token's probability is below the pre-configured threshold.
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


class EntropyThreshold:
    """An operator that backtracks when the entropy of the probabilities is too
    high.

    Attributes:
        max_entropy: The entropy threshold above which backtracking is
            triggered.
        backtrack_count: The number of tokens to backtrack if the condition
            is met.
    """

    def __init__(self,
                 max_entropy: float = 0.2,
                 backtrack_count: int = 2) -> None:
        """Initializes the EntropyThreshold operator.

        Args:
            max_entropy: The entropy threshold. Must be a non-negative number.
            backtrack_count: The number of tokens to remove when backtracking.
                Must be a positive integer.

        Raises:
            ValueError: If `max_entropy` is negative, or if `backtrack_count` is
                not positive.
        """
        if max_entropy < 0.0:
            raise ValueError("`max_entropy` must be non-negative")

        if backtrack_count < 1:
            raise ValueError("`backtrack_count` must be positive")

        self.max_entropy = max_entropy
        self.backtrack_count = backtrack_count

    def __call__(self, tokens: Tensor, probabilities: Tensor, position: int,
                 token: str) -> int:
        """Implements the Operator protocol by checking whether the probability
        distribution's entropy is above a pre-configured threshold.
        """
        non_zero_probabilities = probabilities[probabilities > 0]

        entropy = -(non_zero_probabilities * non_zero_probabilities.log()).sum()

        if entropy.item() > self.max_entropy:
            return self.backtrack_count

        return 0


class ProbabilityMargin:
    """An operator that backtracks if the confidence margin is too small.

    Attributes:
        min_margin: The minimum required difference between the top two
            probabilities. If the actual difference is smaller, backtracking is
            triggered.
        backtrack_count: The number of tokens to backtrack if the condition is
            met.
    """

    def __init__(self,
                 min_margin: float = 0.05,
                 backtrack_count: int = 1) -> None:
        """Initializes the ProbabilityMargin operator.

        Args:
            min_margin: The minimum required difference between the top two
                probabilities. Must be a value between 0.0 and 1.0.
            backtrack_count: The number of tokens to remove when backtracking.
                Must be a positive integer.

        Raises:
            ValueError: If `min_margin` is not between 0.0 and 1.0, or if
                `backtrack_count` is not positive.
        """
        if not 0.0 <= min_margin <= 1.0:
            raise ValueError("`min_margin` must be between 0.0 and 1.0")

        if backtrack_count < 1:
            raise ValueError("`backtrack_count` must be positive")

        self.min_margin = min_margin
        self.backtrack_count = backtrack_count

    def __call__(self, tokens: Tensor, probabilities: Tensor, position: int,
                 token: str) -> int:
        """Implements the Operator protocol by checking whether the margin
        between the top two probabilities is below a pre-configured threshold.
        """
        if probabilities.shape[0] < 2:
            logger.warning(
                "Cannot calculate margin between top 2 probabilities for a "
                "distribution with fewer than 2 elements, got %d.",
                probabilities.shape[0])
            return 0

        top_probabilities, _ = torch.topk(probabilities, k=2)

        difference = top_probabilities[0] - top_probabilities[1]

        if difference.item() < self.min_margin:
            return self.backtrack_count

        return 0


class ProbabilityDrop:
    """An operator that backtracks if the token confidence drops too sharply.

    Attributes:
        max_drop: The maximum allowed relative drop in probability.
        backtrack_count: The number of tokens to backtrack if the condition is
            met.
        _last_probability: Internal state to store the last token's probability.
    """

    def __init__(self, max_drop: float = 0.8, backtrack_count: int = 1) -> None:
        """Initializes the ProbabilityDrop operator.

        Args:
            max_drop: The maximum allowed relative drop. E.g., a value of 0.8
                means a backtrack is triggered if the new probability is less
                than 20% (1.0 - 0.8) of the previous probability. Must be a
                number between 0.0 and 1.0
            backtrack_count: The number of tokens to remove when backtracking.
                Must be a positive number.

        Raises:
            ValueError: If `max_drop` is not between 0.0 and 1.0, or if
                `backtrack_count` is not positive.
        """
        if not 0.0 <= max_drop <= 1.0:
            raise ValueError("`max_drop` must be between 0.0 and 1.0")

        if backtrack_count < 1:
            raise ValueError("`backtrack_count` must be positive")

        self.max_drop = max_drop
        self.backtrack_count = backtrack_count
        self._last_probability: Optional[float] = None

    def __call__(self, tokens: Tensor, probabilities: Tensor, position: int,
                 token: str) -> int:
        """Implements the Operator protocol by comparing the the current token's
        probability to the previous one.
        """
        if not 0 <= position < probabilities.shape[0]:
            logger.warning(
                "Chosen token position %d is out of bounds for "
                "probability tensor of size %d.", position,
                probabilities.shape[0])
            self._last_probability = None
            return 0

        current_probability = probabilities[position].item()
        backtrack = False

        if self._last_probability is not None:
            threshold = self._last_probability * (1.0 - self.max_drop)
            if current_probability < threshold:
                backtrack = True

        self._last_probability = current_probability

        return self.backtrack_count if backtrack else 0
