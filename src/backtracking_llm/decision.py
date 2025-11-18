"""Defines the decision functions.

This module defines the core logic for determining when to backtrack during text
generation process. Decision functions are callable objects that evaluate the
model's output at a given step and return an integer value.
"""

# pylint: disable=unused-argument

import logging
from collections import deque
from typing import Optional, Protocol, runtime_checkable, Set, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@runtime_checkable
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

    def backtrack(self, n_tokens: int) -> None:
        """Called when tokens are removed from the generation.

        Operators with internal state should revert their state as if
        the last n_tokens were never generated. Stateless operators
        can leave this method empty or not implement it at all.

        Args:
            n_tokens: Number of tokens removed from the end.
        """


class Never:
    """A simple operator that never backtracks.

    Used for testing purposes.
    """

    def __call__(self, logits: Tensor, probabilities: Tensor, position: int,
                 token: str) -> int:
        """Always returns 0, indicating no backtracking should occur."""
        return 0

    def backtrack(self, n_tokens: int) -> None:
        pass


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
            raise ValueError('`min_probability` must be between 0.0 and 1.0')

        if backtrack_count < 1:
            raise ValueError('`backtrack_count` must be positive')

        self.min_probability = min_probability
        self.backtrack_count = backtrack_count

    def __call__(self, logits: Tensor, probabilities: Tensor, position: int,
                 token: str) -> int:
        """Implements the Operator protocol by checking whether the last chosen
        token's probability is below the pre-configured threshold.
        """
        if not 0 <= position < probabilities.shape[0]:
            logger.warning(
                'Chosen token position %d is out of bounds for '
                'probability tensor of size %d.', position,
                probabilities.shape[0])
            return 0

        if probabilities[position].item() < self.min_probability:
            return self.backtrack_count

        return 0

    def backtrack(self, n_tokens: int) -> None:
        pass

    def __repr__(self) -> str:
        """Returns an executable representation of the operator."""
        return (f'ProbabilityThreshold(min_probability={self.min_probability!r}'
                f', backtrack_count={self.backtrack_count!r})')

    def __eq__(self, other: object, /) -> bool:
        """Checks if two ProbabilityThreshold operators are equal."""
        if not isinstance(other, ProbabilityThreshold):
            return NotImplemented
        return ((self.min_probability,
                 self.backtrack_count) == (other.min_probability,
                                           other.backtrack_count))

    def __hash__(self) -> int:
        """Computes a hash based on the operator's configuration."""
        return hash((self.min_probability, self.backtrack_count))


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
            raise ValueError('`max_entropy` must be non-negative')

        if backtrack_count < 1:
            raise ValueError('`backtrack_count` must be positive')

        self.max_entropy = max_entropy
        self.backtrack_count = backtrack_count

    def __call__(self, logits: Tensor, probabilities: Tensor, position: int,
                 token: str) -> int:
        """Implements the Operator protocol by checking whether the probability
        distribution's entropy is above a pre-configured threshold.
        """
        non_zero_probabilities = probabilities[probabilities > 0]

        entropy = -(non_zero_probabilities * non_zero_probabilities.log()).sum()

        if entropy.item() > self.max_entropy:
            return self.backtrack_count

        return 0

    def backtrack(self, n_tokens: int) -> None:
        pass

    def __repr__(self) -> str:
        """Returns an executable representation of the operator."""
        return (f'EntropyThreshold(max_entropy={self.max_entropy!r}, '
                f'backtrack_count={self.backtrack_count!r})')

    def __eq__(self, other: object, /) -> bool:
        """Checks if two EntropyThreshold operators are equal."""
        if not isinstance(other, EntropyThreshold):
            return NotImplemented
        return ((self.max_entropy,
                 self.backtrack_count) == (other.max_entropy,
                                           other.backtrack_count))

    def __hash__(self) -> int:
        """Computes a hash based on the operator's configuration."""
        return hash((self.max_entropy, self.backtrack_count))


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
            raise ValueError('`min_margin` must be between 0.0 and 1.0')

        if backtrack_count < 1:
            raise ValueError('`backtrack_count` must be positive')

        self.min_margin = min_margin
        self.backtrack_count = backtrack_count

    def __call__(self, logits: Tensor, probabilities: Tensor, position: int,
                 token: str) -> int:
        """Implements the Operator protocol by checking whether the margin
        between the top two probabilities is below a pre-configured threshold.
        """
        if probabilities.shape[0] < 2:
            logger.warning(
                'Cannot calculate margin between top 2 probabilities for a '
                'distribution with fewer than 2 elements, got %d.',
                probabilities.shape[0])
            return 0

        top_probabilities, _ = torch.topk(probabilities, k=2)

        difference = top_probabilities[0] - top_probabilities[1]

        if difference.item() < self.min_margin:
            return self.backtrack_count

        return 0

    def backtrack(self, n_tokens: int) -> None:
        pass

    def __repr__(self) -> str:
        """Returns an executable representation of the operator."""
        return (f'ProbabilityMargin(min_margin={self.min_margin!r}, '
                f'backtrack_count={self.backtrack_count!r})')

    def __eq__(self, other: object, /) -> bool:
        """Checks if two ProbabilityMargin operators are equal."""
        if not isinstance(other, ProbabilityMargin):
            return NotImplemented
        return ((self.min_margin,
                 self.backtrack_count) == (other.min_margin,
                                           other.backtrack_count))

    def __hash__(self) -> int:
        """Computes a hash based on the operator's configuration."""
        return hash((self.min_margin, self.backtrack_count))


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
            raise ValueError('`max_drop` must be between 0.0 and 1.0')

        if backtrack_count < 1:
            raise ValueError('`backtrack_count` must be positive')

        self.max_drop = max_drop
        self.backtrack_count = backtrack_count
        self._last_probability: Optional[float] = None

    def __call__(self, logits: Tensor, probabilities: Tensor, position: int,
                 token: str) -> int:
        """Implements the Operator protocol by comparing the the current token's
        probability to the previous one.
        """
        if not 0 <= position < probabilities.shape[0]:
            logger.warning(
                'Chosen token position %d is out of bounds for '
                'probability tensor of size %d.', position,
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

    def backtrack(self, n_tokens: int) -> None:
        if n_tokens > 0:
            self._last_probability = None

    def __repr__(self) -> str:
        """Returns an executable representation of the operator."""
        return (f'ProbabilityDrop(max_drop={self.max_drop!r}, '
                f'backtrack_count={self.backtrack_count!r})')

    def __eq__(self, other: object, /) -> bool:
        """Checks if two ProbabilityDrop operators are equal."""
        if not isinstance(other, ProbabilityDrop):
            return NotImplemented
        return ((self.max_drop,
                 self.backtrack_count) == (other.max_drop,
                                           other.backtrack_count))

    def __hash__(self) -> int:
        """Computes a hash based on the operator's configuration."""
        return hash((self.max_drop, self.backtrack_count))


class ProbabilityTrend:
    """An operator that backtracks if the confidence drops below a historical
    average.

    Attributes:
        window_size: The maximum number of recent probabilities to store.
        drop_threshold: A ratio used to check against the historical mean.
            A backtrack is triggered if the current probability is less than
            `mean * relative_drop_threshold`.
        backtrack_count: The number of tokens to backtrack if the condition is
            met.
        _history: A deque used as a sliding window to store recent
            probabilities.
    """

    def __init__(self,
                 window_size: int = 10,
                 drop_threshold: float = 0.5,
                 backtrack_count: int = 1) -> None:
        """Initializes the ProbabilityTrend operator.

        Args:
            window_size: The size of the historical probability window. Must be
                at least 2.
            drop_threshold: The relative threshold for the drop. E.g., a value
                of 0.5 means backtrack is triggered if the current probability
                is less than 50% of the historical average. Must be a value
                between 0.0 and 1.0.
            backtrack_count: The number of tokens to remove when backtracking.
                Must be a positive integer.

        Raises:
            ValueError: If `window_size` is less than 2, `drop_threshold` is not
                between 0.0 and 1.0, or `backtrack_count` is not positive.
        """
        if window_size < 2:
            raise ValueError('`window_size` must be at least 2')

        if not 0.0 < drop_threshold < 1.0:
            raise ValueError('`drop_threshold` must be between 0.0 and 1.0')

        if backtrack_count < 1:
            raise ValueError('`backtrack_count` must be positive')

        self.window_size = window_size
        self.drop_threshold = drop_threshold
        self.backtrack_count = backtrack_count
        self._history: deque[float] = deque(maxlen=self.window_size)

    def __call__(self, logits: Tensor, probabilities: Tensor, position: int,
                 token: str) -> int:
        """Implements the Operator protocol by checking if the current
        probability has dropped below a historical trend.
        """
        if not 0 <= position < probabilities.shape[0]:
            logger.warning(
                'Chosen token position %d is out of bounds for '
                'probability tensor of size %d.', position,
                probabilities.shape[0])
            return 0

        current_probability = probabilities[position].item()
        backtrack = False

        if len(self._history) >= self.window_size // 2:
            historical_mean = sum(self._history) / len(self._history)
            threshold = historical_mean * self.drop_threshold

            if current_probability < threshold:
                backtrack = True

        self._history.append(current_probability)

        return self.backtrack_count if backtrack else 0

    def backtrack(self, n_tokens: int) -> None:
        """Remove last n entries from history."""
        for _ in range(min(n_tokens, len(self._history))):
            if self._history:
                self._history.pop()

    def __repr__(self) -> str:
        """Returns an executable representation of the operator."""
        return (f'ProbabilityTrend(window_size={self.window_size!r}, '
                f'drop_threshold={self.drop_threshold!r}, '
                f'backtrack_count={self.backtrack_count!r})')

    def __eq__(self, other: object, /) -> bool:
        """Checks if two ProbabilityTrend operators are equal."""
        if not isinstance(other, ProbabilityTrend):
            return NotImplemented
        return ((self.window_size, self.drop_threshold,
                 self.backtrack_count) == (other.window_size,
                                           other.drop_threshold,
                                           other.backtrack_count))

    def __hash__(self) -> int:
        """Computes a hash based on the operator's configuration."""
        return hash(
            (self.window_size, self.drop_threshold, self.backtrack_count))


class Repetition:
    """An operator that backtracks on excessive consecutive token repetitions.

    Attributes:
        max_repetitions: The maximum number of consecutive repetitions allowed.
        _last_token: Internal state to store the last token.
        _repeat_count: Internal state to count consecutive repetitions.
    """

    def __init__(self, max_repetitions: int = 3) -> None:
        """Initializes the Repetition operator.

        Args:
            max_repetitions: The maximum number of consecutive repetitions
                allowed before triggering a backtrack. Must be a positive
                integer.

        Raises:
            ValueError: If `max_repetitions` is not positive.
        """
        if max_repetitions < 1:
            raise ValueError('`max_repetitions` must be positive')

        self.max_repetitions = max_repetitions
        self._last_token: Optional[str] = None
        self._repeat_count = 0

    def __call__(self, logits: Tensor, probabilities: Tensor, position: int,
                 token: str) -> int:
        """Implements the Operator protocol by checking for and counting
        consecutive token repetitions.
        """
        if token == self._last_token:
            self._repeat_count += 1
        else:
            self._repeat_count = 1
            self._last_token = token

        if self._repeat_count > self.max_repetitions:
            backtrack_amount = self._repeat_count

            self._repeat_count = 0
            self._last_token = None

            return backtrack_amount

        return 0

    def backtrack(self, n_tokens: int) -> None:
        """Reset state on backtrack."""
        if n_tokens > 0:
            self._repeat_count = 0
            self._last_token = None

    def __repr__(self) -> str:
        """Returns an executable representation of the operator."""
        return (f'Repetition(max_repetitions={self.max_repetitions!r})')

    def __eq__(self, other: object, /) -> bool:
        """Checks if two Repetition operators are equal."""
        if not isinstance(other, Repetition):
            return NotImplemented
        return (self.max_repetitions) == (other.max_repetitions)

    def __hash__(self) -> int:
        """Computes a hash based on the operator's configuration."""
        return hash((self.max_repetitions))


class NGramOverlap:
    """An operator that bactracks when a sequence of tokens is repeated.

    Attributes:
        ngram_size: The size of the token sequences to track.
        backtrack_count: THe number of tokens to remove upon detecting a repeat.
        _window: A sliding window of the `n` most recent tokens.
        _seen_ngrams: A set containing all unique n-grams encountered so far.
    """

    def __init__(self, ngram_size: int = 4, backtrack_count: int = 1) -> None:
        """Initializes the NGramOverlap operator.

        Args:
            ngram_size: The size of the n-gram to check for repetitions. Must be
                an integer greater than 1.
            backtrack_count: The number of tokens to remove when a repetition is
                found. Must be a positive integer.

        Raises:
            ValueError: If `ngram_size` is not greater than 1 or if
                `backtrack_count` is not positive.
        """
        if ngram_size < 2:
            raise ValueError('`ngram_size` must be greater than 1')

        if backtrack_count < 1:
            raise ValueError('`backtrack_count` must be positive')

        self.ngram_size = ngram_size
        self.backtrack_count = backtrack_count
        self._window: deque[str] = deque(maxlen=self.ngram_size)
        self._seen_ngrams: Set[Tuple[str, ...]] = set()
        self._history_log: deque[Optional[Tuple[str, ...]]] = deque()

    def __call__(self, logits: Tensor, probabilities: Tensor, position: int,
                 token: str) -> int:
        """Implements the Operator protocol by checking if the latest n-gram has
        been seen before.
        """

        self._window.append(token)

        if len(self._window) < self.ngram_size:
            self._history_log.append(None)
            return 0

        current_ngram = tuple(self._window)

        if current_ngram in self._seen_ngrams:
            self._history_log.append(None)
            return self.backtrack_count
        else:
            self._history_log.append(current_ngram)
            self._seen_ngrams.add(current_ngram)
            return 0

    def backtrack(self, n_tokens: int) -> None:
        """Remove tokens from window and revert seen_ngrams state."""
        for _ in range(min(n_tokens, len(self._window))):
            if self._window:
                self._window.pop()

            self._seen_ngrams.clear()
        for _ in range(min(n_tokens, len(self._history_log))):
            if self._history_log:
                ngram_to_remove = self._history_log.pop()
                if ngram_to_remove is not None:
                    self._seen_ngrams.discard(ngram_to_remove)

    def __repr__(self) -> str:
        """Returns an executable representation of the operator."""
        return (f'NGramOverlap(ngram_size={self.ngram_size!r}, '
                f'backtrack_count={self.backtrack_count!r})')

    def __eq__(self, other: object, /) -> bool:
        """Checks if two NGramOverlap operators are equal."""
        if not isinstance(other, NGramOverlap):
            return NotImplemented
        return ((self.ngram_size,
                 self.backtrack_count) == (other.ngram_size,
                                           other.backtrack_count))

    def __hash__(self) -> int:
        """Computes a hash based on the operator's configuration."""
        return hash((self.ngram_size, self.backtrack_count))


class LogitThreshold:
    """An operator that backtracks if a token's logit is too low.

    Attributes:
        min_logit: The logit threshold below which backtracking is triggered.
        backtrack_count: The number of tokens to backtrack if the condition
            is met.
    """

    def __init__(self,
                 min_logit: float = -20.0,
                 backtrack_count: int = 1) -> None:
        """Initializes the LogitThreshold operator.

        Args:
            min_logit: The logit threshold.
            backtrack_count: The number of tokens to remove when backtracking.
                Must be a positive integer.

        Raises:
            ValueError: If `backtrack_count` is not positive.
        """
        if backtrack_count < 1:
            raise ValueError('`backtrack_count` must be positive')

        self.min_logit = min_logit
        self.backtrack_count = backtrack_count

    def __call__(self, logits: Tensor, probabilities: Tensor, position: int,
                 token: str) -> int:
        """Implements the Operator protocol by checking whether the last chosen
        token's logit is below the pre-configured threshold.
        """
        if not 0 <= position < logits.shape[0]:
            logger.warning(
                'Chosen token position %d is out of bounds for '
                'logits tensor of size %d.', position, logits.shape[0])
            return 0

        if logits[position].item() < self.min_logit:
            return self.backtrack_count

        return 0

    def backtrack(self, n_tokens: int) -> None:
        pass

    def __repr__(self) -> str:
        """Returns an executable representation of the operator."""
        return (f'LogitThreshold(min_logit={self.min_logit!r}, '
                f'backtrack_count={self.backtrack_count!r})')

    def __eq__(self, other: object, /) -> bool:
        """Checks if two LogitThreshold operators are equal."""
        if not isinstance(other, LogitThreshold):
            return NotImplemented
        return ((self.min_logit,
                 self.backtrack_count) == (other.min_logit,
                                           other.backtrack_count))

    def __hash__(self) -> int:
        """Computes a hash based on the operator's configuration."""
        return hash((self.min_logit, self.backtrack_count))
