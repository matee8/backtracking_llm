import collections
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
from typing import Any, Deque, Dict, Generic, Set, Tuple, TypeVar

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class Key(Generic[T]):
    name: str


LOGITS = Key[Tensor]("LOGITS")
PROBABILITIES = Key[Tensor]("PROBABILITIES")
CHOSEN_OUTPUT = Key[int]("CHOSEN_OUTPUT")
CHOSEN_INDEX = Key[int]("CHOSEN_INDEX")


class Context:

    @classmethod
    def from_data(cls, data: Dict[Key[Any], Any]) -> "Context":
        context = cls()
        context._store = data.copy()
        return context

    def __init__(self) -> None:
        self._store: Dict[Key[Any], Any] = {}

    def __getitem__(self, key: Key[T]) -> T:
        return self._store[key]


@dataclass(frozen=True)
class Outcome:
    should_backtrack: bool
    steps_to_remove: int = 0

    def __post_init__(self) -> None:
        if not self.should_backtrack and self.steps_to_remove != 0:
            raise ValueError("steps_to_remove must be 0 when not "
                             "backtracking.")


class Operator(ABC):

    @property
    @abstractmethod
    def required_inputs(self) -> Set[Key[Any]]:
        pass

    @abstractmethod
    def decide(self, context: Context) -> Outcome:
        pass

    def reset(self) -> None:
        pass


class ProbabilityThreshold(Operator):

    def __init__(self,
                 minimum_probability: float = 0.05,
                 backtrack_count: int = 1) -> None:
        if not 0.0 < minimum_probability < 1.0:
            raise ValueError("Minimum probability must be between 0.0 and 1.0")

        if backtrack_count < 1:
            raise ValueError("Backtrack count must be positive")

        self.minimum_probability = minimum_probability
        self.backtrack_count = backtrack_count

    @property
    def required_inputs(self) -> Set[Key[Any]]:
        return {PROBABILITIES, CHOSEN_INDEX}

    def decide(self, context: Context) -> Outcome:
        probabilities = context[PROBABILITIES]
        chosen_index = context[CHOSEN_INDEX]

        if not 0 <= chosen_index < probabilities.shape[0]:
            logger.warning(
                "Chosen index %d is out of bounds for "
                "probability tensor of size %d.", chosen_index,
                probabilities.shape[0])
            return Outcome(False)

        chosen_probability = probabilities[chosen_index].item()
        if chosen_probability < self.minimum_probability:
            return Outcome(True, self.backtrack_count)

        return Outcome(False)


class EntropyThreshold(Operator):

    def __init__(self,
                 maximum_entropy: float = 2.5,
                 backtrack_count: int = 2) -> None:
        if maximum_entropy <= 0.0:
            raise ValueError("Threshold must be non-negative")

        if backtrack_count < 1:
            raise ValueError("Backtrack count must be positive")

        self.max_entropy = maximum_entropy
        self.backtrack_count = backtrack_count

    @property
    def required_inputs(self) -> Set[Key[Any]]:
        return {PROBABILITIES}

    def decide(self, context: Context) -> Outcome:
        probabilities = context[PROBABILITIES]
        entropy = -(probabilities *
                    torch.log(probabilities + 1e-9)).sum().item()

        if entropy > self.max_entropy:
            return Outcome(True, self.backtrack_count)

        return Outcome(False)


class ProbabilityMargin(Operator):

    def __init__(self,
                 minimum_margin: float = 0.05,
                 backtrack_count: int = 1) -> None:
        if not 0.0 <= minimum_margin <= 1.0:
            raise ValueError("Margin must be between 0.0 and 1.0")

        if backtrack_count < 1:
            raise ValueError("Backtrack count must be positive")

        self.minimum_margin = minimum_margin
        self.backtrack_count = backtrack_count

    @property
    def required_inputs(self) -> Set[Key[Any]]:
        return {PROBABILITIES}

    def decide(self, context: Context) -> Outcome:
        probabilities = context[PROBABILITIES]

        if probabilities.shape[0] < 2:
            logger.warning("Cannot calculate margin for distribution < 2 "
                           "elements.")
            return Outcome(False)

        top_probabilities, _ = torch.topk(probabilities, k=2)
        margin = (top_probabilities[0] - top_probabilities[1]).item()

        if margin < self.minimum_margin:
            return Outcome(True, self.backtrack_count)

        return Outcome(False)


class ProbabilityDrop(Operator):

    def __init__(self,
                 minimum_margin: float = 2.0,
                 backtrack_count: int = 1) -> None:
        if minimum_margin < 1:
            raise ValueError("Drop ratio must be positive")

        if backtrack_count < 1:
            raise ValueError("Backtrack count must be positive")

        self.minimum_margin = minimum_margin
        self.backtrack_count = backtrack_count
        self._last_probability: float | None = None

    @property
    def required_inputs(self) -> Set[Key[Any]]:
        return {PROBABILITIES, CHOSEN_INDEX}

    def decide(self, context: Context) -> Outcome:
        probabilities = context[PROBABILITIES]
        chosen_index = context[CHOSEN_INDEX]

        if not 0 <= chosen_index < probabilities.shape[0]:
            logger.warning(
                "Chosen index %d is out of bounds for "
                "probability tensor of size %d.", chosen_index,
                probabilities.shape[0])
            self._last_probability = None
            return Outcome(False)

        chosen_probability = probabilities[chosen_index].item()

        if self._dropped_below_threshold(chosen_probability):
            self.reset()
            return Outcome(True, self.backtrack_count)

        self._last_probability = chosen_probability

        return Outcome(False)

    def reset(self) -> None:
        self._last_probability = None

    def _dropped_below_threshold(self, chosen_probability: float) -> bool:
        return (self._last_probability is not None and self._last_probability
                > chosen_probability * self.minimum_margin)


class ProbabilityTrend(Operator):

    def __init__(self,
                 window_size: int = 10,
                 minimum_margin: float = 0.5,
                 backtrack_count: int = 1) -> None:
        if window_size < 2:
            raise ValueError("Window size must be at least 2.")

        if not 0.0 < minimum_margin < 1.0:
            raise ValueError("Drop ratio must be between 0.0 and 1.0")

        if backtrack_count < 1:
            raise ValueError("Backtrack count must be a positive integer.")

        self.window_size = window_size
        self.minimum_margin = minimum_margin
        self.backtrack_count = backtrack_count
        self._history_window: Deque[float] = collections.deque(
            maxlen=window_size)

    @property
    def required_inputs(self) -> Set[Key[Any]]:
        return {PROBABILITIES, CHOSEN_INDEX}

    def decide(self, context: Context) -> Outcome:
        probabilities = context[PROBABILITIES]
        chosen_index = context[CHOSEN_INDEX]
        chosen_probability = probabilities[chosen_index].item()

        if (self._has_enough_elements_for_window()
                and self._dropped_below_mean(chosen_probability)):
            return Outcome(True, self.backtrack_count)

        self._history_window.append(chosen_probability)

        return Outcome(False)

    def reset(self) -> None:
        self._history_window.clear()

    def _has_enough_elements_for_window(self) -> bool:
        return len(self._history_window) >= self.window_size // 2

    def _dropped_below_mean(self, chosen_probability: float) -> bool:
        mean_window_probability = self._calculate_mean_window_probability()
        return (chosen_probability
                < mean_window_probability * self.minimum_margin)

    def _calculate_mean_window_probability(self) -> float:
        return sum(self._history_window) / len(self._history_window)


class Repetition(Operator):

    def __init__(self, maximum_repeats: int = 3) -> None:
        if maximum_repeats < 1:
            raise ValueError("Max repetitions must be positive")

        self.maximum_repeats = maximum_repeats
        self._last_chosen_output: int | None = None
        self._number_of_repeats = 0

    @property
    def required_inputs(self) -> Set[Key[Any]]:
        return {CHOSEN_OUTPUT}

    def decide(self, context: Context) -> Outcome:
        chosen_output = context[CHOSEN_OUTPUT]

        if chosen_output == self._last_chosen_output:
            self._number_of_repeats += 1
        else:
            self._update_chosen_output(chosen_output)

        if self._exceeds_maximum_repeats():
            steps_to_remove = self._number_of_repeats
            self.reset()
            return Outcome(True, steps_to_remove)

        return Outcome(False)

    def reset(self) -> None:
        self._last_chosen_output = None
        self._number_of_repeats = 0

    def _update_chosen_output(self, chosen_output: int) -> None:
        self._last_chosen_output = chosen_output
        self._number_of_repeats = 1

    def _exceeds_maximum_repeats(self) -> bool:
        return self._number_of_repeats > self.maximum_repeats


class NGramOverlap(Operator):

    def __init__(self, ngram_size: int = 4) -> None:
        if ngram_size < 2:
            raise ValueError("n-gram size must be greater than 1")

        self.ngram_size = ngram_size
        self._current_ngram: Deque[int] = collections.deque(
            maxlen=self.ngram_size)
        self._seen_ngrams: Set[Tuple[int, ...]] = set()

    @property
    def required_inputs(self) -> Set[Key[Any]]:
        return {CHOSEN_OUTPUT}

    def decide(self, context: Context) -> Outcome:
        chosen_output = context[CHOSEN_OUTPUT]

        self._current_ngram.append(chosen_output)

        if len(self._current_ngram) < self.ngram_size:
            logger.warning("Not enough history to form two n-grams for "
                           "comparison.")
            return Outcome(False)

        current_ngram = tuple(self._current_ngram)

        if current_ngram in self._seen_ngrams:
            self.reset()
            return Outcome(True, self.ngram_size)

        self._seen_ngrams.add(current_ngram)
        return Outcome(False)

    def reset(self) -> None:
        self._current_ngram.clear()
        self._seen_ngrams.clear()

    def _update_current_ngram(self, chosen_output: int) -> None:
        self._current_ngram.append(chosen_output)


class LogitThreshold(Operator):

    def __init__(self,
                 minimum_logit: int = -20,
                 backtrack_count: int = 1) -> None:
        if backtrack_count < 1:
            raise ValueError("Backtrack count must be positive")

        self.minimum_logit = minimum_logit
        self.backtrack_count = backtrack_count

    @property
    def required_inputs(self) -> Set[Key[Any]]:
        return {LOGITS, CHOSEN_INDEX}

    def decide(self, context: Context) -> Outcome:
        logits = context[LOGITS]
        chosen_token_index = context[CHOSEN_INDEX]

        if not 0 <= chosen_token_index < logits.shape[0]:
            logger.warning(
                "Chosen token index %d is out of bounds for logit "
                "tensor of size %d.", chosen_token_index, logits.shape[0])
            return Outcome(False)

        chosen_token_logit = logits[chosen_token_index].item()

        if chosen_token_logit < self.minimum_logit:
            return Outcome(True, self.backtrack_count)

        return Outcome(False)
