# pylint: disable=unused-argument

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
CHOSEN_TOKEN_ID = Key[int]("CHOSEN_TOKEN_ID")
CHOSEN_TOKEN_INDEX = Key[int]("CHOSEN_TOKEN_INDEX")


class Context:

    def __init__(self) -> None:
        self._store: Dict[Key[Any], Any] = {}

    def __getitem__(self, key: Key[T]) -> T:
        return self._store[key]

    def __setitem__(self, key: Key[T], value: T) -> None:
        self._store[key] = value


@dataclass(frozen=True)
class Outcome:
    should_backtrack: bool
    tokens_to_remove: int = 0

    def __post_init__(self) -> None:
        if not self.should_backtrack and self.tokens_to_remove != 0:
            raise ValueError("tokens_to_remove must be 0 when not "
                             "backtracking.")


class DecisionFunction(ABC):

    @abstractmethod
    def __call__(self, z: Tensor, p: Tensor, i_chosen: int, y_hat: int) -> int:
        pass

    def reset(self) -> None:
        pass


class ProbabilityThreshold(DecisionFunction):

    def __init__(self, p_min: float = 0.05, backtrack_k: int = 1) -> None:
        if not 0.0 < p_min < 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        if backtrack_k < 1:
            raise ValueError("Backtrack count must be positive")

        self.p_min = p_min
        self.backtrack_k = backtrack_k

    def __call__(self, z: Tensor, p: Tensor, i_chosen: int, y_hat: int) -> int:
        if not 0 <= i_chosen < p.shape[0]:
            logger.warning(
                "Chosen token index %d is out of bounds for "
                "probability tensor of size %d.", i_chosen, p.shape[0])
            return 0

        if p[i_chosen].item() < self.p_min:
            return self.backtrack_k

        return 0


class EntropyThreshold(DecisionFunction):

    def __init__(self, h_max: float = 2.5, backtrack_k: int = 2) -> None:
        if h_max <= 0.0:
            raise ValueError("Threshold must be non-negative")

        if backtrack_k < 1:
            raise ValueError("Backtrack count must be positive")

        self.h_max = h_max
        self.backtrack_k = backtrack_k

    def __call__(self, z: Tensor, p: Tensor, i_chosen: int, y_hat: int) -> int:
        if -(p * p.log()).sum() > self.h_max:
            return self.backtrack_k

        return 0


class ProbabilityMargin(DecisionFunction):

    def __init__(self, m_min: float = 0.05, backtrack_k: int = 1) -> None:
        if not 0.0 <= m_min <= 1.0:
            raise ValueError("Margin must be between 0.0 and 1.0")

        if backtrack_k < 1:
            raise ValueError("Backtrack count must be positive")

        self.m_min = m_min
        self.backtrack_k = backtrack_k

    def __call__(self, z: Tensor, p: Tensor, i_chosen: int, y_hat: int) -> int:
        if p.shape[0] < 2:
            logger.warning(
                "Cannot calculate probability margin for a "
                "distribution with fewer than 2 elements. Got %d.", p.shape[0])
            return 0

        top_p, _ = torch.topk(p, k=2)

        if (top_p[0] - top_p[1]).item() < self.m_min:
            return self.backtrack_k

        return 0


class ProbabilityDrop(DecisionFunction):

    def __init__(self, m_min: float = 2.0, backtrack_k: int = 1) -> None:
        if m_min < 1:
            raise ValueError("Drop ratio must be positive")

        if backtrack_k < 1:
            raise ValueError("Backtrack count must be positive")

        self.m_min = m_min
        self.backtrack_k = backtrack_k
        self._p_last: float | None = None

    def __call__(self, z: Tensor, p: Tensor, i_chosen: int, y_hat: int) -> int:
        if not 0 <= i_chosen < p.shape[0]:
            logger.warning(
                "Chosen token index %d is out of bounds for "
                "probability tensor of size %d.", i_chosen, p.shape[0])
            self._p_last = None
            return 0

        backtrack = False

        if (self._p_last is not None
                and self._p_last > p[i_chosen].item() * self.m_min):
            backtrack = True

        if backtrack:
            self._p_last = None
            return self.backtrack_k

        self._p_last = p[i_chosen].item()
        return 0

    def reset(self) -> None:
        self._p_last = None


class ProbabilityTrend(DecisionFunction):

    def __init__(self,
                 w: int = 10,
                 m_min: float = 0.5,
                 backtrack_k: int = 1) -> None:
        if w < 2:
            raise ValueError("Window size must be at least 2.")

        if not 0.0 < m_min < 1.0:
            raise ValueError("Drop ratio must be between 0.0 and 1.0")

        if backtrack_k < 1:
            raise ValueError("Backtrack count must be a positive integer.")

        self.w = w
        self.m_min = m_min
        self.backtrack_k = backtrack_k
        self._history: Deque[float] = collections.deque(maxlen=w)

    def __call__(self, z: Tensor, p: Tensor, i_chosen: int, y_hat: int) -> int:
        p_current = p[i_chosen].item()

        if len(self._history) >= self.w // 2:
            mean_p = sum(self._history) / len(self._history)
            if p_current < mean_p * self.m_min:
                return self.backtrack_k

        self._history.append(p_current)

        return 0

    def reset(self) -> None:
        self._history.clear()


class Repetition(DecisionFunction):

    def __init__(self, max_n: int = 3) -> None:
        if max_n < 1:
            raise ValueError("Max repetitions must be positive")

        self.max_n = max_n
        self._v_last: int | None = None
        self._n_repeats = 0

    def __call__(self, z: Tensor, p: Tensor, i_chosen: int, y_hat: int) -> int:
        if y_hat == self._v_last:
            self._n_repeats += 1
        else:
            self._n_repeats = 1
            self._v_last = y_hat

        if self._n_repeats > self.max_n:
            n = self._n_repeats
            self._n_repeats = 0
            self._v_last = None
            return n

        return 0

    def reset(self) -> None:
        self._v_last = None
        self._n_repeats = 0


class NGramOverlap(DecisionFunction):

    def __init__(self, n: int = 4) -> None:
        if n < 2:
            raise ValueError("n-gram size must be greater than 1")

        self.n = n
        self._window: Deque[int] = collections.deque(maxlen=self.n)
        self._seen_ngrams: Set[Tuple[int, ...]] = set()

    def __call__(self, z: Tensor, p: Tensor, i_chosen: int, y_hat: int) -> int:
        self._window.append(y_hat)

        if len(self._window) < self.n:
            logger.warning("Not enough history to form two n-grams for "
                           "comparison.")
            return 0

        last_ngram = tuple(self._window)

        if last_ngram in self._seen_ngrams:
            self._window.clear()
            return self.n
        else:
            self._seen_ngrams.add(last_ngram)
            return 0

    def reset(self) -> None:
        self._window.clear()
        self._seen_ngrams.clear()


class LogitThreshold(DecisionFunction):

    def __init__(self, z_min: int = -20, backtrack_k: int = 1) -> None:
        if backtrack_k < 1:
            raise ValueError("Backtrack count must be positive")

        self.z_min = z_min
        self.backtrack_k = backtrack_k

    def __call__(self, z: Tensor, p: Tensor, i_chosen: int, y_hat: int) -> int:
        if not 0 <= i_chosen < z.shape[0]:
            logger.warning(
                "Chosen token index %d is out of bounds for logit "
                "tensor of size %d.", i_chosen, z.shape[0])
            return 0

        if z[i_chosen].item() < self.z_min:
            return self.backtrack_k

        return 0
