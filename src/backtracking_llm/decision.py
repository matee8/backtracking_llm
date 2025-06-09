# pylint: disable=unused-argument

import collections
import logging
from abc import ABC, abstractmethod
from typing import Deque, List

import torch
from torch import special, Tensor

logger = logging.getLogger(__name__)


class DecisionFunction(ABC):

    @abstractmethod
    def __call__(self, z: Tensor, p: Tensor, i_chosen: int,
                 v_chosen: int) -> int:
        pass


class ProbabilityThreshold(DecisionFunction):

    def __init__(self, theta: float = 0.05, n: int = 1) -> None:
        if not 0.0 < theta < 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        if n < 1:
            raise ValueError("Backtrack count must be positive")

        self.theta = theta
        self.n = n

    def __call__(self, z: Tensor, p: Tensor, i_chosen: int,
                 v_chosen: int) -> int:
        if not 0 <= i_chosen < p.shape[0]:
            logger.warning(
                "Chosen token index %d is out of bounds for "
                "probability tensor of size %d.", i_chosen, p.shape[0])
            return 0

        if p[i_chosen].item() < self.theta:
            return self.n

        return 0


class EntropyThreshold(DecisionFunction):

    def __init__(self, theta: float = 2.5, n: int = 2) -> None:
        if theta <= 0.0:
            raise ValueError("Threshold must be non-negative")

        self.theta = theta
        self.n = n

    def __call__(self, z: Tensor, p: Tensor, i_chosen: int,
                 v_chosen: int) -> int:
        if special.entr(p).item() > self.theta:
            return self.n

        return 0


class ProbabilityMargin(DecisionFunction):

    def __init__(self, theta: float = 0.05, n: int = 1) -> None:
        if not 0.0 <= theta <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        if n < 1:
            raise ValueError("Backtrack count must be positive")

        self.theta = theta
        self.n = n

    def __call__(self, z: Tensor, p: Tensor, i_chosen: int,
                 x_chosen: int) -> int:
        if p.shape[0] < 2:
            logger.warning(
                "Cannot calculate probability margin for a "
                "distribution with fewer than 2 elements. Got %d.", p.shape[0])
            return 0

        top_p, _ = torch.topk(p, k=2)

        if (top_p[0] - top_p[1]).item() < self.theta:
            return self.n

        return 0


class ProbabilityDrop(DecisionFunction):

    def __init__(self, ratio: float = 2.0, n: int = 1) -> None:
        if ratio < 1:
            raise ValueError("Drop ratio must be positive")

        if n < 1:
            raise ValueError("Backtrack count must be positive")

        self.ratio = ratio
        self.n = n
        self._p_last: float | None = None

    def __call__(self, z: Tensor, p: Tensor, i_chosen: int,
                 v_chosen: int) -> int:
        if not 0 <= i_chosen < p.shape[0]:
            logger.warning(
                "Chosen token index %d is out of bounds for "
                "probability tensor of size %d.", i_chosen, p.shape[0])
            self._p_last = None
            return 0

        backtrack = False

        if (self._p_last is not None
                and self._p_last > p[i_chosen].item() * self.ratio):
            backtrack = True

        if backtrack:
            self._p_last = None
            return self.n

        self._p_last = p[i_chosen].item()
        return 0


class ProbabilityTrend(DecisionFunction):

    def __init__(self, w: int = 10, ratio: float = 0.5, n: int = 1) -> None:
        if w < 2:
            raise ValueError("Window size must be at least 2.")

        if not 0.0 < ratio < 1.0:
            raise ValueError("Drop ratio must be between 0.0 and 1.0")

        if n < 1:
            raise ValueError("Backtrack count must be a positive integer.")

        self.w = w
        self.ratio = ratio
        self.n = n
        self._history: Deque[float] = collections.deque(maxlen=w)

    def __call__(self, z: Tensor, p: Tensor, i_chosen: int,
                 v_chosen: int) -> int:
        if len(self._history) == self.w:
            mean_p = sum(self._history) / len(self._history)
            if p[i_chosen].item() < mean_p * self.ratio:
                self._history.clear()
                return self.n

        self._history.append(p[i_chosen].item())
        return 0


class Repetition(DecisionFunction):

    def __init__(self, max_n: int = 3) -> None:
        if max_n < 1:
            raise ValueError("Max repetitions must be positive")

        self.max_n = max_n
        self._v_last: int | None = None
        self._n = 0

    def __call__(self, z: Tensor, p: Tensor, i_chosen: int,
                 v_chosen: int) -> int:
        if v_chosen == self._v_last:
            self._n += 1
        else:
            self._n = 1
            self._v_last = v_chosen

        if self._n >= self.max_n:
            n = self._n
            self._n = 0
            self._v_last = None
            return n

        return 0


class NGramOverlap(DecisionFunction):

    def __init__(self, n: int = 4) -> None:
        if n < 2:
            raise ValueError("n-gram size must be greater than 1")

        self.n = n
        self._history: List[int] = []

    def __call__(self, z: Tensor, p: Tensor, i_chosen: int,
                 v_chosen: int) -> int:
        self._history.append(v_chosen)

        if len(self._history) < 2 * self.n:
            logger.warning("Not enough history to form two n-grams for "
                           "comparison.")
            return 0

        last_ngram = tuple(self._history[-self.n:])
        s = self._history[:-self.n]

        for i in range(len(s) - self.n + 1):
            if tuple(s[i:i + self.n]) == last_ngram:
                self._history = self._history[:-self.n]
                return self.n

        return 0


class LogitThreshold:

    def __init__(self, theta: int = -20, n: int = 1) -> None:
        if n < 1:
            raise ValueError("Backtrack count must be positive")

        self.theta = theta
        self.backtrack_count = n

    def __call__(self, z: Tensor, p: Tensor, i_chosen: int,
                 v_chosen: Tensor) -> int:
        if not 0 <= v_chosen < z.shape[0]:
            logger.warning(
                "Chosen token index %d is out of bounds for logit "
                "tensor of size %d.", i_chosen, z.shape[0])
            return 0

        if z[i_chosen].item() < self.theta:
            return self.backtrack_count

        return 0
