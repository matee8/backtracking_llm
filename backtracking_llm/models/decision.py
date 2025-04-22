import collections
import typing

import torch


class BacktrackStrategy(typing.Protocol):

    def should_backtrack(self, logits: torch.Tensor,
                         probabilities: torch.Tensor, rel_idx: torch.Tensor,
                         token_id: torch.Tensor) -> int:
        ...


class ProbabilityThreshold:

    def __init__(self,
                 threshold: float = 0.05,
                 backtrack_count: int = 1) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        if backtrack_count < 1:
            raise ValueError("Backtrack count must be positive")

        self.threshold = threshold
        self.backtrack_count = backtrack_count

    def should_backtrack(self, logits: torch.Tensor,
                         probabilities: torch.Tensor, rel_idx: torch.Tensor,
                         token_id: torch.Tensor) -> int:
        if not 0 <= rel_idx.item() < len(probabilities):
            return 0

        prob = probabilities[rel_idx].item()

        if prob < self.threshold:
            return self.backtrack_count
        else:
            return 0


class EntropyThreshold:

    def __init__(self,
                 max_entropy: float = 2.5,
                 backtrack_count: int = 2) -> None:
        if max_entropy <= 0.0:
            raise ValueError("Entropy threshold must be non-negative")

        self.max_entropy = max_entropy
        self.backtrack_count = backtrack_count

    def should_backtrack(self, logits: torch.Tensor,
                         probabilities: torch.Tensor, rel_idx: torch.Tensor,
                         token_id: torch.Tensor) -> int:
        try:
            entropy = (-probabilities * probabilities.log()).sum()
        except Exception:
            return 0

        if entropy > self.max_entropy:
            return self.backtrack_count
        else:
            return 0


class ProbabilityMargin:

    def __init__(self,
                 threshold: float = 0.05,
                 backtrack_count: int = 1) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Margin threshold must be between 0.0 and 1.0")

        if backtrack_count < 1:
            raise ValueError("Backtrack count must be positive")

        self.threshold = threshold
        self.backtrack_count = backtrack_count

    def should_backtrack(self, logits: torch.Tensor,
                         probabilities: torch.Tensor, rel_idx: torch.Tensor,
                         token_id: torch.Tensor) -> int:
        if len(probabilities) < 2:
            return 0

        top_probs, _ = torch.topk(probabilities, k=2)

        if (top_probs[0] - top_probs[1]).item() < self.threshold:
            return self.backtrack_count

        return 0


class ProbabilityDrop:

    def __init__(self,
                 threshold_ratio: float = 2.0,
                 backtrack_count: int = 1) -> None:
        if threshold_ratio < 1:
            raise ValueError("Threshold ratio must be positive")

        if backtrack_count < 1:
            raise ValueError("Backtrack count must be positive")

        self.threshold_ratio = threshold_ratio
        self.backtrack_count = backtrack_count
        self._last_chosen_probability: float | None = None

    def should_backtrack(self, logits: torch.Tensor,
                         probabilities: torch.Tensor, rel_idx: torch.Tensor,
                         token_id: torch.Tensor) -> int:
        if not 0 <= rel_idx.item() < len(probabilities):
            self._last_chosen_probability = None
            return 0

        if (self._last_chosen_probability is not None
                and self._last_chosen_probability
                > (self.threshold_ratio * probabilities[rel_idx].item())):
            self._last_chosen_probability = None
            return self.backtrack_count

        return 0


class ProbabilityTrend:

    def __init__(self,
                 window_size: int = 10,
                 drop_ratio: float = 0.5,
                 backtrack_count: int = 1) -> None:
        if window_size < 2:
            raise ValueError("Window size must be greater than or equal to 2")

        if not 0.0 < drop_ratio < 1.0:
            raise ValueError("Drop ratio must be between 0.0 and 1.0")

        if backtrack_count < 1:
            raise ValueError("Backtrack count must be positive")

        self.window_size = window_size
        self.drop_ratio = drop_ratio
        self.backtrack_count = backtrack_count
        self._history: typing.Deque[float] = (collections.deque(
            maxlen=window_size))

    def should_backtrack(self, logits: torch.Tensor,
                         probabilities: torch.Tensor, rel_idx: torch.Tensor,
                         token_id: torch.Tensor) -> int:
        current_prob = probabilities[rel_idx].item()
        if self._history:
            mean_prob = sum(self._history) / len(self._history)
        else:
            mean_prob = 1.0

        self._history.append(current_prob)

        if (len(self._history) >= 2
                and current_prob < mean_prob * self.drop_ratio):
            return self.backtrack_count

        return 0


class Repetition:

    def __init__(self, max_repetitions: int = 3) -> None:
        if max_repetitions < 1:
            raise ValueError("Max repetitions must be positive")

        self.max_repetitions = max_repetitions
        self._last_idx: int | None = None
        self._repeat_count: int = 0

    def should_backtrack(self, logits: torch.Tensor,
                         probabilities: torch.Tensor, rel_idx: torch.Tensor,
                         token_id: torch.Tensor) -> int:
        idx = int(token_id)
        if idx == self._last_idx:
            self._repeat_count += 1
        else:
            self._repeat_count = 1
            self._last_idx = idx

        if self._repeat_count >= self.max_repetitions:
            backtrack_count = self._repeat_count
            self._repeat_count = 0
            self._last_idx = None
            return backtrack_count

        return 0


class NGramOverlap:

    def __init__(self, history: list[int] = [], n: int = 4) -> None:
        if n < 2:
            raise ValueError("n must be greater than 1")

        self.n = n
        self._history = history

    def should_backtrack(self, logits: torch.Tensor,
                         probabilities: torch.Tensor, rel_idx: torch.Tensor,
                         token_id: torch.Tensor) -> int:
        idx = int(token_id)

        self._history.append(idx)

        if len(self._history) < 2 * self.n:
            return 0

        last_ngram = tuple(self._history[-self.n:])
        prior_sequence = self._history[:-self.n]

        for i in range(len(prior_sequence) - self.n + 1):
            if tuple(prior_sequence[i:i + self.n]) == last_ngram:
                return self.n

        return 0


class LogitThreshold:

    def __init__(self, threshold: int = -20, backtrack_count: int = 1) -> None:
        if backtrack_count < 1:
            raise ValueError("Backtrack count must be positive")

        self.threshold = threshold
        self.backtrack_count = backtrack_count

    def should_backtrack(self, logits: torch.Tensor,
                         probabilities: torch.Tensor, rel_idx: torch.Tensor,
                         token_id: torch.Tensor) -> int:
        if not 0 <= rel_idx.item() < len(logits):
            return 0

        logit = logits[rel_idx].item()

        if logit < self.threshold:
            return self.backtrack_count
        else:
            return 0
