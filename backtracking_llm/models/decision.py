import typing

import torch


class BacktrackStrategy(typing.Protocol):

    def should_backtrack(self, logits: torch.Tensor,
                         probabilities: torch.Tensor,
                         token_idx: torch.Tensor) -> int:
        ...


class ProbabilityThresholdDecision:

    def __init__(self,
                 threshold: float = 0.05,
                 backtrack_count: int = 1) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        if backtrack_count < 1:
            raise ValueError("Backtrack count must be positive")

        self.threshold = threshold

    def should_backtrack(self, logits: torch.Tensor,
                         probabilities: torch.Tensor,
                         token_idx: torch.Tensor) -> int:
        if not 0 <= token_idx.item() < len(probabilities):
            return 0

        prob = probabilities[token_idx].item()

        if prob < self.threshold:
            return self.backtrack_count
        else:
            return 0
