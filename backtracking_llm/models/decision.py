import typing

import torch


class BacktrackStrategy(typing.Protocol):

    def should_backtrack(self, logits: torch.Tensor,
                         probabilities: torch.Tensor,
                         token_idx: torch.Tensor) -> tuple[bool, int]:
        ...


class ProbabilityThresholdDecision:

    BACKTRACK_TOKEN_COUNT: typing.Final[int] = 1

    def __init__(self, threshold: float = 0.05) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        self.threshold = threshold

    def should_backtrack(self, logits: torch.Tensor,
                         probabilities: torch.Tensor,
                         token_idx: torch.Tensor) -> tuple[bool, int]:
        if not 0 <= token_idx.item() < len(probabilities):
            return False, 0

        prob = probabilities[token_idx].item()

        if prob < self.threshold:
            return True, self.BACKTRACK_TOKEN_COUNT
        else:
            return False, 0
