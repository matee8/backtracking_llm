import typing

import torch


class BacktrackStrategy(typing.Protocol):
    config: dict[str, typing.Any]

    def should_backtrack(self, logits: torch.Tensor,
                         probabilities: torch.Tensor,
                         token_idx: torch.Tensor) -> tuple[bool, int]:
        ...


class ProbabilityThresholdDecision:

    BACKTRACK_TOKEN_COUNT: typing.Final[int] = 1

    config: dict[str, typing.Any]

    def __init__(self, config: dict[str, typing.Any]) -> None:
        self.config = config

    def should_backtrack(self, logits: torch.Tensor,
                         probabilities: torch.Tensor,
                         token_idx: torch.Tensor) -> tuple[bool, int]:
        chosen_prob = probabilities[token_idx].item()

        if chosen_prob < self.config["probability_threshold"]:
            return True, self.BACKTRACK_TOKEN_COUNT
        else:
            return False, 0
