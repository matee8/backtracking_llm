import typing

import torch

BacktrackingDecisionFunctionType = typing.Callable[
    [torch.Tensor, torch.Tensor, int, dict[str, typing.Any]], tuple[bool, int]]


def simple_threshold_decision(
    top_k_logits_seq: torch.Tensor,
    top_k_probabilites_seq: torch.Tensor,
    chosen_token_relative_idx: int,
    config: dict[str, typing.Any],
) -> tuple[bool, int]:
    BACKTRACK_TOKEN_COUNT: typing.Final[int] = 1
    chosen_prob = (top_k_probabilites_seq[chosen_token_relative_idx].item())

    if chosen_prob < config["probability_threshold"]:
        return True, BACKTRACK_TOKEN_COUNT
    else:
        return False, 0
