import typing

import torch

BacktrackFn = typing.Callable[
    [dict[str, typing.Any], torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[bool, int]]


def simple_threshold_decision(
    config: dict[str, typing.Any],
    top_k_logits_seq: torch.Tensor,
    top_k_probabilites_seq: torch.Tensor,
    chosen_token_relative_idx: torch.Tensor,
) -> tuple[bool, int]:
    BACKTRACK_TOKEN_COUNT: typing.Final[int] = 1
    chosen_prob = top_k_probabilites_seq[chosen_token_relative_idx].item()

    if chosen_prob < config["probability_threshold"]:
        return True, BACKTRACK_TOKEN_COUNT
    else:
        return False, 0
