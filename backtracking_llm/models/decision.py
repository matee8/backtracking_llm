import typing

import torch

BacktrackingDecisionFunctionType = typing.Callable[
    [torch.Tensor, torch.Tensor, int, typing.Dict[str, typing.Any]],
    typing.Tuple[bool, int]]


def simple_threshold_decision(
    top_k_logits_seq: torch.Tensor,
    top_k_probabilites_seq: torch.Tensor,
    chosen_token_relative_idx: int,
    config: typing.Dict[str, typing.Any],
) -> typing.Tuple[bool, int]:
    BACKTRACK_TOKEN_COUNT: int = 1
    chosen_prob: float = top_k_probabilites_seq[
        chosen_token_relative_idx].item()

    if chosen_prob < config["probability_threshold"]:
        return True, BACKTRACK_TOKEN_COUNT
    else:
        return False, 0
