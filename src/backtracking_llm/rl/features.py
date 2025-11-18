"""Provides shared feature extraction logic for RL observations."""

from typing import Any, Sequence

import numpy as np
from torch import Tensor


def compute_observation(probabilities: Tensor, step_count: int,
                        max_seq_length: int,
                        history: Sequence[Any]) -> np.ndarray:
    """Computes the observation vector from generation state.

    Args:
        probabilities: The probability tensor for the current step.
        step_count: The number of new tokens generated so far.
        max_seq_length: The maximum allowed sequence length (for normalization).
        history: A sequence of recent tokens (ids or strings) to check for
            repetition.

    Returns:
        A numpy array of shape (4,) containing:
        - normalized_position
        - top1_probability
        - entropy
        - repetition_penalty
    """
    if probabilities.ndim == 0:
        probabilities = probabilities.unsqueeze(0)

    normalized_pos = min(step_count / max_seq_length, 1.0)

    top1_prob = float(
        probabilities[0].item()) if len(probabilities) > 0 else 0.0

    non_zero_probs = probabilities[probabilities > 0]
    if len(non_zero_probs) > 0:
        entropy = -(non_zero_probs * non_zero_probs.log()).sum().item()
        max_entropy = np.log(len(probabilities))
        distribution_entropy = (entropy /
                                max_entropy if max_entropy > 0 else 0.0)
    else:
        distribution_entropy = 0.0

    repetition_penalty = 0.0
    if len(history) > 0:
        unique = len(set(history))
        repetition_penalty = 1.0 - (unique / len(history))

    return np.array(
        [
            min(normalized_pos, 1.0),
            min(top1_prob, 1.0),
            min(distribution_entropy, 1.0),
            min(repetition_penalty, 1.0),
        ],
        dtype=np.float32,
    )
