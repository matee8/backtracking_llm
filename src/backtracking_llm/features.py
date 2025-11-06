"""Provides utilities for extracting features from the generation state."""

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class FeatureExtractor(Protocol):
    """A protocol for feature extraction classes."""
    feature_dim: int

    def __call__(self, probabilities: Tensor, position: int) -> Tensor:
        """Extracts features from the current generation state.

        Args:
            probabilities: The top-k probability distribution from the model.
            position: The index of the chosen token within the distribution.

        Returns:
            A 1D tensor of engineered features.
        """
        ...


class SimpleFeatureExtractor(FeatureExtractor):
    """A basic feature extractor that provides a small set of common metrics."""
    feature_dim: int = 3

    def __call__(self, probabilities: Tensor, position: int) -> Tensor:
        """Extracts a simple feature set:
        1.  The probability of the chosen token.
        2.  The entropy of the probability distribution.
        3.  The probability margin between the top two tokens.
        """
        chosen_prob = probabilities[position].item()

        non_zero_probs = probabilities[probabilities > 0]
        entropy = -(non_zero_probs * non_zero_probs.log()).sum().item()

        if len(probabilities) > 1:
            top_two = torch.topk(probabilities, 2).values
            margin = (top_two[0] - top_two[1]).item()
        else:
            margin = 1.0

        return torch.tensor([chosen_prob, entropy, margin], dtype=torch.float32)
