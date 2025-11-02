"""Defines classes for extracting features from the model's state."""

import typing
from typing import Protocol, Tuple

import numpy as np
import torch
from torch import Tensor


@typing.runtime_checkable
class FeatureExtractor(Protocol):
    """A protocol for feature extractors.

    An object implementing this protocol must provide a `shape` property and
    a `__call__` method to convert model outputs into a numpy array.
    """

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the numpy array returned by this extractor."""
        ...

    def __call__(self, logits: Tensor, probabilities: Tensor) -> np.ndarray:
        """Extracts features from the current model output.

        Args:
            logits: The raw logits from the model for the current token.
            probabilities: The probabilities for the current token.

        Returns:
            A numpy array representing the observation for the agent.
        """
        ...


class SimpleFeatureExtractor:
    """A simple feature extractor that concatenates top-k logits and probs.

    This class structurally conforms to the `FeatureExtractor` protocol.
    """

    def __init__(self, top_k: int = 10) -> None:
        """Initializes the SimpleFeatureExtractor.

        Args:
            top_k: The number of top logits/probabilities to use as features.
        """
        if top_k < 1:
            raise ValueError('top_k must be a positive integer.')

        self._top_k = top_k
        self._shape = (top_k * 2,)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns the shape of the feature vector."""
        return self._shape

    def __call__(self, logits: Tensor, probabilities: Tensor) -> np.ndarray:
        """Extracts top-k logits and probabilities and concatenates them."""
        logits_cpu = logits.cpu()
        probs_cpu = probabilities.cpu()

        top_k_logits, _ = torch.topk(logits_cpu, self._top_k)
        top_k_probs, _ = torch.topk(probs_cpu, self._top_k)

        features = torch.cat((top_k_logits, top_k_probs)).numpy()
        return features
