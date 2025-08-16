"""Unit tests for the decision function operators."""

import pytest
import torch
from torch import Tensor

from backtracking_llm.decision import Never

# pylint: disable=redefined-outer-name


@pytest.fixture
def base_logits() -> Tensor:
    return torch.tensor([-2.0, 2.5, -0.9, -0.4])


@pytest.fixture
def base_probabilities(base_logits: Tensor) -> Tensor:
    return torch.softmax(base_logits, dim=-1)


@pytest.fixture
def base_position() -> int:
    return 0


@pytest.fixture
def base_token() -> str:
    return "hello"


def test_never_operator_always_returns_0(base_logits: Tensor,
                                         base_probabilities: Tensor,
                                         base_position: int,
                                         base_token: str) -> None:
    """Tests that the Never operator consistently returns 0."""
    never_operator = Never()
    result = never_operator(base_logits, base_probabilities, base_position,
                            base_token)

    assert result is 0
