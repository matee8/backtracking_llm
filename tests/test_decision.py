"""Unit tests for the decision function operators."""

import pytest
import torch
from torch import Tensor

from backtracking_llm.decision import Never, ProbabilityThreshold

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
    op = Never()
    result = op(base_logits, base_probabilities, base_position, base_token)

    assert result is 0


def test_probability_threshold_triggers_backtrack_when_below(
        base_logits: Tensor, base_probabilities: Tensor, base_position: int,
        base_token: str) -> None:
    """
    Tests that backtracking is triggered when the token probability is
    below the minimum threshold.
    """
    op = ProbabilityThreshold(min_probability=0.5, backtrack_count=2)

    result = op(base_logits, base_probabilities, base_position, base_token)

    assert result == 2


def test_probability_threshold_does_not_backtrack_when_above(
        base_logits: Tensor, base_probabilities: Tensor, base_position: int,
        base_token: str) -> None:
    """
    Tests that no backtracking occurs when the token probability is
    above the minimum threshold.
    """
    op = ProbabilityThreshold(min_probability=0.005, backtrack_count=2)

    result = op(base_logits, base_probabilities, base_position, base_token)

    assert result == 0


def test_probability_threshold_does_not_backtrack_when_equal(
        base_logits: Tensor, base_probabilities: Tensor, base_position: int,
        base_token: str) -> None:
    """
    Tests that no backtracking occurs when the token probability is
    exactly equal to the minimum threshold.
    """
    op = ProbabilityThreshold(min_probability=0.0101, backtrack_count=2)

    result = op(base_logits, base_probabilities, base_position, base_token)

    assert result == 0


@pytest.mark.parametrize("invalid_prob", [0.0, 1.0, -0.1, 1.1])
def test_init_raises_error_for_invalid_probability(invalid_prob) -> None:
    """
    Tests that the constructor raises a ValueError for min_probability
    values that are not strictly between 0.0 and 1.0.
    """
    with pytest.raises(ValueError, match="`min_probability` must be between"):
        ProbabilityThreshold(min_probability=invalid_prob)


@pytest.mark.parametrize("invalid_count", [0, -1, -10])
def test_init_raises_error_for_invalid_backtrack_count(invalid_count) -> None:
    """
    Tests that the constructor raises a ValueError for a non-positive
    backtrack_count.
    """
    with pytest.raises(ValueError, match="`backtrack_count` must be positive"):
        ProbabilityThreshold(backtrack_count=invalid_count)


def test_call_handles_out_of_bounds_position(caplog, base_logits: Tensor,
                                             base_probabilities: Tensor,
                                             base_token: str) -> None:
    """
    Tests that a warning is logged and returns 0 for an out-of-bounds position.
    """
    op = ProbabilityThreshold()
    invalid_position = 5

    result = op(base_logits, base_probabilities, invalid_position, base_token)

    assert result == 0
    assert "out of bounds" in caplog.text
