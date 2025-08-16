# pylint: disable=missing-module-docstring

import pytest
import torch
from torch import Tensor

from backtracking_llm.decision import (EntropyThreshold, Never,
                                       ProbabilityMargin, ProbabilityThreshold)

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


@pytest.fixture
def high_entropy_probabilities() -> Tensor:
    """Returns a uniform (high entropy) probability distribution."""
    return torch.tensor([0.25, 0.25, 0.25, 0.25])


@pytest.fixture
def zero_entropy_probabilities() -> Tensor:
    """Returns a certain (zero entropy) probability distribution."""
    return torch.tensor([1.0, 0.0, 0.0, 0.0])


@pytest.fixture
def low_margin_probabilities() -> Tensor:
    """Returns a distribution where the top 2 probabilities are very close."""
    return torch.tensor([0.35, 0.32, 0.18, 0.15])


def test_never_operator_always_returns_0(base_logits: Tensor,
                                         base_probabilities: Tensor,
                                         base_position: int,
                                         base_token: str) -> None:
    op = Never()
    result = op(base_logits, base_probabilities, base_position, base_token)

    assert result is 0


def test_probability_threshold_triggers_backtrack_when_below(
        base_logits: Tensor, base_probabilities: Tensor, base_position: int,
        base_token: str) -> None:
    op = ProbabilityThreshold(min_probability=0.5, backtrack_count=2)

    result = op(base_logits, base_probabilities, base_position, base_token)

    assert result == 2


def test_probability_threshold_does_not_backtrack_when_above(
        base_logits: Tensor, base_probabilities: Tensor, base_position: int,
        base_token: str) -> None:
    op = ProbabilityThreshold(min_probability=0.005, backtrack_count=2)

    result = op(base_logits, base_probabilities, base_position, base_token)

    assert result == 0


def test_probability_threshold_does_not_backtrack_when_equal(
        base_logits: Tensor, base_probabilities: Tensor, base_position: int,
        base_token: str) -> None:
    op = ProbabilityThreshold(min_probability=0.0101, backtrack_count=2)

    result = op(base_logits, base_probabilities, base_position, base_token)

    assert result == 0


@pytest.mark.parametrize("invalid_prob", [0.0, 1.0, -0.1, 1.1])
def test_init_raises_error_for_invalid_probability(invalid_prob) -> None:
    with pytest.raises(ValueError, match="`min_probability` must be between"):
        ProbabilityThreshold(min_probability=invalid_prob)


@pytest.mark.parametrize("invalid_count", [0, -1, -10])
def test_init_raises_error_for_invalid_backtrack_count(invalid_count) -> None:
    with pytest.raises(ValueError, match="`backtrack_count` must be positive"):
        ProbabilityThreshold(backtrack_count=invalid_count)


def test_call_handles_out_of_bounds_position(caplog, base_logits: Tensor,
                                             base_probabilities: Tensor,
                                             base_token: str) -> None:
    op = ProbabilityThreshold()
    invalid_position = 5

    result = op(base_logits, base_probabilities, invalid_position, base_token)

    assert result == 0
    assert "out of bounds" in caplog.text


def test_entropy_threshold_triggers_backtrack_on_high_entropy(
        base_logits: Tensor, high_entropy_probabilities: Tensor,
        base_position: int, base_token: str) -> None:
    op = EntropyThreshold(max_entropy=1.0, backtrack_count=3)

    result = op(base_logits, high_entropy_probabilities, base_position,
                base_token)

    assert result == 3


def test_entropy_threshold_does_not_backtrack_on_low_entropy(
        base_logits: Tensor, base_probabilities: Tensor, base_position: int,
        base_token: str) -> None:
    op = EntropyThreshold(max_entropy=1.0, backtrack_count=2)

    result = op(base_logits, base_probabilities, base_position, base_token)

    assert result == 0


def test_entropy_threshold_does_not_backtrack_when_equal(
        base_logits: Tensor, base_probabilities: Tensor, base_position: int,
        base_token: str) -> None:
    entropy_val = -(base_probabilities * base_probabilities.log()).sum()
    op = EntropyThreshold(max_entropy=entropy_val.item(), backtrack_count=2)

    result = op(base_logits, base_probabilities, base_position, base_token)

    assert result == 0


def test_entropy_threshold_handles_zero_probability_correctly(
        base_logits: Tensor, zero_entropy_probabilities: Tensor,
        base_position: int, base_token: str) -> None:
    op = EntropyThreshold(max_entropy=0.01)

    result = op(base_logits, zero_entropy_probabilities, base_position,
                base_token)

    assert result == 0


@pytest.mark.parametrize("invalid_entropy", [-0.1, -100.0])
def test_entropy_init_raises_error_for_negative_entropy(
        invalid_entropy: float) -> None:
    with pytest.raises(ValueError, match="`max_entropy` must be non-negative"):
        EntropyThreshold(max_entropy=invalid_entropy)


@pytest.mark.parametrize("invalid_count", [0, -1, -10])
def test_entropy_init_raises_error_for_invalid_backtrack_count(
        invalid_count: int) -> None:
    with pytest.raises(ValueError, match="`backtrack_count` must be positive"):
        EntropyThreshold(backtrack_count=invalid_count)


def test_probability_margin_triggers_backtrack_on_low_margin(
        base_logits: Tensor, low_margin_probabilities: Tensor,
        base_position: int, base_token: str) -> None:
    op = ProbabilityMargin(min_margin=0.1, backtrack_count=2)

    result = op(base_logits, low_margin_probabilities, base_position,
                base_token)

    assert result == 2


def test_probability_margin_does_not_backtrack_when_equal(
        base_logits: Tensor, low_margin_probabilities: Tensor,
        base_position: int, base_token: str) -> None:
    op = ProbabilityMargin(min_margin=0.03)

    result = op(base_logits, low_margin_probabilities, base_position,
                base_token)

    assert result == 0


@pytest.mark.parametrize("invalid_margin", [-0.1, 1.1, -10.0])
def test_prob_margin_init_raises_error_for_invalid_margin(
        invalid_margin: float) -> None:
    with pytest.raises(ValueError, match="`min_margin` must be between"):
        ProbabilityMargin(min_margin=invalid_margin)


@pytest.mark.parametrize("invalid_count", [0, -1, -10])
def test_prob_margin_init_raises_error_for_invalid_backtrack_count(
        invalid_count: int) -> None:
    with pytest.raises(ValueError, match="`backtrack_count` must be positive"):
        ProbabilityMargin(backtrack_count=invalid_count)


def test_prob_margin_handles_vocab_size_less_than_2(caplog, base_logits: Tensor,
                                                    base_position: int,
                                                    base_token: str) -> None:
    op = ProbabilityMargin()
    small_probs = torch.tensor([1.0])

    result = op(base_logits, small_probs, base_position, base_token)

    assert result == 0
    assert "fewer than 2 elements" in caplog.text
