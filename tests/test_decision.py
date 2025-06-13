# pylint: disable=redefined-outer-name

import logging

import pytest
import torch
from torch import Tensor

from backtracking_llm.decision import (
    EntropyThreshold,
    ProbabilityMargin,
    ProbabilityThreshold,
)


@pytest.fixture
def base_z() -> Tensor:
    return torch.tensor([-2.0, 2.5, -0.9, -0.4])


@pytest.fixture
def base_p(base_z: Tensor) -> Tensor:
    return torch.softmax(base_z, dim=-1)


def test_default_reset_do_not_fail():
    delta = ProbabilityThreshold()
    try:
        delta.reset()
    except Exception as e:
        pytest.fail(f"Default reset method raised an unexpected error: {e}")


class TestProbabilityThreshold:

    @pytest.mark.parametrize(
        "p_min, backtrack_k, should_raise",
        [
            (0.1, 1, False),
            (0.99, 10, False),
            (1.0, 1, True),
            (0.0, 1, True),
            (0.1, 0, True),
        ],
    )
    def test_init(self, p_min, backtrack_k, should_raise):
        if should_raise:
            with pytest.raises(ValueError):
                ProbabilityThreshold(p_min, backtrack_k)
        else:
            delta = ProbabilityThreshold(p_min, backtrack_k)
            assert delta.p_min == p_min
            assert delta.backtrack_k == backtrack_k

    def test_call_triggers_backtrack(self, base_z, base_p):
        delta = ProbabilityThreshold(0.05, 3)
        result = delta(z=base_z, p=base_p, i_chosen=0, y_hat=123)
        assert result == 3

    def test_call_no_backtrack(self, base_z, base_p):
        delta = ProbabilityThreshold(0.05, 3)
        result = delta(z=base_z, p=base_p, i_chosen=1, y_hat=456)
        assert result == 0

    def test_call_out_of_bounds_index(self, base_p, base_z, caplog):
        delta = ProbabilityThreshold()
        with caplog.at_level(logging.WARNING):
            result = delta(base_z, base_p, i_chosen=99, y_hat=789)
        assert result == 0
        assert "out of bounds" in caplog.text


class TestEntropyThreshold:

    @pytest.mark.parametrize(
        "h_max, backtrack_k, should_raise",
        [
            (2.5, 2, False),
            (0.0, 1, True),
            (2.5, 0, True),
        ],
    )
    def test_init(self, h_max, backtrack_k, should_raise):
        if should_raise:
            with pytest.raises(ValueError):
                EntropyThreshold(h_max=h_max, backtrack_k=backtrack_k)
        else:
            delta = EntropyThreshold(h_max=h_max, backtrack_k=backtrack_k)
            assert delta.h_max == h_max
            assert delta.backtrack_k == backtrack_k

    def test_call_triggers_backtrack(self, base_z):
        p_high_h = torch.tensor([0.25, 0.25, 0.25, 0.25])
        delta = EntropyThreshold(1.0, 2)
        result = delta(z=base_z, p=p_high_h, i_chosen=0, y_hat=123)
        assert result == 2

    def test_call_no_backtrack(self, base_z, base_p):
        delta = EntropyThreshold(1.0, 2)
        result = delta(z=base_z, p=base_p, i_chosen=1, y_hat=456)
        assert result == 0


class TestProbabilityMargin:

    @pytest.mark.parametrize(
        "m_min, backtrack_k, should_raise",
        [
            (0.5, 2, False),
            (-0.5, 1, True),
            (0.5, -1, True),
        ],
    )
    def test_init(self, m_min, backtrack_k, should_raise):
        if should_raise:
            with pytest.raises(ValueError):
                ProbabilityMargin(m_min, backtrack_k)
        else:
            delta = ProbabilityMargin(m_min, backtrack_k)
            assert delta.m_min == m_min
            assert delta.backtrack_k == backtrack_k

    def test_call_triggers_backtrack(self, base_z):
        p_with_close_top_k = torch.tensor([0.4, 0.38, 0.12, 0.1])
        delta = ProbabilityMargin(0.05, 1)
        result = delta(z=base_z, p=p_with_close_top_k, i_chosen=0, y_hat=123)
        assert result == 1

    def test_call_no_backtrack(self, base_z, base_p):
        delta = ProbabilityMargin(0.1, 1)
        result = delta(z=base_z, p=base_p, i_chosen=1, y_hat=456)
        assert result == 0
