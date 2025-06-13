# pylint: disable=redefined-outer-name

import logging

import pytest
import torch
from torch import Tensor

from backtracking_llm.decision import ProbabilityThreshold


@pytest.fixture
def base_z() -> Tensor:
    return torch.tensor([-2.0, 2.5, -0.9, -0.4])


@pytest.fixture
def base_p(base_z: Tensor) -> Tensor:
    return torch.softmax(base_z, dim=-1)


def test_default_reset_do_not_fail():
    df = ProbabilityThreshold()
    try:
        df.reset()
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
            df = ProbabilityThreshold(p_min, backtrack_k)
            assert df.p_min == p_min
            assert df.backtrack_k == backtrack_k

    def test_call_triggers_backtrack(self, base_z, base_p):
        df = ProbabilityThreshold(0.05, 3)
        result = df(z=base_z, p=base_p, i_chosen=0, y_hat=123)
        assert result == 3

    def test_call_no_backtrack(self, base_z, base_p):
        df = ProbabilityThreshold(0.05, 3)
        result = df(z=base_z, p=base_p, i_chosen=1, y_hat=456)
        assert result == 0

    def test_call_out_of_bounds_index(self, base_p, base_z, caplog):
        df = ProbabilityThreshold()
        with caplog.at_level(logging.WARNING):
            result = df(base_z, base_p, i_chosen=99, y_hat=789)
        assert result == 0
        assert "out of bounds" in caplog.text
