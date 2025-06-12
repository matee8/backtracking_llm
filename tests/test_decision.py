import pytest

from backtracking_llm.decision import ProbabilityThreshold


def test_default_reset_do_not_fail():
    df = ProbabilityThreshold()
    try:
        df.reset()
    except Exception as e:
        pytest.fail(f"Default reset method raised an unexpected error: {e}")
