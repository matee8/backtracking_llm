# pylint: disable=missing-module-docstring

import pytest

from backtracking_llm.rl.config import JudgeConfig
from backtracking_llm.rl.judges import MockJudge


@pytest.fixture
def mock_judge():
    return MockJudge(base_score=5.0, repetition_penalty=2.0)


@pytest.fixture
def sample_judge_config():
    return JudgeConfig(model='gpt-3.5-turbo',
                       max_retries=2,
                       timeout=10.0,
                       api_key='asd')
