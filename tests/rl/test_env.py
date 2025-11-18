# pylint: disable=missing-module-docstring

import pytest
import numpy as np
import torch
from unittest.mock import MagicMock

from backtracking_llm.generation import GenerationSession
from backtracking_llm.rl.env import BacktrackingEnv
from backtracking_llm.rl.rewards import RewardShaper
from backtracking_llm.rl.judges import MockJudge
from backtracking_llm.rl.config import EnvConfig, ShapingConfig

# pylint: disable=protected-access
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument


@pytest.fixture
def mock_session_factory():

    def factory():
        model = MagicMock()
        model.device = 'cpu'
        model.config.vocab_size = 50

        def forward(input_ids, past_key_values=None, use_cache=True):
            batch_size, seq_len = input_ids.shape
            logits = torch.randn(batch_size, seq_len, 50)
            cache = MagicMock()
            cache.get_seq_length.return_value = seq_len
            cache.crop = MagicMock()
            return MagicMock(logits=logits, past_key_values=cache)

        model.side_effect = forward

        tokenizer = MagicMock()
        tokenizer.return_value.input_ids = torch.tensor([[1, 2, 3]])
        tokenizer.return_value.to.return_value = tokenizer.return_value
        tokenizer.decode.return_value = 'test'
        tokenizer.eos_token_id = 49

        return GenerationSession(model, tokenizer, prompt='hi')

    return factory


@pytest.fixture
def mock_judge():
    return MockJudge()


@pytest.fixture
def env_config():
    return EnvConfig(max_backtrack=3, max_seq_length=10)


@pytest.fixture
def mock_shaper():
    shaper = MagicMock(spec=RewardShaper)
    shaper.return_value = 0.1
    return shaper


@pytest.fixture
def env(mock_session_factory, mock_judge, env_config):
    shaper = RewardShaper(ShapingConfig())
    return BacktrackingEnv(mock_session_factory, mock_judge, shaper, env_config)


def test_env_initialization(mock_session_factory, mock_judge, mock_shaper,
                            env_config):
    env = BacktrackingEnv(mock_session_factory, mock_judge, mock_shaper,
                          env_config)
    assert env.action_space.n == 4
    assert env.observation_space.shape == (4,)


def test_env_reset(env):
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)
    assert isinstance(info, dict)


def test_env_step_no_action(env):
    env.reset()

    obs, _, terminated, truncated, info = env.step(0)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_env_step_with_backtrack(env):
    env.reset()

    env.step(0)

    obs, _, _, _, _ = env.step(1)

    assert obs.shape == (4,)


def test_env_episode_termination(env):
    env.config.max_seq_length = 2

    env.reset()

    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        _, reward, terminated, truncated, _ = env.step(0)
        total_reward += reward

    assert total_reward != 0
    assert terminated or truncated


def test_env_render(env):
    env.reset()
    env.step(0)

    text = env.render()
    assert isinstance(text, str)


def test_env_step_calls_shaper(mock_session_factory, mock_judge, mock_shaper,
                               env_config):
    env = BacktrackingEnv(mock_session_factory, mock_judge, mock_shaper,
                          env_config)
    env.reset()
    env.step(1)
    mock_shaper.assert_called_once()
    args, _ = mock_shaper.call_args
    assert args[0] == 1
    assert isinstance(args[1], np.ndarray)


def test_env_intermediate_reward_is_from_shaper(mock_session_factory,
                                                mock_judge, mock_shaper,
                                                env_config):
    env = BacktrackingEnv(mock_session_factory, mock_judge, mock_shaper,
                          env_config)
    env.reset()
    _, reward, terminated, truncated, _ = env.step(0)
    assert not terminated and not truncated
    assert reward == 0.1


def test_env_final_reward_is_sum_of_shaper_and_judge(mock_session_factory,
                                                     mock_judge, mock_shaper,
                                                     env_config):
    env_config.max_seq_length = 1
    env = BacktrackingEnv(mock_session_factory, mock_judge, mock_shaper,
                          env_config)
    env.reset()
    mock_judge.score = MagicMock(return_value=3.5)
    _, reward, _, _, _ = env.step(0)
    assert reward == pytest.approx(0.1 + 3.5)
    mock_judge.score.assert_called_once_with('test')
