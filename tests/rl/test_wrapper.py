# pylint: disable=missing-module-docstring, missing-class-docstring, missing-function-docstring

from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest

from backtracking_llm.rl.wrapper import SB3LstmWrapper

# pylint: disable=redefined-outer-name


@pytest.fixture
def mock_env() -> gym.Env:
    env = MagicMock(spec=gym.Env)
    env.step.return_value = (np.array([1.0]), 0.5, False, False, {})
    return env


def test_wrapper_passes_through_normal_step(mock_env: gym.Env):
    wrapper = SB3LstmWrapper(mock_env)
    obs, reward, terminated, truncated, info = wrapper.step(0)

    assert np.array_equal(obs, np.array([1.0]))
    assert reward == 0.5
    assert not terminated
    assert not truncated
    assert 'terminal_observation' not in info
    mock_env.step.assert_called_once_with(0)


def test_wrapper_handles_termination(mock_env: gym.Env):
    mock_env.step.return_value = (np.array([2.0]), -1.0, True, False, {})
    wrapper = SB3LstmWrapper(mock_env)
    obs, reward, terminated, truncated, info = wrapper.step(1)

    assert np.array_equal(obs, np.array([2.0]))
    assert reward == -1.0
    assert terminated
    assert not truncated
    assert 'terminal_observation' not in info


def test_wrapper_handles_truncation(mock_env: gym.Env):
    final_obs = np.array([3.0])
    mock_env.step.return_value = (final_obs, 0.0, False, True, {
        'reason': 'limit'
    })
    wrapper = SB3LstmWrapper(mock_env)
    obs, reward, terminated, truncated, info = wrapper.step(2)

    assert np.array_equal(obs, final_obs)
    assert reward == 0.0
    assert not terminated
    assert truncated
    assert 'terminal_observation' in info
    assert np.array_equal(info['terminal_observation'], final_obs)
    assert info['reason'] == 'limit'


def test_wrapper_handles_termination_and_truncation(mock_env: gym.Env):
    final_obs = np.array([4.0])
    mock_env.step.return_value = (final_obs, 0.0, True, True, {})
    wrapper = SB3LstmWrapper(mock_env)
    _, _, terminated, truncated, info = wrapper.step(3)

    assert terminated
    assert truncated
    assert 'terminal_observation' in info
    assert np.array_equal(info['terminal_observation'], final_obs)
