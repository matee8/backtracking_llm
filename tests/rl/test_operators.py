# pylint: disable=missing-module-docstring

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker

from backtracking_llm.rl.operators import RlPolicyOperator

# pylint: disable=missing-class-docstring
# pylint: disable=unused-argument
# pylint: disable=protected-access


class DummyBacktrackEnv(Env):

    def __init__(self, max_backtrack: int = 5):
        super().__init__()
        self.observation_space = Box(low=0.0,
                                     high=1.0,
                                     shape=(4,),
                                     dtype=np.float32)
        self.action_space = Discrete(max_backtrack + 1)
        self.step_count = 0

    def reset(self, **kwargs):
        self.step_count = 0
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        obs = np.array(
            [
                min(self.step_count / 100.0, 1.0),
                0.5,
                0.3,
                0.0,
            ],
            dtype=np.float32,
        )
        reward = 1.0 if action == 0 else -0.1 * action
        terminated = self.step_count >= 10
        truncated = False
        return obs, reward, terminated, truncated, {}


def test_rl_policy_operator_initialization():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = DummyBacktrackEnv()
        env_checker.check_env(env)

        model = PPO('MlpPolicy', env, n_steps=16, verbose=0)
        model.learn(total_timesteps=32)

        policy_path = Path(tmpdir) / 'test_policy'
        model.save(policy_path)

        operator = RlPolicyOperator(policy_path.with_suffix('.zip'))
        assert operator.action_space_size == 6

        with pytest.raises(FileNotFoundError):
            RlPolicyOperator(Path(tmpdir) / 'nonexistent.zip')


def test_rl_policy_operator_call():

    with tempfile.TemporaryDirectory() as tmpdir:
        env = DummyBacktrackEnv(max_backtrack=3)
        model = PPO('MlpPolicy', env, n_steps=16, verbose=0)
        model.learn(total_timesteps=32)
        policy_path = Path(tmpdir) / 'policy.zip'
        model.save(policy_path)

        operator = RlPolicyOperator(policy_path)

        vocab_size = 100
        logits = torch.randn(vocab_size)
        probabilities = torch.softmax(logits, dim=-1)
        position = 42
        token = 'test_token'

        backtrack_count = operator(logits, probabilities, position, token)
        assert isinstance(backtrack_count, int)
        assert 0 <= backtrack_count <= 3

        backtrack_count_min = operator(logits,
                                       probabilities,
                                       position=0,
                                       token='first')
        backtrack_count_max = operator(logits,
                                       probabilities,
                                       position=vocab_size - 1,
                                       token='last')
        assert isinstance(backtrack_count_min, int)
        assert isinstance(backtrack_count_max, int)


def test_rl_policy_operator_backtrack():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = DummyBacktrackEnv()
        model = PPO('MlpPolicy', env, n_steps=16, verbose=0)
        model.learn(total_timesteps=32)
        policy_path = Path(tmpdir) / 'policy.zip'
        model.save(policy_path)

        operator = RlPolicyOperator(policy_path)

        operator.backtrack(0)
        operator.backtrack(5)
        operator.backtrack(-1)
