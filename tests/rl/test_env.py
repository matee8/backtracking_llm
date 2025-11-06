# pylint: disable=missing-module-docstring

from typing import Dict, Tuple, Union

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete

from backtracking_llm.rl.env import BacktrackingEnv

# pylint: disable=missing-class-docstring
# pylint: disable=broad-exception-caught


class TestBacktrackingEnv:

    @pytest.fixture
    def env_params(self) -> Dict[str, Union[int, Tuple[int]]]:
        return {'num_actions': 3, 'observation_shape': (10,)}

    def test_initialization(self, env_params):
        env = BacktrackingEnv(**env_params)
        assert env is not None

        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == env_params['num_actions']

        assert isinstance(env.observation_space, Box)
        assert env.observation_space.shape == env_params['observation_shape']
        assert env.observation_space.dtype == np.float32

    def test_reset_raises_not_implemented(self, env_params):
        env = BacktrackingEnv(**env_params)
        with pytest.raises(NotImplementedError,
                           match='The `reset` method must be implemented'):
            env.reset(options={})

    def test_step_raises_not_implemented(self, env_params):
        env = BacktrackingEnv(**env_params)
        with pytest.raises(NotImplementedError,
                           match='The `step` method must be implemented'):
            env.step(0)

    def test_render_and_close_are_noop(self, env_params):
        env = BacktrackingEnv(**env_params)
        try:
            env.render()
            env.close()
        except Exception as e:
            pytest.fail(f'render() or close() raised an exception: {e}')
