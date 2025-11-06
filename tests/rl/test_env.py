# pylint: disable=missing-module-docstring

from unittest.mock import MagicMock
from typing import Dict, Tuple, Union

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete

from backtracking_llm.generation import Generator
from backtracking_llm.rl.env import BacktrackingEnv
from backtracking_llm.rl.features import FeatureExtractor

# pylint: disable=missing-class-docstring
# pylint: disable=broad-exception-caught


class TestBacktrackingEnv:

    @pytest.fixture
    def mock_generator(self):
        return MagicMock(spec=Generator)

    @pytest.fixture
    def mock_feature_extractor(self):
        mock = MagicMock(spec=FeatureExtractor)
        mock.shape = (20,)
        return mock

    def test_initialization(self, mock_generator, mock_feature_extractor):
        prompts = ['Once upon a time', 'The future is']
        max_backtrack = 3

        env = BacktrackingEnv(
            generator=mock_generator,
            feature_extractor=mock_feature_extractor,
            prompts=prompts,
            max_backtrack_steps=max_backtrack,
        )

        assert env.generator == mock_generator
        assert env.prompts == prompts

        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == max_backtrack + 1

        assert isinstance(env.observation_space, Box)
        assert env.observation_space.shape == mock_feature_extractor.shape
        assert env.observation_space.dtype == np.float32

    def test_initialization_errors(self, mock_generator,
                                   mock_feature_extractor):
        with pytest.raises(ValueError, match='prompts list cannot be empty'):
            BacktrackingEnv(mock_generator, mock_feature_extractor, [])

        with pytest.raises(ValueError,
                           match='max_backtrack_steps cannot be negative'):
            BacktrackingEnv(mock_generator, mock_feature_extractor, ['a'], -1)

    @pytest.fixture
    def env(self, mock_generator, mock_feature_extractor):
        return BacktrackingEnv(
            generator=mock_generator,
            feature_extractor=mock_feature_extractor,
            prompts=['A test prompt'],
            max_backtrack_steps=2,
        )

    def test_reset_raises_not_implemented(self, env):
        with pytest.raises(NotImplementedError):
            env.reset()

    def test_step_raises_not_implemented(self, env):
        with pytest.raises(NotImplementedError):
            dummy_action = env.action_space.sample()
            env.step(dummy_action)
