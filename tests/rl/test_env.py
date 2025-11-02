# pylint: disable=missing-module-docstring

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Discrete

from backtracking_llm.generation import Generator
from backtracking_llm.rl.env import BacktrackingEnv
from backtracking_llm.rl.features import FeatureExtractor

# pylint: disable=missing-class-docstring
# pylint: disable=broad-exception-caught
# pylint: disable=missing-function-docstring
# pylint: disable=protected-access


class TestBacktrackingEnv:

    @pytest.fixture
    def mock_model_outputs(self):
        outputs = MagicMock()
        outputs.logits = torch.randn(1, 3, 50)
        outputs.past_key_values = MagicMock()
        return outputs

    @pytest.fixture
    def mock_batch_encoding(self):
        mock = MagicMock()
        mock.input_ids = torch.tensor([[10, 20, 30]])
        mock.to.return_value = mock
        mock.__getitem__.side_effect = lambda key: getattr(mock, key)
        return mock

    @pytest.fixture
    def mock_generator(self, mock_batch_encoding, mock_model_outputs):
        generator = MagicMock(spec=Generator)

        generator.tokenizer = MagicMock()
        generator.tokenizer.return_value = mock_batch_encoding

        generator.model = MagicMock()
        generator.model.config = MagicMock()
        generator.model.config.vocab_size = 50
        generator.model.device = 'cpu'

        generator.model.return_value = mock_model_outputs

        return generator

    @pytest.fixture(autouse=True)
    def setup_model_return_value(self, mock_generator, mock_model_outputs):
        mock_generator.model.return_value = mock_model_outputs

    @pytest.fixture
    def mock_feature_extractor(self):
        mock = MagicMock(spec=FeatureExtractor)
        mock.shape = (20,)
        mock.return_value = np.ones(20, dtype=np.float32)
        return mock

    @pytest.fixture
    def env(self, mock_generator, mock_feature_extractor):
        return BacktrackingEnv(
            generator=mock_generator,
            feature_extractor=mock_feature_extractor,
            prompts=['prompt one', 'prompt two'],
            max_backtrack_steps=2,
        )

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

    def test_reset_tokenizes_first_prompt(self, env):
        env.reset()
        env.generator.tokenizer.assert_called_once_with('prompt one',
                                                        return_tensors='pt')

    def test_reset_advances_to_second_prompt(self, env):
        env.reset()
        env.reset()
        env.generator.tokenizer.assert_called_with('prompt two',
                                                   return_tensors='pt')

    def test_reset_cycles_prompts_after_exhaustion(self, env):
        env.reset()
        env.reset()
        env.reset()
        env.generator.tokenizer.assert_called_with('prompt one',
                                                   return_tensors='pt')

    def test_reset_calls_model_with_tokenized_input(self, env):
        tokenized_input_mock = MagicMock()
        tokenized_input_mock.input_ids = torch.tensor([[10, 20]])
        tokenized_input_mock.to.return_value = tokenized_input_mock
        tokenized_input_mock.__getitem__.side_effect = lambda key: getattr(
            tokenized_input_mock, key)
        env.generator.tokenizer.return_value = tokenized_input_mock

        env.reset()

        env.generator.model.assert_called_once_with(
            input_ids=tokenized_input_mock.input_ids, use_cache=True)

    def test_reset_updates_internal_state(self, env, mock_model_outputs):
        env.reset()

        expected_ids = env.generator.tokenizer.return_value.input_ids
        assert torch.equal(env._input_ids, expected_ids)
        assert env._past_key_values == mock_model_outputs.past_key_values

    def test_reset_calls_feature_extractor_with_correct_data(
            self, env, mock_model_outputs):
        env.reset()

        expected_logits = mock_model_outputs.logits[:, -1, :]
        expected_probs = torch.nn.functional.softmax(expected_logits, dim=-1)

        vocab_size = env.generator.model.config.vocab_size
        expected_top_k_logits, _ = torch.topk(expected_logits, vocab_size)
        expected_probs = torch.nn.functional.softmax(expected_top_k_logits,
                                                     dim=-1)

        env.feature_extractor.assert_called_once()
        call_args, _ = env.feature_extractor.call_args
        actual_logits, actual_probs = call_args

        assert torch.allclose(actual_logits, expected_top_k_logits.squeeze(0))
        assert torch.allclose(actual_probs, expected_probs.squeeze(0))

    def test_reset_returns_observation_and_info(self, env):
        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.feature_extractor.shape
        np.testing.assert_array_equal(obs, env.feature_extractor.return_value)
        assert info == {}

    def test_step_raises_not_implemented(self, env):
        with pytest.raises(NotImplementedError):
            dummy_action = env.action_space.sample()
            env.step(dummy_action)
