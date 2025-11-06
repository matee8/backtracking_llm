# pylint: disable=missing-module-docstring

from unittest.mock import MagicMock, patch

import pytest

from backtracking_llm.agents import RLAgentOperator
from backtracking_llm.benchmark.config import (EvaluationConfig,
                                               GenerationConfig)
from backtracking_llm.benchmark.rl_env import RLEnvironment

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
# pylint: disable=missing-class-docstring
# pylint: disable=broad-exception-caught


@pytest.fixture
def mock_generator() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_agent() -> MagicMock:
    return MagicMock(spec=RLAgentOperator)


@pytest.fixture
def valid_eval_config() -> EvaluationConfig:
    return EvaluationConfig(tasks=['test_task'], limit=10)


@pytest.fixture
def valid_gen_config() -> GenerationConfig:
    return GenerationConfig(max_new_tokens=10)


class TestRLEnvironment:

    def test_initialization_success(self, mock_generator: MagicMock,
                                    valid_eval_config: EvaluationConfig,
                                    valid_gen_config: GenerationConfig):
        try:
            env = RLEnvironment(mock_generator, valid_eval_config,
                                valid_gen_config)
            assert env is not None
        except Exception as e:
            pytest.fail(
                f'RLEnvironment initialization failed unexpectedly: {e}')

    def test_initialization_raises_error_on_no_tasks(
            self, mock_generator: MagicMock,
            valid_gen_config: GenerationConfig):
        invalid_eval_config = EvaluationConfig(tasks=[])
        with pytest.raises(ValueError, match='must specify at least one task'):
            RLEnvironment(mock_generator, invalid_eval_config, valid_gen_config)

    @patch('backtracking_llm.benchmark.rl_env.Evaluator')
    def test_run_episode_success(self, mock_evaluator_cls: MagicMock,
                                 mock_generator: MagicMock,
                                 valid_eval_config: EvaluationConfig,
                                 valid_gen_config: GenerationConfig,
                                 mock_agent: MagicMock):
        mock_evaluator_instance = mock_evaluator_cls.return_value
        mock_evaluator_instance.run.return_value = {
            'results': {
                'test_task': {}
            }
        }
        mock_evaluator_cls.extract_primary_score.return_value = 0.75

        env = RLEnvironment(mock_generator, valid_eval_config, valid_gen_config)
        reward = env.run_episode(mock_agent)

        assert reward == 0.75
        mock_evaluator_instance.run.assert_called_once_with(
            mock_generator, valid_gen_config, mock_agent)
        mock_evaluator_cls.extract_primary_score.assert_called_once()

    @patch('backtracking_llm.benchmark.rl_env.Evaluator')
    def test_run_episode_failure(self, mock_evaluator_cls: MagicMock,
                                 mock_generator: MagicMock,
                                 valid_eval_config: EvaluationConfig,
                                 valid_gen_config: GenerationConfig,
                                 mock_agent: MagicMock):
        mock_evaluator_instance = mock_evaluator_cls.return_value
        mock_evaluator_instance.run.return_value = {}

        env = RLEnvironment(mock_generator, valid_eval_config, valid_gen_config)
        reward = env.run_episode(mock_agent)

        assert reward == -1.0

    @patch('backtracking_llm.benchmark.rl_env.Evaluator')
    def test_run_episode_handles_extraction_error(
            self, mock_evaluator_cls: MagicMock, mock_generator: MagicMock,
            valid_eval_config: EvaluationConfig,
            valid_gen_config: GenerationConfig, mock_agent: MagicMock):
        mock_evaluator_instance = mock_evaluator_cls.return_value
        mock_evaluator_instance.run.return_value = {
            'results': {
                'test_task': {}
            }
        }
        mock_evaluator_cls.extract_primary_score.side_effect = KeyError(
            'Metric not found')

        env = RLEnvironment(mock_generator, valid_eval_config, valid_gen_config)
        reward = env.run_episode(mock_agent)

        assert reward == -1.0
