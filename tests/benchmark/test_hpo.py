# pylint: disable=missing-module-docstring

from unittest.mock import MagicMock, patch

import pytest
from optuna import Trial

from backtracking_llm.benchmark.config import (EvaluationConfig,
                                               GenerationConfig, HPOConfig)
from backtracking_llm.benchmark.hpo import HyperparameterOptimizer
from backtracking_llm.decision import ProbabilityThreshold
from backtracking_llm.generation import Generator

# pylint: disable=protected-access
# pylint: disable=unused-argument


@patch('optuna.create_study')
@patch('backtracking_llm.benchmark.hpo.Evaluator')
class TestHyperparameterOptimizer:
    """Tests the HyperparameterOptimizer class."""

    def test_initialization_and_valid_operator(self, mock_evaluator_cls,
                                               mock_create_study):
        hpo_config = HPOConfig(n_trials=1, search_space={})
        eval_config = EvaluationConfig(tasks=['task'])
        optimizer = HyperparameterOptimizer(hpo_config, eval_config,
                                            MagicMock(), MagicMock(),
                                            'ProbabilityThreshold')

        assert optimizer._operator_cls is ProbabilityThreshold
        mock_evaluator_cls.assert_called_once_with(eval_config)

    def test_initialization_invalid_operator_raises_error(
            self, mock_evaluator_cls, mock_create_study):
        with pytest.raises(ValueError,
                           match="'InvalidOperator' is not a valid"):
            HyperparameterOptimizer(MagicMock(), MagicMock(), MagicMock(),
                                    MagicMock(), 'InvalidOperator')

    def test_optimize_runs_study(self, mock_evaluator_cls, mock_create_study):
        mock_study = MagicMock()
        mock_create_study.return_value = mock_study
        hpo_config = HPOConfig(n_trials=10, search_space={})
        optimizer = HyperparameterOptimizer(hpo_config,
                                            EvaluationConfig(tasks=['task']),
                                            MagicMock(), MagicMock(),
                                            'ProbabilityThreshold')

        result_study = optimizer.optimize()

        mock_create_study.assert_called_once_with(direction='maximize')
        mock_study.optimize.assert_called_once_with(optimizer._objective,
                                                    n_trials=10,
                                                    show_progress_bar=True)
        assert result_study is mock_study

    def test_objective_function_success_case(self, mock_evaluator_cls,
                                             mock_create_study):
        mock_evaluator_instance = mock_evaluator_cls.return_value
        mock_evaluator_instance.run.return_value = {'results': {}}
        mock_evaluator_cls.extract_primary_score.return_value = 0.85

        mock_trial = MagicMock(spec=Trial)
        mock_trial.suggest_float.return_value = 0.1
        mock_trial.number = 0

        search_space = {'min_probability': [0.05, 0.5]}
        hpo_config = HPOConfig(n_trials=1, search_space=search_space)
        eval_config = EvaluationConfig(tasks=['arc_easy'])
        gen_config = MagicMock(spec=GenerationConfig)
        generator = MagicMock(spec=Generator)
        optimizer = HyperparameterOptimizer(hpo_config, eval_config, gen_config,
                                            generator, 'ProbabilityThreshold')

        score = optimizer._objective(mock_trial)

        assert score == 0.85
        mock_trial.suggest_float.assert_called_once_with(
            'min_probability', 0.05, 0.5)
        mock_evaluator_instance.run.assert_called_once()
        call_args, _ = mock_evaluator_instance.run.call_args
        assert call_args[0] is generator
        assert call_args[1] is gen_config
        assert isinstance(call_args[2], ProbabilityThreshold)
        assert call_args[2].min_probability == 0.1
        mock_evaluator_cls.extract_primary_score.assert_called_once_with(
            {'results': {}}, 'arc_easy')

    def test_objective_function_handles_exception(self, mock_evaluator_cls,
                                                  mock_create_study):
        mock_evaluator_instance = mock_evaluator_cls.return_value
        mock_evaluator_instance.run.side_effect = RuntimeError('Test Error')
        mock_trial = MagicMock(spec=Trial)
        mock_trial.number = 0

        hpo_config = HPOConfig(n_trials=1, search_space={})
        optimizer = HyperparameterOptimizer(hpo_config,
                                            EvaluationConfig(tasks=['task']),
                                            MagicMock(), MagicMock(),
                                            'ProbabilityThreshold')

        score = optimizer._objective(mock_trial)
        assert score == -1.0

    def test_sample_params_handles_integers(self, mock_evaluator_cls,
                                            mock_create_study):
        mock_trial = MagicMock(spec=Trial)
        search_space = {'backtrack_count': [1, 10]}
        hpo_config = HPOConfig(n_trials=1, search_space=search_space)
        optimizer = HyperparameterOptimizer(hpo_config, MagicMock(),
                                            MagicMock(), MagicMock(),
                                            'ProbabilityThreshold')

        optimizer._sample_params(mock_trial)

        mock_trial.suggest_int.assert_called_once_with('backtrack_count', 1, 10)
        mock_trial.suggest_float.assert_not_called()

    def test_sample_params_raises_type_error_for_unsupported_type(
            self, mock_evaluator_cls, mock_create_study):
        mock_trial = MagicMock(spec=Trial)
        search_space = {'bad_param': ['low', 'high']}
        hpo_config = HPOConfig(n_trials=1, search_space=search_space)
        optimizer = HyperparameterOptimizer(hpo_config, MagicMock(),
                                            MagicMock(), MagicMock(),
                                            'ProbabilityThreshold')

        with pytest.raises(
                TypeError,
                match="Unsupported type for hyperparameter 'bad_param'"):
            optimizer._sample_params(mock_trial)

    def test_get_operator_class_raises_error_for_non_class_attribute(
            self, mock_evaluator_cls, mock_create_study):
        with patch('backtracking_llm.decision.logger', 'not a class'):
            with pytest.raises(
                    ValueError,
                    match="'logger' is not a valid Operator class name"):
                HyperparameterOptimizer._get_operator_class('logger')
