# pylint: disable=missing-module-docstring

from unittest.mock import MagicMock, patch

import pytest

from backtracking_llm.benchmark.config import EvaluationConfig, GenerationConfig
from backtracking_llm.benchmark.evaluator import Evaluator
from backtracking_llm.decision import Never

# pylint: disable=protected-access
# pylint: disable=unused-argument


@patch('lm_eval.simple_evaluate')
class TestEvaluator:
    """Tests the Evaluator class."""

    def test_initialization(self, mock_simple_evaluate: MagicMock):
        eval_config = EvaluationConfig(tasks=['task1'])
        evaluator = Evaluator(eval_config)
        assert evaluator.config is eval_config

    def test_run_calls_simple_evaluate_correctly(
            self, mock_simple_evaluate: MagicMock, mock_generator: MagicMock):
        operator = Never()
        eval_config = EvaluationConfig(tasks=['gsm8k'], num_fewshot=5, limit=10)
        gen_config = GenerationConfig(max_new_tokens=50,
                                      temperature=0.7,
                                      top_k=40)
        evaluator = Evaluator(eval_config)

        evaluator.run(mock_generator, gen_config, operator)

        mock_simple_evaluate.assert_called_once()
        _, call_kwargs = mock_simple_evaluate.call_args

        assert 'model' in call_kwargs
        model_instance = call_kwargs['model']
        assert model_instance._generator is mock_generator
        assert model_instance._operator is operator

        assert call_kwargs['tasks'] == ['gsm8k']
        assert call_kwargs['num_fewshot'] == 5
        assert call_kwargs['limit'] == 10
        expected_gen_kwargs = {
            'max_gen_toks': 50,
            'temperature': 0.7,
            'top_k': 40
        }
        assert call_kwargs['gen_kwargs'] == expected_gen_kwargs

    def test_extract_primary_score_finds_metric(
            self, mock_simple_evaluate: MagicMock):
        results = {
            'results': {
                'task1': {
                    'acc,none': 0.5,
                    'some_other_metric': 0.9
                }
            }
        }
        score = Evaluator.extract_primary_score(results, 'task1')
        assert score == 0.5

    def test_extract_primary_score_finds_normalized_metric(
            self, mock_simple_evaluate: MagicMock):
        results = {
            'results': {
                'task1': {
                    'acc,none': 0.5,
                    'acc_norm,none': 0.45
                }
            }
        }
        score = Evaluator.extract_primary_score(results, 'task1')
        assert score == 0.45

    def test_extract_primary_score_raises_error_on_missing_task(
            self, mock_simple_evaluate: MagicMock):
        results = {'results': {'other_task': {}}}
        with pytest.raises(ValueError, match="'task1' not found"):
            Evaluator.extract_primary_score(results, 'task1')

    def test_extract_primary_score_raises_error_on_missing_metric(
            self, mock_simple_evaluate: MagicMock):
        results = {'results': {'task1': {'non_standard_metric': 0.1}}}
        with pytest.raises(KeyError, match='No standard primary metric'):
            Evaluator.extract_primary_score(results, 'task1')

    def test_run_handles_none_result_from_lm_eval(
            self, mock_simple_evaluate: MagicMock, mock_generator: MagicMock):
        mock_simple_evaluate.return_value = None

        eval_config = EvaluationConfig(tasks=['task'])
        evaluator = Evaluator(eval_config)

        results = evaluator.run(mock_generator, MagicMock())

        assert results == {}
        mock_simple_evaluate.assert_called_once()
