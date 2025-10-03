# pylint: disable=missing-module-docstring

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from backtracking_llm.benchmark.config import (BenchmarkingConfig,
                                               EvaluationConfig, HPOConfig)
from backtracking_llm.benchmark.runner import (BenchmarkRunner,
                                               _save_results_json,
                                               _sanitize_for_json)
from backtracking_llm.decision import Never

# pylint: disable=protected-access
# pylint: disable=unused-argument


@patch('backtracking_llm.benchmark.runner.Generator.from_pretrained')
class TestBenchmarkRunner:
    """Tests the BenchmarkRunner class."""

    def test_run_calls_baseline_only(self, mock_from_pretrained):
        config = BenchmarkingConfig(model_name_or_path='test',
                                    run_baseline=True,
                                    evaluation=EvaluationConfig(tasks=['task']))

        runner = BenchmarkRunner(config)
        with patch.object(runner, '_run_baseline',
                          MagicMock()) as mock_baseline, patch.object(
                              runner, '_run_hpo', MagicMock()) as mock_hpo:
            runner.run()
            mock_baseline.assert_called_once()
            mock_hpo.assert_not_called()

    def test_run_calls_hpo_only(self, mock_from_pretrained):
        config = BenchmarkingConfig(model_name_or_path='test',
                                    run_baseline=False,
                                    evaluation=EvaluationConfig(tasks=['task']),
                                    operator_to_tune='Op',
                                    hpo=HPOConfig(n_trials=1, search_space={}))

        runner = BenchmarkRunner(config)
        with patch.object(runner, '_run_baseline',
                          MagicMock()) as mock_baseline, \
             patch.object(runner, '_run_hpo', MagicMock()) as mock_hpo:
            runner.run()
            mock_baseline.assert_not_called()
            mock_hpo.assert_called_once()

    def test_run_calls_both(self, mock_from_pretrained):
        config = BenchmarkingConfig(model_name_or_path='test',
                                    run_baseline=True,
                                    evaluation=EvaluationConfig(tasks=['task']),
                                    operator_to_tune='Op',
                                    hpo=HPOConfig(n_trials=1, search_space={}))

        runner = BenchmarkRunner(config)
        with patch.object(runner, '_run_baseline',
                          MagicMock()) as mock_baseline, \
             patch.object(runner, '_run_hpo', MagicMock()) as mock_hpo:
            runner.run()
            mock_baseline.assert_called_once()
            mock_hpo.assert_called_once()

    @patch('backtracking_llm.benchmark.runner.Evaluator')
    def test_run_baseline_calls_evaluator_correctly(self, mock_evaluator_cls,
                                                    mock_from_pretrained):
        config = BenchmarkingConfig(model_name_or_path='test',
                                    evaluation=EvaluationConfig(
                                        tasks=['task'],
                                        output_dir='/tmp/results'))

        mock_evaluator_instance = mock_evaluator_cls.return_value
        mock_evaluator_instance.run.return_value = {
            'results': {
                'task': {
                    'acc,none': 0.5
                }
            }
        }

        runner = BenchmarkRunner(config)
        with patch('backtracking_llm.benchmark.runner._save_results_json'
                  ) as mock_save:
            runner._run_baseline()

            mock_evaluator_cls.assert_called_once_with(config.evaluation)
            mock_evaluator_instance.run.assert_called_once()
            op_arg = mock_evaluator_instance.run.call_args[0][2]
            assert isinstance(op_arg, Never)

            mock_save.assert_called_once_with(
                {'results': {
                    'task': {
                        'acc,none': 0.5
                    }
                }}, Path('/tmp/results/baseline_results.json'))

    @patch('backtracking_llm.benchmark.runner.HyperparameterOptimizer')
    def test_run_hpo_calls_optimizer_correctly(self, mock_optimizer_cls,
                                               mock_from_pretrained):
        hpo_config = HPOConfig(n_trials=1, search_space={})
        config = BenchmarkingConfig(model_name_or_path='test',
                                    evaluation=EvaluationConfig(
                                        tasks=['task'], output_dir='/tmp/hpo'),
                                    operator_to_tune='TestOp',
                                    hpo=hpo_config)

        mock_study = MagicMock()
        mock_study.best_value = 0.9
        mock_study.best_params = {'p': 1}
        mock_optimizer_instance = mock_optimizer_cls.return_value
        mock_optimizer_instance.optimize.return_value = mock_study

        runner = BenchmarkRunner(config)
        with patch('backtracking_llm.benchmark.runner._save_results_json'
                  ) as mock_save:
            runner._run_hpo()

            mock_optimizer_cls.assert_called_once_with(
                hpo_config=config.hpo,
                eval_config=config.evaluation,
                gen_config=config.generation,
                generator=runner.generator,
                operator_name='TestOp')
            mock_optimizer_instance.optimize.assert_called_once()

            expected_results = {
                'best_value': 0.9,
                'best_params': {
                    'p': 1
                },
                'operator': 'TestOp'
            }
            mock_save.assert_called_once_with(expected_results,
                                              Path('/tmp/hpo/hpo_results.json'))

    def test_run_warns_if_operator_to_tune_but_no_hpo_config(
            self, mock_from_pretrained, caplog):
        config = BenchmarkingConfig(model_name_or_path='test',
                                    run_baseline=False,
                                    evaluation=EvaluationConfig(tasks=['task']),
                                    operator_to_tune='Op',
                                    hpo=None)

        runner = BenchmarkRunner(config)
        runner.run()

        assert 'Skipping HPO' in caplog.text
        assert ('was specified to tune, but no HPO configuration was provided'
                in caplog.text)

    @patch('backtracking_llm.benchmark.runner.Evaluator')
    def test_run_baseline_handles_evaluator_failure(self, mock_evaluator_cls,
                                                    mock_from_pretrained,
                                                    caplog):
        config = BenchmarkingConfig(model_name_or_path='test',
                                    evaluation=EvaluationConfig(tasks=['task']))

        mock_evaluator_instance = mock_evaluator_cls.return_value
        mock_evaluator_instance.run.return_value = {}

        runner = BenchmarkRunner(config)
        runner._run_baseline()

        assert 'Baseline evaluation failed to produce results' in caplog.text

    @patch('backtracking_llm.benchmark.runner.Evaluator')
    def test_run_baseline_handles_score_extraction_failure(
            self, mock_evaluator_cls, mock_from_pretrained, caplog):
        config = BenchmarkingConfig(model_name_or_path='test',
                                    evaluation=EvaluationConfig(tasks=['task']))

        mock_evaluator_cls.extract_primary_score.side_effect = KeyError(
            'Metric not found')

        mock_evaluator_instance = mock_evaluator_cls.return_value
        mock_evaluator_instance.run.return_value = {'results': {'task': {}}}

        runner = BenchmarkRunner(config)
        with patch('backtracking_llm.benchmark.runner._save_results_json'):
            runner._run_baseline()

        assert 'Could not extract primary score for baseline' in caplog.text

    @patch('pathlib.Path.mkdir')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('json.dump')
    def test_save_results_json_creates_dirs_and_writes_file(
            self, mock_json_dump, mock_open, mock_mkdir, mock_from_pretrained):
        test_path = Path('/fake/dir/results.json')
        test_data = {'key': 'value'}

        _save_results_json(test_data, test_path)

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_open.assert_called_once_with(test_path, 'w', encoding='utf-8')

        mock_file_handle = mock_open.return_value.__enter__.return_value
        mock_json_dump.assert_called_once_with(test_data,
                                               mock_file_handle,
                                               indent=4)

    @patch('backtracking_llm.benchmark.runner.HyperparameterOptimizer')
    def test_run_hpo_returns_early_if_no_hpo_config(self, mock_optimizer_cls,
                                                    mock_from_pretrained):
        config = BenchmarkingConfig(model_name_or_path='test',
                                    evaluation=EvaluationConfig(tasks=['task']),
                                    operator_to_tune='Op',
                                    hpo=None)

        runner = BenchmarkRunner(config)
        runner._run_hpo()

        mock_optimizer_cls.assert_not_called()

    @patch('backtracking_llm.benchmark.runner.HyperparameterOptimizer')
    def test_run_hpo_returns_early_if_no_operator_to_tune(
            self, mock_optimizer_cls, mock_from_pretrained):
        config = BenchmarkingConfig(model_name_or_path='test',
                                    evaluation=EvaluationConfig(tasks=['task']),
                                    operator_to_tune=None,
                                    hpo=HPOConfig(n_trials=1, search_space={}))

        runner = BenchmarkRunner(config)
        runner._run_hpo()

        mock_optimizer_cls.assert_not_called()


class _NonSerializableObject:

    def __repr__(self) -> str:
        return 'CustomObjectRepresentation'


@pytest.mark.parametrize('input_data, expected_output', [
    ({
        'a': 1,
        'b': 'hello',
        'c': 3.14,
        'd': True,
        'e': None
    }, {
        'a': 1,
        'b': 'hello',
        'c': 3.14,
        'd': True,
        'e': None
    }),
    ({
        'a': [1, 2, 3],
        'b': (4, 5, 6)
    }, {
        'a': [1, 2, 3],
        'b': [4, 5, 6]
    }),
    ({
        123: 'value',
        True: 'bool_key'
    }, {
        '123': 'value',
        'True': 'bool_key'
    }),
    ({
        'numpy_int': np.int64(100),
        'numpy_float': np.float32(99.9),
        'numpy_dtype': np.dtype('float64'),
        'a_set': {1, 2, 3}
    }, {
        'numpy_int': '100',
        'numpy_float': '99.9',
        'numpy_dtype': 'float64',
        'a_set': str({1, 2, 3})
    }),
    ({
        'custom_obj': _NonSerializableObject()
    }, {
        'custom_obj': 'CustomObjectRepresentation'
    }),
    ({
        'level1': [{
            'a': 1
        }, {
            'b': {
                np.int32(2): _NonSerializableObject()
            }
        }]
    }, {
        'level1': [{
            'a': 1
        }, {
            'b': {
                '2': 'CustomObjectRepresentation'
            }
        }]
    }),
])
def test_sanitize_for_json(input_data, expected_output):
    sanitized = _sanitize_for_json(input_data)

    if 'a_set' in sanitized:
        assert sanitized['a_set'] in ('{1, 2, 3}', '{1, 3, 2}', '{2, 1, 3}',
                                      '{2, 3, 1}', '{3, 1, 2}', '{3, 2, 1}')

        del sanitized['a_set']
        del expected_output['a_set']

    assert sanitized == expected_output
