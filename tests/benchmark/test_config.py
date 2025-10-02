# pylint: disable=missing-module-docstring

import pytest

from backtracking_llm.benchmark import config

# pylint: disable=no-value-for-parameter


class TestGenerationConfig:
    """Tests the GenerationConfig dataclass."""

    def test_initialization_with_defaults(self):
        gen_config = config.GenerationConfig()
        assert gen_config.max_new_tokens == 100
        assert gen_config.temperature == 1.0
        assert gen_config.top_k == 50
        assert gen_config.backtrack_every_n == 1

    def test_initialization_with_custom_values(self):
        gen_config = config.GenerationConfig(max_new_tokens=200,
                                             temperature=0.5,
                                             top_k=30,
                                             backtrack_every_n=5)
        assert gen_config.max_new_tokens == 200
        assert gen_config.temperature == 0.5
        assert gen_config.top_k == 30
        assert gen_config.backtrack_every_n == 5


class TestEvaluationConfig:
    """Tests the EvaluationConfig dataclass."""

    def test_initialization_with_required_fields(self):
        eval_config = config.EvaluationConfig(tasks=['task1', 'task2'])
        assert eval_config.tasks == ['task1', 'task2']
        assert eval_config.num_fewshot == 0
        assert eval_config.limit is None
        assert eval_config.output_dir == 'benchmark_results'

    def test_initialization_with_all_fields(self):
        eval_config = config.EvaluationConfig(tasks=['task1'],
                                              num_fewshot=5,
                                              limit=10,
                                              output_dir='/tmp/results')
        assert eval_config.tasks == ['task1']
        assert eval_config.num_fewshot == 5
        assert eval_config.limit == 10
        assert eval_config.output_dir == '/tmp/results'

    def test_missing_tasks_raises_error(self):
        with pytest.raises(TypeError):
            config.EvaluationConfig()


class TestHPOConfig:
    """Tests the HPOConfig dataclass."""

    def test_initialization(self):
        search_space = {'param1': [0.1, 0.9], 'param2': [1, 10]}
        hpo_config = config.HPOConfig(n_trials=50, search_space=search_space)
        assert hpo_config.n_trials == 50
        assert hpo_config.search_space == search_space

    def test_missing_fields_raises_error(self):
        with pytest.raises(TypeError):
            config.HPOConfig(n_trials=50)
        with pytest.raises(TypeError):
            config.HPOConfig(search_space={})


class TestBenchmarkingConfig:
    """Tests the root BenchmarkingConfig dataclass."""

    def test_initialization_with_minimal_required_fields(self):
        eval_config = config.EvaluationConfig(tasks=['task1'])
        bench_config = config.BenchmarkingConfig(
            model_name_or_path='test-model', evaluation=eval_config)

        assert bench_config.model_name_or_path == 'test-model'
        assert bench_config.evaluation is eval_config
        assert bench_config.device == 'cpu'
        assert bench_config.operator_to_tune is None
        assert bench_config.run_baseline is True
        assert isinstance(bench_config.generation, config.GenerationConfig)
        assert bench_config.hpo is None

    def test_initialization_with_all_fields(self):
        gen_config = config.GenerationConfig(max_new_tokens=50)
        eval_config = config.EvaluationConfig(tasks=['task1'], limit=50)
        hpo_config = config.HPOConfig(n_trials=10, search_space={'p': [0, 1]})

        bench_config = config.BenchmarkingConfig(
            model_name_or_path='test-model',
            device='cuda:0',
            operator_to_tune='ProbabilityThreshold',
            run_baseline=False,
            generation=gen_config,
            evaluation=eval_config,
            hpo=hpo_config)

        assert bench_config.model_name_or_path == 'test-model'
        assert bench_config.device == 'cuda:0'
        assert bench_config.operator_to_tune == 'ProbabilityThreshold'
        assert bench_config.run_baseline is False
        assert bench_config.generation is gen_config
        assert bench_config.evaluation is eval_config
        assert bench_config.hpo is hpo_config
