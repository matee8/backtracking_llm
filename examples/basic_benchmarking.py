#!/usr/bin/env python3
"""A complete example of how to configure and run the benchmarking pipeline
programmatically.
"""

import logging
import sys

from backtracking_llm.benchmark.config import (BenchmarkingConfig,
                                               EvaluationConfig,
                                               GenerationConfig, HPOConfig)
from backtracking_llm.benchmark.runner import BenchmarkRunner

# This is a simple example, so we'll catch broad exceptions.
# pylint: disable=broad-exception-caught

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)


def main() -> None:
    """Configures and runs a benchmark."""
    print('--- Backtracking LLM Benchmarking Example ---')
    print('This script will run a small benchmark pipeline:')
    print('1. Run a baseline evaluation on the "arc_easy" task.')
    print('2. Run a 5-trial hyperparameter search for the '
          '"ProbabilityThreshold" operator.')
    print(
        '\nNOTE: This will download a model (~1GB) and datasets on the first run.'
    )

    try:
        # 1. Define the configuration for the entire run.
        generation_config = GenerationConfig(max_new_tokens=128,
                                             temperature=0.8,
                                             top_k=40)

        evaluation_config = EvaluationConfig(
            tasks=['arc_easy'],
            limit=25,
            output_dir='benchmark_example_results')

        hpo_config = HPOConfig(n_trials=5,
                               search_space={
                                   'min_probability': [0.01, 0.25],
                                   'backtrack_count': [1, 3]
                               })

        config = BenchmarkingConfig(
            model_name_or_path='Qwen/Qwen2-0.5B-Instruct',
            device='cpu',
            run_baseline=True,
            operator_to_tune='ProbabilityThreshold',
            generation=generation_config,
            evaluation=evaluation_config,
            hpo=hpo_config)

        runner = BenchmarkRunner(config)

        print('\nStarting the benchmark pipeline... This may take a while.')
        runner.run()

        print('\nBenchmark pipeline finished successfully!')
        print(
            f"Results have been saved to the '{config.evaluation.output_dir}' directory."
        )
    except Exception as e:
        logging.error('The benchmark example failed: %s', e, exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
