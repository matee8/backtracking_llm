"""The command-line interface for the benchmarking module."""

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

import yaml
from backtracking_llm.benchmark.config import (BenchmarkingConfig,
                                               EvaluationConfig,
                                               GenerationConfig, HPOConfig)
from backtracking_llm.benchmark.runner import BenchmarkRunner

# pylint: disable=broad-exception-caught


def _setup_logging(level: int = logging.INFO) -> None:
    """Configures the root logger for the application."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=level, format=log_format, stream=sys.stdout)


def _load_config(config_path: Path) -> BenchmarkingConfig:
    """Loads, parses, and validates the YAML configuration file."""
    if not config_path.is_file():
        raise FileNotFoundError(f'Config file not found at: {config_path}')

    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f'Error parsing YAML file: {e}') from e

    try:
        gen_conf = GenerationConfig(**raw_config.get('generation', {}))
        eval_conf = EvaluationConfig(**raw_config['evaluation'])
        hpo_conf = HPOConfig(
            **raw_config['hpo']) if 'hpo' in raw_config else None

        nested_keys = {'generation', 'evaluation', 'hpo'}
        root_keys = {
            f.name
            for f in dataclasses.fields(BenchmarkingConfig)
            if f.name not in nested_keys
        }
        root_args = {k: v for k, v in raw_config.items() if k in root_keys}

        return BenchmarkingConfig(generation=gen_conf,
                                  evaluation=eval_conf,
                                  hpo=hpo_conf,
                                  **root_args)
    except (TypeError, KeyError) as e:
        raise ValueError(
            f'Missing or invalid configuration parameter: {e}') from e


def main() -> None:
    """The main entry point for the benchmarking CLI."""
    parser = argparse.ArgumentParser(
        description='Run the backtracking-llm benchmarking pipeline.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to the YAML configuration file for the benchmark run.')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Enable verbose DEBUG logging.')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    _setup_logging(log_level)

    try:
        logging.info('Loading configuration from: %s', args.config)
        config = _load_config(args.config)
        runner = BenchmarkRunner(config)
        runner.run()
    except (FileNotFoundError, ValueError) as e:
        logging.error('Failed to run benchmark: %s', e)
        sys.exit(1)
    except Exception as e:
        logging.error('An unexpected error occurred: %s', e, exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
