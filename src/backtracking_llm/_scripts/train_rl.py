"""The command-line interface for the RL training module."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from backtracking_llm.rl.config import RLConfig
from backtracking_llm.rl.data import TextFilePromptProvider
from backtracking_llm.rl.trainers import RLTrainer

# pylint: disable=broad-exception-caught


def _setup_logging(level: int = logging.INFO) -> None:
    """Configures the root logger for the application."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=level, format=log_format, stream=sys.stdout)


def _load_config(config_path: Path) -> RLConfig:
    """Loads, parses, and validates the RL YAML configuration file."""
    if not config_path.is_file():
        raise FileNotFoundError(f'Config file not found at: {config_path}')

    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f'Error parsing YAML file: {e}') from e

    if not isinstance(raw_config, dict):
        raise ValueError('YAML config root must be a dictionary.')

    try:
        return RLConfig(**raw_config)
    except (TypeError, KeyError, ValueError) as e:
        raise ValueError(
            f'Missing or invalid configuration parameter: {e}') from e


def main() -> None:
    """The main entry point for the RL training CLI."""
    parser = argparse.ArgumentParser(
        description='Run the backtracking-llm reinforcement learning trainer.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to the YAML configuration file for the RL run.')
    parser.add_argument(
        '--prompts',
        type=Path,
        required=True,
        help='Path to a text file containing one prompt per line.')
    parser.add_argument(
        '--shuffle-prompts',
        action='store_true',
        help='Randomly shuffle the prompts before starting training.')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Enable verbose DEBUG logging.')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    _setup_logging(log_level)

    try:
        logging.info('Loading configuration from: %s', args.config)
        config = _load_config(args.config)

        logging.info('Loading prompts from: %s', args.prompts)
        prompt_provider = TextFilePromptProvider(args.prompts,
                                                 shuffle=args.shuffle_prompts)

        trainer = RLTrainer(config)
        trainer.train(prompt_provider)

    except (FileNotFoundError, ValueError) as e:
        logging.error('Failed to run RL training: %s', e)
        sys.exit(1)
    except Exception as e:
        logging.error('An unexpected error occurred: %s', e, exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
