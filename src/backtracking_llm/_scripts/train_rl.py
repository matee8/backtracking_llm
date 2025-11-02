#!/usr/bin/env python3
"""The command-line interface for training an RL agent."""

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.optim import Adam

from backtracking_llm.agents import RLAgentOperator
from backtracking_llm.benchmark.config import (EvaluationConfig, FullRLConfig,
                                               GenerationConfig,
                                               RLTrainingConfig)
from backtracking_llm.benchmark.rl_env import RLEnvironment
from backtracking_llm.generation import Generator

# pylint: disable=broad-exception-caught

logger = logging.getLogger(__name__)


def _setup_logging(level: int = logging.INFO) -> None:
    """Configures the root logger for the application."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=level, format=log_format, stream=sys.stdout)


def _load_config(config_path: Path) -> FullRLConfig:
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
        rl_conf = RLTrainingConfig(**raw_config.get('rl_training', {}))

        nested_keys = {'generation', 'evaluation', 'rl_training'}
        root_keys = {
            f.name
            for f in dataclasses.fields(FullRLConfig)
            if f.name not in nested_keys
        }
        root_args = {k: v for k, v in raw_config.items() if k in root_keys}

        return FullRLConfig(generation=gen_conf,
                            evaluation=eval_conf,
                            rl_training=rl_conf,
                            **root_args)
    except (TypeError, KeyError) as e:
        raise ValueError(
            f'Missing or invalid configuration parameter: {e}') from e


class RLTrainer:
    """Orchestrates the end-to-end RL training process."""

    def __init__(self, config: FullRLConfig) -> None:
        """Initializes the RLTrainer."""
        self.config = config

        logger.info('Initializing generator for model: %s',
                    config.model_name_or_path)
        generator = Generator.from_pretrained(config.model_name_or_path,
                                              **config.model_kwargs)
        generator.model.to(config.device)  # type: ignore

        self.env = RLEnvironment(generator, config.evaluation,
                                 config.generation)
        self.agent = RLAgentOperator(
            input_dim=config.rl_training.feature_dim,
            num_actions=config.rl_training.num_actions).to(config.device)
        self.optimizer = Adam(self.agent.parameters(),
                              lr=config.rl_training.learning_rate)

    def train(self) -> None:
        """Executes the REINFORCE training loop."""
        logger.info('Starting RL training for %d episodes.',
                    self.config.rl_training.num_episodes)

        for i_episode in range(self.config.rl_training.num_episodes):
            self.agent.train()

            reward = self.env.run_episode(self.agent)

            if not self.agent.saved_log_probs:
                logger.warning(
                    'Episode %d: No actions were taken by the agent. '
                    'Skipping policy update.', i_episode + 1)
                continue

            policy_loss = []
            for log_prob in self.agent.saved_log_probs:
                policy_loss.append(-log_prob * reward)

            self.optimizer.zero_grad()
            loss = torch.cat(policy_loss).sum()
            loss.backward()
            self.optimizer.step()

            del self.agent.saved_log_probs[:]

            logger.info('Episode %d finished. Loss: %.4f', i_episode + 1,
                        loss.item())

        self.save_model()

    def save_model(self) -> None:
        """Saves the trained agent's state dictionary to a file."""
        output_path = self.config.rl_training.output_model_path
        logger.info('RL training finished. Saving model to %s.', output_path)
        torch.save(self.agent.state_dict(), output_path)


def main() -> None:
    """The main entry point for the RL training CLI."""
    parser = argparse.ArgumentParser(
        description='Run the backtracking-llm RL training pipeline.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to the YAML configuration file for the training run.')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Enable verbose DEBUG logging.')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    _setup_logging(log_level)

    try:
        config = _load_config(args.config)
        trainer = RLTrainer(config)
        trainer.train()
    except (FileNotFoundError, ValueError) as e:
        logging.error('Failed to run training: %s', e)
        sys.exit(1)
    except Exception as e:
        logging.error('An unexpected error occurred: %s', e, exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
