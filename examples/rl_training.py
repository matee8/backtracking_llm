#!/usr/bin/env python3
"""A complete example of how to configure and run the RL training pipeline
programmatically.
"""

import logging
import sys
import os
from pathlib import Path

from backtracking_llm.rl.config import (RLConfig, JudgeConfig, EnvConfig,
                                        TrainingConfig, ShapingConfig)
from backtracking_llm.rl.data import TextFilePromptProvider
from backtracking_llm.rl.trainers import RLTrainer

# This is a simple example, so we'll catch broad exceptions.
# pylint: disable=broad-exception-caught

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)


def create_dummy_prompts(file_path: Path) -> None:
    """Creates a dummy prompt file for the example."""
    prompts = [
        'Once upon a time',
        'The capital of France is',
        'Write a python function to',
        'Explain quantum mechanics',
        'The quick brown fox',
    ]
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(prompts))
    logging.info('Created dummy prompt file at: %s', file_path)


def main() -> None:
    """Configures and runs an RL training session."""
    print('--- Backtracking LLM RL Training Example ---')
    print('This script will run a short RL training session:')
    print('1. Create a dummy prompts file.')
    print('2. Configure the PPO agent and environment.')
    print('3. Train for a few steps using a small model.')
    print('\nNOTE: This example requires an OpenAI API key for the Judge.')
    print('      Set OPENAI_API_KEY in your environment variables.')

    if 'OPENAI_API_KEY' not in os.environ:
        logging.error('OPENAI_API_KEY environment variable not set. '
                      'The Judge requires it to score generations.')
        sys.exit(1)

    prompt_file = Path('example_prompts.txt')

    try:
        # 1. Setup Data
        if not prompt_file.exists():
            create_dummy_prompts(prompt_file)

        prompt_provider = TextFilePromptProvider(prompt_file, shuffle=True)

        # 2. Define Configuration
        # We use small values to make this example run quickly on CPU.
        config = RLConfig(
            model_name_or_path='Qwen/Qwen2.5-0.5B-Instruct',
            device='cpu',
            output_dir=Path('rl_example_output'),
            judge=JudgeConfig(
                model='gpt-4-turbo-preview',
                # api_key is read from env automatically if not provided
            ),
            env=EnvConfig(max_backtrack=3, max_seq_length=64),
            training=TrainingConfig(
                total_timesteps=512,  # Very short run
                n_steps=128,
                batch_size=32,
                n_epochs=2,
                policy_type='MlpLstmPolicy'),
            shaping=ShapingConfig(backtrack_action_penalty=0.05,
                                  repetition_penalty_weight=0.1))

        # 3. Initialize Trainer
        trainer = RLTrainer(config)

        # 4. Start Training
        print('\nStarting RL training... This may take a minute.')
        trainer.train(prompt_provider)

        print('\nRL training finished successfully!')
        print(f'Policy saved to: {config.output_dir}/policy.zip')

    except Exception as e:
        logging.error('The RL training example failed: %s', e, exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
