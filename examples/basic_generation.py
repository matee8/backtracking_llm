#!/usr/bin/env python3
"""A basic example demonstrating how to use the Generator with a backtracking
Operator to generate text.
"""

import logging
import sys

from backtracking_llm.decision import Repetition
from backtracking_llm.generation import Generator

# pylint: disable=broad-exception-caught

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def main() -> None:
    print('Basic generation example.')
    print('This example demonstrates using the Generator with a Repetition'
          'operator.')

    print("\nLoading model 'gpt2'... This may take a moment.")
    try:
        generator = Generator.from_pretrained('openai-community/gpt2')
    except Exception as e:
        print(f'Failed to load model: {e}')
        sys.exit(1)

    prompt = ('In a shocking finding, scientists discovered a herd of '
              'unicorns living in a remote, previously unexplored valley, '
              'in the Andes Mountains. The unicorns were unique because '
              'they could speak English. When asked about their origin, '
              'one unicorn replied,')

    repetition_operator = Repetition(max_repetitions=2)

    generated_text = generator.generate(prompt,
                                        operator=repetition_operator,
                                        max_new_tokens=100,
                                        temperature=0.8,
                                        top_k=50,
                                        backtrack_every_n=1)

    print(f'Prompt:\n{prompt}')
    print(f'Generated completion:\n{generated_text}')


if __name__ == '__main__':
    main()
