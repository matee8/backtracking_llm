"""A dedicated script for profiling the performance of the Generator.generate
method.
"""

import cProfile
import logging

from backtracking_llm.decision import ProbabilityThreshold
from backtracking_llm.generation import Generator

logging.disable(logging.CRITICAL)


def run_baseline_task() -> None:
    print('Loading model...')
    generator = Generator.from_pretrained('openai-community/gpt2')

    prompt = 'The primary purpose of a profiler in software engineering is to'
    generator.generate(prompt, max_new_tokens=100)

    print('Generation complete')


def run_complex_task() -> None:
    print('Loading model...')
    generator = Generator.from_pretrained('openai-community/gpt2')

    prompt = 'The primary purpose of a profiler in software engineering is to'
    generator.generate(prompt,
                       max_new_tokens=100,
                       operator=ProbabilityThreshold(min_probability=0.01),
                       backtrack_every_n=1,
                       stop_sequences=['\n', ' a', ' the'])

    print('Generation complete')


def main() -> None:
    print('Running baseline profile.')
    cProfile.run('run_baseline_task()', 'baseline.prof')
    print('Baseline generation complete.')

    print('Running complex profile.')
    cProfile.run('run_complex_task()', 'complex.prof')
    print('Complex generation complete.')


if __name__ == '__main__':
    main()
