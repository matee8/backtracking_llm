"""A dedicated script for profiling the performance of the Generator.generate
method.
"""

import cProfile
import logging

from backtracking_llm.generation import Generator

logging.disable(logging.CRITICAL)

def run_profiling_task() -> None:
    print('Loading model...')
    generator = Generator.from_pretrained('openai-community/gpt2')

    prompt = 'The primary purpose of a profiler in software engineering is to'
    generator.generate(prompt, max_new_tokens=100)

    print('Generation complete')

if __name__ == '__main__':
    cProfile.run('run_profiling_task()', 'program.prof')
