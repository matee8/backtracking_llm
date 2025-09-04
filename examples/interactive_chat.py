#!/usr/bin/env python3
"""An example of how to use the ChatPipeline for interactive, multi-turn
conversation.
"""

import logging
import sys

from backtracking_llm.chat import ChatPipeline, ConversationHistory
from backtracking_llm.decision import ProbabilityThreshold
from backtracking_llm.generation import Generator

# pylint: disable=broad-exception-caught

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def main() -> None:
    print('Interactive chat example')
    model_name = 'qwen/qwen2.5-0.5b-instruct'

    print(f"\nLoading model '{model_name}'... This may take a moment.")
    try:
        generator = Generator.from_pretrained(model_name)
    except Exception as e:
        print(f'Failed to load model: {e}')
        sys.exit(1)

    # The pipeline is a stateless helper that wraps the Generator.
    chat_pipeline = ChatPipeline(generator)

    # The user is responsible for storing the history.
    history: ConversationHistory = []

    question = 'What is the capital of France?'

    probthreshold_operator = ProbabilityThreshold(min_probability=0.5)

    generation_kwargs = {
        'operator': probthreshold_operator,
        'max_new_tokens': 100,
        'temperature': 0.8,
        'top_k': 50,
        'backtrack_every_n': 1
    }

    answer, history = chat_pipeline.run_turn(question, history,
                                             **generation_kwargs)

    print(f'Question: {question}')
    print(f'Answer: {answer}')


if __name__ == '__main__':
    main()
