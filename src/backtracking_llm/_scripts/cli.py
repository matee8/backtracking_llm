#!/usr/bin/env python3
"""Provides an interactive command-line interface for chatting with a model
using the backtracking_llm library.
"""

import logging
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Dict, Type

from backtracking_llm.chat import ChatPipeline, ConversationHistory
from backtracking_llm.decision import (EntropyThreshold, LogitThreshold,
                                       NGramOverlap, Operator, ProbabilityDrop,
                                       ProbabilityMargin, ProbabilityThreshold,
                                       ProbabilityTrend, Repetition)
from backtracking_llm.generation import Generator
from backtracking_llm._scripts.ui import Spinner

# pylint: disable=broad-exception-caught

OPERATOR_FACTORIES: Dict[str, Type[Operator]] = {
    'probability_threshold': ProbabilityThreshold,
    'entropy_threshold': EntropyThreshold,
    'probability_margin': ProbabilityMargin,
    'probability_drop': ProbabilityDrop,
    'probability_trend': ProbabilityTrend,
    'repetition': Repetition,
    'ngram_overlap': NGramOverlap,
    'logit_threshold': LogitThreshold,
}


def setup_logging(verbosity: int) -> None:
    if verbosity == 1:
        log_level = logging.INFO
    elif verbosity >= 2:
        log_level = logging.DEBUG
    else:
        log_level = logging.WARNING

    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')


def main() -> None:
    parser = ArgumentParser(
        description=('run an interactive chat session with a'
                     'backtracking-enabled LLM'),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'model_name',
        type=str,
        help='the name or path of the model from the Hugging Face Hub')
    parser.add_argument(
        '--operator',
        type=str,
        default='none',
        choices=OPERATOR_FACTORIES.keys(),
        help='the backtracking operator to use during generation')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=200,
        help='the maximum number of new tokens to generate in each turn')
    parser.add_argument('--temperature',
                        type=float,
                        default=1.0,
                        help='controls the randomness of the output')
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help=
        ('controls the sampling strategy by limiting the next token prediction'
         'pool'))
    parser.add_argument('-v',
                        '--verbose',
                        action='count',
                        default=0,
                        help='increase logging verbosity')

    args = parser.parse_args()
    logger = logging.getLogger('backtracking_llm_cli')

    logger.info("Loading model '%s'... This may take a moment.",
                args.model_name)
    with Spinner(f"Loading model '{args.model_name}'..."):
        try:
            generator = Generator.from_pretrained(args.model_name)
        except Exception as e:
            logger.error('Failed to load the model: %s.', e, exc_info=True)
            logger.error('Please ensure the model name is correct, dependencies'
                         'are installed, and you have an internet connection.')
            sys.exit(1)

    chat_pipeline = ChatPipeline(generator)
    history: ConversationHistory = []
    operator_factory = OPERATOR_FACTORIES.get(args.operator)
    operator = operator_factory() if operator_factory else None

    generation_kwargs = {
        'operator': operator,
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_k': args.top_k,
    }

    print('Model loaded. Starting interactive chat session.')
    print("Type 'exit' or 'quit' to end\n")

    while True:
        try:
            question = input('User: ')

            if question.lower() in ['exit', 'quit']:
                break

            answer = ''
            with Spinner('AI is thinking...'):
                answer, history = chat_pipeline.run_turn(
                    question, history, **generation_kwargs)

            print(f'AI: {answer}')
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            logger.error('\nAn unexpected error occured: %s.', e, exc_info=True)
            break

    print('\nExiting chat session.')


if __name__ == '__main__':
    main()
