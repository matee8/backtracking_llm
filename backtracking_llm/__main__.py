#!/usr/bin/env python3

import argparse
import functools
import logging
import typing
import sys

import transformers

from backtracking_llm.models import decision, inference, question_answering

DEFAULT_MAX_LENGTH: typing.Final[int] = 100
DEFAULT_TOP_K: typing.Final[int] = 50
DEFAULT_TEMPERATURE: typing.Final[float] = 1.
DEFAULT_BACKTRACK_EVERY_N: typing.Final[int] = 5
DEFAULT_PROBABILITY_THRESHOLD: typing.Final[float] = .5


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on any model and analyze token logits "
        "and probabilites",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model", type=str, help="Model name to use")

    parser.add_argument("--max-answer-length",
                        type=int,
                        default=DEFAULT_MAX_LENGTH,
                        help="Maximum number of tokens to generate for each "
                        " answer")

    parser.add_argument("--top-k",
                        type=int,
                        default=DEFAULT_TOP_K,
                        help="Number of top tokens to analyze")

    parser.add_argument("--verbose",
                        action="store_true",
                        help="Enable verbose logging")

    parser.add_argument("--temperature",
                        type=float,
                        default=DEFAULT_TEMPERATURE,
                        help="Sampling temperature")

    parser.add_argument(
        "--backtrack-every-n",
        type=int,
        default=DEFAULT_BACKTRACK_EVERY_N,
        help="Check for backtracking every N generated tokens.")

    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=DEFAULT_PROBABILITY_THRESHOLD,
        help="Probability threshold for the simple backtracking"
        "decision function (default: %(default)s)")

    return parser.parse_args()


def _setup_logging(verbose: bool = False) -> logging.Logger:
    if verbose:
        log_level: int = logging.DEBUG
    else:
        log_level: int = logging.INFO

    log_format: str = "%(asctime)s  - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(level=log_level,
                        format=log_format,
                        handlers=[logging.StreamHandler(sys.stdout)],
                        force=True)

    return logging.getLogger(__name__)


def _main() -> None:
    args: argparse.Namespace = _parse_arguments()
    logger: logging.Logger = _setup_logging(args.verbose)

    try:
        model: transformers.PreTrainedModel
        tokenizer: transformers.PreTrainedTokenizer
        model, tokenizer = inference.load_model_and_tokenizer(
            args.model, logger)
    except Exception:
        logger.critical("Please ensure the model name is correct and you have"
                        " an internet connection if needed.")
        sys.exit(1)

    decision_function_config: typing.Dict[str, typing.Any] = {
        "probability_threshold": args.probability_threshold,
    }

    configured_decision_func: functools.partial = functools.partial(
        decision.simple_threshold_decision, config=decision_function_config)

    question_answering.run_qa_loop(
        model=model,
        tokenizer=tokenizer,
        logger=logger,
        max_length_per_turn=args.max_answer_length,
        temperature=args.temperature,
        top_k=args.top_k,
        backtrack_every_n=args.backtrack_every_n,
        backtracking_decision_function=configured_decision_func)


if __name__ == "__main__":
    _main()
