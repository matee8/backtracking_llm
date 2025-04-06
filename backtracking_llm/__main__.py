#!/usr/bin/env python3

import argparse
import logging
import sys

import transformers

from backtracking_llm.models import inference, question_answering

DEFAULT_MAX_LENGTH = 100
DEFAULT_TOP_K = 50
DEFAULT_TEMPERATURE = 1.


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on any model and analyze token logits "
        "and probabilites")

    parser.add_argument("--model", type=str, help="Model name to use")

    parser.add_argument(
        "--max-answer-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum number of tokens to generate for each answer "
        "(default: %(default)s)")

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of top tokens to analyze (default: %(default)s)")

    parser.add_argument("--verbose",
                        action="store_true",
                        help="Enable verbose logging")

    parser.add_argument("--temperature",
                        type=float,
                        default=DEFAULT_TEMPERATURE,
                        help="Sampling temperature (default: %(default)s)")

    parser.add_argument("--answer-start",
                        type=str,
                        help="The start of the answer.")

    return parser.parse_args()


def _setup_logging(verbose: bool = False) -> logging.Logger:
    log_level: int = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(level=log_level,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    return logging.getLogger(__name__)


def _main() -> None:
    args: argparse.Namespace = _parse_arguments()
    logger: logging.Logger = _setup_logging(args.verbose)

    try:
        model: transformers.PreTrainedModel
        tokenizer: transformers.PreTrainedTokenizer
        model, tokenizer = inference.load_model_and_tokenizer(args.model)
    except Exception:
        logger.critical("Failed to load model %s", args.model, exc_info=True)
        logger.critical("Please ensure the model name is correct and you have"
                        " an internet connection if needed.")
        sys.exit(1)

    question_answering.run_qa_loop(model=model,
                                   tokenizer=tokenizer,
                                   logger=logger,
                                   max_length_per_turn=args.max_answer_length,
                                   top_k=args.top_k,
                                   temperature=args.temperature)


if __name__ == "__main__":
    _main()
