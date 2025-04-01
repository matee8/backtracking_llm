#!/usr/bin/env python3

import argparse
import logging
import sys

import transformers

from backtracking_llm.models import inference


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on GPT-2 and analyze token logits " \
        "and probabilites"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name to use (default: gpt2)",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt to start generation",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (default: 10)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top tokens to analyze (default: 50)",
    )

    parser.add_argument("--verbose",
                        action="store_true",
                        help="Enable verbose logging")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )

    return parser.parse_args()


def _setup_logging(verbose: bool = False) -> logging.Logger:
    log_level: int = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    return logging.getLogger(__name__)


def _main() -> None:
    args: argparse.Namespace = _parse_arguments()
    logger: logging.Logger = _setup_logging(args.verbose)

    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    model, tokenizer = inference.load_model_and_tokenizer(args.model)

    inference.run_inference_loop(model=model,
                              tokenizer=tokenizer,
                              prompt=args.prompt,
                              max_length=args.max_length,
                              top_k=args.top_k,
                              logger=logger,
                              temperature=args.temperature)


if __name__ == "__main__":
    _main()
