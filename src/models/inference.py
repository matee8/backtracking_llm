#!/usr/bin/env python3

from argparse import Namespace, ArgumentParser
import logging
from logging import StreamHandler
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
import typing
import sys


def load_model_and_tokenizer(
    model_name: str = "gpt2",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> typing.Tuple[PreTrainedModel, PreTrainedTokenizer]:
    try:
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name)
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name)
        model.to(device)
        model.eval()

        return model, tokenizer
    except Exception as e:
        logging.error("Failed to load model or tokenizer: %s", str(e))
        raise ValueError(f"Error loading model or tokenizer: {str(e)}") from e


def _parse_arguments() -> Namespace:
    parser = ArgumentParser(
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
        help="Prompt to start generation",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        defualt=100,
        help="Maximum number of tokens to generate (default: 10)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top tokens to analyze (default: 50)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )

    parser.add_argument("--verbose",
                        action="store_true",
                        help="Enable verbose logging")

    return parser.parse_args()


def _setup_logging(verbose: bool = False) -> None:
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[StreamHandler(sys.stdout)],
    )


if __name__ == "__main__":
    args: Namespace = _parse_arguments()
    _setup_logging(args.verbose)
