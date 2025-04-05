#!/usr/bin/env python3

import argparse
import logging
import typing
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
        help="Model name to use (default: %(default)s)",
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
        help="Maximum number of tokens to generate (default: %(default)s)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top tokens to analyze (default: %(default)s)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: %(default)s)",
    )

    parser.add_argument(
        "--answer-start",
        type=str,
        help="The start of the answer."
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

    try:
        model: transformers.PreTrainedModel
        tokenizer: transformers.PreTrainedTokenizer
        model, tokenizer = inference.load_model_and_tokenizer(args.model)
    except Exception:
        logger.error("Failed to load model %s", args.model, exc_info=True)
        logger.error("Please ensure the model name is correct and you have an" \
                     " internet connection if needed.")
        sys.exit(1)

    try:
        chat: typing.List[typing.Dict[str, str]] = [
            {"role": "user", "content": args.prompt},
        ]

        if args.answer_start is not None:
            chat.append({"role": "assistant", "content": args.answer_start})

        formatted_prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            continue_final_message=True
        )

        if not isinstance(formatted_prompt, str):
            logger.error(
                "Failed to apply chat template for model %s", args.model)
            sys.exit(1)

        inference.run_inference_loop(model=model,
                                tokenizer=tokenizer,
                                prompt=formatted_prompt,
                                max_length=args.max_length,
                                top_k=args.top_k,
                                logger=logger,
                                temperature=args.temperature)
    except Exception as e:
        logger.error("An error occurred during the inference loop: %e",
                     e,
                     exc_info=True)
        sys.exit(1) 


if __name__ == "__main__":
    _main()
