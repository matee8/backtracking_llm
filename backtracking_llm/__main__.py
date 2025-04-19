#!/usr/bin/env python3

import argparse
import logging
import os
import typing
import sys

import platformdirs

from backtracking_llm.models import decision, inference, question_answering

DEFAULT_MAX_LENGTH: typing.Final[int] = 100
DEFAULT_TOP_K: typing.Final[int] = 50
DEFAULT_TEMPERATURE: typing.Final[float] = 1.
DEFAULT_BACKTRACK_EVERY_N: typing.Final[int] = 5
DEFAULT_PROBABILITY_THRESHOLD: typing.Final[float] = .5
DEFAULT_DEVICE: typing.Final[str] = "cpu"


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on any model and use backtracking to remove "
        "already generated tokens",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model",
                        type=str,
                        help="id of the pre-trained model "
                        "hosted on the Hugging Face model hub")

    parser.add_argument("--max-answer-length",
                        type=int,
                        default=DEFAULT_MAX_LENGTH,
                        help="maximum number of tokens to generate for each "
                        " answer")

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="controls the sampling strategy by limiting the "
        "next token prediction pool to the k most likely tokens")

    parser.add_argument("--verbose",
                        action="store_true",
                        help="enable verbose logging")

    parser.add_argument("--temperature",
                        type=float,
                        default=DEFAULT_TEMPERATURE,
                        help="controls the creativity or randomness of the "
                        "generated text")

    parser.add_argument("--backtrack-every-n",
                        type=int,
                        default=DEFAULT_BACKTRACK_EVERY_N,
                        help="check for backtracking every N generated tokens")

    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=DEFAULT_PROBABILITY_THRESHOLD,
        help="probability threshold for the simple backtracking"
        "decision function")

    parser.add_argument("--device",
                        type=str,
                        default=DEFAULT_DEVICE,
                        help="device which the model inference will run on")

    parser.add_argument("--log-stdout",
                        action="store_true",
                        help="wrte logs to stdout instead of file")

    return parser.parse_args()


def _setup_logging(verbose: bool = False,
                   log_stdout: bool = False) -> logging.Logger:
    if verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    log_format = "%(asctime)s  - %(name)s - %(levelname)s - %(message)s"

    if log_stdout:
        handler = logging.StreamHandler(sys.stdout)
    else:
        log_dir = platformdirs.user_log_dir("backtracking_llm",
                                            appauthor="matee8")
        log_filename = "backtracking_llm.log"
        full_log_path = os.path.join(log_dir, log_filename)
        os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(full_log_path)

    logging.basicConfig(level=log_level,
                        format=log_format,
                        handlers=[handler],
                        force=True)

    return logging.getLogger(__name__)


def _main() -> None:
    args = _parse_arguments()
    logger = _setup_logging(args.verbose, args.log_stdout)

    inference_config = inference.InferenceConfig(
        max_answer_length=args.max_answer_length,
        top_k=args.top_k,
        temperature=args.temperature,
        backtrack_every_n=args.backtrack_every_n,
        backtrack_fn=decision.simple_threshold_decision,
        backtrack_fn_config={
            "probability_threshold": args.probability_threshold,
        },
        device=args.device)

    try:
        engine = inference.InferenceEngine(model_name=args.model,
                                           logger=logger,
                                           config=inference_config)
    except inference.ModelInitializationError as e:
        logger.error("Failed to initialize the model or tokenizer: %s", e)
        logger.error(
            "Please ensure the model name ('%s') is correct, "
            "dependencies are installed, and you have an internet "
            "connection if needed.", args.model)
        sys.exit(1)

    chat_session = question_answering.ChatSession(engine=engine, logger=logger)
    try:
        chat_session.run()
    except Exception as e:
        logger.error("An error occured during question answering loop: %s",
                     e,
                     exc_info=True)


if __name__ == "__main__":
    _main()
