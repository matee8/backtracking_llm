#!/usr/bin/env python3

import argparse
import logging
import pathlib
import sys
import typing

from backtracking_llm.benchmark import evaluate, runner
from backtracking_llm.models import decision

DECISION_MAP: typing.Final[dict[str,
                                typing.Type[decision.BacktrackStrategy]]] = {
                                    "probability_threshold":
                                    decision.ProbabilityThreshold,
                                    "entropy_threshold":
                                    decision.EntropyThreshold,
                                    "probability_margin":
                                    decision.ProbabilityMargin,
                                    "probability_drop":
                                    decision.ProbabilityDrop,
                                    "probability_trend":
                                    decision.ProbabilityTrend,
                                    "repetition": decision.Repetition,
                                    "ngram_overlap": decision.NGramOverlap,
                                    "logit_threshold": decision.LogitThreshold,
                                }

DEFAULT_MODEL_NAME: typing.Final[str] = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TASK_NAMES: typing.Final[list[str]] = ["hendrycks_math_algebra"]
DEFAULT_FEWSHOT: typing.Final[int] = 8
DEFAULT_BACKTRACK_EVERY_N: typing.Final[int] = 5
DEFAULT_OUTPUT_DIR: typing.Final[pathlib.Path] = pathlib.Path(
    "benchmark_results")
DEFAULT_DEVICE: typing.Final[str] = "cpu"
DEFAULT_BASE_LIMIT: typing.Final[int | None] = None
DEFAULT_SEARCH_LIMIT: typing.Final[int | None] = None
DEFAULT_MAX_ANSWER_LENGTH: typing.Final[int] = 1024
DEFAULT_TOP_K: typing.Final[int] = 50
DEFAULT_TEMPERATURE: typing.Final[float] = 1.0

BOOTSTRAP_ITERS = 1000
BATCH_SIZE = 1


def _setup_logger(verbose: bool = False) -> logging.Logger:
    if verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    log_format = "%(asctime)s  - %(name)s - %(levelname)s - %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=level,
                        format=log_format,
                        handlers=[handler],
                        force=True)
    return logging.getLogger(__name__)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="benchmark base vs backtracking language models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model-name",
                        type=str,
                        default=DEFAULT_MODEL_NAME,
                        help="name of the model to run benchmarks with")

    parser.add_argument("--task-names",
                        nargs="+",
                        type=str,
                        default=DEFAULT_TASK_NAMES,
                        help="list of evaluation tasks")

    parser.add_argument("--fewshot",
                        type=int,
                        default=DEFAULT_FEWSHOT,
                        help="number of examples in a few-shot context")

    parser.add_argument("--backtrack-every-n",
                        type=int,
                        default=DEFAULT_BACKTRACK_EVERY_N,
                        help="count of tokens between calling backtracking"
                        "decision function.")

    parser.add_argument("--output-dir",
                        type=pathlib.Path,
                        default=DEFAULT_OUTPUT_DIR,
                        help="directory which results will be saved to")

    parser.add_argument("--device",
                        type=str,
                        default=DEFAULT_DEVICE,
                        help="device which benchmarks will run on")

    parser.add_argument("--base-limit",
                        type=int,
                        default=DEFAULT_BASE_LIMIT,
                        help="limit of instances passed to the baseline "
                        "evaluation")

    parser.add_argument("--search-limit",
                        type=int,
                        default=DEFAULT_SEARCH_LIMIT,
                        help="limit of instances passed to the best strategy "
                        "evaluation")

    parser.add_argument(
        "--max-answer-length",
        type=int,
        default=DEFAULT_MAX_ANSWER_LENGTH,
        help="max length of the generated answer by bactracking"
        " model in evaluation")

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="controls the sampling strategy by limiting the "
        "next token prediction pool to the k most likely tokens")

    parser.add_argument("--temperature",
                        type=float,
                        default=DEFAULT_TEMPERATURE,
                        help="controls the creativity or randomness of the "
                        "generated text")

    parser.add_argument("--skip-base",
                        action="store_true",
                        help="skip baseline model benchmarking")

    parser.add_argument("--decision-functions",
                        nargs="+",
                        help="the name of the decision function to use",
                        choices=DECISION_MAP.keys())

    parser.add_argument("--verbose",
                        action="store_true",
                        help="print more useful debugging logs")

    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    logger = _setup_logger(args.verbose)

    evaluator_config = evaluate.Config(task_names=args.task_names,
                                       fewshot=args.fewshot,
                                       bootstrap_iters=BOOTSTRAP_ITERS,
                                       output_dir=args.output_dir,
                                       device=args.device)

    benchmark_config = runner.Config(model_name=args.model_name,
                                     backtrack_every_n=args.backtrack_every_n,
                                     skip_base=args.skip_base,
                                     baseline_limit=args.base_limit,
                                     search_limit=args.search_limit,
                                     max_answer_length=args.max_answer_length,
                                     top_k=args.top_k,
                                     temperature=args.temperature,
                                     batch_size=BATCH_SIZE,
                                     device=args.device)

    if args.decision_functions is not None:
        benchmark_config.decision_strategies = []
        for decision_strategy in args.decision_functions:
            benchmark_config.decision_strategies.append(
                DECISION_MAP[decision_strategy])

    benchmark_runner = runner.BenchmarkRunner(benchmark_config,
                                              evaluator_config, logger)

    benchmark_runner.run()


if __name__ == "__main__":
    main()
