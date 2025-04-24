#!/usr/bin/env python3

import argparse
import logging
import pathlib
import sys
import typing

from backtracking_llm.benchmark import config, runner
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
                        default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="name of the model to run benchmarks with")

    parser.add_argument("--task-names",
                        nargs="+",
                        type=str,
                        default=["hendrycks_math_algebra"],
                        help="list of evaluation tasks")

    parser.add_argument("--fewshot",
                        type=int,
                        default=8,
                        help="number of examples in a few-shot context")

    parser.add_argument("--backtrack-every-n",
                        type=int,
                        default=5,
                        help="count of tokens between calling backtracking"
                        "decision function.")

    parser.add_argument("--output-dir",
                        type=pathlib.Path,
                        default=pathlib.Path("benchmark_results"),
                        help="directory which results will be saved to")

    parser.add_argument("--device",
                        type=str,
                        default="cpu",
                        help="device which benchmarks will run on")

    parser.add_argument("--base-limit",
                        type=int,
                        default=None,
                        help="limit of instances passed to the baseline "
                        "evaluation")

    parser.add_argument("--search-limit",
                        type=int,
                        default=500,
                        help="limit of instances passed to the best strategy "
                        "evaluation")

    parser.add_argument("--final-limit",
                        type=int,
                        default=None,
                        help="limit of instances passed to the best strategy "
                        "evaluation")

    parser.add_argument(
        "--max-answer-length",
        type=int,
        default=64,
        help="max length of the generated answer by bactracking"
        " model in evaluation")

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="controls the sampling strategy by limiting the "
        "next token prediction pool to the k most likely tokens")

    parser.add_argument("--temperature",
                        type=float,
                        default=1.0,
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

    benchmark_config = config.BenchmarkConfig(
        model_name=args.model_name,
        task_names=args.task_names,
        fewshot=args.fewshot,
        backtrack_every_n=args.backtrack_every_n,
        output_dir=args.output_dir,
        device=args.device,
        skip_base=args.skip_base,
        baseline_limit=args.base_limit,
        search_limit=args.search_limit,
        final_limit=args.final_limit,
        max_answer_length=args.max_answer_length,
        top_k=args.top_k,
        temperature=args.temperature)

    if args.decision_functions is not None:
        benchmark_config.decision_strategies = []
        for decision_strategy in args.decision_functions:
            benchmark_config.decision_strategies.append(
                DECISION_MAP[decision_strategy])

    benchmark_runner = runner.BenchmarkRunner(benchmark_config, logger)

    benchmark_runner.run()


if __name__ == "__main__":
    main()
