#!/usr/bin/env python3

import argparse
import logging
import pathlib
import sys

from backtracking_llm.benchmark import config, runner


def _setup_logger() -> logging.Logger:
    level = logging.DEBUG
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
                        desc="name of the model to run benchmarks with")

    parser.add_argument("--task-names",
                        type=list[str],
                        default=["hendrycks_math_algebra"],
                        desc="list of evaluation tasks")

    parser.add_argument("--fewshot",
                        type=int,
                        default=8,
                        desc="number of examples in a few-shot context")

    parser.add_argument("--backtrack-every-n",
                        type=int,
                        default=5,
                        desc="count of tokens between calling backtracking"
                        "decision function.")

    parser.add_argument("--output-dir",
                        type=pathlib.Path,
                        default=pathlib.Path("benchmark_results"),
                        desc="directory which results will be saved to")

    parser.add_argument("--device",
                        type=str,
                        default="cpu",
                        desc="device which benchmarks will run on")

    parser.add_argument("--baseline-limit",
                        type=int | None,
                        default=None,
                        desc="limit of instances passed to the baseline "
                        "evaluation")

    parser.add_argument("--search-limit",
                        type=int | None,
                        default=500,
                        desc="limit of instances passed to the best strategy "
                        "evaluation")

    parser.add_argument("--final-limit",
                        type=int,
                        default=None,
                        desc="limit of instances passed to the best strategy "
                        "evaluation")

    parser.add_argument(
        "--max-answer-length",
        type=int,
        default=64,
        desc="max length of the generated answer by bactracking"
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

    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    logger = _setup_logger()

    benchmark_config = config.BenchmarkConfig(
        model_name=args.model_name,
        task_names=args.task_names,
        fewshot=args.fewshot,
        backtrack_every_n=args.backtrack_every_n,
        output_dir=args.output_dir,
        device=args.device,
        baseline_limit=args.baseline_limit,
        search_limit=args.search_limit,
        final_limit=args.final_limit,
        backtracking_max_answer_length=args.max_answer_length,
        backtracking_top_k=args.top_k,
        backtracking_temperature=args.temperature)

    benchmark_runner = runner.BenchmarkRunner(benchmark_config, logger)

    benchmark_runner.run()


if __name__ == "__main__":
    main()
