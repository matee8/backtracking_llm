#!/usr/bin/env python3

import logging
import json
import os
import time
import typing
import sys

import lm_eval
from lm_eval.models import huggingface
from lm_eval.api import instance

from backtracking_llm.models import inference, decision

MODEL_NAME: typing.Final[str] = "Qwen/Qwen2.5-0.5B-Instruct"
TASK_NAME: typing.Final[str] = "hendrycks_math_algebra"
NUM_FEWSHOT: typing.Final[int] = 8
LIMIT_FOR_SEARCH: typing.Final[int] = 500
BACKTRACK_EVERY_N: typing.Final[int] = 5
OUTPUT_DIR: typing.Final[str] = "benchmark_results"
DEVICE: typing.Final[str] = "cpu"
DECISION_STRATEGIES: (
    typing.Final[list[typing.Type[decision.BacktrackStrategy]]]) = [
        decision.ProbabilityThreshold,
        decision.EntropyThreshold,
        decision.ProbabilityMargin,
        decision.ProbabilityDrop,
        decision.ProbabilityTrend,
        decision.Repetition,
        decision.NGramOverlap,
        decision.LogitThreshold,
    ]


class BacktrackingLM(huggingface.HFLM):

    def __init__(self,
                 backtracking_config: inference.BacktrackingInferenceConfig,
                 logger: logging.Logger, **kwargs) -> None:
        super().__init__(**kwargs)

        self._backtracking_config = backtracking_config
        self.logger = logger

        self.engine = self._setup_engine()
        logger.info("Initialized BacktrackingLM with engine using config: %s",
                    backtracking_config)

    def _setup_engine(self) -> inference.BacktrackingInferenceEngine:
        self._backtracking_config.device = str(self.device)

        try:
            if not isinstance(self.pretrained, str):
                raise TypeError("Model's pretrained field is a model instead "
                                "of a string")

            engine = inference.BacktrackingInferenceEngine(
                model_name=self.pretrained,
                logger=self.logger,
                config=self._backtracking_config)

            return engine
        except Exception:
            self.logger.error("Failed to initialize "
                              "BacktrackingInferenceEngine")
            raise

    def generate_until(self,
                       /,
                       requests: list[instance.Instance],
                       disable_tqdm: bool = False) -> list[str]:
        self.logger.debug("Received %d generation requests.", len(requests))

        results = []

        for req in requests:
            context = req.args[0]

            if not isinstance(req.args[0], str):
                self.logger.warning("Context is not of type 'str'. Skipping.")
                results.append("Generation error")
                continue

            try:
                self.logger.debug("Generating for context: '%s...'", context)

                token_ids = self.engine.generate(context, None)
                if token_ids is None:
                    self.logger.warning("Model did not return any generated "
                                        "text. Skipping.")
                    results.append("Generation error")
                    continue

                decoded = self.tokenizer.decode(token_ids[0])

                self.logger.debug("Raw generated text: '%s...'", decoded)

                results.append(decoded.strip())
            except Exception:
                self.logger.error(
                    "Error during generation for context '%s...'",
                    context[:50])
                results.append("Generation error")

        return results


def _run_evaluation(
        logger: logging.Logger,
        model_name: str,
        model_args: dict[str, typing.Any],
        task_names: list[str | dict | object],
        num_fewshot: int,
        limit: int | None = None,
        description: str = "eval",
        output_filename: str | None = None,
        bootstrap_iters: int = 1000) -> dict[str, typing.Any] | None:
    start_time = time.time()
    logger.info(
        "Starting evaulation: %s - Model: %s, Tasks: %s, Limit: %s, "
        "Fewshot: %d", description, model_name, task_names, limit, num_fewshot)

    model_args["device"] = DEVICE

    try:
        results = lm_eval.simple_evaluate(model=model_name,
                                          model_args=model_args,
                                          tasks=task_names,
                                          num_fewshot=num_fewshot,
                                          limit=limit,
                                          bootstrap_iters=bootstrap_iters)

        if results is None:
            raise ValueError("Simple evaluate returned None.")

        end_time = time.time()

        results["config"]["eval_details"] = {
            "description": description,
            "duration_seconds": end_time - start_time,
            "limit": limit,
            "num_fewshot": num_fewshot,
            "tasks": task_names,
            "model_type": model_name,
            "model_args": {
                k: str(v)
                for k, v in model_args.items()
            }
        }

        if output_filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            desc = "".join(c if c.isalnum() else "_" for c in description)
            output_filename = f"{desc}{timestamp}.json"

        output_path = os.path.join(OUTPUT_DIR, output_filename)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([results["results"], results["samples"]], f, indent=4)

        logger.info("Evaluation %s finished in %.2f seconds.", description,
                    end_time - start_time)
        logger.info("Results saved to: %s.", output_path)
        logger.info("Results summary:\n%s",
                    json.dumps(results["results"], indent=2))

        return results
    except Exception as e:
        logger.error("Evaluation %s failed.", description)

        error_info = {
            "error": str(e),
            "config": {
                "description": description,
                "limit": limit,
                "num_fewshot": num_fewshot,
                "tasks": task_names,
                "model_type": model_name,
                "model_args": {
                    k: str(v)
                    for k, v in model_args.items()
                }
            }
        }

        output_path = os.path.join(OUTPUT_DIR, f"ERROR_{description}.json")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(error_info, f, indent=4)

        return None


def _setup_logger() -> logging.Logger:
    log_level = logging.DEBUG

    log_format = "%(asctime)s  - %(name)s - %(levelname)s - %(message)s"

    handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(level=log_level,
                        format=log_format,
                        handlers=[handler],
                        force=True)

    return logging.getLogger(__name__)


def _main() -> None:
    logger = _setup_logger()

    logger.info("Starting benchmarking pipeline.")
    logger.info("Step 1: Running baseline evaluation (no backtracking).")

    baseline_model_args = {"pretrained": MODEL_NAME}

    baseline_results = _run_evaluation(
        logger,
        model_name="hf",
        model_args=baseline_model_args,
        task_names=[TASK_NAME],
        num_fewshot=NUM_FEWSHOT,
        limit=None,
        description="baseline_full_dataset",
        output_filename="results_baseline_full.json")

    if baseline_results is None:
        logger.error("Baseline evaluation failed. Aborting pipeline.")
        return

    try:
        score = (baseline_results["results"][TASK_NAME].get(
            "acc_norm,none",
            baseline_results["results"][TASK_NAME].get("acc,none")))

        if score is None:
            raise KeyError("Could not find standard accuracy metric "
                           "(acc_norm or acc).")

        logger.info("Baseline %s Accuracy (acc_norm/acc): %.4f", TASK_NAME,
                    score)
    except (KeyError, StopIteration) as e:
        logger.warning(
            "Could not extract baseline primary score "
            "automatically: %s", e)
        score = None

    logger.info("Step 1: Comparing decision strategies.")
    strategy_results = {}
    best_strategy_cls = None
    best_strategy_score = -1.0

    if not DECISION_STRATEGIES:
        raise ValueError("No decision strategies defined in "
                         "DECISION_STRATEGIES.")

    for strategy_cls in DECISION_STRATEGIES:
        strategy_name = strategy_cls.__name__
        logger.info("Evaluating strategy: %s", strategy_name)

        try:
            strategy_instance = strategy_cls()

            backtracking_config = inference.BacktrackingInferenceConfig(
                max_answer_length=64,
                top_k=50,
                temperature=1.0,
                backtrack_every_n=BACKTRACK_EVERY_N,
                backtrack_strategy=strategy_instance,
                device=DEVICE)

            model_args = {
                "pretrained": MODEL_NAME,
                "backtracking_config": backtracking_config,
            }

            results = _run_evaluation(
                logger,
                model_name="hf",
                model_args=model_args,
                task_names=[TASK_NAME],
                num_fewshot=NUM_FEWSHOT,
                limit=LIMIT_FOR_SEARCH,
                description=f"strategy_comparison_{strategy_name}",
                output_filename=(f"results_strategy_{strategy_name}_limit_"
                                 f"{LIMIT_FOR_SEARCH}.json"))

            if not results:
                logger.error("Strategy evaluation failed. Aborting pipeline.")
                return

            strategy_results[strategy_name] = results

            try:
                task_name = next(iter(results["results"].keys()))

                score = (results["results"][task_name].get(
                    "acc_norm,none",
                    results["results"][task_name].get("acc,none")))

                if score is None:
                    raise KeyError("Could not find standard accuracy metric "
                                   "(acc_norm or acc).")

                logger.info("Strategy %s score: %.4f", strategy_name, score)

                if score > best_strategy_score:
                    best_strategy_score = score
                    best_strategy_cls = strategy_cls
                    logger.info("New best strategy found.")
            except (KeyError, StopIteration) as e:
                logger.warning(
                    "Could not extract baseline primary score "
                    "automatically: %s", e)

        except Exception:
            logger.error("Failed to evaluate strategy: %s", strategy_name)

    if best_strategy_cls:
        logger.info("\nBest initial strategy: %s with score %.4f",
                    best_strategy_cls.__name__, best_strategy_score)


if __name__ == "__main__":
    _main()
