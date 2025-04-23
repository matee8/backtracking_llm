import json
import logging
import pathlib
import time
import typing

import lm_eval
from lm_eval import tasks
from lm_eval.api import model

from backtracking_llm.benchmark import config, utils


class Evaluator:

    def __init__(self, benchmark_config: config.BenchmarkConfig,
                 logger: logging.Logger) -> None:
        self.config = benchmark_config
        self.logger = logger

    def run(self, lm: str | model.LM, model_args: dict[str, typing.Any] | None,
            limit: int | None, description: str,
            output_filename: str | None) -> dict[str, typing.Any] | None:
        lm_name = self._get_model_name(lm, model_args)

        self.logger.info(
            "Starting evaluation '%s' - Model: %s, Tasks: %s, "
            "Fewshot: %s, Limit: %s", description, lm_name,
            self.config.task_names, self.config.fewshot, limit)

        start_time = time.time()

        try:
            prepared_args = self._prepare_model_args(lm, model_args)

            results = self._execute_evaluation(lm=lm,
                                               model_args=model_args,
                                               limit=limit)
            if results is None:
                raise ValueError("'lm_eval.simple_evaluate()' returned None")

            end_time = time.time()

            processed_results = self._process_results(results=results,
                                                      description=description,
                                                      start_time=start_time,
                                                      end_time=end_time,
                                                      limit=limit,
                                                      lm_name=lm_name,
                                                      model_args=prepared_args)

            output_path = self._save_results(results, description,
                                             output_filename)

            self.logger.info("Evaluation %s finished in %.2f seconds.",
                             description, end_time - start_time)
            self.logger.info("Results saved to: %s.", output_path)

            if "results" in results:
                self.logger.info("Results summary:\n%s",
                                 json.dumps(results["results"], indent=2))

            return processed_results
        except Exception as e:
            self.logger.error("Evaluation %s failed: %s",
                              description,
                              e,
                              exc_info=True)

            self._handle_error(error=e,
                               description=description,
                               limit=limit,
                               lm_name=lm_name)

            return None

    def _get_model_name(self, lm: str | model.LM,
                        model_args: dict[str, typing.Any] | None) -> str:
        if isinstance(lm, str):
            if model_args:
                return model_args.get("pretrained", lm)
            else:
                return lm
        elif hasattr(lm, "pretrained"):
            if isinstance(lm.pretrained, str):
                return lm.pretrained
            elif hasattr(lm.pretrained, "name_or_path"):
                return lm.pretrained.name_or_path

        raise ValueError("Unknown model name.")

    def _prepare_model_args(
            self, lm: str | model.LM,
            model_args: dict[str, typing.Any] | None) -> dict[str, typing.Any]:
        if not isinstance(lm, str):
            return {}

        if model_args is None:
            return {"device": self.config.device}
        else:
            model_args["device"] = self.config.device
            return model_args

    def _execute_evaluation(self, lm: str | model.LM,
                            model_args: dict[str, typing.Any] | None,
                            limit: int | None) -> dict[str, typing.Any] | None:
        if isinstance(lm, str):
            if model_args is None:
                raise ValueError("'model_args' cannot be None if model isn't "
                                 "initialized")

            return lm_eval.simple_evaluate(
                model=lm,
                model_args=model_args,
                tasks=self.config.task_names,
                num_fewshot=self.config.fewshot,
                limit=limit,
                bootstrap_iters=self.config.bootstrap_iters)
        else:
            return lm_eval.simple_evaluate(
                model=lm,
                tasks=self.config.task_names,
                num_fewshot=self.config.fewshot,
                limit=limit,
                bootstrap_iters=self.config.bootstrap_iters)

    def _process_results(self, results: dict[str, typing.Any],
                         description: str, start_time: float, end_time: float,
                         limit: int | None, lm_name: str,
                         model_args: dict) -> dict[str, typing.Any]:
        if "config" not in results:
            results["config"] = {}

        results["config"]["eval_details"] = {
            "description": description,
            "duration_seconds": end_time - start_time,
            "tasks": tasks,
            "model_type": lm_name,
            "model_args": {
                k: str(v)
                for k, v in model_args.items()
            },
            "num_fewshot": self.config.fewshot,
            "limit": limit
        }

        return results

    def _save_results(self, results: dict[str, typing.Any], description: str,
                      output_filename: str | None) -> pathlib.Path:
        output_data = [results["results"], results["samples"]]

        if output_filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = (self.config.output_dir /
                           f"{description}_{timestamp}.json")
        else:
            output_path = self.config.output_dir / output_filename

        utils._save_json(output_data, output_path)

        return output_path

    def _handle_error(self, error: Exception, description: str,
                      limit: int | None, lm_name: str | None) -> None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        error_info = {
            "error_message": str(error),
            "evaluation_run_details": {
                "description": description,
                "limit": limit,
                "num_fewshot": self.config.fewshot,
                "tasks": self.config.task_names,
                "model_evaluated": lm_name,
                "timestamp": timestamp
            }
        }

        safe_desc = "".join(c if c.isalnum() else "_" for c in description)
        error_path = (self.config.output_dir /
                      f"ERROR_{safe_desc}_{timestamp}.json")

        utils._save_json(error_info, error_path)
        self.logger.info("Error details saved to: %s.", error_path)
