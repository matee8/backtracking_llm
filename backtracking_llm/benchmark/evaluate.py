import dataclasses
import json
import logging
import pathlib
import time
import typing

import lm_eval
from lm_eval import tasks
from lm_eval.api import model

from backtracking_llm.benchmark import utils


@dataclasses.dataclass
class Config:
    task_names: list[str | dict | object]
    fewshot: int
    bootstrap_iters: int
    output_dir: pathlib.Path
    device: str


class Evaluator:

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def run(self, lm: str | model.LM, model_args: dict[str, typing.Any] | None,
            limit: int | None, description: str, output_filename: str | None,
            gen_kwargs: dict[str, typing.Any] | None,
            save: dict[str, typing.Any] | None
            ) -> dict[str, typing.Any] | None:
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
                                               limit=limit,
                                               gen_kwargs=gen_kwargs)
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
                                             output_filename, save)

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

        pretrained = getattr(lm, "pretrained", None)
        if pretrained is None:
            raise ValueError("Unknown model name.")

        if isinstance(pretrained, str):
            return pretrained
        elif hasattr(pretrained, "name_or_path"):
            return pretrained.name_or_path
        else:
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

    def _execute_evaluation(
        self, lm: str | model.LM, model_args: dict[str, typing.Any] | None,
        limit: int | None, gen_kwargs: dict[str, typing.Any] | None
    ) -> dict[str, typing.Any] | None:
        if gen_kwargs is None:
            gen_kwargs_str = None
        else:
            gen_kwargs_list = []
            for key, value in gen_kwargs.items():
                gen_kwargs_list.append(f"{key}={str(value)}")

            gen_kwargs_str = ",".join(gen_kwargs_list)

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
                bootstrap_iters=self.config.bootstrap_iters,
                gen_kwargs=gen_kwargs_str,
                confirm_run_unsafe_code=True)
        else:
            print(type(lm))

            return lm_eval.simple_evaluate(
                model=lm,
                tasks=self.config.task_names,
                num_fewshot=self.config.fewshot,
                limit=limit,
                bootstrap_iters=self.config.bootstrap_iters,
                gen_kwargs=gen_kwargs_str,
                confirm_run_unsafe_code=True)

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
                      output_filename: str | None,
                      save: dict[str, typing.Any] | None) -> pathlib.Path:
        output_data = [results["results"], save, results["samples"]]

        if output_filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = (self.config.output_dir /
                           f"{description}_{timestamp}.json")
        else:
            output_path = self.config.output_dir / output_filename

        utils.save_json(output_data, output_path)

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

        utils.save_json(error_info, error_path)
        self.logger.info("Error details saved to: %s.", error_path)
