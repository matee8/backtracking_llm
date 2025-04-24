import logging
import typing

from backtracking_llm.benchmark import config, evaluate, model, utils
from backtracking_llm.models import decision, inference


class BenchmarkRunner:

    def __init__(self, benchmark_config: config.BenchmarkConfig,
                 logger: logging.Logger) -> None:
        self.config = benchmark_config
        self.logger = logger

        self.evaluator = evaluate.Evaluator(benchmark_config, logger)

    def run(self) -> None:
        self.logger.info("Starting benchmarking pipeline.")

        self.logger.info("-- Step 1: Running baseline evaluation. --")
        if self.config.skip_base:
            self.logger.info("Skipping due to configuration.")
        else:
            baseline_results = self._run_baseline()
            if baseline_results is None:
                self.logger.error(
                    "Baseline evaluation failed. Aborting pipeline.")
                return

        self.logger.info("-- Step 2: Comparing decision strategies. --")
        best_strategy, score = self._run_strategies()

        self.logger.info("Benchmarking pipeline finished.")

    def _run_baseline(self) -> dict[str, typing.Any] | None:
        model_args = {
            "pretrained": self.config.model_name,
            "batch_size": 1,
            "device": self.config.device,
        }

        gen_kwargs = {
            "do_sample": True,
            "temperature": self.config.temperature,
            "top_k": self.config.top_k,
            "top_p": 1.0
        }

        desc = f"baseline_{self.config.model_name.replace('/', '_')}"
        limit = self.config.baseline_limit
        if limit is not None:
            desc += f"_limit_{limit}"

        filename = f"results_{desc}.json"

        self.logger.info("Running baseline evaluation.")

        results = self.evaluator.run(lm="hf",
                                     model_args=model_args,
                                     limit=limit,
                                     description=desc,
                                     output_filename=filename,
                                     gen_kwargs=gen_kwargs)

        if results is None:
            return None

        for task in self.config.task_names:
            if isinstance(task, str):
                try:
                    score = utils.extract_primary_score(
                        results["results"], task)
                    self.logger.info("Baseline accuracy: %.4f", score)
                except Exception as e:
                    self.logger.warning(
                        "Could not extract primary baseline "
                        "score: %s", e)

        return results

    def _run_strategies(
            self
    ) -> tuple[typing.Type[decision.BacktrackStrategy] | None, float]:
        if not self.config.decision_strategies:
            raise ValueError("No decision strategies defined in config")

        base_config = inference.BacktrackingInferenceConfig(
            max_answer_length=self.config.max_answer_length,
            top_k=self.config.top_k,
            temperature=self.config.temperature,
            backtrack_every_n=self.config.backtrack_every_n,
            device=self.config.device)

        model_args = {
            "pretrained": self.config.model_name,
            "batch_size": 1,
            "backtracking_config": base_config,
            "logger": self.logger,
            "device": self.config.device,
        }

        lm = model.BacktrackingLM.create_from_arg_obj(model_args)

        limit = self.config.search_limit

        best_strategy = None
        best_score = float("-inf")

        for strategy_cls in self.config.decision_strategies:
            strategy_name = strategy_cls.__name__
            self.logger.info("Evaluating strategy: %s", strategy_name)

            try:
                lm.engine.config.backtrack_strategy = strategy_cls()

                desc = f"strategy_search_{strategy_name}"
                filename = f"results_{desc}_limit_{limit}.json"

                results = self.evaluator.run(lm=lm,
                                             model_args=None,
                                             limit=limit,
                                             description=desc,
                                             output_filename=filename,
                                             gen_kwargs=None)

                if not results:
                    self.logger.error("Strategy evaluation failed. Skipping.")
                    continue

                for task in self.config.task_names:
                    if isinstance(task, str):
                        try:
                            score = utils.extract_primary_score(
                                results["results"], task)

                            self.logger.info("Strategy %s accuracy: %.4f",
                                             strategy_name, score)

                            if score > best_score:
                                best_strategy = strategy_cls
                                best_score = score
                                self.logger.info("-- New best strategy found. "
                                                 "--")
                        except Exception as e:
                            self.logger.warning(
                                "Could not extract primary "
                                "score for strategy %s: %s", strategy_name, e)
            except Exception as e:
                self.logger.error("Failed to evaluate strategy %s: %s",
                                  strategy_name,
                                  e,
                                  exc_info=True)

        if best_strategy:
            self.logger.info("Best strategy: %s with score %.4f.",
                             best_strategy.__name__, best_score)

        return best_strategy, best_score
