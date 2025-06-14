import dataclasses
import inspect
import logging
import random
import typing

from backtracking_llm.benchmark import evaluate, model, utils
from backtracking_llm.models import decision, inference


@dataclasses.dataclass
class Config:
    model_name: str
    backtrack_every_n: int
    batch_size: int
    device: str
    skip_base: bool
    skip_decision: bool
    skip_hparam_search: bool
    baseline_limit: int | None
    search_limit: int | None
    max_answer_length: int
    top_k: int
    temperature: float
    random_search_iters: int
    decision_strategies: list[typing.Type[decision.BacktrackStrategy]] = (
        dataclasses.field(default_factory=lambda: [
            decision.ProbabilityThreshold,
            decision.EntropyThreshold,
            decision.ProbabilityMargin,
            decision.ProbabilityDrop,
            decision.ProbabilityTrend,
            decision.Repetition,
            decision.NGramOverlap,
            decision.LogitThreshold,
        ]))


class BenchmarkRunner:

    def __init__(self, config: Config, evaluator_config: evaluate.Config,
                 logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

        self.evaluator = evaluate.Evaluator(evaluator_config, logger)

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
        if self.config.skip_decision:
            self.logger.info("Skipping due to configuration.")
            best_strategy = None
            best_score = None
        else:
            best_strategy, best_score = self._run_strategies()

        self.logger.info("-- Step 3: Random hyperparameter search. --")
        if self.config.skip_hparam_search:
            self.logger.info("Skipping due to configuration.")
            best_params_score = None
        else:
            best_params, best_params_score = self._run_hparam_search(
                best_strategy)
            self.logger.info("Random search best params=%s, score=%.4f",
                             best_params, best_params_score)

        if best_score is not None and best_params_score is not None:
            self.logger.info("Random search improved result by %d.",
                             best_params_score - best_score)

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

        # pylint: disable=inconsistent-quotes
        desc = f"baseline_{self.config.model_name.replace('/', '_')}"
        # pylint: enable=inconsistent-quotes

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
                                     gen_kwargs=gen_kwargs,
                                     save=None)

        if results is None:
            return None

        for task in self.evaluator.config.task_names:
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

        base_config = inference.BacktrackConfig(
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
                                             gen_kwargs=None,
                                             save=None)

                if not results:
                    self.logger.error("Strategy evaluation failed. Skipping.")
                    continue

                for task in self.evaluator.config.task_names:
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

    def _run_hparam_search(
        self, best_strategy: typing.Type[decision.BacktrackStrategy] | None
    ) -> tuple[dict[str, typing.Any], float]:
        base_config = inference.BacktrackConfig(
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

        best_score = float("-inf")
        best_params: dict[str, typing.Any] = {}

        freq_min = 1
        freq_max = max(self.config.backtrack_every_n, 1)

        for i in range(self.config.random_search_iters):
            random.seed()

            if best_strategy is None:
                strategy_cls = random.choice(self.config.decision_strategies)
            else:
                strategy_cls = best_strategy

            strategy_name = strategy_cls.__name__

            hparams = self._sample_hyperparams(strategy_cls)

            try:
                strategy = strategy_cls(**hparams)
            except Exception as e:
                self.logger.warning(
                    "Iteration %d: failed to build %s with %s: %s", i, i,
                    strategy_name, hparams, e)

                continue

            freq = random.randint(freq_min, freq_max)

            lm.engine.config.backtrack_strategy = strategy
            lm.engine.config.backtrack_every_n = freq

            desc = f"random_search_{strategy_name}_iter_{i}"
            filename = f"results_{desc}_limit_{self.config.search_limit}.json"

            save = {**hparams, "frequency": freq}

            self.logger.info("Iteration %d: eval %s; freq=%d; hparams=%s", i,
                             strategy_name, freq, hparams)

            results = self.evaluator.run(lm=lm,
                                         model_args=None,
                                         limit=self.config.search_limit,
                                         description=desc,
                                         output_filename=filename,
                                         gen_kwargs=None,
                                         save=save)
            if not results:
                self.logger.error("Iteration %d: evaluation failed, skipping.",
                                  i)
                continue

            for task in self.evaluator.config.task_names:
                if not isinstance(task, str):
                    continue

                try:
                    score = utils.extract_primary_score(
                        results["results"], task)
                    self.logger.info("Iteration %d: %s score=%.4f", i,
                                     strategy_name, score)

                    if best_score < score:
                        best_score = score
                        best_params = {
                            "strategy": strategy_name,
                            "frequency": freq,
                            **hparams
                        }

                        self.logger.info("New best: iter %d: %s (score=%.4f)",
                                         i, best_params, best_score)
                except Exception as e:
                    self.logger.warning(
                        "Iter %d: could not extract score for "
                        "%s: %s", i, task, e)

        return best_params, best_score

    def _sample_hyperparams(
        self, strategy_cls: typing.Type[decision.BacktrackStrategy]
    ) -> dict[str, typing.Any]:
        hparams: dict[str, typing.Any] = {}
        sig = inspect.signature(strategy_cls.__init__)

        for name, param in sig.parameters.items():
            if name == "self" or param.default is inspect.Parameter.empty:
                continue

            default = param.default

            if isinstance(default, bool):
                hparams[name] = random.choice([True, False])
            elif isinstance(default, float):
                low = 0.0
                if 0.0 <= default <= 1.0:
                    high = 1.0
                else:
                    high = default * 2
                hparams[name] = random.uniform(low, high)
            elif isinstance(default, int):
                low = 1
                high = max(default * 2, 1)
                hparams[name] = random.randint(low, high)

        return hparams
