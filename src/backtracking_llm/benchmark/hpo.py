"""Provides a class for running hyperparameter optimization using Optuna."""

import logging
from typing import Any, Dict, Type

import optuna
from optuna.study import Study
from optuna.trial import Trial

from backtracking_llm import decision
from backtracking_llm.benchmark.config import (EvaluationConfig,
                                               GenerationConfig, HPOConfig)
from backtracking_llm.benchmark.evaluator import Evaluator
from backtracking_llm.generation import Generator

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Manages the hyperparameter optimization process using Optuna."""

    def __init__(self, hpo_config: HPOConfig, eval_config: EvaluationConfig,
                 gen_config: GenerationConfig, generator: Generator,
                 operator_name: str) -> None:
        """Initializes the optimizer.

        Args:
            hpo_config: The HPO configuration, including the search space.
            eval_config: The configuration for the evaluation run in each trial.
            gen_config: The configuration for generation.
            generator: The generator instance to use for all trials.
            operator_name: The string name of the Operator class to optimize.

        Raises:
            ValueError: If the `operator_name` is not a valid decision Operator.
        """
        self._hpo_config = hpo_config
        self._eval_config = eval_config
        self._gen_config = gen_config
        self._generator = generator
        self._operator_cls = self._get_operator_class(operator_name)
        self._evaluator = Evaluator(self._eval_config)

    def optimize(self) -> Study:
        """Runs the Optuna optimization study.

        Returns:
            The completed Optuna study object, which contains information about
            all trials and the best parameters found.
        """
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective,
                       n_trials=self._hpo_config.n_trials,
                       show_progress_bar=True)
        return study

    def _objective(self, trial: Trial) -> float:
        """The objective function for a single Optuna trial.

        This method samples hyperparameters, instantiates the operator, runs an
        evaluation, and returns the primary score.

        Args:
            trial: An Optuna Trial object used for sampling parameters.

        Returns:
            The primary score for the evaluation task.
        """
        try:
            params = self._sample_params(trial)
            operator = self._operator_cls(**params)
            logger.info('Trial %d: testing %s with params: %s', trial.number,
                        self._operator_cls.__name__, params)

            results = self._evaluator.run(self._generator, self._gen_config,
                                          operator)

            primary_task = self._eval_config.tasks[0]
            score = Evaluator.extract_primary_score(results, str(primary_task))

            logger.info('Trial %d score: %.4f', trial.number, score)
            return score
        except (ValueError, TypeError, KeyError, RuntimeError):
            logger.exception('Trial %d failed.', trial.number)
            return -1.0

    def _sample_params(self, trial: Trial) -> Dict[str, Any]:
        """Samples parameters for a trial based on the search_space config."""
        params: Dict[str, Any] = {}

        for name, range_list in self._hpo_config.search_space.items():
            low, high = range_list

            if isinstance(low, float) or isinstance(high, float):
                params[name] = trial.suggest_float(name, float(low),
                                                   float(high))
            elif isinstance(low, int) and isinstance(high, int):
                params[name] = trial.suggest_int(name, low, high)
            else:
                raise TypeError(
                    f"Unsupported type for hyperparameter '{name}'. "
                    'Only int and float ranges are supported.')

        return params

    @staticmethod
    def _get_operator_class(name: str) -> Type[decision.Operator]:
        """Retrieves an Operator class from the decision module by its name."""
        if not hasattr(decision, name):
            raise ValueError(f"'{name}' is not a valid Operator class name.")

        operator_cls = getattr(decision, name)
        if not isinstance(operator_cls, type) or not issubclass(
                operator_cls, decision.Operator):
            raise ValueError(f"'{name}' is not a valid Operator class name.")

        return operator_cls
