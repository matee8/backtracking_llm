"""Provides a stateless wrapper for runnning lm-eval."""

import logging
from typing import Any, Dict, Optional

import lm_eval

from backtracking_llm.benchmark.config import EvaluationConfig, GenerationConfig
from backtracking_llm.benchmark.model import BacktrackingLM
from backtracking_llm.decision import Operator
from backtracking_llm.generation import Generator

logger = logging.getLogger(__name__)


class Evaluator:
    """A wrapper for running a single evaluation using lm-eval."""

    def __init__(self, config: EvaluationConfig) -> None:
        """Initializes the evaluator."""
        self.config = config

    def run(self,
            generator: Generator,
            generation_config: GenerationConfig,
            operator: Optional[Operator] = None) -> Dict[str, Any]:
        """Executes a single evaluation run.

        Args:
            generator: The generator instance to use for the evaluation.
            generation_config: The parameters for text generation.
            operator: The decision operator to use. If None, a baseline run
                without backtracking is performed.

        Returns:
            A dictionary containing the full evaluation results from lm-eval.
        """
        logger.info('Starting evaluation for tasks: %s', self.config.tasks)

        lm = BacktrackingLM(generator=generator, operator=operator)

        gen_kwargs = {
            'max_gen_toks': generation_config.max_new_tokens,
            'temperature': generation_config.temperature,
            'top_k': generation_config.top_k,
        }

        results = lm_eval.simple_evaluate(model=lm,
                                          tasks=self.config.tasks,
                                          limit=self.config.limit,
                                          num_fewshot=self.config.num_fewshot,
                                          confirm_run_unsafe_code=True,
                                          gen_kwargs=gen_kwargs)

        if results is None:
            logger.warning('lm_eval.simple_evaluate returned None. '
                           'This indicates a potential critical error.')
            return {}

        logger.info('Evaluation finished.')
        return results

    @staticmethod
    def extract_primary_score(results: Dict[str, Any], task_name: str) -> float:
        """Extracts the primary score from an lm-eval results dictionary.

        Args:
            results: The complete results dictionary from `simple_evaluate`.
            task_name: The name of the task to extract the score from.

        Returns:
            The float value of the primary metric.

        Raises:
            ValueError: If the results dictionary is invalid or the task is
                not found.
            KeyError: If no standard accuracy metric is found for the task.
        """
        if 'results' not in results or task_name not in results['results']:
            raise ValueError(f"Task '{task_name}' not found in results.")

        task_results = results['results'][task_name]
        metric_keys = [
            'acc_norm,none',
            'acc,none',
            'exact_match,none',
            'pass@1,create_test',
            'bleu,none',
            'rouge2,none',
            'exact_match,strict-match',
        ]

        for key in metric_keys:
            if key in task_results:
                return float(task_results[key])

        raise KeyError(
            f"No standard primary metric found for task '{task_name}'. "
            f'Available metrics: {list(task_results.keys())}')
