"""Provides the main orchestrator for the benchmarking pipeline."""

import json
import logging
from typing import Any, Dict
from pathlib import Path

from backtracking_llm.benchmark.config import BenchmarkingConfig
from backtracking_llm.benchmark.evaluator import Evaluator
from backtracking_llm.benchmark.hpo import HyperparameterOptimizer
from backtracking_llm.decision import Never
from backtracking_llm.generation import Generator

logger = logging.getLogger(__name__)


def _sanitize_for_json(data: Any) -> Any:
    """Recursively sanitizes a data structure to make it JSON serializable."""
    if isinstance(data, (int, float, str, bool, type(None))):
        return data
    if isinstance(data, dict):
        return {str(k): _sanitize_for_json(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_sanitize_for_json(item) for item in data]

    return str(data)


def _save_results_json(data: Dict[str, Any], output_path: Path) -> None:
    """Saves a dictionary to a JSON file, creating parent dirs if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sanitized_data = _sanitize_for_json(data)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sanitized_data, f, indent=4)
    logger.info('Results saved to %s', output_path)


class BenchmarkRunner:
    """Orchestrates the end-to-end benchmarking process."""

    def __init__(self, config: BenchmarkingConfig) -> None:
        """Initializes the BenchmarkRunner.

        Args:
            config: The main configuration object for the entire run.
        """
        self.config = config
        self.generator = Generator.from_pretrained(config.model_name_or_path,
                                                   **config.model_kwargs)
        self.generator.model.to(config.device)

    def run(self) -> None:
        """Executes the full benchmarking pipeline based on the config."""
        logger.info('--- Starting benchmarking pipeline for model: %s ---',
                    self.config.model_name_or_path)

        if self.config.run_baseline:
            self._run_baseline()

        if self.config.hpo and self.config.operator_to_tune:
            self._run_hpo()
        elif self.config.operator_to_tune:
            logger.warning(
                "An operator '%s' was specified to tune, but no HPO "
                'configuration was provided. Skipping HPO.',
                self.config.operator_to_tune)

        logger.info('--- Benchmarking pipeline finished. ---')

    def _run_baseline(self) -> None:
        """Runs the baseline evaluation without any backtracking."""
        logger.info('--- Step: Running Baseline Evaluation ---')
        evaluator = Evaluator(self.config.evaluation)
        baseline_operator = Never()

        results = evaluator.run(self.generator, self.config.generation,
                                baseline_operator)

        if results:
            output_dir = Path(self.config.evaluation.output_dir)
            _save_results_json(results, output_dir / 'baseline_results.json')
            try:
                primary_task = self.config.evaluation.tasks[0]
                score = Evaluator.extract_primary_score(results,
                                                        str(primary_task))
                logger.info("Baseline score for task '%s': %.4f", primary_task,
                            score)
            except (KeyError, ValueError, IndexError) as e:
                logger.warning(
                    'Could not extract primary score for baseline: %s', e)
        else:
            logger.error('Baseline evaluation failed to produce results.')

    def _run_hpo(self) -> None:
        """Runs the hyperparameter optimization process."""
        if not self.config.hpo or not self.config.operator_to_tune:
            return

        logger.info('--- Step: Running HPO for Operator: %s ---',
                    self.config.operator_to_tune)
        optimizer = HyperparameterOptimizer(
            hpo_config=self.config.hpo,
            eval_config=self.config.evaluation,
            gen_config=self.config.generation,
            generator=self.generator,
            operator_name=self.config.operator_to_tune)

        study = optimizer.optimize()

        logger.info('HPO finished. Best trial:')
        logger.info('  Value: %.4f', study.best_value)
        logger.info('  Params: %s', study.best_params)

        hpo_results = study.trials_dataframe()
        output_dir = Path(self.config.evaluation.output_dir)
        hpo_results.to_csv(output_dir / 'full_results.csv')
        best_results = {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'operator': self.config.operator_to_tune,
        }
        _save_results_json(best_results, output_dir / 'hpo_results.json')
