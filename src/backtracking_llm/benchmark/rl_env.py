"""Defines the environment for training a backtracking reinforcement learning
agent."""

import logging

from backtracking_llm.agents import RLAgentOperator
from backtracking_llm.benchmark.config import EvaluationConfig, GenerationConfig
from backtracking_llm.benchmark.evaluator import Evaluator
from backtracking_llm.generation import Generator

logger = logging.getLogger(__name__)


class RLEnvironment:
    """Wraps the evaluation pipeline to serve as an environment for an RL
    agent."""

    def __init__(self, generator: Generator, eval_config: EvaluationConfig,
                 gen_config: GenerationConfig) -> None:
        """Initializes the RL Environment.

        Args:
            generator: The pre-configured Generator instance.
            eval_config: The configuration for the evaluation tasks.
            gen_config: The configuration for the text generation process.

        Raises:
            ValueError: If the `eval_config` does not specify any tasks.
        """
        self._generator = generator
        self._eval_config = eval_config
        self._gen_config = gen_config
        self._evaluator = Evaluator(self._eval_config)

        if not self._eval_config.tasks:
            raise ValueError(
                'EvaluationConfig must specify at least one task for the RL '
                'environment.')

    def run_episode(self, agent: RLAgentOperator) -> float:
        """Runs a single episode of generation and evaluation.

        This method executes the full evaluation pipeline using the provided
        agent as the decision operator. The primary score from the first
        configured task is returned as the reward.

        Args:
            agent: The RL agent operator whose policy will be used for the run.

        Returns:
            The terminal reward for the episode, which is the primary score of
            the main evaluation task. Returns -1.0 if the evaluation fails or a
            score cannot be extracted.
        """
        logger.info('Starting new RL episode')

        try:
            results = self._evaluator.run(self._generator, self._gen_config,
                                          agent)

            if not results or 'results' not in results:
                logger.error('RL episode failed: Evaluation produced no '
                             'results.')
                return -1.0

            primary_task = str(self._eval_config.tasks[0])
            reward = Evaluator.extract_primary_score(results, primary_task)

            logger.info('RL episode finished. Reward: %.4f.', reward)

            return reward
        except (ValueError, KeyError, IndexError, RuntimeError) as e:
            logger.error('RL episode failed with an exception: %s',
                         e,
                         exc_info=True)
            return -1.0
