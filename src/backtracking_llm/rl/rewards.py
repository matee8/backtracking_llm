"""Provides the reward shaping logic for the RL environment."""

import numpy as np

from backtracking_llm.rl.config import ShapingConfig


class RewardShaper:
    """Calculates intermediate rewards to guide the RL agent.

    This class encapsulates the reward shaping logic, turning state and action
    information into a scalar reward signal. This helps mitigate the sparse
    reward problem by providing the agent with more frequent feedback than just
    the final score from the judge.
    """

    def __init__(self, config: ShapingConfig) -> None:
        """Initializes the RewardShaper.

        Args:
            config: The configuration dataclass containing weights and
                thresholds for the reward components.
        """
        self.config = config

    def calculate(self, action: int, observation: np.ndarray) -> float:
        """Calculates the shaping reward for a single environment step.

        Args:
            action: The action taken by the agent (the number of tokens to
                backtrack).
            observation: The observation vector of the resulting state.

        Returns:
            A scalar reward value for the given step.
        """
        if action <= 0:
            return 0.0

        penalty = (self.config.backtrack_action_penalty +
                   action * self.config.backtrack_token_penalty)

        return -penalty
