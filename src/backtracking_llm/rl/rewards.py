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

    def __call__(self, action: int, observation: np.ndarray) -> float:
        """Calculates the shaping reward for a single environment step.

        Args:
            action: The action taken by the agent (the number of tokens to
                backtrack).
            observation: The observation vector of the resulting state.

        Returns:
            A scalar reward value for the given step.
        """
        reward = 0.0

        if action > 0:
            penalty = (self.config.backtrack_action_penalty +
                       action * self.config.backtrack_token_penalty)
            reward -= penalty

        repetition_feature = observation[3]
        reward -= repetition_feature * self.config.repetition_penalty_weight

        top1_prob = observation[1]  # Index 1 is top1_probability
        if top1_prob > self.config.high_confidence_threshold:
            reward += self.config.high_confidence_reward

        return reward
