"""Defines the Gymnasium environment for training a backtracking agent."""

from typing import Any, Dict, Optional, Tuple, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete


class BacktrackingEnv(Env):
    """A Gymnasium environment for learning a backtracking policy.

    This environment simulates the process of generating text token-by-token.
    At each step, an RL agent can choose to backtrack a certain number of
    tokens. The goal is to learn a policy that maximizes a final reward,
    typically derived from a benchmark score (e.g., from lm-eval).
    """

    metadata = {'render_modes': {}}

    def __init__(self, num_actions: int,
                 observation_shape: Sequence[int]) -> None:
        """Initializes the BacktrackingEnv.

        In subsequent commits, this will be updated to accept a Generator,
        evaluation configuration, a feature extractor, and other necessary
        components.
        """
        super().__init__()

        self.action_space = Discrete(num_actions)
        self.observation_space = Box(low=-np.inf,
                                     high=np.inf,
                                     shape=observation_shape,
                                     dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to the beginning of a new episode.

        This involves selecting a new prompt from the evaluation dataset and
        initializing the generation state.

        Returns:
            A tuple containing the initial observation and an info dictionary.
        """
        super().reset(seed=seed)

        raise NotImplementedError(
            'The `reset` method must be implemented to start a new episode.')

    def step(
            self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one step in the environment.

        Args:
            action: The action chosen by the agent (e.g., number of tokens to
                backtrack).

        Returns:
            A tuple containing the next observation, reward, terminated flag,
            truncated flag, and an info dictionary.
        """
        raise NotImplementedError(
            'The `step` method must be implemented to advance the episode.')
