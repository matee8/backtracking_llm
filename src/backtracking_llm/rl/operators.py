"""RL policy operators for executing trained reinforcement learning policies.

This module provides operators that load and execute Stable Baselines 3 policies
to make backtracking decisions during text generation. These operators bridge
trained RL agents with the existing generation pipeline.
"""

import logging
from collections import deque
from pathlib import Path

from stable_baselines3 import PPO
from torch import Tensor

from backtracking_llm.rl import features

logger = logging.getLogger(__name__)


class RlPolicyOperator:
    """An operator that executes a trained RL policy to make backtracking
    decisions.

    This operator loads a Stable Baselines 3 policy model and uses it to
    determine when to backtrack during text generation. The policy is called
    at each generation step and returns the number of tokens to remove based
    on the current generation state features.

    Attributes:
        model: The loaded Stable Baselines 3 policy model.
        action_space_size: The maximum number of tokens that can be backtracked
            in a single action (derived from policy action space).
    """

    def __init__(self, policy_path: Path, max_seq_length: int = 512) -> None:
        """Initialize the RL policy operator.

        Args:
            policy_path: Path to the saved Stable Baselines 3 policy model.
            max_seq_length: The maximum sequence length used during training.

        Raises:
            FileNotFoundError: If the policy file does not exist.
            ValueError: If the policy has invalid action space.
        """
        if not policy_path.exists():
            raise FileNotFoundError(f'Policy file not found at {policy_path}')

        self.model = PPO.load(policy_path)

        action_space = self.model.action_space
        if not hasattr(action_space, 'n'):
            raise ValueError('Policy action space must be discrete with'
                             '`n` attribute.')

        action_space_size = getattr(action_space, 'n', None)

        if action_space_size is None:
            raise ValueError('Policy actions space must be discrete with '
                             '`n` attribute.')

        self.action_space_size = action_space_size

        self.max_seq_length = max_seq_length
        self._step_count = 0
        self._history: deque[str] = deque(maxlen=5)

        self.policy_path = policy_path

    def __call__(self, logits: Tensor, probabilities: Tensor, position: int,
                 token: str) -> int:
        """Execute the RL policy to decide whether to backtrack.

        This method constructs an observation vector from the current generation
        state and passes it to the trained policy. The policy returns an action
        indicating how many tokens to backtrack (0 = no backtrack).

        Args:
            logits: The raw logits from the model for the current token.
                Shape: (vocab_size,)
            probabilities: The probabilities for the current token.
                Shape: (vocab_size,)
            position: The position of the chosen token in the probabilities
                tensor.
            token: The string representation of the chosen token.

        Returns:
            Number of tokens to backtrack (0 to max_backtrack).

        Raises:
            ValueError: If observation construction fails.
        """
        self._step_count += 1
        self._history.append(token)

        observation = features.compute_observation(
            probabilities=probabilities,
            step_count=self._step_count,
            max_seq_length=self.max_seq_length,
            history=self._history)

        action, _ = self.model.predict(observation, deterministic=True)
        backtrack_count = int(action)

        return max(0, min(backtrack_count, self.action_space_size - 1))

    def backtrack(self, n_tokens: int) -> None:
        """Handle backtracking events.

        Updates the internal step count and history buffer.

        Args:
            n_tokens: Number of tokens removed from the generation.
        """
        self._step_count = max(0, self._step_count - n_tokens)
        for _ in range(min(n_tokens, len(self._history))):
            self._history.pop()

    def __repr__(self) -> str:
        """Returns a string representation of the operator."""
        return (f'RlPolicyOperator(policy_path={self.policy_path!r}, '
                f'max_seq_length={self.max_seq_length!r})')
