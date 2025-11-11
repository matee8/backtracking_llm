"""RL policy operators for executing trained reinforcement learning policies.

This module provides operators that load and execute Stable Baselines 3 policies
to make backtracking decisions during text generation. These operators bridge
trained RL agents with the existing generation pipeline.
"""

import logging
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from torch import Tensor

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

    def __init__(self, policy_path: Path) -> None:
        """Initialize the RL policy operator.

        Args:
            policy_path: Path to the saved Stable Baselines 3 policy model.

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
        observation = self._build_observation(logits, probabilities, position)

        action, _ = self.model.predict(observation, deterministic=True)
        backtrack_count = int(action)

        return max(0, min(backtrack_count, self.action_space_size - 1))

    def backtrack(self, n_tokens: int) -> None:
        """Handle backtracking events.

        This operator maintains no internal state, so backtracking doesn't
        require any state updates. This method exists to satisfy the Operator
        protocol.

        Args:
            n_tokens: Number of tokens removed from the generation.
        """
        pass

    def _build_observation(self, logits: Tensor, probabilities: Tensor,
                           position: int) -> np.ndarray:
        """Build observation vector from generation state.

        Constructs a 4-dimensional feature vector matching the observation
        space used during training in `BacktrackingEnv`:
        1. chosen_token_confidence: Probability of the selected token
        2. distribution_entropy: Entropy of the token probability distribution
        3. position_ratio: Position of chosen token relative to vocabulary size
        4. logit_magnitude: Standardized logit value of chosen token

        Args:
            logits: Raw logits tensor of shape (vocab_size,).
            probabilities: Probability tensor of shape (vocab_size,).
            position: Index of the chosen token.

        Returns:
            Normalized feature vector of shape (4,) with dtype float32.

        Raises:
            ValueError: If tensors have incompatible shapes or sizes.
        """
        if len(probabilities) == 0:
            raise ValueError(
                'Cannot build observation from empty probabilities')

        if position < 0 or position >= len(probabilities):
            raise ValueError(
                f'Position {position} out of bounds for probabilities of '
                f'size {len(probabilities)}')

        chosen_token_confidence = float(probabilities[position])

        non_zero_probs = probabilities[probabilities > 0]
        if len(non_zero_probs) > 0:
            entropy = -(non_zero_probs * non_zero_probs.log()).sum().item()

            max_entropy = np.log(len(probabilities))
            distribution_entropy = (entropy /
                                    max_entropy if max_entropy > 0 else 0.0)
        else:
            distribution_entropy = 0.0

        position_ratio = float(position) / len(probabilities)

        chosen_logit = float(logits[position])
        logit_std = float(logits.std()) if logits.numel() > 1 else 1.0
        logit_magnitude = chosen_logit / logit_std if logit_std > 0 else 0.0

        observation = np.array(
            [
                min(chosen_token_confidence, 1.0),
                min(max(distribution_entropy, 0.0), 1.0),
                min(position_ratio, 1.0),
                min(abs(logit_magnitude) / 10.0, 1.0),
            ],
            dtype=np.float32,
        )

        return observation
