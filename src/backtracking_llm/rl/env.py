"""Defines the Gymnasium environment for training a backtracking agent."""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from torch import Tensor
from transformers import DynamicCache

from backtracking_llm.generation import Generator
from backtracking_llm.rl.features import FeatureExtractor


class BacktrackingEnv(Env):
    """A Gymnasium environment for learning a backtracking policy.

    This environment simulates the process of generating text token-by-token.
    At each step, an RL agent can choose to backtrack a certain number of
    tokens. The goal is to learn a policy that maximizes a final reward,
    typically derived from a benchmark score (e.g., from lm-eval).
    """

    metadata = {'render_modes': {}}

    def __init__(self,
                 generator: Generator,
                 feature_extractor: FeatureExtractor,
                 prompts: List[str],
                 max_backtrack_steps: int = 2) -> None:
        """Initializes the BacktrackingEnv.

        Args:
            generator: The pre-configured Generator instance.
            feature_extractor: The object to extract observation features.
            prompts: A list of initial prompts to use for episodes.
            max_backtrack_steps: The maximum number of tokens to backtrack.
                The action space will be `max_backtrack_steps + 1`.

        Raises:
            ValueError: if `prompts` is an empty list, or if
                `max_backtrack_steps` is negative.
        """
        super().__init__()
        if not prompts:
            raise ValueError('prompts list cannot be empty.')
        if max_backtrack_steps < 0:
            raise ValueError('max_backtrack_steps cannot be negative.')

        self.generator = generator
        self.feature_extractor = feature_extractor
        self.prompts = prompts
        self.max_backtrack_steps = max_backtrack_steps

        self.action_space = Discrete(self.max_backtrack_steps + 1)
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=self.feature_extractor.shape,
            dtype=np.float32,
        )

        self._current_prompt_idx: int = 0
        self._current_generated_text: str = ''

        self._prompt_iterator = iter(self.prompts)
        self._input_ids: Optional[Tensor] = None
        self._past_key_values: Optional[DynamicCache] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to the beginning of a new episode.

        This involves selecting the next prompt from the list, tokenizing it,
        and performing an initial forward pass to get the first observation.

        Returns:
            A tuple containing the initial observation and an info dictionary.
        """
        super().reset(seed=seed)

        try:
            prompt = next(self._prompt_iterator)
        except StopIteration:
            self._prompt_iterator = iter(self.prompts)
            prompt = next(self._prompt_iterator)

        device = self.generator.model.device
        inputs = self.generator.tokenizer(prompt,
                                          return_tensors='pt').to(device)
        self._input_ids = inputs.input_ids

        context_manager = (torch.inference_mode() if hasattr(
            torch, 'inference_mode') else torch.no_grad())

        with context_manager:
            outputs = self.generator.model(input_ids=self._input_ids,
                                           use_cache=True)
            next_token_logits = outputs.logits[:, -1, :]
            self._past_key_values = outputs.past_key_values

        top_k_logits, _ = torch.topk(next_token_logits,
                                     self.generator.model.config.vocab_size)
        top_k_probs = torch.nn.functional.softmax(top_k_logits, dim=-1)

        observation = self.feature_extractor(top_k_logits.squeeze(0),
                                             top_k_probs.squeeze(0))

        return observation, {}

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
