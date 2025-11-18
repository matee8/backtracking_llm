"""OpenAI Gym environment wrapper for backtracking generation."""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from gymnasium import Env, spaces

from backtracking_llm.generation import GenerationSession
from backtracking_llm.rl import features
from backtracking_llm.rl.config import EnvConfig
from backtracking_llm.rl.rewards import RewardShaper
from backtracking_llm.rl.judges import Judge

logger = logging.getLogger(__name__)


class BacktrackingEnv(Env):
    """Gymnasium environment for training RL agents to learn backtracking
    policies.

    The environment wraps a GenerationSession and exposes:
    - Observations: Vector of generation state features
    - Actions: Discrete backtrack count (0 = no-op)
    - Rewards: Final text quality score from Judge (sparse)

    Attributes:
        session_factory: Callable that creates new GenerationSession instances
        judge: Judge that scores final generations
        config: Environment hyperparameters
        observation_space: Box of shape (4,) with normalized state features
        action_space: Discrete(max_backtrack + 1)
    """

    def __init__(
        self,
        session_factory: Callable[[], GenerationSession],
        judge: Judge,
        shaper: RewardShaper,
        config: EnvConfig,
    ) -> None:
        """Initialize environment.

        Args:
            session_factory: Zero-argument callable returning a fresh session
            judge: LLM-as-a-judge implementation for scoring
            shaper: The reward shaper for calculating intermediate rewards.
            config: Environment configuration (max_backtrack, max_seq_length,
                etc.)
        """
        super().__init__()

        self.session_factory = session_factory
        self.judge = judge
        self.shaper = shaper
        self.config = config

        self.session: Optional[GenerationSession] = None
        self._last_step_result: Optional[Any] = None

        self.action_space = spaces.Discrete(config.max_backtrack + 1)

        self.observation_space = spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(4,),
                                            dtype=np.float32)

        logger.info(
            'BacktrackingEnv initialized: max_backtrack=%d, max_seq_len=%d',
            config.max_backtrack, config.max_seq_length)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and start new generation episode.

        Returns:
            observation: Initial observation vector
            info: Empty dict (future: metadata about episode)
        """
        super().reset(seed=seed)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if self.session is not None:
            del self.session
            self._last_step_result = None
            torch.cuda.empty_cache()

        self.session = self.session_factory()

        observation = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        logger.debug('Environment reset complete')
        return observation, {}

    def step(
            self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step: apply action, generate token, compute
        reward.

        Args:
            action: Number of tokens to backtrack (0 = generate normally)

        Returns:
            observation: Next state features
            reward: 0.0 until episode end, then judge score
            terminated: Whether generation finished (EOS, max tokens, etc.)
            truncated: Whether episode exceeded max_seq_length
            info: Empty dict for future metadata
        """
        if self.session is None or self.session.done:
            raise RuntimeError('step() called on terminated environment')

        if action > 0:
            logger.debug('Agent action: backtrack %d tokens', action)
            self.session.backtrack(action)

        step_result = self.session.step()
        self._last_step_result = step_result

        history_tokens = self.session.token_ids[0, -5:].tolist()
        observation = features.compute_observation(
            probabilities=step_result.probabilities,
            step_count=self.session.generated_token_count,
            max_seq_length=self.config.max_seq_length,
            history=history_tokens)
        reward = self.shaper(action, observation)

        terminated = self.session.done
        truncated = (self.session.generated_token_count
                     >= self.config.max_seq_length)

        final_score = 0.0

        if terminated or truncated:
            generated_text = self.session.get_decoded_text()
            if not generated_text or not generated_text.strip():
                final_score = -1.0
                logger.info(
                    'Episode end: empty text produced, penalized '
                    'with score %.2f', reward)
            else:
                final_score = float(self.judge.score(generated_text))
                logger.info('Episode end: scored %.2f', reward)

        reward += final_score

        return observation, reward, terminated, truncated, {}

    def render(self) -> Optional[str]:
        """Return current generated text for monitoring."""
        if self.session is None:
            return None

        return self.session.get_decoded_text()
