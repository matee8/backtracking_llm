"""OpenAI Gym environment wrapper for backtracking generation."""

import logging
from collections import deque
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from gymnasium import Env, spaces

from backtracking_llm.generation import GenerationSession
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
        self._feature_count = 4
        self.history_len = config.history_len
        self.judge = judge
        self.shaper = shaper
        self.config = config

        self.session: Optional[GenerationSession] = None
        self._last_step_result: Optional[Any] = None

        self.observation_history = deque(maxlen=self.history_len)
        self.action_space = spaces.Discrete(config.max_backtrack + 1)

        self.observation_space = spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(self.history_len *
                                                   self._feature_count,),
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

        self.session = self.session_factory()
        self._last_step_result = None

        self.observation_history.clear()
        for _ in range(self.history_len):
            self.observation_history.append(
                np.zeros(self._feature_count, dtype=np.float32))

        logger.debug('Environment reset complete')
        return np.array(self.observation_history).flatten(), {}

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

        single_step_observation = self._build_observation(step_result)
        self.observation_history.append(single_step_observation)
        observation = np.array(self.observation_history).flatten()
        reward = self.shaper.calculate(action, observation)

        terminated = self.session.done
        truncated = (self.session.generated_token_count
                     >= self.config.max_seq_length)

        final_score = 0.0

        if terminated or truncated:
            generated_text = self.session.get_decoded_text()
            if not generated_text or not generated_text.strip():
                final_score = 0.0
                logger.info(
                    'Episode end: empty text produced, penalized '
                    'with score %.2f', reward)
            else:
                final_score = float(self.judge.score(generated_text))
                logger.info('Episode end: scored %.2f', reward)

        reward += final_score

        return observation, reward, terminated, truncated, {}

    def _build_observation(self, step_result) -> np.ndarray:
        """Compute feature vector from generation state.

        Features:
        - normalized_position: How far through max sequence we are
        - top1_probability: Confidence in current token
        - entropy: Uncertainty of current distribution
        - repetition_penalty: 0 if last token is unique, increases with repeats
        """
        if self.session is None:
            return np.zeros(self._feature_count, dtype=np.float32)

        seq_len = self.session.token_ids.shape[1]
        prompt_len = self.session.prompt_length
        max_len = prompt_len + self.config.max_seq_length
        normalized_pos = (seq_len - prompt_len) / max_len

        if step_result:
            probs = step_result.probabilities
            top1_prob = probs[0].item() if len(probs) > 0 else 0.0

            non_zero = probs[probs > 0]
            if len(non_zero) > 0:
                entropy = -(non_zero * non_zero.log()).sum().item()
                entropy = entropy / np.log(len(probs))
            else:
                entropy = 0.0
        else:
            top1_prob = 0.0
            entropy = 0.0

        repetition_penalty = self._compute_repetition_penalty()

        observation = np.array([
            min(normalized_pos, 1.0),
            min(top1_prob, 1.0),
            min(entropy, 1.0),
            min(repetition_penalty, 1.0),
        ],
                               dtype=np.float32)

        return observation

    def _compute_repetition_penalty(self) -> float:
        """Computes a repetition penalty based on the uniqueness of recent
        tokens.

        The repetition penalty is calculated by comparing the number of unique
        tokens in the most recent 5 generated tokens against the total number of
        recent tokens. A higher penalty indicates more repetition in the recent
        output.

        Returns:
            The repetition penalty value between 0.0 (no repetition) and 1.0
                (complete repetition). Returns 0.0 if there's no session or if
                there aren't enough tokens to compute the penalty.
        """
        if self.session is None or self.session.generated_token_count < 2:
            return 0.0

        token_ids = self.session.token_ids[0].cpu().numpy()
        prompt_len = self.session.prompt_length

        if len(token_ids) <= prompt_len + 1:
            return 0.0

        recent = token_ids[-5:]
        unique = len(set(recent))
        repetition_penalty = 1.0 - (unique / len(recent))

        return repetition_penalty

    def render(self) -> Optional[str]:
        """Return current generated text for monitoring."""
        if self.session is None:
            return None

        return self.session.get_decoded_text()
