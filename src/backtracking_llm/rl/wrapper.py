"""Provides Gymnasium wrappers for compatibility with external libraries."""

from typing import Any, Dict, SupportsFloat, Tuple

from gymnasium import Wrapper


class SB3LstmWrapper(Wrapper):
    """A wrapper to make environments compatible with Stable Baselines 3's
    recurrent policies.

    Stable Baselines 3's stateful policies (like `MlpLstmPolicy`) have two
    specific requirements for the environment's `step` method:

    1.  It expects a single `done` boolean flag, not separate `terminated` and
        `truncated` flags.
    2.  When an episode ends due to truncation, the final observation must be
        stored in the `info` dictionary under the key `"terminal_observation"`.
        This allows the value function to be bootstrapped correctly.

    This wrapper handles this transformation, keeping the underlying environment
    logic clean and framework-agnostic.
    """

    def step(
            self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Performs a step in the environment and adds the terminal observation
        to the info dict if the episode was truncated.

        Args:
            action: The action to take in the environment.

        Returns:
            A 5-tuple of (observation, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        if truncated:
            info['terminal_observation'] = obs

        return obs, reward, terminated, truncated, info
