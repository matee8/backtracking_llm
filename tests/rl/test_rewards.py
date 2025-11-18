# pylint: disable=missing-module-docstring, missing-class-docstring, missing-function-docstring

import numpy as np
import pytest

from backtracking_llm.rl.config import ShapingConfig
from backtracking_llm.rl.rewards import RewardShaper

# pylint: disable=redefined-outer-name


@pytest.fixture
def default_config() -> ShapingConfig:
    return ShapingConfig(backtrack_action_penalty=0.01,
                         backtrack_token_penalty=0.005,
                         repetition_penalty_weight=0.02,
                         high_confidence_reward=0.01,
                         high_confidence_threshold=0.9)


@pytest.fixture
def dummy_observation() -> np.ndarray:
    return np.zeros(4, dtype=np.float32)


def test_reward_shaper_no_reward_for_neutral_step(
        default_config: ShapingConfig, dummy_observation: np.ndarray):
    shaper = RewardShaper(default_config)
    reward = shaper(action=0, observation=dummy_observation)
    assert reward == 0.0


def test_reward_shaper_penalty_for_single_token_backtrack(
        default_config: ShapingConfig, dummy_observation: np.ndarray):
    shaper = RewardShaper(default_config)
    expected_penalty = -(default_config.backtrack_action_penalty +
                         1 * default_config.backtrack_token_penalty)
    reward = shaper(action=1, observation=dummy_observation)
    assert reward == pytest.approx(expected_penalty)


def test_reward_shaper_penalty_for_multiple_token_backtrack(
        default_config: ShapingConfig, dummy_observation: np.ndarray):
    shaper = RewardShaper(default_config)
    action = 5
    expected_penalty = -(default_config.backtrack_action_penalty +
                         action * default_config.backtrack_token_penalty)
    reward = shaper(action=action, observation=dummy_observation)
    assert reward == pytest.approx(expected_penalty)


def test_reward_shaper_uses_custom_config_values(dummy_observation: np.ndarray):
    custom_config = ShapingConfig(backtrack_action_penalty=0.5,
                                  backtrack_token_penalty=0.1)
    shaper = RewardShaper(custom_config)
    action = 3
    expected_penalty = -(custom_config.backtrack_action_penalty +
                         action * custom_config.backtrack_token_penalty)
    reward = shaper(action=action, observation=dummy_observation)
    assert reward == pytest.approx(expected_penalty)


def test_reward_shaper_ignores_negative_action(default_config: ShapingConfig,
                                               dummy_observation: np.ndarray):
    shaper = RewardShaper(default_config)
    reward = shaper(action=-1, observation=dummy_observation)
    assert reward == 0.0


def test_reward_shaper_applies_repetition_penalty(
        default_config: ShapingConfig):
    shaper = RewardShaper(default_config)
    observation = np.array([0.5, 0.5, 0.5, 0.8], dtype=np.float32)
    expected_penalty = -0.8 * default_config.repetition_penalty_weight
    reward = shaper(action=0, observation=observation)
    assert reward == pytest.approx(expected_penalty)


def test_reward_shaper_applies_high_confidence_reward(
        default_config: ShapingConfig):
    shaper = RewardShaper(default_config)
    observation = np.array([0.5, 0.95, 0.5, 0.0], dtype=np.float32)
    reward = shaper(action=0, observation=observation)
    assert reward == pytest.approx(default_config.high_confidence_reward)


def test_reward_shaper_combines_all_reward_components(
        default_config: ShapingConfig):
    shaper = RewardShaper(default_config)
    observation = np.array([0.5, 0.95, 0.5, 0.8], dtype=np.float32)
    backtrack_penalty = -(default_config.backtrack_action_penalty +
                          2 * default_config.backtrack_token_penalty)
    repetition_penalty = -0.8 * default_config.repetition_penalty_weight
    confidence_reward = default_config.high_confidence_reward
    expected_reward = backtrack_penalty + repetition_penalty + confidence_reward
    reward = shaper(action=2, observation=observation)
    assert reward == pytest.approx(expected_reward)
