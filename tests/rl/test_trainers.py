# pylint: disable=missing-module-docstring

from pathlib import Path
from unittest import mock

import pytest
from stable_baselines3 import PPO

from backtracking_llm.rl.config import (EnvConfig, JudgeConfig, RLConfig,
                                        TrainingConfig)
from backtracking_llm.rl.data import PromptProvider
from backtracking_llm.rl.trainers import RLTrainer

# pylint: disable=redefined-outer-name


@pytest.fixture
def mock_config(tmp_path: Path) -> RLConfig:
    return RLConfig(
        model_name_or_path='mock-model',
        output_dir=tmp_path / 'output',
        judge=JudgeConfig(model='mock-judge'),
        env=EnvConfig(max_seq_length=50),
        training=TrainingConfig(total_timesteps=100, n_steps=10),
    )


@pytest.fixture
def mock_prompt_provider() -> mock.Mock:
    provider = mock.Mock(spec=PromptProvider)
    provider.get_prompt.return_value = 'This is a test prompt.'
    return provider


@mock.patch('backtracking_llm.rl.trainer.Generator.from_pretrained')
@mock.patch('backtracking_llm.rl.trainer.OpenAIJudge')
def test_trainer_initialization(mock_judge_cls, mock_generator_cls,
                                mock_config):
    mock_generator_instance = mock.Mock()
    mock_generator_cls.return_value = mock_generator_instance

    trainer = RLTrainer(config=mock_config)

    mock_generator_cls.assert_called_once_with(mock_config.model_name_or_path)
    mock_generator_instance.model.to.assert_called_once_with(mock_config.device)
    mock_judge_cls.assert_called_once_with(mock_config.judge)
    assert trainer.config == mock_config


@mock.patch('backtracking_llm.rl.trainer.Generator.from_pretrained')
@mock.patch('backtracking_llm.rl.trainer.OpenAIJudge')
@mock.patch('backtracking_llm.rl.trainer.BacktrackingEnv')
@mock.patch('backtracking_llm.rl.trainer.check_env')
@mock.patch('backtracking_llm.rl.trainer.PPO')
@mock.patch('backtracking_llm.rl.trainer.GenerationSession')
def test_train_method_orchestration(
    mock_session_cls,
    mock_ppo_cls,
    mock_check_env,
    mock_env_cls,
    mock_config: RLConfig,
    mock_prompt_provider: mock.Mock,
):
    mock_agent_instance = mock.Mock(spec=PPO)
    mock_ppo_cls.return_value = mock_agent_instance
    mock_env_instance = mock.Mock()
    mock_env_cls.return_value = mock_env_instance

    trainer = RLTrainer(config=mock_config)
    trainer.train(prompt_provider=mock_prompt_provider)

    mock_env_cls.assert_called_once()
    _, env_kwargs = mock_env_cls.call_args
    assert 'session_factory' in env_kwargs
    assert env_kwargs['judge'] is trainer.judge
    assert env_kwargs['config'] == mock_config.env
    mock_check_env.assert_called_once_with(mock_env_instance)

    session_factory = env_kwargs['session_factory']
    session_factory()
    mock_prompt_provider.get_prompt.assert_called_once()
    mock_session_cls.assert_called_once_with(
        model=trainer.generator.model,
        tokenizer=trainer.generator.tokenizer,
        prompt=mock_prompt_provider.get_prompt.return_value,
        max_new_tokens=mock_config.env.max_seq_length,
    )

    mock_ppo_cls.assert_called_once_with(
        policy=mock_config.training.policy_type,
        env=mock_env_instance,
        learning_rate=mock_config.training.learning_rate,
        n_steps=mock_config.training.n_steps,
        batch_size=mock_config.training.batch_size,
        n_epochs=mock_config.training.n_epochs,
        gamma=mock_config.training.gamma,
        seed=mock_config.training.seed,
        device=mock_config.device,
        verbose=1,
    )
    mock_agent_instance.learn.assert_called_once_with(
        total_timesteps=mock_config.training.total_timesteps,
        progress_bar=True,
    )

    expected_save_path = mock_config.output_dir / 'policy.zip'
    mock_agent_instance.save.assert_called_once_with(expected_save_path)
    assert mock_config.output_dir.exists()
