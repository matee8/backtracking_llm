# pylint: disable=missing-module-docstring

from pathlib import Path
from unittest import mock

import pytest
from sb3_contrib import RecurrentPPO

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
        training=TrainingConfig(total_timesteps=100, n_steps=10, ent_coef=0.05),
    )


@pytest.fixture
def mock_prompt_provider() -> mock.Mock:
    provider = mock.Mock(spec=PromptProvider)
    provider.__next__.return_value = 'This is a test prompt.'
    return provider


@mock.patch('backtracking_llm.rl.trainers.Generator.from_pretrained')
@mock.patch('backtracking_llm.rl.trainers.OpenAIJudge')
@mock.patch('backtracking_llm.rl.trainers.RewardShaper')
def test_trainer_initialization(mock_shaper_cls, mock_judge_cls,
                                mock_generator_cls, mock_config):
    mock_generator_instance = mock.Mock()
    mock_generator_cls.return_value = mock_generator_instance

    trainer = RLTrainer(config=mock_config)

    mock_generator_cls.assert_called_once_with(mock_config.model_name_or_path)
    mock_generator_instance.model.to.assert_called_once_with(mock_config.device)
    mock_judge_cls.assert_called_once_with(mock_config.judge)
    mock_shaper_cls.assert_called_once_with(mock_config.shaping)
    assert trainer.config == mock_config


@mock.patch('backtracking_llm.rl.trainers.Generator.from_pretrained')
@mock.patch('backtracking_llm.rl.trainers.OpenAIJudge')
@mock.patch('backtracking_llm.rl.trainers.DummyVecEnv')
@mock.patch('backtracking_llm.rl.trainers.SB3LstmWrapper')
@mock.patch('backtracking_llm.rl.trainers.env_checker')
@mock.patch('backtracking_llm.rl.trainers.RecurrentPPO')
@mock.patch('backtracking_llm.rl.trainers.GenerationSession')
@mock.patch('backtracking_llm.rl.trainers.BacktrackingEnv')
def test_train_method_orchestration(
    mock_env_cls,
    mock_session_cls,
    mock_recurrent_ppo_cls,
    mock_check_env,
    mock_wrapper_cls,
    mock_dummy_vec_env_cls,
    mock_config: RLConfig,
    mock_prompt_provider: mock.Mock,
):
    mock_agent_instance = mock.Mock(spec=RecurrentPPO)
    mock_recurrent_ppo_cls.return_value = mock_agent_instance

    trainer = RLTrainer(config=mock_config)
    trainer.train(prompt_provider=mock_prompt_provider)

    mock_dummy_vec_env_cls.assert_called_once()
    env_factory = mock_dummy_vec_env_cls.call_args.args[0][0]
    created_env = env_factory()

    mock_env_cls.assert_called_with(session_factory=mock.ANY,
                                    judge=trainer.judge,
                                    shaper=trainer.shaper,
                                    config=mock_config.env)
    mock_wrapper_cls.assert_called_with(mock.ANY)
    mock_check_env.check_env.assert_called_with(created_env)

    session_factory = mock_env_cls.call_args.kwargs['session_factory']
    session_factory()
    mock_session_cls.assert_called_once_with(
        model=trainer.generator.model,
        tokenizer=trainer.generator.tokenizer,
        prompt=mock_prompt_provider.__next__.return_value,
        max_new_tokens=mock_config.env.max_seq_length,
    )

    mock_recurrent_ppo_cls.assert_called_once_with(
        policy=mock_config.training.policy_type,
        env=mock.ANY,
        learning_rate=mock_config.training.learning_rate,
        n_steps=mock_config.training.n_steps,
        batch_size=mock_config.training.batch_size,
        n_epochs=mock_config.training.n_epochs,
        gamma=mock_config.training.gamma,
        ent_coef=mock_config.training.ent_coef,
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
