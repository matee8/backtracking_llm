# pylint: disable=missing-module-docstring

import dataclasses
import pathlib
import tempfile
import pytest
from pathlib import Path

from backtracking_llm.rl.config import (
    EnvConfig,
    JudgeConfig,
    ShapingConfig,
    RLConfig,
    TrainingConfig,
)

# pylint: disable=missing-class-docstring
# pylint: disable=unexpected-keyword-arg
# pylint: disable=no-value-for-parameter


class TestJudgeConfig:

    def test_default_values(self):
        config = JudgeConfig(model='gpt-4')
        assert config.model == 'gpt-4'
        assert config.max_retries == 3
        assert config.timeout == 30.0
        assert config.api_key is None
        assert config.base_url is None

    def test_custom_values(self):
        config = JudgeConfig(model='gpt-3.5-turbo',
                             max_retries=5,
                             timeout=60.0,
                             api_key='sk-123',
                             base_url='https://custom.api.com')

        assert config.model == 'gpt-3.5-turbo'
        assert config.max_retries == 5
        assert config.timeout == 60.0
        assert config.api_key == 'sk-123'
        assert config.base_url == 'https://custom.api.com'

    def test_model_empty_string_raises(self):
        with pytest.raises(ValueError, match='`model` cannot be empty'):
            JudgeConfig(model='')

    def test_model_whitespace_raises(self):
        with pytest.raises(ValueError, match='`model` cannot be empty'):
            JudgeConfig(model='   ')

    def test_model_none_raises(self):
        with pytest.raises(ValueError, match='`model` cannot be None'):
            JudgeConfig(model=None)  # type: ignore

    def test_negative_retries_raises(self):
        with pytest.raises(ValueError,
                           match='`max_retries` must be non-negative'):
            JudgeConfig(model='gpt-4', max_retries=-1)

    def test_negative_timeout_raises(self):
        with pytest.raises(ValueError, match='`timeout` must be non-negative'):
            JudgeConfig(model='gpt-4', timeout=-1)

    def test_model_string_stripped(self):
        config = JudgeConfig(model='  gpt-4  ')
        assert config.model == 'gpt-4'

    def test_string_conversion(self):
        config = JudgeConfig(model=123)  # type: ignore
        assert config.model == '123'


class TestEnvConfig:

    def test_default_values(self):
        config = EnvConfig()
        assert config.max_backtrack == 5
        assert config.max_seq_length == 512
        assert config.judge_prompt_template is None

    def test_custom_values(self):
        config = EnvConfig(max_backtrack=10,
                           max_seq_length=1024,
                           judge_prompt_template='Rate this: {text}')
        assert config.max_backtrack == 10
        assert config.max_seq_length == 1024
        assert config.judge_prompt_template == 'Rate this: {text}'

    def test_zero_max_backtrack_raises(self):
        with pytest.raises(ValueError,
                           match='`max_backtrack` must be non-negative'):
            EnvConfig(max_backtrack=0)

    def test_negative_max_backtrack_raises(self):
        with pytest.raises(ValueError,
                           match='`max_backtrack` must be non-negative'):
            EnvConfig(max_backtrack=-5)

    def test_zero_max_seq_length_raises(self):
        with pytest.raises(ValueError,
                           match='`max_seq_length` must be non-negative'):
            EnvConfig(max_seq_length=0)

    def test_negative_max_seq_length_raises(self):
        with pytest.raises(ValueError,
                           match='`max_seq_length` must be non-negative'):
            EnvConfig(max_seq_length=-1)

    def test_empty_prompt_template_becomes_none(self):
        config = EnvConfig(judge_prompt_template='')
        assert config.judge_prompt_template is None

    def test_whitespace_prompt_template_becomes_none(self):
        config = EnvConfig(judge_prompt_template='   ')
        assert config.judge_prompt_template is None


class TestTrainingConfig:

    def test_default_values(self):
        config = TrainingConfig()
        assert config.policy_type == 'MlpLstmPolicy'
        assert config.total_timesteps == 10000
        assert config.learning_rate == 3e-4
        assert config.n_steps == 2048
        assert config.batch_size == 64
        assert config.n_epochs == 10
        assert config.gamma == 0.99
        assert config.ent_coef == 0.01
        assert config.seed is None

    def test_custom_values(self):
        config = TrainingConfig(policy_type='CnnPolicy',
                                total_timesteps=50000,
                                learning_rate=1e-3,
                                n_steps=1024,
                                batch_size=32,
                                n_epochs=5,
                                gamma=0.95,
                                ent_coef=0.05,
                                seed=42)
        assert config.policy_type == 'CnnPolicy'
        assert config.total_timesteps == 50000
        assert config.learning_rate == 1e-3
        assert config.n_steps == 1024
        assert config.batch_size == 32
        assert config.n_epochs == 5
        assert config.gamma == 0.95
        assert config.ent_coef == 0.05
        assert config.seed == 42

    def test_zero_timesteps_raises(self):
        with pytest.raises(ValueError,
                           match='`total_timesteps` must be positive'):
            TrainingConfig(total_timesteps=0)

    def test_negative_timesteps_raises(self):
        with pytest.raises(ValueError,
                           match='`total_timesteps` must be positive'):
            TrainingConfig(total_timesteps=-100)

    def test_zero_learning_rate_raises(self):
        with pytest.raises(ValueError,
                           match='`learning_rate` must be positive'):
            TrainingConfig(learning_rate=0)

    def test_negative_learning_rate_raises(self):
        with pytest.raises(ValueError,
                           match='`learning_rate` must be positive'):
            TrainingConfig(learning_rate=-1e-4)

    def test_invalid_policy_type_raises(self):
        with pytest.raises(ValueError, match='Unsupported policy_type'):
            TrainingConfig(policy_type='InvalidPolicy')

    def test_zero_n_steps_raises(self):
        with pytest.raises(ValueError, match='`n_steps` must be positive'):
            TrainingConfig(n_steps=0)

    def test_negative_n_steps_raises(self):
        with pytest.raises(ValueError, match='`n_steps` must be positive'):
            TrainingConfig(n_steps=-1)

    def test_zero_batch_size_raises(self):
        with pytest.raises(ValueError, match='`batch_size` must be positive'):
            TrainingConfig(batch_size=0)

    def test_negative_batch_size_raises(self):
        with pytest.raises(ValueError, match='`batch_size` must be positive'):
            TrainingConfig(batch_size=-1)

    def test_zero_n_epochs_raises(self):
        with pytest.raises(ValueError, match='`n_epochs` must be positive'):
            TrainingConfig(n_epochs=0)

    def test_negative_n_epochs_raises(self):
        with pytest.raises(ValueError, match='`n_epochs` must be positive'):
            TrainingConfig(n_epochs=-1)

    def test_gamma_below_zero_raises(self):
        with pytest.raises(ValueError, match='`gamma` must be between 0 and 1'):
            TrainingConfig(gamma=-0.1)

    def test_gamma_above_one_raises(self):
        with pytest.raises(ValueError, match='`gamma` must be between 0 and 1'):
            TrainingConfig(gamma=1.1)

    def test_supported_policy_types(self):
        for policy in ('MlpPolicy', 'CnnPolicy', 'MultiInputPolicy'):
            config = TrainingConfig(policy_type=policy)
            assert config.policy_type == policy

    def test_seed_zero_is_valid(self):
        config = TrainingConfig(seed=0)
        assert config.seed == 0

    def test_negative_seed_is_valid(self):
        config = TrainingConfig(seed=-42)
        assert config.seed == -42


class TestRLConfig:

    def test_default_values(self):
        config = RLConfig(model_name_or_path='gpt2')
        assert config.model_name_or_path == 'gpt2'
        assert isinstance(config.judge, JudgeConfig)
        assert isinstance(config.env, EnvConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.shaping, ShapingConfig)
        assert config.output_dir == pathlib.Path('rl_output')
        assert config.device == 'auto'

    def test_custom_values(self):
        judge_config = JudgeConfig(model='gpt-4')
        env_config = EnvConfig(max_backtrack=10)
        training_config = TrainingConfig(total_timesteps=50000)
        shaping_config = ShapingConfig(backtrack_action_penalty=0.5)

        config = RLConfig(model_name_or_path='gpt2',
                          judge=judge_config,
                          env=env_config,
                          training=training_config,
                          shaping=shaping_config,
                          output_dir=pathlib.Path('custom_output'),
                          device='cuda:0')

        assert config.model_name_or_path == 'gpt2'
        assert config.judge is judge_config
        assert config.env is env_config
        assert config.training is training_config
        assert config.shaping is shaping_config
        assert config.output_dir == pathlib.Path('custom_output')
        assert config.device == 'cuda:0'

    def test_empty_model_name_raises(self):
        with pytest.raises(ValueError,
                           match='`model_name_or_path` cannot be empty'):
            RLConfig(model_name_or_path='')

    def test_whitespace_model_name_raises(self):
        with pytest.raises(ValueError,
                           match='`model_name_or_path` cannot be empty'):
            RLConfig(model_name_or_path='   ')

    def test_output_dir_file_raises(self):
        with tempfile.NamedTemporaryFile() as tmp:
            with pytest.raises(ValueError, match='is not a directory'):
                RLConfig(model_name_or_path='gpt2',
                         output_dir=pathlib.Path(tmp.name))

    def test_device_lowercased(self):
        config = RLConfig(model_name_or_path='gpt2', device='CPU')
        assert config.device == 'cpu'

    def test_device_empty_raises(self):
        with pytest.raises(ValueError, match='`device` cannot be empty'):
            RLConfig(model_name_or_path='gpt2', device='')

    def test_device_whitespace_raises(self):
        with pytest.raises(ValueError, match='`device` cannot be empty'):
            RLConfig(model_name_or_path='gpt2', device='   ')

    def test_config_serializable(self):
        config = RLConfig(model_name_or_path='gpt2')

        serialized = dataclasses.asdict(config)

        assert serialized['model_name_or_path'] == 'gpt2'
        assert isinstance(serialized['judge'], dict)
        assert isinstance(serialized['env'], dict)
        assert isinstance(serialized['training'], dict)
        assert isinstance(serialized['shaping'], dict)
        assert isinstance(serialized['output_dir'], Path)
        assert serialized['device'] == 'auto'

    def test_nested_config_validation_propagated(self):
        with pytest.raises(ValueError, match='`model` cannot be empty'):
            RLConfig(model_name_or_path='gpt2', judge=JudgeConfig(model=''))

    def test_nested_dict_conversion(self):
        config = RLConfig(
            model_name_or_path='gpt2',
            judge={
                'model': 'gpt-3.5-turbo',
                'max_retries': 5
            },  # type: ignore
            env={'max_backtrack': 10},  # type: ignore
            training={'total_timesteps': 50000},  # type: ignore
            shaping={'backtrack_action_penalty': 0.05}  # type: ignore
        )

        assert isinstance(config.judge, JudgeConfig)
        assert config.judge.model == 'gpt-3.5-turbo'
        assert config.judge.max_retries == 5
        assert isinstance(config.env, EnvConfig)
        assert config.env.max_backtrack == 10
        assert isinstance(config.training, TrainingConfig)
        assert config.training.total_timesteps == 50000
        assert isinstance(config.shaping, ShapingConfig)
        assert config.shaping.backtrack_action_penalty == 0.05
        assert config.shaping.repetition_penalty_weight == 0.02

    def test_nested_dict_invalid_raises(self):
        with pytest.raises(ValueError,
                           match='`max_backtrack` must be non-negative'):
            RLConfig(
                model_name_or_path='gpt2',
                judge={'model': 'gpt-4'},  # type: ignore
                env={'max_backtrack': 0}  # type: ignore
            )

    def test_output_dir_string_converted(self):
        config = RLConfig(model_name_or_path='gpt2',
                          output_dir='my_output')  # type: ignore
        assert isinstance(config.output_dir, pathlib.Path)
        assert str(config.output_dir) == 'my_output'

    def test_model_name_string_conversion(self):
        config = RLConfig(model_name_or_path=123)  # type: ignore
        assert config.model_name_or_path == '123'

    def test_partial_nested_dict_merged_with_defaults(self):
        config = RLConfig(
            model_name_or_path='gpt2',
            judge={'model': 'gpt-4'}  # type: ignore
        )

        assert config.judge.model == 'gpt-4'
        assert config.judge.max_retries == 3
        assert config.judge.timeout == 30.0

    def test_config_equality(self):
        config1 = RLConfig(model_name_or_path='gpt2')
        config2 = RLConfig(model_name_or_path='gpt2')
        config3 = RLConfig(model_name_or_path='gpt2', device='cuda')

        assert config1 == config2
        assert config1 != config3

    def test_config_repr(self):
        config = RLConfig(model_name_or_path='gpt2')
        repr_str = repr(config)

        assert 'RLConfig' in repr_str
        assert 'model_name_or_path' in repr_str
        assert 'gpt2' in repr_str

    def test_output_dir_none_raises(self):
        with pytest.raises(ValueError, match='`output_dir` cannot be None'):
            RLConfig(model_name_or_path='gpt2', output_dir=None)  # type: ignore

    def test_negative_ent_coef_raises(self):
        with pytest.raises(ValueError, match='`ent_coef` must be non-negative'):
            TrainingConfig(ent_coef=-0.1)
