"""Configuration dataclasses for RL operators."""

import dataclasses
from pathlib import Path
from typing import Optional

# pylint: disable=not-a-mapping


@dataclasses.dataclass
class JudgeConfig:
    """Configuration for the judge LLM.

    This configures the LLM-as-a-Judge system that provides reward signals.

    Attributes:
        model: The model identifier (e.g., 'gpt-4-turbo-preview').
        max_retries: Maximum number of API call retries.
        timeout: Timeout for API calls in seconds.
        api_key: Optional API key. If None, uses environment variable.
        base_url: Optional base URL for API endpoint.
    """

    model: str = 'gpt-4-turbo-preview'
    max_retries: int = 3
    timeout: float = 30.0
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.model is None:
            raise ValueError('`model` cannot be None')
        self.model = str(self.model).strip()
        if len(self.model) == 0:
            raise ValueError('`model` cannot be empty')

        if self.max_retries < 0:
            raise ValueError('`max_retries` must be non-negative.')

        if self.timeout < 0:
            raise ValueError('`timeout` must be non-negative.')


@dataclasses.dataclass
class EnvConfig:
    """Configuration for the RL environment.

    Attributes:
        max_backtrack: Maximum tokens that can be removed in one action.
        max_seq_length: Maximum sequence length before truncation.
        judge_prompt_template: Optional custom prompt template for judge.
    """

    max_backtrack: int = 5
    max_seq_length: int = 512
    judge_prompt_template: Optional[str] = None

    def __post_init__(self):
        """Validate environment configuration."""
        if self.max_backtrack < 1:
            raise ValueError('`max_backtrack` must be non-negative')
        if self.max_seq_length < 1:
            raise ValueError('`max_seq_length` must be non-negative')

        if self.judge_prompt_template is not None:
            self.judge_prompt_template = str(self.judge_prompt_template).strip()
            if not self.judge_prompt_template:
                self.judge_prompt_template = None


@dataclasses.dataclass
class TrainingConfig:
    """Configuration for RL training.

    Attributes:
        policy_type: SB3 policy class name (MlpPolicy, CnnPolicy, 
            MultiInputPolicy).
        total_timesteps: Total environment steps to train for.
        learning_rate: Learning rate for optimizer.
        n_steps: Number of steps per rollout.
        batch_size: Minibatch size for updates.
        n_epochs: Number of epochs per update.
        gamma: Discount factor for rewards.
        ent_coef: Entropy coefficient for the loss calculation. Encourages
            exploration.
        seed: Random seed for reproducibility.
    """

    policy_type: str = 'MlpLstmPolicy'
    total_timesteps: int = 10000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    ent_coef: float = 0.01
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate training configuration."""
        supported_policies = ('MlpPolicy', 'CnnPolicy', 'MultiInputPolicy',
                              'MlpLstmPolicy')
        if self.policy_type not in supported_policies:
            raise ValueError(f'Unsupported policy_type `{self.policy_type}`. '
                             f"Supported: {'`, `'.join(supported_policies)}")

        if self.total_timesteps < 1:
            raise ValueError('`total_timesteps` must be positive')
        if self.learning_rate <= 0:
            raise ValueError('`learning_rate` must be positive')
        if self.n_steps < 1:
            raise ValueError('`n_steps` must be positive')
        if self.batch_size < 1:
            raise ValueError('`batch_size` must be positive')
        if self.n_epochs < 1:
            raise ValueError('`n_epochs` must be positive')
        if not 0 <= self.gamma <= 1:
            raise ValueError('`gamma` must be between 0 and 1')
        if self.ent_coef < 0:
            raise ValueError('`ent_coef` must be non-negative')


@dataclasses.dataclass
class ShapingConfig:
    """Configuration for reward shaping.

    These values define the weights and thresholds for intermediate rewards
    and penalties provided to the agent at each step to guide its learning
    process in a sparse reward environment.

    Attributes:
        backtrack_action_penalty: A fixed penalty applied for any backtrack
            action.
        backtrack_token_penalty: An additional penalty per token backtracked.
        repetition_penalty_weight: A multiplier for the repetition feature
            to turn it into a penalty.
        high_confidence_reward: A reward for generating a token with high
            confidence.
        high_confidence_threshold: The probability threshold to trigger the
            high confidence reward.
    """
    backtrack_action_penalty: float = 0.01
    backtrack_token_penalty: float = 0.005
    repetition_penalty_weight: float = 0.02
    high_confidence_reward: float = 0.01
    high_confidence_threshold: float = 0.9


@dataclasses.dataclass
class RLConfig:
    """Root configuration for RL operator training.

    Combines all configuration components into a single object.

    Attributes:
        model_name_or_path: Hugging Face model identifier.
        judge: Judge configuration.
        env: Environment configuration.
        training: Training configuration.
        output_dir: Directory for saving outputs.
        device: Device for training ("auto", "cpu", "cuda", etc.).
    """

    model_name_or_path: str
    judge: JudgeConfig = dataclasses.field(default_factory=JudgeConfig)
    env: EnvConfig = dataclasses.field(default_factory=EnvConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    shaping: ShapingConfig = dataclasses.field(default_factory=ShapingConfig)
    output_dir: Path = dataclasses.field(
        default_factory=lambda: Path('rl_output'))
    device: str = 'auto'

    def __post_init__(self):
        """Validate root configuration."""

        if self.model_name_or_path is None:
            raise ValueError('`model_name_or_path` cannot be None')
        self.model_name_or_path = str(self.model_name_or_path).strip()
        if not self.model_name_or_path:
            raise ValueError('`model_name_or_path` cannot be empty')

        self.device = str(self.device).strip().lower()
        if not self.device:
            raise ValueError('`device` cannot be empty')

        if self.output_dir is None:
            raise ValueError('`output_dir` cannot be None')
        self.output_dir = Path(self.output_dir)
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise ValueError(
                f'`output_dir` {self.output_dir} is not a directory')

        if isinstance(self.judge, dict):
            self.judge = JudgeConfig(**self.judge)
        if isinstance(self.env, dict):
            self.env = EnvConfig(**self.env)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if isinstance(self.shaping, dict):
            self.shaping = ShapingConfig(**self.shaping)
