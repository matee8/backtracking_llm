"""Provides the main orchestrator for the RL training pipeline."""

import logging

from sb3_contrib import RecurrentPPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv

from backtracking_llm.generation import Generator, GenerationSession
from backtracking_llm.rl.config import RLConfig
from backtracking_llm.rl.data import PromptProvider
from backtracking_llm.rl.rewards import RewardShaper
from backtracking_llm.rl.env import BacktrackingEnv
from backtracking_llm.rl.judges import Judge, OpenAIJudge
from backtracking_llm.rl.wrapper import SB3LstmWrapper

logger = logging.getLogger(__name__)


class RLTrainer:
    """Orchestrates the end-to-end reinforcement learning training process.

    This class handles the setup of all components required for training,
    including the model generator, the scoring judge, the Gym environment,
    and the RL agent. It then manages the training loop and saves the
    resulting policy.
    """

    def __init__(self, config: RLConfig) -> None:
        """Initializes the RLTrainer.

        Args:
            config: The main configuration object for the entire RL run.
        """
        self.config = config
        self.generator = Generator.from_pretrained(config.model_name_or_path)
        self.generator.model.to(config.device)  # type: ignore
        self.generator.model.eval()
        self.judge: Judge = OpenAIJudge(config.judge)
        self.shaper = RewardShaper(config.shaping)

    def train(self, prompt_provider: PromptProvider) -> None:
        """Executes the full RL training pipeline.

        Args:
            prompt_provider: A provider that supplies prompts for each
                training episode.
        """
        logger.info('Starting RL Training Pipeline for model: %s.',
                    self.config.model_name_or_path)

        def session_factory() -> GenerationSession:
            """Creates a new GenerationSession for each episode."""
            prompt = next(prompt_provider)
            return GenerationSession(
                model=self.generator.model,
                tokenizer=self.generator.tokenizer,
                prompt=prompt,
                max_new_tokens=self.config.env.max_seq_length)

        def env_factory() -> SB3LstmWrapper:
            """Creates the base environment for the VecEnv."""
            env = BacktrackingEnv(session_factory=session_factory,
                                  judge=self.judge,
                                  shaper=self.shaper,
                                  config=self.config.env)
            return SB3LstmWrapper(env)

        logger.info('Verifying environment compatibility...')
        env_checker.check_env(env_factory())
        logger.info('Environment check passed.')

        env = DummyVecEnv([env_factory])

        agent = RecurrentPPO(
            policy=self.config.training.policy_type,
            env=env,
            learning_rate=self.config.training.learning_rate,
            n_steps=self.config.training.n_steps,
            batch_size=self.config.training.batch_size,
            n_epochs=self.config.training.n_epochs,
            gamma=self.config.training.gamma,
            ent_coef=self.config.training.ent_coef,
            seed=self.config.training.seed,
            device=self.config.device,
            verbose=1,
        )

        logger.info('Starting agent training for %d timesteps...',
                    self.config.training.total_timesteps)
        agent.learn(total_timesteps=self.config.training.total_timesteps,
                    progress_bar=True)

        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / 'policy.zip'
        agent.save(model_path)

        logger.info('RL Training Pipeline Finished.')
        logger.info('Trained policy saved to: %s', model_path)
