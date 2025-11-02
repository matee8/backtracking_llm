"""Defines machine-learning based operators."""

<<<<<<< HEAD
from typing import List

=======
import torch
>>>>>>> parent of baec8be (feat(scripts): implement basic REINFORCE training loop)
from torch import Tensor, nn
from torch.nn import functional as F

from backtracking_llm.features import FeatureExtractor


class RLAgentOperator(nn.Module):
    """An operator that uses a trained RL agent to decide when to backtrack.

    This class wraps a simple MPL that acts as the policy network. It takes
    engineered features from the generation state and outputs a probability
    distribution over a discrete set of actions.

    The agent is not trainable by default; it must be trained in a separate
    RL loop and its state dictionary loaded for inference.
    """

    def __init__(self,
                 feature_extractor: FeatureExtractor,
                 num_actions: int,
                 hidden_dim: int = 64) -> None:
        """Initializes the RL Agent Operator

        Args:
            feature_extractor: An object that can convert a generation state
                into a feature tensor.
            num_actions: The number of discrete backtrack actions to choose
                from. For example, a value of 3 corresponds to the action
                space {0, 1, 2}.
            hidden_dim: The size of the hidden layer in the MLP.
        """
        super().__init__()

        self.feature_extractor = feature_extractor
        self.num_actions = num_actions

        self.network = nn.Sequential(
            nn.Linear(feature_extractor.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_actions))

    def forward(self, state_features: Tensor) -> Tensor:
        """Performs a forward pass to get the logits for each action.

        Args:
            state_features: A tensor representing the engineered features of
                the current generation state.

        Returns:
            A tensor of logits for each possible backtrack action.
        """
        return self.network(state_features)

    def __call__(self, logits: Tensor, probabilities: Tensor, position: int,
                 token: str) -> int:
        """Determines the backtrack count by sampling an action from the policy.

        This method conforms to the `Operator` protocol.

        Args:
            logits: The raw top-k logits from the model.
            probabilities: The top-k probabilities for the current token.
            position: The position (index) of the chosen token.
            token: The string representation of the chosen token (unused by
                this agent but required by the protocol).

        Returns:
            The number of tokens to backtrack, as chosen by the agent.
        """
        device = next(self.parameters()).device
        features = self.feature_extractor(probabilities, position).to(device)

        context_manager = (torch.inference_mode() if hasattr(
            torch, 'inference_mode') else torch.no_grad())

        with context_manager:
            device = next(self.parameters()).device
            features = self._extract_features(probabilities,
                                              position).to(device)

            action_logits = self.forward(features)

            action_probs = F.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_probs, num_samples=1).item()

        return int(action)
