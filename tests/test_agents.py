# pylint: disable=missing-module-docstring

import pytest
import torch
from unittest.mock import MagicMock

from backtracking_llm.agents import RLAgentOperator
from backtracking_llm.features import FeatureExtractor

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
# pylint: disable=missing-class-docstring


@pytest.fixture
def mock_feature_extractor() -> MagicMock:
    extractor = MagicMock(spec=FeatureExtractor)
    extractor.feature_dim = 3
    extractor.return_value = torch.randn(3)
    return extractor


@pytest.fixture
def agent(mock_feature_extractor: MagicMock) -> RLAgentOperator:
    return RLAgentOperator(feature_extractor=mock_feature_extractor,
                           num_actions=3,
                           hidden_dim=16)


class TestRLAgentOperator:

    def test_initialization(self, agent: RLAgentOperator):
        assert agent.num_actions == 3
        assert isinstance(agent.network[0], torch.nn.Linear)
        assert agent.network[
            0].in_features == agent.feature_extractor.feature_dim
        assert agent.network[0].out_features == 16
        assert agent.network[2].in_features == 16
        assert agent.network[2].out_features == 3

    def test_forward_pass(self, agent: RLAgentOperator):
        features = torch.randn(1, 3)
        logits = agent.forward(features)
        assert logits.shape == (1, 3)

    def test_call_inference_mode(self, agent: RLAgentOperator,
                                 mock_feature_extractor: MagicMock):
        agent.eval()

        logits = torch.randn(50)
        probabilities = torch.softmax(logits, dim=-1)
        position = 10
        token = 'test'

        assert not agent.saved_log_probs

        action = agent(logits, probabilities, position, token)

        mock_feature_extractor.assert_called_once_with(probabilities, position)
        assert isinstance(action, int)
        assert 0 <= action < agent.num_actions
        assert not agent.saved_log_probs

    def test_call_training_mode(self, agent: RLAgentOperator,
                                mock_feature_extractor: MagicMock):
        agent.train()

        logits = torch.randn(50)
        probabilities = torch.softmax(logits, dim=-1)
        position = 10
        token = 'test'

        assert not agent.saved_log_probs

        action = agent(logits, probabilities, position, token)

        mock_feature_extractor.assert_called_once_with(probabilities, position)
        assert isinstance(action, int)
        assert 0 <= action < agent.num_actions

        assert len(agent.saved_log_probs) == 1
        assert isinstance(agent.saved_log_probs[0], torch.Tensor)

    def test_multiple_calls_in_training_mode(self, agent: RLAgentOperator):
        agent.train()
        for _ in range(5):
            agent(torch.randn(10), torch.softmax(torch.randn(10), -1), 0, '')

        assert len(agent.saved_log_probs) == 5
