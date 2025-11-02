# pylint: disable=missing-module-docstring

import pytest
import torch

from backtracking_llm.agents import RLAgentOperator

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
# pylint: disable=missing-class-docstring

@pytest.fixture
def agent() -> RLAgentOperator:
    return RLAgentOperator(input_dim=3, num_actions=3, hidden_dim=16)


class TestRLAgentOperator:

    def test_initialization(self, agent: RLAgentOperator):
        assert agent.input_dim == 3
        assert agent.num_actions == 3
        assert isinstance(agent.network[0], torch.nn.Linear)
        assert agent.network[0].in_features == 3
        assert agent.network[0].out_features == 16
        assert agent.network[2].in_features == 16
        assert agent.network[2].out_features == 3

    def test_forward_pass(self, agent: RLAgentOperator):
        features = torch.randn(1, 3)
        logits = agent.forward(features)
        assert logits.shape == (1, 3)

    def test_extract_features_placeholder(self, agent: RLAgentOperator):
        probabilities = torch.tensor([0.1, 0.2, 0.5, 0.15, 0.05])
        position = 2

        features = agent._extract_features(probabilities, position)
        assert features.shape == (3,)

        assert torch.isclose(features[0], torch.tensor(0.5))

        expected_entropy = -torch.sum(probabilities * torch.log(probabilities))
        assert torch.isclose(features[1], expected_entropy)

        assert torch.isclose(features[2], torch.tensor(0.3))

    def test_extract_features_margin_edge_case(self, agent: RLAgentOperator):
        probabilities = torch.tensor([1.0])
        position = 0
        features = agent._extract_features(probabilities, position)
        assert features[2].item() == 1.0

    def test_extract_features_mismatched_dim_raises_error(self):
        agent_wrong_dim = RLAgentOperator(input_dim=5, num_actions=3)
        with pytest.raises(ValueError, match='produces 3 features'):
            agent_wrong_dim._extract_features(torch.tensor([0.5, 0.5]), 0)

    def test_call_inference_mode(self, agent: RLAgentOperator):
        agent.eval()

        logits = torch.randn(50)
        probabilities = torch.softmax(logits, dim=-1)
        position = 10
        token = 'test'

        assert not agent.saved_log_probs

        action = agent(logits, probabilities, position, token)

        assert isinstance(action, int)
        assert 0 <= action < agent.num_actions
        assert not agent.saved_log_probs

    def test_call_training_mode(self, agent: RLAgentOperator):
        agent.train()

        logits = torch.randn(50)
        probabilities = torch.softmax(logits, dim=-1)
        position = 10
        token = 'test'

        assert not agent.saved_log_probs

        action = agent(logits, probabilities, position, token)

        assert isinstance(action, int)
        assert 0 <= action < agent.num_actions

        assert len(agent.saved_log_probs) == 1
        assert isinstance(agent.saved_log_probs[0], torch.Tensor)

    def test_multiple_calls_in_training_mode(self, agent: RLAgentOperator):
        agent.train()
        for _ in range(5):
            agent(torch.randn(10), torch.softmax(torch.randn(10), -1), 0, '')

        assert len(agent.saved_log_probs) == 5
