# pylint: disable=missing-module-docstring

import pytest
import torch

from backtracking_llm.features import SimpleFeatureExtractor

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

class TestSimpleFeatureExtractor:

    @pytest.fixture
    def extractor(self) -> SimpleFeatureExtractor:
        return SimpleFeatureExtractor()

    def test_feature_extraction(self, extractor: SimpleFeatureExtractor):
        probabilities = torch.tensor([0.1, 0.2, 0.5, 0.15, 0.05])
        position = 2

        features = extractor(probabilities, position)

        assert features.shape == (extractor.feature_dim,)
        assert extractor.feature_dim == 3
        assert torch.isclose(features[0], torch.tensor(0.5))
        expected_entropy = -torch.sum(probabilities * torch.log(probabilities))
        assert torch.isclose(features[1], expected_entropy)
        assert torch.isclose(features[2], torch.tensor(0.3))

    def test_margin_edge_case(self, extractor: SimpleFeatureExtractor):
        probabilities = torch.tensor([1.0])
        position = 0
        features = extractor(probabilities, position)
        assert features[2].item() == 1.0
