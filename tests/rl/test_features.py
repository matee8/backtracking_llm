# pylint: disable=missing-module-docstring

import numpy as np
import pytest
import torch

from backtracking_llm.rl.features import (
    FeatureExtractor,
    SimpleFeatureExtractor,
)

# pylint: disable=missing-class-docstring


class TestSimpleFeatureExtractor:

    def test_protocol_conformance(self):
        extractor = SimpleFeatureExtractor()
        assert isinstance(extractor, FeatureExtractor)

    def test_initialization(self):
        extractor = SimpleFeatureExtractor(top_k=5)
        assert extractor.shape == (10,)

        with pytest.raises(ValueError,
                           match='top_k must be a positive integer'):
            SimpleFeatureExtractor(top_k=0)

    def test_feature_extraction_logic(self):
        extractor = SimpleFeatureExtractor(top_k=3)
        vocab_size = 50
        logits = torch.randn(vocab_size)
        probs = torch.softmax(logits, dim=-1)

        features = extractor(logits, probs)

        assert isinstance(features, np.ndarray)
        assert features.shape == extractor.shape
        assert features.dtype == np.float32

        expected_top_logits, _ = torch.topk(logits, 3)
        expected_top_probs, _ = torch.topk(probs, 3)
        expected_features = torch.cat(
            (expected_top_logits, expected_top_probs)).numpy()

        np.testing.assert_array_equal(features, expected_features)
