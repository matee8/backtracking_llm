# pylint: disable=missing-module-docstring

from unittest.mock import MagicMock, Mock

import pytest
import torch
from torch.nn import functional as F

from backtracking_llm.generation import Generator

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
# pylint: disable=missing-function-docstring


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.device = 'cpu'
    model.return_value.logits = torch.randn(1, 1, 10)
    model.return_value.past_key_values = None
    return model


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 0

    mock_inputs = MagicMock()
    mock_inputs.input_ids = torch.tensor([[0, 1, 2]])
    mock_inputs.to.return_value = mock_inputs

    tokenizer.return_value = mock_inputs
    tokenizer.decode.return_value = 'decoded text'

    return tokenizer


def test_generator_init(mock_model, mock_tokenizer):
    generator = Generator(mock_model, mock_tokenizer)
    assert generator.model is mock_model
    assert generator.tokenizer is mock_tokenizer


def test_calculate_top_k_distribution_no_temp():
    generator = Generator(Mock(), Mock())
    logits = torch.tensor([1.0, 2.0, 10.0, 5.0, 9.0])
    _, probs, indices = generator._calculate_top_k_distribution(logits,
                                                                temperature=0.0,
                                                                top_k=3)

    assert torch.equal(indices, torch.tensor([2, 4, 3]))
    assert torch.allclose(probs,
                          F.softmax(torch.tensor([10.0, 9.0, 5.0]), dim=-1))


def test_calculate_top_k_distribution_with_temp():
    generator = Generator(Mock(), Mock())
    logits = torch.tensor([1.0, 2.0, 10.0, 5.0, 9.0])
    _, probs, _ = generator._calculate_top_k_distribution(logits,
                                                          temperature=2.0,
                                                          top_k=3)
    scaled_logits = torch.tensor([10.0, 9.0, 5.0]) / 2.0
    expected_probs = F.softmax(scaled_logits, dim=-1)
    assert torch.allclose(probs, expected_probs)


def test_generate_raises_for_invalid_backtrack_every_n(mock_model,
                                                       mock_tokenizer):
    generator = Generator(mock_model, mock_tokenizer)
    with pytest.raises(ValueError, match='must be a positive integer'):
        generator.generate('prompt', backtrack_every_n=0)


def test_generate_stops_at_max_new_tokens(mock_model, mock_tokenizer):
    mock_model.return_value.logits = torch.full((1, 1, 10), -10.0)
    mock_model.return_value.logits[0, 0, 5] = 10.0

    generator = Generator(mock_model, mock_tokenizer)
    generator.generate('prompt', max_new_tokens=5, top_k=10)

    final_call_args = mock_tokenizer.decode.call_args[0][0]
    print(final_call_args.shape)
    assert final_call_args.shape[0] == 8


def test_generate_stops_at_eos_token(mock_model, mock_tokenizer):
    logits_first = torch.full((1, 1, 10), -10.0)
    logits_first[0, 0, 5] = 10.0
    logits_second = torch.full((1, 1, 10), -10.0)
    logits_second[0, 0, 0] = 10.0
    mock_model.return_value.logits = logits_second
    mock_model.side_effect = [
        MagicMock(logits=logits_first, past_key_values=None),
        MagicMock(logits=logits_second, past_key_values=Mock()),
    ]

    generator = Generator(mock_model, mock_tokenizer)
    generator.generate('prompt', max_new_tokens=10, top_k=10)

    final_call_args = mock_tokenizer.decode.call_args[0][0]
    assert final_call_args.shape[0] == 4


def test_generate_uses_kv_cache(mock_model, mock_tokenizer):
    generator = Generator(mock_model, mock_tokenizer)
    generator.generate('prompt', max_new_tokens=3, top_k=3)
    first_call_input_ids = mock_model.call_args_list[0].kwargs['input_ids']
    assert first_call_input_ids.shape[1] == 3

    second_call_input_ids = mock_model.call_args_list[1].kwargs['input_ids']
    assert second_call_input_ids.shape[1] == 1
