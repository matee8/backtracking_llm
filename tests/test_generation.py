# pylint: disable=missing-module-docstring

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from transformers import DynamicCache, PreTrainedTokenizer, PreTrainedModel

from backtracking_llm.generation import Generator, GenerationSession
from backtracking_llm.decision import Operator

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
# pylint: disable=missing-function-docstring


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.device = 'cpu'
    model.return_value.logits = torch.randn(1, 1, 10)
    model.return_value.past_key_values = MagicMock(spec=DynamicCache)
    model.config.vocab_size = 32000
    return model


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 9

    mock_inputs = MagicMock()
    mock_inputs.input_ids = torch.tensor([[1, 2, 3]])
    mock_inputs.to.return_value = mock_inputs

    tokenizer.return_value = mock_inputs
    tokenizer.decode.return_value = 'decoded text'

    return tokenizer


def test_generator_init(mock_model, mock_tokenizer):
    generator = Generator(mock_model, mock_tokenizer)
    assert generator.model is mock_model
    assert generator.tokenizer is mock_tokenizer


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
    assert final_call_args.shape[0] == 5


def test_generate_stops_at_eos_token(mock_model, mock_tokenizer):
    logits_first = torch.full((1, 1, 10), -10.0)
    logits_first[0, 0, 5] = 10.0
    logits_second = torch.full((1, 1, 10), -10.0)
    logits_second[0, 0, 9] = 10.0
    mock_model.return_value.logits = logits_second
    mock_model.side_effect = [
        MagicMock(logits=logits_first, past_key_values=None),
        MagicMock(logits=logits_second, past_key_values=Mock()),
    ]

    generator = Generator(mock_model, mock_tokenizer)
    generator.generate('prompt', max_new_tokens=10, top_k=10)

    final_call_args = mock_tokenizer.decode.call_args[0][0]
    assert final_call_args.shape[0] == 2


def test_generate_uses_kv_cache(mock_model, mock_tokenizer):
    mock_model.return_value.logits = torch.full((1, 1, 10), -10.0)
    mock_model.return_value.logits[0, 0, 5] = 10.0

    generator = Generator(mock_model, mock_tokenizer)
    generator.generate('prompt', max_new_tokens=3, top_k=3)
    first_call_input_ids = mock_model.call_args_list[0].kwargs['input_ids']
    assert first_call_input_ids.shape[1] == 3

    second_call_input_ids = mock_model.call_args_list[1].kwargs['input_ids']
    assert second_call_input_ids.shape[1] == 1


def test_generate_applies_backtracking(mock_model, mock_tokenizer):
    mock_model.return_value.logits = torch.full((1, 1, 10), -10.0)
    mock_model.return_value.logits[0, 0, 5] = 10.0
    mock_model.return_value.past_key_values.get_seq_length.return_value = 10

    mock_operator = Mock(spec=Operator)
    mock_operator.side_effect = [0, 1, 0, 0, 0]

    generator = Generator(mock_model, mock_tokenizer)
    generator.generate('prompt',
                       operator=mock_operator,
                       max_new_tokens=4,
                       backtrack_every_n=1,
                       top_k=5)

    assert mock_operator.call_count == 5
    final_call_args = mock_tokenizer.decode.call_args[0][0]
    assert final_call_args.shape[0] == 4


def test_generate_discards_token_on_clipped_backtrack(mock_model,
                                                      mock_tokenizer):
    mock_model.return_value.logits = torch.full((1, 1, 10), -10.0)
    mock_model.return_value.logits[0, 0, 5] = 10.0
    mock_model.return_value.past_key_values.get_seq_length.return_value = 10

    mock_operator = Mock(spec=Operator)
    mock_operator.side_effect = [2, 0]

    generator = Generator(mock_model, mock_tokenizer)
    generator.generate('prompt',
                       operator=mock_operator,
                       max_new_tokens=1,
                       backtrack_every_n=1,
                       top_k=5)

    assert mock_model.call_count == 2
    assert mock_operator.call_count == 2

    final_call_args = mock_tokenizer.decode.call_args[0][0]
    assert final_call_args.shape[0] == 1


def test_generator_repr_with_name_attributes(mock_model, mock_tokenizer):
    mock_model.config._name_or_path = 'test-model'
    mock_tokenizer.name_or_path = 'test-tokenizer'

    generator = Generator(mock_model, mock_tokenizer)
    expected_repr = "<Generator model='test-model', tokenizer='test-tokenizer'>"

    assert repr(generator) == expected_repr


def test_generator_repr_fallback_on_missing_attributes():
    mock_model = MagicMock(spec=PreTrainedModel)
    del mock_model.config

    mock_tokenizer = MagicMock(spec=PreTrainedTokenizer)
    del mock_tokenizer.name_or_path

    generator = Generator(mock_model, mock_tokenizer)

    model_class_name = mock_model.__class__.__name__
    tokenizer_class_name = mock_tokenizer.__class__.__name__
    expected_repr = (f"<Generator model='{model_class_name}', "
                     f"tokenizer='{tokenizer_class_name}'>")

    assert repr(generator) == expected_repr


@patch('backtracking_llm.generation.AutoTokenizer.from_pretrained')
@patch('backtracking_llm.generation.AutoModelForCausalLM.from_pretrained')
def test_from_pretrained_calls_dependencies_correctly(
        mock_model_from_pretrained, mock_tokenizer_from_pretrained):
    mock_model = Mock()
    mock_tokenizer = Mock()
    mock_model_from_pretrained.return_value = mock_model
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer

    model_name = 'gpt2'

    generator = Generator.from_pretrained(model_name)

    mock_model_from_pretrained.assert_called_once_with(model_name)
    mock_tokenizer_from_pretrained.assert_called_once_with(model_name)

    assert isinstance(generator, Generator)
    assert generator.model is mock_model
    assert generator.tokenizer is mock_tokenizer


@patch('backtracking_llm.generation.AutoTokenizer.from_pretrained')
@patch('backtracking_llm.generation.AutoModelForCausalLM.from_pretrained')
def test_from_pretrained_passes_model_kwargs(mock_model_from_pretrained,
                                             mock_tokenizer_from_pretrained):
    model_name = 'gpt2'
    model_kwargs = {'device_map': 'auto', 'torch_dtype': torch.bfloat16}

    Generator.from_pretrained(model_name, **model_kwargs)

    mock_model_from_pretrained.assert_called_once_with(model_name,
                                                       **model_kwargs)
    mock_tokenizer_from_pretrained.assert_called_once_with(model_name)


def test_call_is_alias_for_generate(mock_model, mock_tokenizer):
    generator = Generator(mock_model, mock_tokenizer)

    assert generator.__call__ == generator.generate


@patch('backtracking_llm.generation.torch.topk')
def test_generate_caps_top_k_at_vocab_size(mock_topk, mock_model,
                                           mock_tokenizer):
    vocab_size = 30
    mock_model.config.vocab_size = vocab_size

    requested_top_k = 100

    mock_topk.return_value = (torch.tensor([[1.0]]), torch.tensor([[5]]))
    mock_model.return_value.logits = torch.randn(1, 1, vocab_size)

    generator = Generator(mock_model, mock_tokenizer)

    generator.generate('prompt', max_new_tokens=1, top_k=requested_top_k)

    called_k = mock_topk.call_args[0][1]
    assert called_k == vocab_size
    assert called_k != requested_top_k


def test_generate_stops_on_single_token_stop_sequence(mock_model,
                                                      mock_tokenizer):
    token_sequence = [5, 6, 7, 8]
    side_effects = []
    test_vocab_size = 10
    for token_id in token_sequence:
        logits = torch.full((1, 1, 10), -10.0)
        logits[0, 0, token_id] = 10.0
        side_effects.append(MagicMock(logits=logits, past_key_values=Mock()))

    mock_model.side_effect = side_effects
    mock_model.config.vocab_size = test_vocab_size

    def decode_side_effect(ids, skip_special_tokens=True):
        _ = skip_special_tokens
        text = ''.join([f'<{i}>' for i in ids.tolist()])
        return text.replace('<7>', ' STOP')

    mock_tokenizer.decode.side_effect = decode_side_effect
    stop_sequences = [' STOP']

    generator = Generator(mock_model, mock_tokenizer)

    generator.generate('prompt',
                       max_new_tokens=10,
                       stop_sequences=stop_sequences)

    assert mock_model.call_count == 3


def test_generate_stops_on_multi_token_stop_sequence(mock_model,
                                                     mock_tokenizer):
    token_sequence = [5, 6, 7]
    side_effects = []
    for token_id in token_sequence:
        logits = torch.full((1, 1, 10), -10.0)
        logits[0, 0, token_id] = 10.0
        side_effects.append(MagicMock(logits=logits, past_key_values=Mock()))

    mock_model.side_effect = side_effects
    mock_model.config.vocab_size = 10

    decode_outputs = ['token5', 'token5 User:', 'token5 User: t7']
    mock_tokenizer.decode.side_effect = decode_outputs

    stop_sequences = ['User:']
    generator = Generator(mock_model, mock_tokenizer)

    generator.generate('prompt',
                       max_new_tokens=10,
                       top_k=5,
                       stop_sequences=stop_sequences)

    assert mock_model.call_count == 2


@patch('backtracking_llm.generation.torch.topk')
def test_generate_applies_temperature_scaling_to_logits(mock_topk, mock_model,
                                                        mock_tokenizer):
    original_logits = torch.tensor([[[0.0, 2.0, 4.0]]])
    mock_model.return_value.logits = original_logits
    mock_model.config.vocab_size = 3

    mock_topk.return_value = (torch.tensor([[1.0]]), torch.tensor([[2]]))

    temperature = 2.0
    expected_scaled_logits = original_logits / temperature

    generator = Generator(mock_model, mock_tokenizer)

    generator.generate('prompt', temperature=temperature, max_new_tokens=1)

    actual_logits_passed_to_topk = mock_topk.call_args[0][0]
    assert torch.allclose(actual_logits_passed_to_topk, expected_scaled_logits)


@patch('backtracking_llm.generation.torch.topk')
def test_generate_skips_temperature_scaling_when_zero(mock_topk, mock_model,
                                                      mock_tokenizer):
    original_logits = torch.tensor([[[0.0, 2.0, 4.0]]])
    mock_model.return_value.logits = original_logits
    mock_model.config.vocab_size = 3
    mock_topk.return_value = (torch.tensor([[1.0]]), torch.tensor([[2]]))

    temperature = 0.0

    generator = Generator(mock_model, mock_tokenizer)

    generator.generate('prompt', temperature=temperature, max_new_tokens=1)

    actual_logits_passed_to_topk = mock_topk.call_args[0][0]
    assert torch.allclose(actual_logits_passed_to_topk, original_logits)


def test_generation_session_exists_and_is_callable():
    assert callable(GenerationSession)


def test_generation_session_backtrack_method(mock_model, mock_tokenizer):
    mock_model.return_value.past_key_values.get_seq_length.return_value = 10
    original_logits = torch.tensor([[[0.0, 2.0, 4.0]]])
    mock_model.return_value.logits = original_logits
    mock_model.config.vocab_size = 3
    session = GenerationSession(mock_model,
                                mock_tokenizer,
                                prompt='hi',
                                top_k=10)
    session.step()
    session.step()

    assert session._input_ids.shape[1] == 5

    session.backtrack(10)

    assert session._input_ids.shape[1] == 3
    assert session._generated_token_count == 0


def test_generation_session_done_persists_after_backtrack(
        mock_model, mock_tokenizer):
    mock_model.return_value.past_key_values.get_seq_length.return_value = 10
    session = GenerationSession(mock_model,
                                mock_tokenizer,
                                prompt='hi',
                                max_new_tokens=1,
                                top_k=10)

    session.step()
    assert session.done

    session.backtrack(1)
    assert not session.done


def test_generation_session_raises_after_done(mock_model, mock_tokenizer):
    session = GenerationSession(mock_model, mock_tokenizer, prompt='hi')
    session._done = True

    with pytest.raises(RuntimeError, match='done'):
        session.step()


def test_generation_backtrack_returns_on_negative(mock_model, mock_tokenizer,
                                                  caplog):
    session = GenerationSession(mock_model, mock_tokenizer, prompt='hi')

    shape_before = session._input_ids.shape

    session.backtrack(-1)

    shape_after = session._input_ids.shape

    assert 'negative' in caplog.text
    assert shape_before == shape_after


def test_generation_backtrack_returns_on_no_tokens(mock_model, mock_tokenizer,
                                                   caplog):
    session = GenerationSession(mock_model, mock_tokenizer, prompt='hi')

    shape_before = session._input_ids.shape

    session.backtrack(1)

    shape_after = session._input_ids.shape

    assert 'No tokens' in caplog.text
    assert shape_before == shape_after


def test_generation_backtrack_returns_on_0_tokens(mock_model, mock_tokenizer,
                                                  caplog):
    session = GenerationSession(mock_model,
                                mock_tokenizer,
                                prompt='hi',
                                top_k=10)
    session.step()

    shape_before = session._input_ids.shape

    session.backtrack(0)

    shape_after = session._input_ids.shape

    assert '0 tokens' in caplog.text
    assert shape_before == shape_after


def test_generation_backtrack_rolls_back_kv_cache(mock_model, mock_tokenizer):
    session = GenerationSession(mock_model, mock_tokenizer, prompt='hi')
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])
    mock_cache = MagicMock(spec=DynamicCache)
    mock_cache.get_seq_length.return_value = 5

    session._input_ids = input_ids
    session._past_key_values = mock_cache

    session.backtrack(2)

    mock_cache.crop.assert_called_once_with(3)


def test_generation_backtrack_empties_kv_cache(mock_model, mock_tokenizer):
    session = GenerationSession(mock_model, mock_tokenizer, prompt='')
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])
    mock_cache = MagicMock(spec=DynamicCache)
    mock_cache.get_seq_length.return_value = 5

    session._input_ids = input_ids
    session._past_key_values = mock_cache
    session.prompt_length = 0

    session.backtrack(5)

    assert session._past_key_values is None
