# pylint: disable=missing-module-docstring

import copy
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from transformers import (DynamicCache, PreTrainedTokenizer, PreTrainedModel,
                          AutoModelForCausalLM, AutoTokenizer)

from backtracking_llm.generation import Generator, GenerationState
from backtracking_llm.decision import Operator

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


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


def test_apply_backtracking_ids_only():
    generator = Generator(Mock(), Mock())
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])
    truncated_ids, _ = generator._apply_backtracking(input_ids, None, 2)
    assert torch.equal(truncated_ids, torch.tensor([[0, 1, 2]]))


def test_apply_backtracking_with_cache():
    generator = Generator(Mock(), Mock())
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])

    mock_cache = MagicMock(spec=DynamicCache)
    mock_cache.get_seq_length.return_value = 5

    generator._apply_backtracking(input_ids, mock_cache, 2)

    mock_cache.crop.assert_called_once_with(3)


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
    assert final_call_args.shape[0] == 1


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

    mock_operator = Mock(spec=Operator)
    mock_operator.side_effect = [0, 1, 0, 0, 0]

    generator = Generator(mock_model, mock_tokenizer)
    generator.generate('prompt',
                       operator=mock_operator,
                       max_new_tokens=3,
                       backtrack_every_n=1,
                       top_k=5)

    assert mock_operator.call_count == 5
    final_call_args = mock_tokenizer.decode.call_args[0][0]
    assert final_call_args.shape[0] == 3


def test_generate_discards_token_on_clipped_backtrack(mock_model,
                                                      mock_tokenizer):
    mock_model.return_value.logits = torch.full((1, 1, 10), -10.0)
    mock_model.return_value.logits[0, 0, 5] = 10.0

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


class TestGenerationStateCreation:

    def test_default_creation(self):
        input_ids = torch.tensor([[1, 2, 3]])
        state = GenerationState(input_ids=input_ids)

        assert torch.equal(state.input_ids, input_ids)
        assert state.past_key_values is None
        assert state.prompt_length == 0
        assert state.generated_count == 0
        assert state.max_new_tokens == 100
        assert state.temperature == 1.0
        assert state.top_k == 50

    def test_custom_values(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        mock_cache = MagicMock(spec=DynamicCache)

        state = GenerationState(
            input_ids=input_ids,
            past_key_values=mock_cache,
            prompt_length=2,
            generated_count=3,
            max_new_tokens=200,
            temperature=0.8,
            top_k=30,
        )

        assert torch.equal(state.input_ids, input_ids)
        assert state.past_key_values is mock_cache
        assert state.prompt_length == 2
        assert state.generated_count == 3
        assert state.max_new_tokens == 200
        assert state.temperature == 0.8
        assert state.top_k == 30

    def test_1d_input_ids_converted_to_2d(self):
        input_ids = torch.tensor([1, 2, 3])
        state = GenerationState(input_ids=input_ids)

        assert state.input_ids.dim() == 2
        assert state.input_ids.shape == (1, 3)
        assert torch.equal(state.input_ids[0], torch.tensor([1, 2, 3]))

    def test_2d_input_ids_unchanged(self):
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        state = GenerationState(input_ids=input_ids)

        assert state.input_ids.dim() == 2
        assert torch.equal(state.input_ids, input_ids)


class TestGenerationStateValidation:

    def test_invalid_input_ids_dim_raises(self):
        input_ids = torch.ones(2, 3, 4)
        with pytest.raises(ValueError, match='must be 1D or 2D tensor'):
            GenerationState(input_ids=input_ids)

    def test_generated_count_exceeds_available_raises(self):
        input_ids = torch.tensor([[1, 2, 3]])
        with pytest.raises(ValueError, match='exceeds actual generated tokens'):
            GenerationState(input_ids=input_ids,
                            prompt_length=0,
                            generated_count=4)

    def test_valid_generated_count_within_bounds(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        state = GenerationState(input_ids=input_ids,
                                prompt_length=2,
                                generated_count=3)
        assert state.generated_count == 3

    def test_zero_generated_count_valid(self):
        input_ids = torch.tensor([[1, 2, 3]])
        state = GenerationState(input_ids=input_ids,
                                prompt_length=0,
                                generated_count=0)
        assert state.generated_count == 0

    def test_prompt_length_equals_sequence_length_and_generated_count_zero(
            self):
        input_ids = torch.tensor([[1, 2, 3]])
        state = GenerationState(input_ids=input_ids,
                                prompt_length=3,
                                generated_count=0)
        assert state.prompt_length == 3
        assert state.generated_count == 0


class TestGenerationStateProperties:

    def test_device_property(self):
        input_ids = torch.tensor([[1, 2, 3]], device='cpu')
        state = GenerationState(input_ids=input_ids)
        assert state.device == torch.device('cpu')

        if torch.cuda.is_available():
            input_ids_cuda = torch.tensor([[1, 2, 3]], device='cuda')
            state_cuda = GenerationState(input_ids=input_ids_cuda)
            assert state_cuda.device.type == 'cuda'

    def test_batch_size_property(self):
        input_ids = torch.tensor([[1, 2, 3]])
        state = GenerationState(input_ids=input_ids)
        assert state.batch_size == 1

        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        state = GenerationState(input_ids=input_ids)
        assert state.batch_size == 2

    def test_sequence_length_property(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        state = GenerationState(input_ids=input_ids)
        assert state.sequence_length == 5

    def test_generated_ids_property_no_generation(self):
        input_ids = torch.tensor([[1, 2, 3]])
        state = GenerationState(input_ids=input_ids,
                                prompt_length=3,
                                generated_count=0)
        generated = state.generated_ids

        assert generated.shape == (1, 0)
        assert generated.numel() == 0

    def test_generated_ids_property_with_generation(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        state = GenerationState(input_ids=input_ids,
                                prompt_length=2,
                                generated_count=3)
        generated = state.generated_ids

        assert generated.shape == (1, 3)
        assert torch.equal(generated[0], torch.tensor([3, 4, 5]))

    def test_generated_ids_property_all_generated(self):
        input_ids = torch.tensor([[1, 2, 3]])
        state = GenerationState(input_ids=input_ids,
                                prompt_length=0,
                                generated_count=3)
        generated = state.generated_ids

        assert torch.equal(generated, input_ids)


class TestGenerationStateEdgeCases:

    def test_too_large_input_ids_raises(self):
        input_ids = torch.empty(1, 1, 1)
        with pytest.raises(ValueError, match='must be 1D or 2D tensor'):
            GenerationState(input_ids=input_ids)

    def test_empty_2d_input_ids(self):
        input_ids = torch.empty(1, 0).long()

        state = GenerationState(input_ids=input_ids,
                                prompt_length=0,
                                generated_count=0)
        assert state.sequence_length == 0

    def test_large_batch_dimension(self):
        input_ids = torch.randn(8, 10)
        state = GenerationState(input_ids=input_ids)
        assert state.batch_size == 8
        assert state.sequence_length == 10

    def test_dataclass_fields_accessible(self):
        input_ids = torch.tensor([[1, 2, 3]])
        state = GenerationState(input_ids=input_ids)

        assert hasattr(state, 'input_ids')
        assert hasattr(state, 'past_key_values')
        assert hasattr(state, 'prompt_length')
        assert hasattr(state, 'generated_count')
        assert hasattr(state, 'max_new_tokens')
        assert hasattr(state, 'temperature')
        assert hasattr(state, 'top_k')

        state.prompt_length = 5
        assert state.prompt_length == 5

    def test_state_with_mock_past_key_values(self):
        input_ids = torch.tensor([[1, 2, 3]])
        mock_cache = MagicMock(spec=DynamicCache)

        state = GenerationState(input_ids=input_ids, past_key_values=mock_cache)
        assert state.past_key_values is mock_cache

    def test_generated_count_zero_state(self):
        input_ids = torch.tensor([[1, 2, 3]])
        state = GenerationState(
            input_ids=input_ids,
            prompt_length=3,
            generated_count=0,
            max_new_tokens=100,
        )

        assert state.prompt_length == 3
        assert state.generated_count == 0
        assert state.max_new_tokens == 100
        assert state.generated_ids.numel() == 0


@pytest.fixture
def generator():
    model = AutoModelForCausalLM.from_pretrained('gpt2',
                                                 torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    return Generator(model, tokenizer)


class TestInitializeStateBasic:

    def test_initialize_state_creates_state(self, generator):
        generator._initialize_state('Hello world')

        assert generator._state is not None
        assert isinstance(generator._state, GenerationState)

    def test_initialize_state_prompt_tokenized(self, generator):
        generator._initialize_state('Hello world')

        assert generator._state.input_ids is not None
        assert generator._state.input_ids.dim() == 2
        assert generator._state.input_ids.shape[0] == 1
        assert generator._state.input_ids.shape[1] > 0

    def test_initialize_state_prompt_length_recorded(self, generator):
        test_prompt = 'Hello world'
        generator._initialize_state(test_prompt)

        expected_length = generator.tokenizer(
            test_prompt, return_tensors='pt').input_ids.shape[1]
        assert generator._state.prompt_length == expected_length
        assert generator._state.generated_count == 0

    def test_initialize_state_default_params(self, generator):
        generator._initialize_state('Test')

        assert generator._state.max_new_tokens == 100
        assert generator._state.temperature == 1.0
        assert generator._state.top_k == 50

    def test_initialize_state_custom_params(self, generator):
        generator._initialize_state(
            'Test',
            max_new_tokens=200,
            temperature=0.8,
            top_k=30,
        )

        assert generator._state.max_new_tokens == 200
        assert generator._state.temperature == 0.8
        assert generator._state.top_k == 30


class TestInitializeStateValidation:

    def test_empty_prompt_raises(self, generator):
        with pytest.raises(ValueError, match='Prompt cannot be empty'):
            generator._initialize_state('')

    def test_whitespace_prompt_raises(self, generator):
        with pytest.raises(ValueError, match='Prompt cannot be empty'):
            generator._initialize_state('   ')

    def test_invalid_max_new_tokens_raises(self, generator):
        with pytest.raises(ValueError, match='`max_new_tokens` must be positive'):
            generator._initialize_state('Test', max_new_tokens=0)

        with pytest.raises(ValueError, match='`max_new_tokens` must be positive'):
            generator._initialize_state('Test', max_new_tokens=-1)

    def test_invalid_temperature_raises(self, generator):
        with pytest.raises(ValueError, match='`temperature` must be positive'):
            generator._initialize_state('Test', temperature=0.0)

        with pytest.raises(ValueError, match='`temperature` must be positive'):
            generator._initialize_state('Test', temperature=-0.5)

    def test_invalid_top_k_raises(self, generator):
        with pytest.raises(ValueError, match='`top_k` must be positive'):
            generator._initialize_state('Test', top_k=0)

        with pytest.raises(ValueError, match='`top_k` must be positive'):
            generator._initialize_state('Test', top_k=-10)


class TestInitializeStateDeviceHandling:

    def test_default_device_is_model_device(self, generator):
        generator._initialize_state('Test')

        assert generator._state.input_ids.device == generator.model.device

    def test_custom_device_override(self, generator):
        generator._initialize_state('Test', device='cpu')

        assert generator._state.input_ids.device.type == 'cpu'

    def test_device_matches_input(self, generator):
        generator._initialize_state('Test')

        assert generator._state.input_ids.device == generator._state.device


class TestInitializeStateTensorHandling:

    def test_1d_input_converted_to_2d(self, generator):
        generator._initialize_state('Test')

        assert generator._state.input_ids.dim() == 2
        assert generator._state.batch_size == 1

    def test_tokenizer_truncation_respected(self, generator):
        long_prompt = 'word ' * 1000

        generator._initialize_state(long_prompt, max_seq_length=50)

        assert generator._state.input_ids.shape[1] <= 50

    def test_tokenizer_no_truncation_by_default(self, generator):
        normal_prompt = 'Short prompt'
        generator._initialize_state(normal_prompt)

        expected_length = generator.tokenizer(
            normal_prompt, return_tensors='pt').input_ids.shape[1]
        assert generator._state.input_ids.shape[1] == expected_length

    def test_past_key_values_initialized_none(self, generator):
        generator._initialize_state('Test')

        assert generator._state.past_key_values is None

    def test_all_state_fields_populated(self, generator):
        generator._initialize_state(
            'Test prompt',
            max_new_tokens=150,
            temperature=0.7,
            top_k=40,
        )

        state = generator._state

        assert isinstance(state.input_ids, torch.Tensor)
        assert state.past_key_values is None
        assert isinstance(state.prompt_length, int)
        assert isinstance(state.generated_count, int)
        assert isinstance(state.max_new_tokens, int)
        assert isinstance(state.temperature, float)
        assert isinstance(state.top_k, int)


class TestInitializeStateIntegration:

    def test_initialize_state_does_not_affect_generate(self, generator):
        generator._initialize_state('Test prompt')

        result = generator.generate('Another prompt', max_new_tokens=10)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_multiple_initialize_state_calls_override(self, generator):
        generator._initialize_state('First prompt')
        first_length = generator._state.prompt_length

        generator._initialize_state('Second prompt that is longer')
        second_length = generator._state.prompt_length

        assert second_length > first_length
        assert 'Second prompt' in generator.tokenizer.decode(
            generator._state.input_ids[0])

    def test_initialize_state_preserves_model_state(self, generator):
        original_params = {
            k: v.clone() for k, v in generator.model.named_parameters()
        }

        generator._initialize_state('Test')

        for k, v in generator.model.named_parameters():
            assert torch.equal(v, original_params[k])

    def test_state_isolation_between_generators(self):
        model1 = AutoModelForCausalLM.from_pretrained('gpt2',
                                                      torch_dtype=torch.float32)
        tokenizer1 = AutoTokenizer.from_pretrained('gpt2')
        gen1 = Generator(model1, tokenizer1)

        model2 = AutoModelForCausalLM.from_pretrained('gpt2',
                                                      torch_dtype=torch.float32)
        tokenizer2 = AutoTokenizer.from_pretrained('gpt2')
        gen2 = Generator(model2, tokenizer2)

        gen1._initialize_state('Prompt one')
        gen2._initialize_state('Prompt two.')

        assert not torch.equal(gen1._state.input_ids, gen2._state.input_ids)
        assert gen1._state.prompt_length != gen2._state.prompt_length


class TestInitializeStateEdgeCases:

    def test_single_token_prompt(self, generator):
        generator._initialize_state('Hi')

        assert generator._state.prompt_length == 1
        assert generator._state.generated_count == 0

    def test_very_long_prompt(self, generator):
        long_prompt = 'word ' * 100
        generator._initialize_state(long_prompt)

        assert generator._state.prompt_length > 0
        assert generator._state.prompt_length < 200

    def test_prompt_with_special_characters(self, generator):
        special_prompt = 'Hello!\nNew line\tTab 😊'
        generator._initialize_state(special_prompt)

        assert generator._state.input_ids.shape[1] > 0

    def test_prompt_unicode_handling(self, generator):
        unicode_prompt = 'Hello 世界 🌍'
        generator._initialize_state(unicode_prompt)

        assert generator._state.input_ids is not None

    def test_empty_generation_params_uses_defaults(self, generator):
        generator._initialize_state('Test', **{})

        assert generator._state.max_new_tokens == 100
        assert generator._state.temperature == 1.0
        assert generator._state.top_k == 50

    def test_partial_generation_params_merged_with_defaults(self, generator):
        generator._initialize_state('Test', temperature=0.5)

        assert generator._state.temperature == 0.5
        assert generator._state.max_new_tokens == 100
        assert generator._state.top_k == 50


def test_model_device_attribute_missing(mock_model, mock_tokenizer):
    if hasattr(mock_model, 'device'):
        delattr(mock_model, 'device')

    generator = Generator(mock_model, mock_tokenizer)

    generator.model = mock_model

    with pytest.raises(RuntimeError,
                       match='Model does not have a device attribute'):
        generator._initialize_state('Test')
