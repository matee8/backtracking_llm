# pylint: disable=missing-module-docstring

from unittest.mock import MagicMock, call
from lm_eval.api.instance import Instance

from backtracking_llm.benchmark.model import BacktrackingLM
from backtracking_llm.decision import Never

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
# pylint: disable=missing-function-docstring


class TestBacktrackingLM:
    """Tests the BacktrackingLM class."""

    def test_initialization(self, mock_generator: MagicMock):
        operator = Never()
        model = BacktrackingLM(generator=mock_generator, operator=operator)

        assert model._generator is mock_generator
        assert model._operator is operator
        assert model.model is mock_generator.model
        assert model.tokenizer is mock_generator.tokenizer

    def test_generate_until_calls_generator_correctly(
            self, mock_generator: MagicMock):
        requests = [
            Instance(request_type='generate_until',
                     doc={},
                     arguments=('prompt1', {
                         'until': ['stop1'],
                         'max_gen_toks': 100
                     }),
                     idx=0),
            Instance(request_type='generate_until',
                     doc={},
                     arguments=('prompt2', {
                         'until': None,
                         'max_gen_toks': 50
                     }),
                     idx=1)
        ]
        operator = Never()
        model = BacktrackingLM(generator=mock_generator,
                               operator=operator,
                               batch_size=1)

        results = model.generate_until(requests)

        assert results == [' a generated response.', ' a generated response.']
        expected_calls = [
            call(prompt='prompt1',
                 operator=operator,
                 max_new_tokens=100,
                 stop_sequences=['stop1']),
            call(prompt='prompt2',
                 operator=operator,
                 max_new_tokens=50,
                 stop_sequences=None)
        ]
        mock_generator.generate.assert_has_calls(expected_calls)
        assert mock_generator.generate.call_count == 2

    def test_generate_until_with_no_requests(self, mock_generator: MagicMock):
        model = BacktrackingLM(generator=mock_generator)
        results = model.generate_until([])
        assert not results
        mock_generator.generate.assert_not_called()

    def test_generate_until_handles_generator_exception(
            self, mock_generator: MagicMock):
        mock_generator.generate.side_effect = ValueError('Test Exception')
        requests = [
            Instance(request_type='generate_until',
                     doc={},
                     arguments=('prompt1', {
                         'until': None,
                         'max_gen_toks': 100
                     }),
                     idx=0)
        ]
        model = BacktrackingLM(generator=mock_generator)

        results = model.generate_until(requests)

        assert results == ['GENERATION_ERROR']
        mock_generator.generate.assert_called_once()

    def test_generate_until_handles_request_with_empty_args(
            self, mock_generator: MagicMock):
        requests = [
            Instance(request_type='generate_until', doc={}, arguments=(), idx=0)
        ]
        model = BacktrackingLM(generator=mock_generator)

        results = model.generate_until(requests)

        assert results == ['GENERATION_ERROR']
        mock_generator.generate.assert_not_called()

    def test_generate_until_handles_malformed_request_args(
            self, mock_generator: MagicMock):
        requests = [
            Instance(request_type='generate_until',
                     doc={},
                     arguments=('a prompt', 'not-a-dictionary'),
                     idx=0)
        ]
        operator = Never()
        model = BacktrackingLM(generator=mock_generator, operator=operator)

        model.generate_until(requests)

        mock_generator.generate.assert_called_once_with(prompt='a prompt',
                                                        operator=operator,
                                                        max_new_tokens=128,
                                                        stop_sequences=None)
