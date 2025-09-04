# pylint: disable=missing-module-docstring

from unittest.mock import MagicMock, ANY

import pytest

from backtracking_llm.chat import ChatPipeline
from backtracking_llm.decision import Operator
from backtracking_llm.generation import Generator

# pylint: disable=redefined-outer-name


@pytest.fixture
def mock_generator() -> MagicMock:
    """Provides a mock Generator instance with necessary attributes."""
    generator = MagicMock(spec=Generator)

    tokenizer_mock = MagicMock()

    generator.tokenizer = tokenizer_mock

    generator.tokenizer.chat_template = 'TEMPLATE'
    generator.tokenizer.eos_token = '<|endoftext|>'

    return generator


def test_pipeline_initialization(mock_generator: MagicMock):
    chat_pipeline = ChatPipeline(mock_generator)

    assert chat_pipeline.generator is mock_generator


def test_run_turn_happy_path(mock_generator: MagicMock):
    question = 'What is the capital of France?'
    initial_history = []

    formatted_prompt = 'User: What is the capital of France?\nAI:'
    generated_answer = 'Paris'
    mock_generator.tokenizer.apply_chat_template.return_value = formatted_prompt
    mock_generator.generate.return_value = generated_answer

    chat_pipeline = ChatPipeline(mock_generator)

    answer, new_history = chat_pipeline.run_turn(question, initial_history)

    expected_history_for_template = [{'role': 'user', 'content': question}]
    mock_generator.tokenizer.apply_chat_template.assert_called_once_with(
        expected_history_for_template,
        tokenize=False,
        add_generation_prompt=True)

    mock_generator.generate.assert_called_once_with(formatted_prompt)

    assert answer == generated_answer

    expected_new_history = [{
        'role': 'user',
        'content': question
    }, {
        'role': 'assistant',
        'content': generated_answer
    }]
    assert new_history == expected_new_history


def test_run_turn_with_existing_history(mock_generator: MagicMock):
    question = 'And its population?'
    initial_history = [{
        'role': 'user',
        'content': 'What is the capital of France?'
    }, {
        'role': 'assistant',
        'content': 'Paris'
    }]

    chat_pipeline = ChatPipeline(mock_generator)

    _, new_history = chat_pipeline.run_turn(question, initial_history)

    expected_history_for_template = initial_history + [{
        'role': 'user',
        'content': question
    }]
    mock_generator.tokenizer.apply_chat_template.assert_called_once_with(
        expected_history_for_template,
        tokenize=False,
        add_generation_prompt=True)

    assert len(new_history) == 4
    assert new_history[-1]['role'] == 'assistant'


def test_run_turn_raises_error_if_no_chat_template(mock_generator: MagicMock):
    mock_generator.tokenizer.chat_template = None
    chat_pipeline = ChatPipeline(mock_generator)

    with pytest.raises(ValueError, match='does not have a chat template'):
        chat_pipeline.run_turn('A question', [])


def test_run_turn_forwards_generation_kwargs(mock_generator: MagicMock):
    question = 'Test question'
    initial_history = []

    mock_operator = MagicMock(spec=Operator)

    test_kwargs = {
        'operator': mock_operator,
        'temperature': 0.123,
        'max_new_tokens': 99
    }

    mock_generator.tokenizer.apply_chat_template.return_value = (
        'formatted_prompt')
    mock_generator.generate.return_value = 'test answer'

    chat_pipeline = ChatPipeline(mock_generator)

    chat_pipeline.run_turn(question, initial_history, **test_kwargs)

    mock_generator.generate.assert_called_once_with(ANY, **test_kwargs)
