"""Fixtures for benchmarking tests."""

from unittest.mock import MagicMock, create_autospec

import pytest
from transformers import (PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizerFast)

from backtracking_llm.generation import Generator

# pylint: disable=missing-function-docstring


@pytest.fixture
def mock_generator() -> MagicMock:
    generator = create_autospec(Generator, instance=True)
    generator.generate.return_value = ' a generated response.'

    mock_model = create_autospec(PreTrainedModel, instance=True)
    mock_config = create_autospec(PretrainedConfig, instance=True)

    mock_config.model_parallel = False
    mock_config.architectures = ['MockForCausalLM']
    mock_model.config = mock_config
    mock_model.name_or_path = 'mock-model-name'

    generator.model = mock_model
    mock_tokenizer = create_autospec(PreTrainedTokenizerFast, instance=True)
    mock_tokenizer.pad_token = None
    mock_tokenizer.unk_token = None
    mock_tokenizer.eos_token = '</s>'
    mock_tokenizer.eos_token_id = 9
    generator.tokenizer = mock_tokenizer
    return generator
