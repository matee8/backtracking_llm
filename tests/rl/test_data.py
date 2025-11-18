# pylint: disable=missing-module-docstring

import random
from pathlib import Path

import pytest

from backtracking_llm.rl.data import PromptProvider, TextFilePromptProvider

# pylint: disable=redefined-outer-name


@pytest.fixture
def prompt_file(tmp_path: Path) -> Path:

    file_path = tmp_path / 'prompts.txt'
    content = 'First prompt.\n\nSecond prompt.\n  \nThird prompt.\n'
    file_path.write_text(content, encoding='utf-8')
    return file_path


def test_initialization_loads_prompts(prompt_file: Path):

    provider = TextFilePromptProvider(prompt_file)
    assert len(provider) == 3
    assert provider.prompts == [
        'First prompt.', 'Second prompt.', 'Third prompt.'
    ]
    assert isinstance(provider, PromptProvider)


def test_get_prompt_cycles_through_prompts(prompt_file: Path):
    provider = TextFilePromptProvider(prompt_file)
    assert next(provider) == 'First prompt.'
    assert next(provider) == 'Second prompt.'
    assert next(provider) == 'Third prompt.'
    assert next(provider) == 'First prompt.'


def test_shuffle_randomizes_prompt_order(prompt_file: Path):
    random.seed(42)

    provider_no_shuffle = TextFilePromptProvider(prompt_file, shuffle=False)
    provider_shuffle = TextFilePromptProvider(prompt_file, shuffle=True)

    assert len(provider_no_shuffle) == len(provider_shuffle)
    assert set(provider_no_shuffle.prompts) == set(provider_shuffle.prompts)

    assert provider_no_shuffle.prompts != provider_shuffle.prompts


def test_empty_file_raises_value_error(tmp_path: Path):
    empty_file = tmp_path / 'empty.txt'
    empty_file.write_text('')
    with pytest.raises(ValueError, match='No non-empty prompts found'):
        TextFilePromptProvider(empty_file)


def test_file_with_only_whitespace_raises_value_error(tmp_path: Path):
    ws_file = tmp_path / 'whitespace.txt'
    ws_file.write_text('  \n\t\n ')
    with pytest.raises(ValueError, match='No non-empty prompts found'):
        TextFilePromptProvider(ws_file)


def test_file_not_found_raises_error():
    non_existent_path = Path('non_existent_file_12345.txt')
    with pytest.raises(FileNotFoundError):
        TextFilePromptProvider(non_existent_path)
