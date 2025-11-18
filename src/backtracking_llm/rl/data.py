"""Provides data feeder abstractions for RL training episodes."""

import random
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class PromptProvider(Protocol):
    """A protocol for classes that provide prompts for training episodes."""

    def __iter__(self) -> Iterator[str]:
        ...

    def __next__(self) -> str:
        """Returns the next prompt string for the start of an episode.
        Implementations should handle their own internal state, such as
        cycling through a dataset.

        Returns:
            A prompt string.
        """
        ...


class TextFilePromptProvider:
    """A prompt provider that reads prompts from a plain text file.

    This provider loads all non-empty lines from a specified text file into
    memory. It can cycle through these prompts sequentially or in a shuffled
    order.

    Attributes:
        prompts: The list of loaded prompts.
    """

    def __init__(self, file_path: Path, shuffle: bool = False) -> None:
        """Initializes the TextFilePromptProvider.

        Args:
            file_path: The path to the text file containing prompts, one per
                line.
            shuffle: If True, the order of prompts will be randomized upon
                loading.

        Raises:
            FileNotFoundError: If the specified `file_path` does not exist.
            ValueError: If the file is successfully read but contains no
                non-empty lines.
        """
        if not file_path.is_file():
            raise FileNotFoundError(f'Prompt file not found at: {file_path}')

        with open(file_path, 'r', encoding='utf-8') as f:
            self.prompts = [line.strip() for line in f if line.strip()]

        if not self.prompts:
            raise ValueError(f'No non-empty prompts found in file: {file_path}')

        if shuffle:
            random.shuffle(self.prompts)

        self._current_index = 0

    def __iter__(self) -> 'TextFilePromptProvider':
        """Returns self as the iterator."""
        return self

    def __next__(self) -> str:
        """Returns the next prompt, cycling through the list if necessary.

        Returns:
            The next prompt string from the loaded list.
        """
        prompt = self.prompts[self._current_index]
        self._current_index = (self._current_index + 1) % len(self.prompts)
        return prompt

    def __len__(self) -> int:
        """Returns the total number of loaded prompts."""
        return len(self.prompts)
