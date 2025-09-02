"""Provides a high-level, stateless pipeline for conversational chat."""

import typing
from typing import Dict, List, Tuple

from backtracking_llm.generation import Generator

ConversationHistory = List[Dict[str, str]]


class ChatPipeline:
    """A stateless pipeline for managing conversational interactions with a
    model.

    This pipeline simplifies the process of building a multi-turn conversation.
    It uses the tokenizer's chat template to correctly format the conversation
    history for the model, ensuring optimal performance.

    The pipeline is stateless, meaning the user is responsible for storing and
    passing the conversation history into each turn.
    """

    def __init__(self, generator: Generator) -> None:
        """Initializes the ChatPipeline.

        Args:
            generator: A pre-configured Generator instance that will be used to
                generate responses.
        """
        self.generator = generator

    def run_turn(self, question: str, history: ConversationHistory,
                 **generation_kwargs) -> Tuple[str, ConversationHistory]:
        """Runs a single turn of conversation.

        This method is stateless. It takes the current history, adds the new
        user question, formats them into a single prompt using the model's
        chat template, generates an answer, and returns the answer along with
        the new, updated history.

        Args:
            question: The user's new question for this turn.
            history: A list of dictionaries representing the conversation so
                far, following the Hugging Face format (e.g.,
                `[{'role': 'user', ...}]`).
            **generation_kwargs: Additional keyword arguments to be passed
                directly to the `generator.generate()` method.

        Returns:
            A tuple containing:
            - The newly generated answer string.
            - The updated conversation history including the new turn.

        Raises:
            ValueError: If the tokenizer does not have a chat template
                configured.
        """
        if not self.generator.tokenizer.chat_template:
            raise ValueError(
                'The tokenizer for this model does not have a chat template. '
                'This pipeline requires a model that supports chat templating.')

        current_turn_history = history + [{'role': 'user', 'content': question}]

        prompt = self.generator.tokenizer.apply_chat_template(
            current_turn_history, add_generation_prompt=True, tokenize=False)
        prompt = typing.cast(str, prompt)

        answer = self.generator.generate(prompt, **generation_kwargs)

        new_history = history + [{
            'role': 'user',
            'content': question
        }, {
            'role': 'assistant',
            'content': answer
        }]

        return answer, new_history
