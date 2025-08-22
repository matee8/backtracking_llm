"""Defines the core generation logic with backtracking capabilities."""

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer, DynamicCache


class Generator:
    """Orchestrates token-by-token text generation with a backtracking
    mechanism.

    This class wraps a `transformers` model and tokenizer, decoupling the
    generation logic from model loading and configuration. Its primary role is
    to execute a custom generation loop that can undo previous generation steps
    based on the logic provided by a given `Operator`.

    Attributes:
        model: The `PreTrainedModel` used for generating token logits. Note that
            it is the user's responsibility to ensure the model is on the
            correct device.
        tokenizer: The `PreTrainedTokenizer` for the model, for encoding prompts
            and decoding generated sequences.
    """

    def __init__(self, model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer) -> None:
        """Initializes the Generator.

        Args:
            model: A pre-loaded Hugging Face model to be used for generation.
            tokenizer: The corresponding tokenizer for the model.
        """
        self.model = model
        self.tokenizer = tokenizer

    def generate(self,
                 prompt: str,
                 max_new_tokens: int = 100,
                 backtrack_every_n: int = 1,
                 temperature: float = 1.0,
                 top_k: int = 50) -> str:
        """Generates text from a prompt using the backtracking strategy.

        Args:
            prompt: The initial text to start generation from.
            operator: The decision function to be called to determine if
                backtracking should occur.
            max_new_tokens: The maximum number of new tokens to generate.
            backtrack_every_n: The frequency (in tokens) at which the decision
                `operator` is called. A value of 1 means it's called for every
                new token. Must be a positive integer.
            temperature: The value used to modulate the next token
                probabilities.
            top_k: The number of highest probability vocabulary tokens to keep
                for top-k-filtering.

        Returns:
            The generated text, including the initial prompt.

        Raises:
            ValueError: If `backtrack_every_n` is not a positive integer.
        """
        if backtrack_every_n < 1:
            raise ValueError('`backtrack_every_n` must be a positive integer')

        device = self.model.device
        inputs = self.tokenizer(prompt, return_tensors='pt').to(device)
        input_ids: Tensor = inputs.input_ids
        model_inputs = input_ids

        past_key_values: Optional[DynamicCache] = None

        context_manager = (torch.inference_mode() if hasattr(
            torch, 'inference_mode') else torch.no_grad())

        with context_manager:
            for _ in range(max_new_tokens):
                outputs = self.model(input_ids=model_inputs,
                                     past_key_values=past_key_values,
                                     use_cache=True)
                next_token_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                _, top_k_probs, top_k_indices = (
                    self._calculate_top_k_distribution(next_token_logits,
                                                       temperature, top_k))

                chosen_index = torch.multinomial(top_k_probs, num_samples=1)
                next_token_id = top_k_indices[0, chosen_index].item()

                if next_token_id == self.tokenizer.eos_token_id:
                    break

                input_ids = torch.cat(
                    [input_ids,
                     torch.tensor([[next_token_id]], device=device)],
                    dim=-1)
                model_inputs = input_ids[:, -1:]

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def _calculate_top_k_distribution(
            self, logits: Tensor, temperature: float,
            top_k: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Filters the logits using temperature and top-k, returning a new
        distribution.

        Args:
            logits: The raw, full-vocabulary logits from the model.
            temperature: The value for modulating token probabilities.
            top_k: The number of highest probability tokens to keep.

        Returns:
            A tuple containing:
            - top_k_logits: A tensor of logits for only the top-k candidates.
            - top_k_probs: The softmax probabilities of the top-k logits.
            - top_k_indices: The original vocabulary indices of the top-k
              candidates.
        """
        if temperature > 0:
            logits = logits / temperature

        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        return top_k_logits, top_k_probs, top_k_indices
