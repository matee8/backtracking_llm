"""Defines the core generation logic with backtracking capabilities."""

import logging
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer, DynamicCache,
                          PreTrainedModel, PreTrainedTokenizer)
from backtracking_llm.decision import Operator

logger = logging.getLogger(__name__)


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
                 operator: Optional[Operator] = None,
                 max_new_tokens: int = 100,
                 backtrack_every_n: int = 1,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 stop_sequences: Optional[List[str]] = None) -> str:
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
            stop_sequences: A list of strings that, if generated, will cause the
                generation to stop.

        Returns:
            The generated text, including the initial prompt.

        Raises:
            ValueError: If `backtrack_every_n` is not a positive integer.
        """
        if backtrack_every_n < 1:
            raise ValueError('`backtrack_every_n` must be a positive integer')

        vocab_size = self.model.config.vocab_size
        top_k = min(top_k, vocab_size)

        device = self.model.device
        inputs = self.tokenizer(prompt, return_tensors='pt').to(device)
        input_ids: Tensor = inputs.input_ids
        model_inputs = input_ids
        prompt_length = input_ids.shape[1]

        past_key_values: Optional[DynamicCache] = None
        generated_token_count = 0

        logger.info("Starting text generation from prompt: '%s...'.",
                    prompt[:50])
        context_manager = (torch.inference_mode() if hasattr(
            torch, 'inference_mode') else torch.no_grad())

        with context_manager:
            while generated_token_count < max_new_tokens:
                logger.debug('Generation step %d.', generated_token_count + 1)
                outputs = self.model(input_ids=model_inputs,
                                     past_key_values=past_key_values,
                                     use_cache=True)
                next_token_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                if temperature > 0:
                    next_token_logits = next_token_logits / temperature

                top_k_logits, top_k_indices = torch.topk(
                    next_token_logits, top_k)
                top_k_probs = F.softmax(top_k_logits, dim=-1)

                chosen_index = torch.multinomial(top_k_probs, num_samples=1)
                next_token_id = top_k_indices[0, chosen_index].item()
                logger.debug('Sampled token ID: %d', next_token_id)

                if next_token_id == self.tokenizer.eos_token_id:
                    logger.info('EOS token detected. Stopping generation.')
                    break

                backtrack_count = 0
                if (operator is not None and
                    (generated_token_count + 1) % backtrack_every_n == 0):
                    token_str = self.tokenizer.decode(next_token_id)
                    backtrack_count = operator(top_k_logits.squeeze(),
                                               top_k_probs.squeeze(),
                                               int(chosen_index.item()),
                                               token_str)

                if backtrack_count > 0:
                    logger.info('Operator requested backtrack of %d tokens.',
                                backtrack_count)
                    max_backtrack = input_ids.shape[1] - prompt_length
                    backtrack_count = min(backtrack_count, max_backtrack)

                    if backtrack_count > 0:
                        input_ids, past_key_values = self._apply_backtracking(
                            input_ids, past_key_values, backtrack_count)
                        generated_token_count -= backtrack_count
                    else:
                        logger.debug('Backtrack clipped to 0. Discarding token '
                                     'and continuing.')
                    continue

                input_ids = torch.cat(
                    [input_ids,
                     torch.tensor([[next_token_id]], device=device)],
                    dim=-1)
                model_inputs = input_ids[:, -1:]

                generated_token_count += 1

                if stop_sequences:
                    current_output = self.tokenizer.decode(
                        input_ids[0, prompt_length:], skip_special_tokens=True)
                    if (any(
                            current_output.endswith(seq)
                            for seq in stop_sequences)):
                        logger.info('Stopping generation due to detected stop '
                                    'sequence.')
                        break

        logger.info('Generation finished. Total tokens generated: %d.',
                    generated_token_count)
        newly_generated_ids = input_ids[0, prompt_length:]
        return self.tokenizer.decode(newly_generated_ids,
                                     skip_special_tokens=True)

    def __repr__(self) -> str:
        """Returns a developer-friendly representation of the Generator."""
        try:
            model_name = self.model.config._name_or_path
        except AttributeError:
            model_name = self.model.__class__.__name__

        try:
            tokenizer_name = self.tokenizer.name_or_path
        except AttributeError:
            tokenizer_name = self.tokenizer.__class__.__name__
        return f"<Generator model='{model_name}', tokenizer='{tokenizer_name}'>"

    __call__ = generate

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **model_kwargs):
        """Instantiates a Generator from a pretrained model and tokenizer.

        Args:
            model_name_or_path: The name or path of the model on the Hugging
                Face hub.
            **model_kwargs: Additional keyword arguments to pass to the model's
                `from_pretrained` method.
        """
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                     **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        return cls(model, tokenizer)

    def _apply_backtracking(
            self, input_ids: Tensor, past_key_values: Optional[DynamicCache],
            backtrack_count: int) -> Tuple[Tensor, Optional[DynamicCache]]:
        """Truncates the input tensor and past key-value cache.

        Args:
            input_ids: The tensor of token IDs for the entire sequence.
            past_key_values: The model's KV cache.
            backtrack_count: The number of tokens to remove.

        Returns:
            A tuple containing the truncated `input_ids` and `past_key_values`.
        """
        truncated_ids = input_ids[:, :-backtrack_count]

        if past_key_values is None or backtrack_count == 0:
            return truncated_ids, past_key_values

        current_length = past_key_values.get_seq_length()
        new_length = current_length - backtrack_count
        past_key_values.crop(new_length)

        return truncated_ids, past_key_values
