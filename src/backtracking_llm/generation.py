"""Defines the core generation logic with backtracking capabilities."""

import dataclasses
import logging
from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer, DynamicCache,
                          PreTrainedModel, PreTrainedTokenizer)
from backtracking_llm.decision import Operator

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class StepResult:
    """Result of a single generation step.

    Attributes:
        token_id: The generated token ID.
        backtrack_count: Number of tokens to backtrack if requested.
    """
    token_id: int
    backtrack_count: int
    probabilities: Tensor


class GenerationSession:
    """Stateful, non-iterator session for controlled text generation.

    This class provides explicit step-by-step control over the generation
    process, including fine-grained backtracking capabilities. It is designed
    for RL training and research scenarios where the generation state must be
    inspectable and mutable.

    Attributes:
        model: The model used for generation.
        tokenizer: The tokenizer for encoding/decoding.
        prompt: The initial prompt string.
        prompt_length: Length of the tokenized prompt (read-only).
        token_ids: The full sequence of generated token IDs (read-only
            snapshot).
        done: Whether generation has terminated (EOS, max tokens, or stop
            sequence).
        max_new_tokens: Maximum tokens to generate (read-only).
        stop_sequences: List of strings that terminate generation (read-only).
    """

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 prompt: str,
                 operator: Optional[Operator] = None,
                 backtrack_every_n: int = 1,
                 max_new_tokens: int = 100,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 stop_sequences: Optional[List[str]] = None) -> None:
        """Initialize a generation session.

        Args:
            model: Pre-trained causal LM.
            tokenizer: Corresponding tokenizer.
            prompt: Initial text to start generation.
            operator: Optional decision function to evaluate after generation.
            backtrack_every_n: The frequency (in tokens) at which the decision
                `operator` is called. A value of 1 means it's called for every
                new token. Must be a positive integer.
            max_new_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering parameter.
            stop_sequences: Strings that trigger termination.
        Raises:
            ValueError: If `backtrack_every_n` is not a positive integer.
        """
        if backtrack_every_n < 1:
            raise ValueError('`backtrack_every_n` must be a positive integer')

        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.operator = operator
        self.backtrack_every_n = backtrack_every_n
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.stop_sequences = stop_sequences or []

        device = model.device
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        self._input_ids: Tensor = inputs.input_ids
        self.prompt_length = self._input_ids.shape[1]

        self._past_key_values: Optional[DynamicCache] = None
        self._generated_token_count = 0
        self._done = False

        self._current_output: Optional[str] = None

        logger.info('GenerationSession initialized with prompt: "%s..."',
                    prompt[:50])

    @property
    def done(self) -> bool:
        """Whether generation has terminated."""
        return self._done

    @property
    def token_ids(self) -> Tensor:
        return self._input_ids

    @property
    def generated_token_count(self) -> int:
        return self._generated_token_count

    def step(self) -> StepResult:
        """Execute one generation step.

        Returns:
            StepResult with token information and backtracking request.

        Raises:
            RuntimeError: If called after generation is done.
        """
        if self._done:
            raise RuntimeError('Cannot step: generation is already done.')

        context_manager = (torch.inference_mode() if hasattr(
            torch, 'inference_mode') else torch.no_grad())

        with context_manager:
            outputs = self.model(
                input_ids=(self._input_ids[:,
                                           -1:] if self._generated_token_count
                           > 0 else self._input_ids),
                past_key_values=self._past_key_values,
                use_cache=True)
            next_token_logits = outputs.logits[:, -1, :]
            self._past_key_values = outputs.past_key_values

        if self.temperature > 0:
            next_token_logits = next_token_logits / self.temperature

        vocab_size = self.model.config.vocab_size
        top_k = min(self.top_k, vocab_size)

        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        chosen_index = torch.multinomial(top_k_probs, num_samples=1)
        next_token_id = top_k_indices[0, chosen_index].item()
        logger.debug('Sampled token ID: %d', next_token_id)

        backtrack_count = 0
        if (self.operator is not None and
            (self._generated_token_count + 1) % self.backtrack_every_n == 0):
            token_str = self.tokenizer.decode(next_token_id)
            backtrack_count = self.operator(top_k_logits.squeeze(),
                                            top_k_probs.squeeze(),
                                            int(chosen_index.item()), token_str)
            if backtrack_count > 0:
                logger.info('Operator requested backtrack of %d tokens.',
                            backtrack_count)

        self._input_ids = torch.cat([
            self._input_ids,
            torch.tensor([[next_token_id]], device=self.model.device)
        ],
                                    dim=-1)
        self._generated_token_count += 1

        self._done = self._should_stop()
        if self._done:
            logger.info('Generation finished. Total tokens generated: %d.',
                        self._generated_token_count)

        return StepResult(token_id=int(next_token_id),
                          probabilities=top_k_probs.squeeze(),
                          backtrack_count=backtrack_count)

    def backtrack(self, n_tokens: int) -> None:
        """Remove the last n tokens from the generation.

        Args:
            n_tokens: Number of tokens to remove. Clipped to prompt length.

        Raises:
            RuntimeError: If n_tokens is negative.
        """
        if n_tokens < 0:
            logger.warning('Cannot backtrack negative tokens.')
            return

        max_backtrack = self._input_ids.shape[1] - self.prompt_length
        if max_backtrack <= 0:
            logger.warning('No tokens to backtrack (max_backtrack=%d)',
                           max_backtrack)
            return

        n_tokens = min(n_tokens, max_backtrack)
        if n_tokens == 0:
            logger.warning('Backtrack of 0 tokens requested, ignoring')
            return

        logger.info('Backtracking %d tokens', n_tokens)

        self._input_ids = self._input_ids[:, :-n_tokens]

        if self._past_key_values is not None:
            current_length = self._past_key_values.get_seq_length()
            new_length = current_length - n_tokens
            if new_length > 0:
                self._past_key_values.crop(new_length)
            else:
                self._past_key_values = None

        self._generated_token_count -= n_tokens
        self._done = False

        if self.operator is not None:
            self.operator.backtrack(n_tokens)

        logger.debug('Backtrack complete. New sequence length: %d',
                     self._input_ids.shape[1])

    def get_decoded_text(self) -> str:
        """Get the currently generated text (excluding prompt)."""
        newly_generated_ids = self._input_ids[0, self.prompt_length:]
        return self.tokenizer.decode(newly_generated_ids,
                                     skip_special_tokens=True)

    def __repr__(self) -> str:
        """Returns a developer-friendly representation of the session state."""
        return (f'<GenerationSession generated={self._generated_token_count}/'
                f'{self.max_new_tokens}, done={self._done}, '
                f'prompt_len={self.prompt_length}>')

    def _should_stop(self) -> bool:
        """Check if generation should stop based on stop sequences or limits."""
        if self._generated_token_count >= self.max_new_tokens:
            logger.debug('Stopping: max_new_tokens reached')
            return True

        if self._input_ids[0, -1].item() == self.tokenizer.eos_token_id:
            logger.debug('Stopping: EOS token detected')
            return True

        if self.stop_sequences:
            newly_generated_ids = self._input_ids[0, self.prompt_length:]
            current_output = self.tokenizer.decode(newly_generated_ids,
                                                   skip_special_tokens=True)
            if (any(
                    current_output.endswith(seq)
                    for seq in self.stop_sequences)):
                logger.debug('Stopping: stop sequence detected')
                return True

        return False


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
        session = GenerationSession(model=self.model,
                                    tokenizer=self.tokenizer,
                                    prompt=prompt,
                                    operator=operator,
                                    max_new_tokens=max_new_tokens,
                                    temperature=temperature,
                                    top_k=top_k,
                                    stop_sequences=stop_sequences,
                                    backtrack_every_n=backtrack_every_n)

        while not session.done:
            result = session.step()
            if result.backtrack_count > 0:
                session.backtrack(result.backtrack_count)

        return session.get_decoded_text()

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
