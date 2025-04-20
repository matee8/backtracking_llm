import dataclasses
import enum
import logging
import typing

import torch
import torch.nn.functional as F
import transformers
from transformers import modeling_outputs

from backtracking_llm.models import decision


class GenerationEventType(enum.Enum):
    TOKEN = 0
    END = 1
    ERROR = 2


@dataclasses.dataclass
class GenerationEvent:
    type: GenerationEventType
    data: typing.Any


class InferenceEngine(typing.Protocol):
    tokenizer: transformers.PreTrainedTokenizer

    def generate(
        self, prompt: str | torch.Tensor,
        token_callback: typing.Callable[[GenerationEvent], None] | None
    ) -> torch.Tensor | None:
        ...


class ModelInitializationError(RuntimeError):
    pass


class GenerationError(RuntimeError):
    pass


@dataclasses.dataclass
class BacktrackingInferenceConfig:
    max_answer_length: int = 64
    top_k: int = 50
    temperature: float = 1.0
    backtrack_every_n: int = 5
    backtrack_strategy: decision.BacktrackStrategy = (
        decision.ProbabilityThresholdDecision())
    device: str | None = None

    def __post_init__(self):
        if self.max_answer_length <= 0:
            raise ValueError("max_answer_length must be positive")

        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")

        if self.temperature < 0.0:
            raise ValueError("temperature cannot be negative")

        if self.backtrack_every_n < 1:
            raise ValueError("backtrack_every_n must be at least 1")


class BacktrackingInferenceEngine:

    def __init__(
        self,
        model_name: str,
        logger: logging.Logger,
        config: BacktrackingInferenceConfig = BacktrackingInferenceConfig()
    ) -> None:
        self.logger = logger
        self.config = config

        self.device = self._setup_device(config.device)
        self.model, self.tokenizer = self._load_model_and_tokenizer(model_name)

        self.logger.info("Moving model to device: %s", self.device)
        try:
            self.model.to(self.device)  # type: ignore[reportArgumentType]
            self.model.eval()
        except Exception as e:
            msg = f"Failed to move model to device {self.device}"
            self.logger.error("%s: %s", msg, e, exc_info=True)
            raise ModelInitializationError(msg) from e

    def generate(
        self, prompt: str | torch.Tensor,
        token_callback: typing.Callable[[GenerationEvent], None] | None
    ) -> torch.Tensor | None:
        input_ids: torch.Tensor | None = None
        generated: torch.Tensor | None = None

        def _send_event(event_type: GenerationEventType,
                        data: typing.Any) -> None:
            if token_callback:
                try:
                    token_callback(GenerationEvent(type=event_type, data=data))
                except Exception as e:
                    self.logger.error(
                        "Error executing generation callback: %s",
                        e,
                        exc_info=True)

        try:
            input_ids = self._prepare_input_ids(prompt).to(self.device)

            generated = input_ids
            past: transformers.DynamicCache | None = None
            current = input_ids

            for step in range(self.config.max_answer_length):
                logits, indices, past = self._predict_next_token(current, past)

                seq_logits = logits[0]
                seq_indices = indices[0]

                probs = self._calculate_probabilities(seq_logits)

                rel_idx = self._sample_next_token(probs)

                token_id = seq_indices[rel_idx].unsqueeze(0)

                if (step + 1) % self.config.backtrack_every_n == 0:
                    should, num = self._handle_backtrack(generated_count=step,
                                                         logits=logits,
                                                         probabilities=probs,
                                                         rel_idx=rel_idx)
                    if should:
                        self.logger.debug(
                            "Backtracking triggered after %d "
                            "tokens. Removing %d tokens.", step, num)

                        if num > 1:
                            generated = generated[:, :-(num - 1)]

                        past = self._trim_past(past, num)

                        current = generated[:, -1:]
                        step -= num

                        continue

                if (self.tokenizer.eos_token_id is not None
                        and token_id.item() == self.tokenizer.eos_token_id):
                    self.logger.debug("EOS at step %d; stopping.", step)
                    _send_event(GenerationEventType.END, None)
                    break

                try:
                    decoded = self.tokenizer.decode(token_id.item(),
                                                    skip_special_tokens=True)
                    _send_event(GenerationEventType.TOKEN, decoded)
                except Exception as e:
                    self.logger.warning("Failed to decode token ID %d: %s",
                                        token_id.item(), e)
                    _send_event(GenerationEventType.ERROR,
                                "Failed to decode token.")

                generated = torch.cat([generated, token_id.view(1, 1)], dim=-1)
                current = token_id.view(1, 1)

            if generated is None or generated.shape[1] <= input_ids.shape[1]:
                self.logger.warning("Generation stopped early or failed. "
                                    "No new tokens were generated.")
                return None

            _send_event(GenerationEventType.END, None)
            return generated
        except KeyboardInterrupt:
            msg = "Generation interrupted by user."

            self.logger.warning(msg)
            _send_event(GenerationEventType.ERROR, msg)

            if (generated is not None and input_ids is not None
                    and generated.shape[1] > input_ids.shape[1]):
                self.logger.info("Returning partially generated sequence.")
                return generated

            return None
        except GenerationError as e:
            msg = "Generation failed due to GenerationError"

            self.logger.error("%s: %s", msg, e, exc_info=True)

            _send_event(GenerationEventType.ERROR, msg)

            if (generated is not None and input_ids is not None
                    and generated.shape[1] > input_ids.shape[1]):
                self.logger.info("Returning partially generated sequence.")
                return generated

            raise
        except Exception as e:
            self.logger.error(
                "An unexpected error occurred during the "
                "generation loop: %s",
                e,
                exc_info=True)

            _send_event(GenerationEventType.ERROR, "Unexpected error.")

            if (generated is not None and input_ids is not None
                    and generated.shape[1] > input_ids.shape[1]):
                self.logger.info("Returning partially generated sequence.")
                return generated

            raise

    def _setup_device(self, device_str: str | None) -> torch.device:
        if (device_str and "cuda" in device_str
                and not torch.cuda.is_available()):
            msg = f"Specified device '{device_str}' but CUDA is not available."
            self.logger.error(msg)
            raise ModelInitializationError(msg)

        if device_str:
            dev = torch.device(device_str)
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")

        self.logger.info("Using device '%s'", dev)
        return dev

    def _load_model_and_tokenizer(
        self, model_name: str
    ) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        try:
            self.logger.info("Loading '%s'...", model_name)
            tok = transformers.AutoTokenizer.from_pretrained(model_name)
            mdl = transformers.AutoModelForCausalLM.from_pretrained(model_name)

            if tok.pad_token is None and tok.eos_token is not None:
                self.logger.info("Setting pad_token to eos_token.")
                tok.pad_token = tok.eos_token
                mdl.config.pad_token_id = mdl.config.eos_token_id
            elif tok.pad_token_id is None:
                self.logger.warning(
                    "No pad or eos token for '%s'; padding may "
                    "fail.", model_name)

            return mdl, tok
        except Exception as e:
            self.logger.error(
                "Failed to load model or tokenizer '%s' due to: "
                "%s",
                model_name,
                e,
                exc_info=True)
            raise ModelInitializationError("Failed to load model or tokenizer "
                                           f"{model_name}") from e

    def _prepare_input_ids(self, prompt: str | torch.Tensor) -> torch.Tensor:
        try:
            if isinstance(prompt, str):
                self.logger.debug("Tokenizing prompt of length %d",
                                  len(prompt))
                ids: torch.Tensor = (self.tokenizer(
                    prompt, return_tensors="pt").input_ids)
            else:
                self.logger.debug("Using raw tensor prompt.")
                ids = prompt

            if ids.ndim == 1:
                ids = ids.unsqueeze(0)
            elif ids.ndim != 2:
                raise GenerationError("Input tensor must be 1D or 2D, "
                                      f"got {ids.ndim}D")

            return ids
        except Exception as e:
            msg = "Failed to prepare input IDs"
            self.logger.error("%s: %s", msg, e, exc_info=True)
            raise GenerationError(msg) from e

    def _predict_next_token(
        self, input_ids: torch.Tensor,
        past_key_values: transformers.DynamicCache | None
    ) -> tuple[torch.Tensor, torch.Tensor, transformers.DynamicCache]:
        try:
            if hasattr(torch, "inference_mode"):
                context_manager = torch.inference_mode()
            else:
                context_manager = torch.no_grad()

            with context_manager:
                out: modeling_outputs.CausalLMOutputWithCrossAttentions = (
                    self.model(input_ids=input_ids,
                               past_key_values=past_key_values,
                               use_cache=True))

            logits = out.logits[:, -1, :]  # type: ignore

            if out.past_key_values is None:
                raise GenerationError("Model did not return updated past key "
                                      "value cache.")
            else:
                cache = transformers.DynamicCache(out.past_key_values)

            top_logits, top_indices = torch.topk(logits, self.config.top_k)

            return top_logits, top_indices, cache
        except Exception as e:
            msg = ("Error during next token prediction "
                   f"(input shape: {input_ids.shape})")
            self.logger.error("%s: %s", msg, e, exc_info=True)
            raise GenerationError(msg) from e

    def _calculate_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        try:
            if logits.numel() == 0:
                raise GenerationError("Cannot sample from empty logits.")

            if self.config.temperature == 0.0:
                probs = F.softmax(logits, dim=-1)
            else:
                probs = F.softmax(logits / self.config.temperature, dim=-1)

            return probs
        except Exception as e:
            msg = "Error during calculating probabilities"
            self.logger.error("%s: %s", msg, e, exc_info=True)
            raise GenerationError(msg) from e

    def _sample_next_token(self, probabilities: torch.Tensor) -> torch.Tensor:
        if probabilities.dim() != 1:
            raise GenerationError("Probabilities must be a 1D tensor for "
                                  f"sampling, got shape {probabilities.shape}")

        if probabilities.numel() == 0:
            raise GenerationError("Cannot sample from empty probabilities")

        if self.config.temperature == 0.0:
            rel = torch.argmax(probabilities)
        else:
            try:
                rel = (torch.multinomial(probabilities,
                                         num_samples=1).squeeze())
            except RuntimeError as e:
                self.logger.error(
                    "Multinomial sampling failed: %s. Falling "
                    "back to greedy.",
                    e,
                    exc_info=True)
                rel = torch.argmax(probabilities)

        return rel

    def _handle_backtrack(self, generated_count: int, logits: torch.Tensor,
                          probabilities: torch.Tensor,
                          rel_idx: torch.Tensor) -> tuple[bool, int]:
        if generated_count <= 0:
            self.logger.debug("Cannot backtrack if nothing is generated.")
            return False, 0

        try:
            should, num = self.config.backtrack_strategy.should_backtrack(
                logits, probabilities, rel_idx)
        except Exception as e:
            self.logger.error(
                "Error calling backtracking decision function at"
                "token %d: %s",
                generated_count,
                e,
                exc_info=True)
            return False, 0

        if num < 0:
            self.logger.debug(
                "Backtracking triggered but num_to_remove=%d. "
                "No changes made", num)
            return False, 0

        return should, min(num, generated_count)

    def _trim_past(self, past: transformers.DynamicCache | None,
                   num: int) -> transformers.DynamicCache | None:
        if not past or num <= 0:
            self.logger.warning(
                "Trimming requested with None or num_to_remove="
                "%d. No changes made.", num)
            return past

        try:
            new_cache: list[tuple[torch.Tensor, ...]] = []

            for layer in past:
                trimmed: list[torch.Tensor] = []

                for tensor in layer:
                    if tensor.dim() < 2:
                        raise ValueError("Cannot trim 1D tensor.")

                    length = tensor.shape[-2] - num
                    trimmed.append(tensor[..., :length, :])

                new_cache.append(tuple(trimmed))

            return transformers.DynamicCache(new_cache)
        except Exception as e:
            self.logger.error("Unexpected error during cache trimming: %s.",
                              "Resetting cache.",
                              e,
                              exc_info=True)
            return None
