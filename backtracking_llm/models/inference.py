import functools
import typing
import logging

import torch
import torch.nn.functional as F
import transformers
from transformers import modeling_outputs


def load_model_and_tokenizer(
    model_name: str, logger: logging.Logger
) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            logger.info(
                "Tokenizer for '%s' has no pad token set. "
                "Setting to EOS token.", model_name)

            tokenizer.pad_token = tokenizer.eos_token

            if hasattr(model, "config"):
                model.config.pad_token_id = model.config.eos_token_id
            else:
                logger.warning("Could not set model.config.pad_token_id as "
                               "model has no 'config' attribute.")
        elif tokenizer.pad_token_id is None and tokenizer.eos_token is None:
            logger.warning(
                "Tokenizer for '%s' has neither pad_token nor "
                "eos_token. Padding may not work correctly.", model_name)

        logger.info("Successfully loaded model and tokenizer: %s", model_name)
        return model, tokenizer
    except (OSError, ValueError, RuntimeError) as e:
        logger.error("Failed to load model or tokenizer '%s' due to: %s",
                     model_name,
                     e,
                     exc_info=True)
        raise
    except Exception as e:
        logger.error(
            "An unexpected error occured while loading model '%s': %s",
            model_name,
            e,
            exc_info=True)
        raise


def run_inference_loop(model: transformers.PreTrainedModel,
                       tokenizer: transformers.PreTrainedTokenizer,
                       prompt: str | torch.Tensor,
                       max_answer_length: int,
                       top_k: int,
                       logger: logging.Logger,
                       temperature: float = 1.,
                       backtrack_every_n: int = 5,
                       backtracking_decision_function: functools.partial
                       | None = None,
                       device: str | None = None) -> torch.Tensor | None:
    try:
        selected_device = _setup_device(device, logger)
        model.to(selected_device)  # type: ignore
        model.eval()

        input_ids = _prepare_input_ids(prompt, tokenizer, logger)

        if input_ids is None:
            return None

        input_ids.to(selected_device)

        generated_ids = input_ids
        prompt_length = input_ids.shape[1]
        past_key_values = None
        current_input_ids = input_ids

        for step in range(max_answer_length):
            next_token_stats = _predict_next_token(
                model=model,
                input_ids=current_input_ids,
                top_k=top_k,
                past_key_values=past_key_values,
                logger=logger,
            )

            if next_token_stats is None:
                logger.error("Prediction failed at step %d. Stopping.", step)
                if generated_ids.numel() > 0:
                    return generated_ids
                else:
                    return None

            top_k_logits = next_token_stats[0]
            top_k_indices = next_token_stats[1]
            past_key_values = next_token_stats[2]

            top_k_logits_seq = top_k_logits[0]
            top_k_indices_seq = top_k_indices[0]

            sample_result = _sample_next_token(
                top_k_logits_seq=top_k_logits_seq,
                top_k_indices_seq=top_k_indices_seq,
                temperature=temperature,
                logger=logger)
            if sample_result is None:
                logger.error("Sampling failed at step %d. Stopping", step)
                return generated_ids

            chosen_token_id, chosen_token_relative_idx, probabilities = (
                sample_result)

            _log_generation_details(
                step=step,
                tokenizer=tokenizer,
                chosen_token_id=chosen_token_id,
                top_k_logits_seq=top_k_logits_seq,
                probabilities=probabilities,
                chosen_token_relative_idx=chosen_token_relative_idx,
                logger=logger)

            if (backtracking_decision_function is not None
                    and (step + 1) % backtrack_every_n == 0):
                backtrack_ids_result, num_tokens_removed = (
                    _handle_backtracking(
                        generated_ids=generated_ids,
                        prompt_length=prompt_length,
                        backtracking_decision_function=
                        backtracking_decision_function,
                        top_k_logits_seq=top_k_logits_seq,
                        probabilities=probabilities,
                        chosen_token_relative_idx=chosen_token_relative_idx,
                        logger=logger))

                if num_tokens_removed > 0:
                    logger.debug("Backtracking: %d token(s) removed.",
                                 num_tokens_removed)

                    past_key_values = _trim_past_key_values(
                        past_key_values, num_tokens_removed, logger)

                    if backtrack_ids_result is not None:
                        generated_ids = backtrack_ids_result

                    if generated_ids.shape[1] > 0:
                        current_input_ids = generated_ids[:, -1:]
                    else:
                        logger.error("Backtracking removed all tokens, "
                                     "including prompt. Stopping.")
                        return None

                    continue

            if (tokenizer.eos_token_id is not None
                    and chosen_token_id.item() == tokenizer.eos_token_id):
                logging.debug(
                    "EOS token detected at step %d. Stopping inference.",
                    step + 1)
                break

            generated_ids = torch.cat(
                [generated_ids, chosen_token_id.view(1, 1)], dim=-1)
            current_input_ids = chosen_token_id.view(1, 1)

        return generated_ids
    except KeyboardInterrupt:
        logging.warning("Inference interrupted by user.")
        if ("generated_ids" in locals()
                and generated_ids.numel() > input_ids.numel()):
            return generated_ids
        else:
            return None
    except (RuntimeError, ValueError) as e:
        logger.error("Error during inference loop: %s", e, exc_info=True)
        return None
    except Exception as e:
        logging.critical(
            "An unexpected error occured during the inference loop: %s",
            e,
            exc_info=True)
        if ("generated_ids" in locals()
                and generated_ids.shape[1] > prompt_length):
            logger.warning(
                "Attempting to return partially generated sequence after error."
            )
            return generated_ids.to("cpu")
        return None


def _setup_device(device_str: str | None,
                  logger: logging.Logger) -> torch.device:
    if device_str:
        if "cuda" in device_str and not torch.cuda.is_available():
            logger.error("Specified device '%s' but CUDA is not available.",
                         device_str)
            raise RuntimeError("CUDA not available for specified device"
                               f"{device_str}.")
        selected_device = torch.device(device_str)
        logger.info("Using specified device: %s.", selected_device)
    elif torch.cuda.is_available():
        selected_device: torch.device = torch.device("cuda")
        logger.info("Auto-detected CUDA, using GPU: %s.", selected_device)
    else:
        selected_device: torch.device = torch.device("cpu")
        logger.info("CUDA not available, using CPU.")

    return selected_device


def _prepare_input_ids(prompt: typing.Union[str, torch.Tensor],
                       tokenizer: transformers.PreTrainedTokenizer,
                       logger: logging.Logger) -> torch.Tensor | None:
    try:
        if isinstance(prompt, str):
            try:
                input_ids: torch.Tensor = tokenizer(
                    prompt, return_tensors="pt").input_ids
            except Exception as e:
                logger.error("Failed to tokenize prompt '%s': %s",
                             prompt,
                             e,
                             exc_info=True)
                raise
        else:
            input_ids = prompt

        if not isinstance(input_ids, torch.Tensor):
            logger.error("Tokenizer did not return a Tensor.")
            return None

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        elif input_ids.ndim != 2:
            logger.error("Input tensor must be 1D or 2D, got %dD.",
                         input_ids.ndim)
            return None

        return input_ids
    except Exception as e:
        logger.error("Failed to prepare input_ids: %s", e, exc_info=True)
        return None


def _predict_next_token(
    model: transformers.PreTrainedModel, input_ids: torch.Tensor, top_k: int,
    past_key_values: transformers.DynamicCache | None, logger: logging.Logger
) -> tuple[torch.Tensor, torch.Tensor, transformers.DynamicCache
           | None] | None:
    try:
        model.eval()

        if hasattr(torch, "inference_mode"):
            context_manager = torch.inference_mode()
        else:
            context_manager = torch.no_grad()

        with context_manager:
            outputs: modeling_outputs.CausalLMOutputWithCrossAttentions = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True)
            next_token_logits: torch.Tensor = outputs.logits[:, -1, :]
            updated_past_key_values = outputs.past_key_values

            if updated_past_key_values is not None:
                updated_cache = transformers.DynamicCache(
                    updated_past_key_values)
            else:
                updated_cache = None

            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)

            return top_k_logits, top_k_indices, updated_cache
    except (RuntimeError, ValueError, IndexError) as e:
        logger.error(
            "Error during next token prediction (input shape: %s): %s",
            input_ids.shape,
            e,
            exc_info=True)
        return None
    except Exception as e:
        logger.error(
            "An unexpected error occured during next token prediction: "
            "(input shape: %s): %s",
            input_ids.shape,
            e,
            exc_info=True)
        raise


def _sample_next_token(
    top_k_logits_seq: torch.Tensor, top_k_indices_seq: torch.Tensor,
    temperature: float, logger: logging.Logger
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if top_k_logits_seq.dim() != 1 or top_k_indices_seq.dim() != 1:
        logger.error("Logits and indices must be 1D tensors for sampling.")
        return None

    if top_k_logits_seq.shape != top_k_indices_seq.shape:
        logger.error("Logits and indices shapes must match for sampling.")
        return None

    if top_k_logits_seq.numel() == 0 or top_k_indices_seq.numel() == 0:
        logger.error("Cannot sample from empty logits/indices.")
        return None

    if temperature == 0.:
        probabilities = F.softmax(top_k_logits_seq, dim=-1)

        chosen_token_relative_idx = torch.argmax(top_k_logits_seq)
    else:
        probabilities = F.softmax(top_k_logits_seq / temperature, dim=-1)

        try:
            chosen_token_relative_idx = torch.multinomial(
                probabilities, num_samples=1).squeeze()
        except RuntimeError as e:
            logger.error(
                "Multinomial sampling failed: %s. Falling back to"
                " greedy.",
                e,
                exc_info=True)

            chosen_token_relative_idx = torch.argmax(probabilities)

    chosen_token_id = top_k_indices_seq[chosen_token_relative_idx].unsqueeze(0)

    return chosen_token_id, chosen_token_relative_idx, probabilities


def _log_generation_details(step: int,
                            tokenizer: transformers.PreTrainedTokenizer,
                            chosen_token_id: torch.Tensor,
                            top_k_logits_seq: torch.Tensor,
                            probabilities: torch.Tensor,
                            chosen_token_relative_idx: torch.Tensor,
                            logger: logging.Logger) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return

    logger.debug("Iteration %d.", step)

    try:
        stats = _calculate_statistics(top_k_logits_seq, probabilities)

        for key, value in stats.items():
            logging.debug("%s: %.4f.", key, value)

        logging.debug("Chosen token: %s, logit: %d, probability: %.4f.",
                      tokenizer.decode(chosen_token_id.item()),
                      top_k_logits_seq[chosen_token_relative_idx].item(),
                      probabilities[chosen_token_relative_idx])
    except Exception as e:
        logging.error("Failed to calculate or log statistics at step %d: %s",
                      step + 1,
                      e,
                      exc_info=True)


def _calculate_statistics(
    top_k_logits: torch.Tensor,
    top_k_probabilities: torch.Tensor,
) -> dict[str, float]:
    if top_k_logits.dim() != 1 or top_k_probabilities.dim() != 1:
        raise ValueError("Input tensors must be 1D. Got shapes: "
                         f"{top_k_logits.shape}, {top_k_probabilities.shape}.")

    if top_k_logits.numel() == 0 or top_k_probabilities.numel() == 0:
        raise ValueError("Cannot calculate statistics on empty logit or "
                         "probability tensors.")

    if top_k_logits.shape != top_k_probabilities.shape:
        raise ValueError(f"Logits shape {top_k_logits.shape} must match "
                         f"probabilites shape {top_k_probabilities.shape}.")

    top_logit_value = top_k_logits[0].item()

    if top_k_logits.numel() > 1:
        second_logit_value = top_k_logits[1].item()
    else:
        second_logit_value = float("inf")

    top_prob_value = top_k_probabilities[0].item()

    if top_k_probabilities.numel() > 1:
        second_prob_value = top_k_probabilities[1].item()
    else:
        second_prob_value = 0.

    stats = {}

    epsilon = 1e-10

    stats["highest_logit"] = top_logit_value

    stats["highest_prob"] = top_prob_value

    stats["logit_diff"] = top_logit_value - second_logit_value

    if second_logit_value != 0:
        stats["logit_ratio"] = top_logit_value / second_logit_value
    else:
        stats["logit_ratio"] = float("inf")

    stats["prob_diff"] = top_prob_value - second_prob_value

    if second_prob_value != 0:
        stats["prob_ratio"] = top_prob_value / second_prob_value
    else:
        stats["prob_ratio"] = float("inf")

    stats["prob_entropy"] = -torch.sum(
        top_k_probabilities * torch.log(top_k_probabilities + epsilon)).item()

    return stats


def _handle_backtracking(
        generated_ids: torch.Tensor, prompt_length: int,
        backtracking_decision_function: functools.partial,
        top_k_logits_seq: torch.Tensor, probabilities: torch.Tensor,
        chosen_token_relative_idx: torch.Tensor,
        logger: logging.Logger) -> tuple[torch.Tensor | None, int]:
    num_generated_tokens = generated_ids.shape[1] - prompt_length
    if num_generated_tokens > 0:
        logger.debug("Calling backtrack decision function at the %d. token.",
                     num_generated_tokens + 1)
        try:
            should_backtrack: bool
            num_to_remove: int
            should_backtrack, num_to_remove = backtracking_decision_function(
                top_k_logits_seq, probabilities, chosen_token_relative_idx)
        except Exception as e:
            logger.error("Error calling the decision function at the %d. ",
                         "token: %s",
                         num_generated_tokens + 1,
                         e,
                         exc_info=True)
            should_backtrack = False
            num_to_remove = 0

        if should_backtrack:
            actual_num_to_remove = min(num_to_remove, num_generated_tokens)

            logger.debug(
                "Backtracking triggered at the %d. token. Removing %d "
                "token(s).", num_generated_tokens + 1, actual_num_to_remove)

            if actual_num_to_remove <= 0:
                logger.debug(
                    "Backtracking triggered but actual_num_to_remove "
                    "is %d. No changes made.", actual_num_to_remove)
                return None, 0

            if actual_num_to_remove > 1:
                generated_ids = generated_ids[:, :-(actual_num_to_remove - 1)]

            return generated_ids, actual_num_to_remove
        else:
            return None, 0
    else:
        return None, 0


def _trim_past_key_values(
        past_key_values: transformers.DynamicCache | None, num_to_remove: int,
        logger: logging.Logger) -> transformers.DynamicCache | None:
    if past_key_values is None:
        logger.warning("Attempted to trim a None past_key_values cache.")
        return None

    if num_to_remove <= 0:
        logger.debug(
            "Trimming requested with num_to_remove=%d. No changes"
            " needed.", num_to_remove)
        return past_key_values

    try:
        new_past: list[tuple[torch.Tensor, ...]] = []
        for layer_past in past_key_values:
            new_layer_past: list[torch.Tensor] = []
            for state_tensor in layer_past:
                if state_tensor.dim() < 2:
                    logger.error(
                        "Cannot trim state tensor with dimension < 2."
                        " Shape: %s", state_tensor.shape)
                    return None

                current_seq_len = state_tensor.shape[-2]

                if num_to_remove > current_seq_len:
                    logger.warning(
                        "Attempting to remove %d tokens, but cache "
                        "sequence length is only %d. Removing all.",
                        num_to_remove, current_seq_len)
                    return None

                new_seq_len = current_seq_len - num_to_remove

                trimmed_state = state_tensor[..., :new_seq_len, :]
                new_layer_past.append(trimmed_state)
            new_past.append(tuple(new_layer_past))

        return transformers.DynamicCache(new_past)
    except (IndexError, ValueError, TypeError) as e:
        logger.error(
            "Failed to trim past_key_values. Cache structure might be "
            "unexpected. Resetting cache. Reason: %s.",
            e,
            exc_info=True)
        return None
    except Exception as e:
        logger.critical("Unexpected error during past_key_values trimming: %s",
                        e,
                        exc_info=True)
        return None
