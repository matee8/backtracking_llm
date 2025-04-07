import functools
import typing
import logging

import torch
import torch.nn.functional as F
import transformers
from transformers import modeling_outputs

PastKeyValuesType = typing.Optional[typing.Tuple[typing.Tuple[torch.Tensor]]]


def load_model_and_tokenizer(
    model_name: str, logger: logging.Logger
) -> typing.Tuple[transformers.PreTrainedModel,
                  transformers.PreTrainedTokenizer]:
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


def run_inference_loop(
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        prompt: typing.Union[str, torch.Tensor],
        max_answer_length: int,
        top_k: int,
        logger: logging.Logger,
        temperature: float = 1.,
        backtrack_every_n: int = 5,
        backtracking_decision_function: typing.Optional[
            functools.partial] = None,
        device: typing.Optional[str] = None) -> typing.Optional[torch.Tensor]:
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("Auto-deteced CUDA, using GPU.")
        else:
            device = "cpu"
            logger.info("CUDA not available, using CPU.")
    try:
        model.to(device)

        if isinstance(prompt, str):
            try:
                input_ids: torch.Tensor = tokenizer(
                    prompt, return_tensors="pt").input_ids.to(device)
            except Exception as e:
                logger.error("Failed to tokenize prompt '%s': %s",
                             prompt,
                             e,
                             exc_info=True)
                raise
        else:
            input_ids: torch.Tensor = prompt.to(device)

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        elif input_ids.ndim != 2:
            raise ValueError("Input tensor must be 1D or 2D, got "
                             f"{input_ids.ndim}D.")

        generated_ids: torch.Tensor = input_ids
        prompt_length: int = input_ids.shape[1]

        past_key_values: PastKeyValuesType = None
        current_input_ids: torch.Tensor = input_ids

        for step in range(max_answer_length):
            try:
                next_token_stats: typing.Optional[typing.Dict[
                    str,
                    typing.Union[torch.Tensor,
                                 PastKeyValuesType]]] = _predict_next_token(
                                     model=model,
                                     input_ids=current_input_ids,
                                     top_k=top_k,
                                     past_key_values=past_key_values,
                                     logger=logger)

                if next_token_stats is None:
                    return None

                top_k_logits: torch.Tensor = next_token_stats["logits"]
                top_k_indices: torch.Tensor = next_token_stats["indices"]
                past_key_values = next_token_stats["past_key_values"]

                top_k_logits_seq: torch.Tensor = top_k_logits[0]
                top_k_indices_seq: torch.Tensor = top_k_indices[0]
            except Exception as e:
                logger.error("Prediction failed at step %d: %s",
                             step,
                             e,
                             exc_info=True)
                if generated_ids.numel() > prompt_length:
                    return generated_ids
                else:
                    return None

            if temperature == 0.:
                probabilities: torch.Tensor = F.softmax(top_k_logits_seq,
                                                        dim=-1)

                chosen_token_relative_idx: torch.Tensor = torch.argmax(
                    top_k_logits_seq)
            else:
                probabilities: torch.Tensor = F.softmax(top_k_logits_seq /
                                                        temperature,
                                                        dim=-1)

                try:
                    chosen_token_relative_idx: torch.Tensor = torch.multinomial(
                        probabilities, num_samples=1).squeeze()
                except RuntimeError as e:
                    logger.error(
                        "Multinomial sampling failed at step %d: %s."
                        " Probabilities: %s", step, e, probabilities)

                    chosen_token_relative_idx: torch.Tensor = torch.argmax(
                        probabilities)

            chosen_token_id = top_k_indices_seq[
                chosen_token_relative_idx].unsqueeze(0)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Iteration %d.", step)

                stats: typing.Dict[str, float] = _calculate_statistics(
                    top_k_logits_seq, probabilities)

                for key, value in stats.items():
                    logging.debug("%s: %.4f.", key, value)

                logging.debug(
                    "Chosen token: %s, logit: %d, probability: %.4f.",
                    tokenizer.decode(chosen_token_id.item()),
                    top_k_logits_seq[chosen_token_relative_idx].item(),
                    probabilities[chosen_token_relative_idx])

            if backtracking_decision_function is not None:
                num_generated_tokens: int = generated_ids.shape[
                    1] - prompt_length

                if num_generated_tokens > 0 and (step +
                                                 1) % backtrack_every_n == 0:
                    logger.debug(
                        "Calling backtrack decision function at step %d.",
                        step + 1)

                    try:
                        backtrack: typing.Tuple[bool, int]
                        backtrack = backtracking_decision_function(
                            top_k_logits_seq, probabilities,
                            chosen_token_relative_idx)

                        should_backtrack: bool
                        num_to_remove: int

                        should_backtrack, num_to_remove = backtrack
                    except Exception as e:
                        logger.error(
                            "Error calling the decision functiona at step"
                            " %d: %s",
                            step + 1,
                            e,
                            exc_info=True)
                        should_backtrack = False
                        num_to_remove = 0

                    if should_backtrack:
                        actual_num_to_remove: int = min(
                            num_to_remove, num_generated_tokens)

                        logger.debug(
                            "Backtracking triggered at step %d. "
                            "Removing %d token(s).", step + 1,
                            actual_num_to_remove)

                        if actual_num_to_remove > 1:
                            generated_ids = generated_ids[:, :-(
                                actual_num_to_remove - 1)]

                        past_key_values = None

                        current_input_ids = generated_ids

                        continue

            if tokenizer.eos_token_id is not None and chosen_token_id.item(
            ) == tokenizer.eos_token_id:
                logging.debug(
                    "EOS token detected at step %d. Stopping inference.",
                    step + 1)
                break

            generated_ids: torch.Tensor = torch.cat(
                [generated_ids, chosen_token_id.view(1, 1)], dim=-1)
            current_input_ids = chosen_token_id.view(1, 1)

        logging.debug(
            "Generated text:\n%s",
            tokenizer.decode(generated_ids[0], skip_special_tokens=True))

        return generated_ids
    except KeyboardInterrupt:
        logging.warning("Inference interrupted by user.")
        if "generated_ids" in locals() and generated_ids.numel(
        ) > input_ids.numel():
            return generated_ids
        else:
            return None
    except (RuntimeError, ValueError) as e:
        logger.error("Error during inference loop: %s", e, exc_info=True)
        return None
    except Exception as e:
        logging.error(
            "An unexpected error occured during the inference loop: %s",
            e,
            exc_info=True)
        raise


def _predict_next_token(
    model: transformers.PreTrainedModel, input_ids: torch.Tensor, top_k: int,
    past_key_values: PastKeyValuesType, logger: logging.Logger
) -> typing.Optional[typing.Dict[str, typing.Union[torch.Tensor,
                                                   PastKeyValuesType]]]:
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
            updated_past_key_values: PastKeyValuesType = outputs.past_key_values

            top_k_logits: torch.Tensor
            top_k_indices: torch.Tensor
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)

            return {
                "logits": top_k_logits,
                "indices": top_k_indices,
                "past_key_values": updated_past_key_values
            }
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


def _calculate_statistics(
    top_k_logits: torch.Tensor,
    top_k_probabilities: torch.Tensor,
) -> typing.Dict[str, float]:
    if top_k_logits.dim() != 1 or top_k_probabilities.dim() != 1:
        raise ValueError("Input tensors must be 1D. Got shapes: "
                         f"{top_k_logits.shape}, {top_k_probabilities.shape}.")

    if top_k_logits.numel() == 0 or top_k_probabilities.numel() == 0:
        raise ValueError(
            "Cannot calculate statistics on empty logit or probability tensors."
        )

    if top_k_logits.shape != top_k_probabilities.shape:
        raise ValueError(
            f"Logits shape {top_k_logits.shape} must match probabilites "
            f"shape {top_k_probabilities.shape}.")

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
