import typing
import logging
import torch
import torch.nn.functional as F
import transformers
from transformers import modeling_outputs


def load_model_and_tokenizer(
    model_name: str = "gpt2"
) -> typing.Tuple[transformers.PreTrainedModel,
                  transformers.PreTrainedTokenizer]:
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        return model, tokenizer
    except Exception as e:
        logging.error("Failed to load model or tokenizer: %s", str(e))
        raise


def predict_next_token(
    model: transformers.PreTrainedModel,
    input_ids: torch.Tensor,
    top_k: int,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    try:
        model.eval()

        with torch.no_grad():
            outputs: modeling_outputs.CausalLMOutputWithCrossAttentions = model(
                input_ids)
            logits: torch.Tensor = outputs.logits[:, -1, :]

            return torch.topk(logits, top_k)
    except Exception as e:
        logging.error("Error during next token prediction: %s", str(e))
        raise


def run_inference_loop(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    prompt: str,
    max_length: int,
    top_k: int,
    logger: logging.Logger,
    temperature: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> typing.Optional[torch.Tensor]:
    try:
        input_ids: torch.Tensor = tokenizer(
            prompt, return_tensors="pt").input_ids.to(device)

        for _ in range(max_length - input_ids.shape[-1]):
            top_k_indices: torch.Tensor
            top_k_logits: torch.Tensor
            top_k_logits, top_k_indices = predict_next_token(
                model, input_ids, top_k)
            top_k_logits = top_k_logits.ravel()
            top_k_indices = top_k_indices.ravel()

            if temperature == 0.:
                probabilities: torch.Tensor = F.softmax(top_k_logits, dim=-1)

                chosen_token_relative_idx: torch.Tensor = torch.argmax(
                    top_k_logits)
            else:
                probabilities: torch.Tensor = F.softmax(top_k_logits /
                                                        temperature,
                                                        dim=-1)

                chosen_token_relative_idx: torch.Tensor = torch.multinomial(
                    probabilities, num_samples=1)

            chosen_token_id = top_k_indices[chosen_token_relative_idx]

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Iteration %d",
                    len(input_ids[0]) - len(tokenizer(prompt).input_ids))

                for i in range(top_k):
                    logging.debug(
                        "Token: %s, logit: %r, probability: %r",
                        tokenizer.decode(top_k_indices[i].item()),
                        top_k_logits[i].item(),
                        probabilities[i].item(),
                    )

                stats: typing.Dict[str, float] = _calculate_statistics(
                    top_k_logits, probabilities)

                for key, value in stats.items():
                    logging.debug("%s: %r", key, value)

                logging.debug("Chosen token: %s",
                              tokenizer.decode(chosen_token_id.item()))

            input_ids: torch.Tensor = torch.cat(
                [input_ids, chosen_token_id.reshape(1, -1)], dim=-1)

        logging.debug("Generated text: %s",
                      tokenizer.decode(input_ids[0], skip_special_tokens=True))
        return input_ids
    except KeyboardInterrupt:
        logging.info("Inference interrupted by user")
        return None
    except Exception as e:
        logging.error("Error during inference: %s", str(e))
        raise


def _calculate_statistics(
    logits: torch.Tensor,
    probabilities: torch.Tensor,
) -> typing.Dict[str, float]:
    if probabilities.numel() == 0 or logits.numel() == 0:
        raise ValueError("Cannot calculate statistics on empty tensors")

    top_logit_value = logits[0].item()

    if logits.numel() > 1:
        second_logit_value = logits[1].item()
    else:
        second_logit_value = float("inf")

    top_prob_value = probabilities[0].item()

    if probabilities.numel() > 1:
        second_prob_value = probabilities[1].item()
    else:
        second_prob_value = 0.0

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
        probabilities * torch.log(probabilities + epsilon)).item()

    return stats
