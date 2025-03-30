#!/usr/bin/env python3

from argparse import Namespace, ArgumentParser
import logging
from logging import Logger, StreamHandler
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from typing import Any
import sys


def load_model_and_tokenizer(
    model_name: str = "gpt2", ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    try:
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name)

        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        model.eval()

        return model, tokenizer
    except Exception as e:
        logging.error("Failed to load model or tokenizer: %s", str(e))
        raise ValueError(f"Error loading model or tokenizer: {str(e)}") from e


def predict_next_token(
    model: PreTrainedModel,
    input_ids: Tensor,
    top_k: int,
) -> list[dict[str, Any]]:
    try:
        model.eval()

        with torch.no_grad():
            outputs: CausalLMOutputWithCrossAttentions = model(input_ids)
            logits: Tensor = outputs.logits[:, -1, :]
            probabilites: Tensor = F.softmax(logits, dim=-1)

            top_k_probs: Tensor
            top_k_ids: Tensor
            top_k_probs, top_k_ids = torch.topk(probabilites, top_k)

            res = []

            for i in range(top_k):
                res.append({
                    "token_id": top_k_ids[0][i],
                    "logit": logits[0, top_k_ids[0, i]].item(),
                    "probability": top_k_probs[0, i].item(),
                })

            return res
    except Exception as e:
        logging.error("Error during next token prediction: %s", str(e))
        raise


def _calculate_statistics(
    logits: Tensor,
    probabilities: Tensor,
) -> dict[str, float]:
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


def run_inference_loop(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_length: int,
    top_k: int,
    logger: Logger,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tensor | None:
    try:
        input_ids: Tensor = tokenizer(prompt,
                                      return_tensors="pt").input_ids.to(device)

        for _ in range(max_length - input_ids.shape[-1]):
            tokens: list[dict[str, Any]] = predict_next_token(
                model, input_ids, top_k)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Iteration %d",
                    len(input_ids[0]) - len(tokenizer(prompt).input_ids))

                logits = torch.empty(len(tokens))
                probabilities = torch.empty(len(tokens))

                for i in range(top_k):
                    logit = tokens[i]["logit"]
                    probability = tokens[i]["probability"]

                    logging.debug(
                        "Token: %s, logit: %r, probability: %r",
                        tokenizer.decode(tokens[i]["token_id"]),
                        logit,
                        probability,
                    )

                    logits[i] = torch.tensor([logit])
                    probabilities[i] = torch.tensor([probability])

                stats: dict[str, float] = _calculate_statistics(
                    logits, probabilities)

                for key, value in stats.items():
                    logging.debug("%s: %r", key, value)

                logging.debug("Chosen token: %s",
                              tokenizer.decode(tokens[0]["token_id"]))

            input_ids: Tensor = torch.cat(
                [input_ids, tokens[0]["token_id"].reshape(1, -1)], dim=-1)

        logging.debug("Generated text: %s",
                      tokenizer.decode(input_ids[0], skip_special_tokens=True))
        return input_ids
    except KeyboardInterrupt:
        logging.info("Inference interrupted by user")
        return None
    except Exception as e:
        logging.error("Error during inference: %s", str(e))
        raise


def _parse_arguments() -> Namespace:
    parser = ArgumentParser(
        description="Run inference on GPT-2 and analyze token logits " \
        "and probabilites"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name to use (default: gpt2)",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt to start generation",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (default: 10)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top tokens to analyze (default: 50)",
    )

    parser.add_argument("--verbose",
                        action="store_true",
                        help="Enable verbose logging")

    return parser.parse_args()


def _setup_logging(verbose: bool = False) -> Logger:
    log_level: int = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[StreamHandler(sys.stdout)],
    )

    return logging.getLogger(__name__)


def _main() -> None:
    args: Namespace = _parse_arguments()
    logger: Logger = _setup_logging(args.verbose)

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    model, tokenizer = load_model_and_tokenizer(args.model)

    run_inference_loop(model=model,
                       tokenizer=tokenizer,
                       prompt=args.prompt,
                       max_length=args.max_length,
                       top_k=args.top_k,
                       logger=logger)


if __name__ == "__main__":
    _main()
