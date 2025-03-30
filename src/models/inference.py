#!/usr/bin/env python3

from argparse import Namespace, ArgumentParser
import logging
from logging import StreamHandler
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
import typing
import sys


def load_model_and_tokenizer(
    model_name: str = "gpt2",
) -> typing.Tuple[PreTrainedModel, PreTrainedTokenizer]:
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


def run_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_length: int,
    top_k: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str | None:
    try:
        input_ids: Tensor = tokenizer(prompt,
                                      return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[-1]):
                outputs: CausalLMOutputWithCrossAttentions = model(input_ids)
                logits: Tensor = outputs.logits[:, -1, :]
                probabilites: Tensor = F.softmax(logits)

                top_k_probs: Tensor
                top_k_ids: Tensor
                top_k_probs, top_k_ids = torch.topk(probabilites, top_k)

                top_k_tokens: list[str] = tokenizer.convert_ids_to_tokens(
                    top_k_ids[0].tolist())

                logging.debug(
                    "Iteration %d",
                    len(input_ids[0]) - len(tokenizer(prompt).input_ids))
                for i in range(top_k):
                    logging.debug("Token: %s, Logit: %d, Probability: %r",
                                  top_k_tokens[i], logits[0,
                                                          top_k_ids[0,
                                                                    i]].item(),
                                  top_k_probs[0, i].item())

                input_ids: Tensor = torch.cat(
                    [input_ids, top_k_ids[0][0].reshape(1, -1)], dim=-1)

                top_logit_value = logits[0, 0].item()
                second_logit_value = logits[
                    0, 1].item() if top_k > 1 else float("-inf")
                top_prob_value = top_k_probs[0, 0].item()
                second_prob_value = top_k_probs[0,
                                                1].item() if top_k > 1 else 0.0
                epsilon = 1e-10

                highest_logit_abs = abs(top_logit_value)
                logging.debug("Highest logit (abs): %r", highest_logit_abs)

                highest_prob = top_prob_value
                logging.debug("Highest probability: %r", highest_prob)

                logit_diff = top_logit_value - second_logit_value
                logging.debug("Logit difference (1st-2nd): %r", logit_diff)

                if second_logit_value != 0:
                    logit_ratio = top_logit_value / second_logit_value
                else:
                    logit_ratio = float("inf")
                logging.debug("Logit ratio (1st/2nd): %r", logit_ratio)

                prob_diff = top_prob_value - second_prob_value
                logging.debug("Probability difference (1st-2nd): %r",
                              prob_diff)

                if second_prob_value != 0:
                    prob_ratio = top_prob_value / second_prob_value
                else:
                    prob_ratio = float("inf")
                logging.debug("Probability ratio (1st/2nd): %r", prob_ratio)

                prob_entropy = -torch.sum(
                    probabilites * torch.log(probabilites + epsilon)).item()
                logging.debug("Probability distribution entropy: %r",
                              prob_entropy)

                logging.debug(
                    "Chosen token: %s",
                    top_k_tokens[0]
                )
                logging.debug(
                    "Generated text so far: %s",
                    tokenizer.decode(input_ids[0], skip_special_tokens=True))

            return tokenizer.decode(input_ids[0], skip_special_tokens=True)
    except KeyboardInterrupt:
        logging.info("Inference interrupted by user")
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


def _setup_logging(verbose: bool = False) -> None:
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[StreamHandler(sys.stdout)],
    )


def _main() -> None:
    args: Namespace = _parse_arguments()
    _setup_logging(args.verbose)

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    model, tokenizer = load_model_and_tokenizer(args.model)

    generated_text: str | None = run_inference(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        top_k=args.top_k,
    )
    logging.info("Generated text: %s", generated_text)


if __name__ == "__main__":
    _main()
