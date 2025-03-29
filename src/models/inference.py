#!/usr/bin/env python3

import logging
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
import typing


def load_model_and_tokenizer(
    model_name: str = "gpt2",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> typing.Tuple[PreTrainedModel, PreTrainedTokenizer]:
    try:
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name)
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name)
        model.to(device)
        model.eval()

        return model, tokenizer
    except Exception as e:
        logging.error("Failed to load model or tokenizer: %s", str(e))
        raise ValueError(f"Error loading model or tokenizer: {str(e)}") from e


if __name__ == "__main__":
    pass
