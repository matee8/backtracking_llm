import logging
from abc import ABC, abstractmethod
from typing import Type

from torch import cuda, device, Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel

logger = logging.getLogger(__name__)


class TextGenerator(ABC):

    def __init__(self, model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls: Type["TextGenerator"],
                        model_name: str,
                        device_str: str | None = None,
                        **kwargs) -> "TextGenerator":
        if device_str:
            if "cuda" in device_str and not cuda.is_available():
                raise ValueError(
                    f"Specified device '{device_str}' but CUDA is "
                    "not available.")

            dev = device(device_str)
        elif cuda.is_available():
            dev = device("cuda")
        else:
            dev = device("cpu")

        logger.info("Using device '%s'.", dev)

        logger.info("Loading model and tokenizer for '%s'...", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            logger.info("Setting pad_token to eos_token for generation.")
            tokenizer.pad_token = tokenizer.eos_token

        model.to(dev)
        model.eval()

        return cls(model, tokenizer, **kwargs)

    @abstractmethod
    def generate(self, prompt: str | Tensor, **kwargs) -> Tensor | None:
        pass
