"""Integrates the custom backtracking generator with lm-evaluation-harness."""

import logging
from typing import List, Optional

from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

from backtracking_llm.decision import Operator
from backtracking_llm.generation import Generator

logger = logging.getLogger(__name__)


@register_model('backtracking_lm')
class BacktrackingLM(HFLM):
    """An lm-evaluation-harness model for the backtracking generator.

    This class adapts the custom `Generator` to be compatible with the lm-eval
    framework. It is initialized with a pre-configured `Generator` instance
    and an optional decision `Operator`. Its primary role is to override the
    `generate_until` method to use the backtracking generation logic.
    """

    def __init__(self,
                 generator: Generator,
                 operator: Optional[Operator] = None,
                 **kwargs) -> None:
        """Initializes the BacktrackingLM.

        Args:
            generator: The pre-configured Generator instance.
            operator: The decision operator for backtracking.
            **kwargs: Additional keyword arguments to pass to the parent HFLM
                class. Common arguments include `batch_size`.
        """
        super().__init__(pretrained=generator.model,
                         tokenizer=generator.tokenizer,
                         **kwargs)
        self._generator = generator
        self._operator = operator
        logger.info('BacktrackingLM initialized with operator: %s', operator)

    def generate_until(self,
                       requests: List[Instance],
                       disable_tqdm: bool = False) -> List[str]:
        """Generates text for a list of requests using the backtracking
        generator.

        This is the core method called by lm-eval to get model generations.

        Args:
            requests: A list of Instance objects, each containing a context
                string and generation arguments.
            disable_tqdm: Whether to disable the progress bar (ignored, as
                generation is not batched in this implementation).

        Returns:
            A list of generated strings, one for each request.
        """
        if not requests:
            return []

        logger.info('Received %d generation requests.', len(requests))
        results = []

        for request in requests:
            if not request.args:
                logger.warning('Request received with no arguments. Skipping.')
                results.append('GENERATION_ERROR')
                continue

            context = request.args[0]

            request_args = {}
            if len(request.args) > 1 and isinstance(request.args[1], dict):
                request_args = request.args[1]
            stop_sequences = request_args.get('until', None)
            max_tokens = request_args.get('max_gen_toks', 128)

            try:
                generated_text = self._generator.generate(
                    prompt=context,
                    operator=self._operator,
                    max_new_tokens=max_tokens,
                    stop_sequences=stop_sequences)
                results.append(generated_text)
            except (ValueError, RuntimeError) as e:
                logger.error("Error during generation for context '%s...': %s",
                             context[:50],
                             e,
                             exc_info=True)
                results.append('GENERATION_ERROR')

        return results
