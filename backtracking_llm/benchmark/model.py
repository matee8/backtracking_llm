import logging

import tqdm
from lm_eval.api import instance, registry
from lm_eval.models import huggingface

from backtracking_llm.models import inference


@registry.register_model("backtracking_lm")
class BacktrackingLM(huggingface.HFLM):

    def __init__(self, pretrained: str,
                 backtracking_config: inference.BacktrackConfig,
                 logger: logging.Logger, **kwargs) -> None:
        self.backtracking_config = backtracking_config
        self.logger = logger

        self.engine = inference.BacktrackEngine(model_name=pretrained,
                                                config=backtracking_config,
                                                logger=self.logger)

        kwargs["pretrained"] = self.engine.model
        kwargs["tokenizer"] = self.engine.tokenizer

        super().__init__(**kwargs)

        self.pretrained = pretrained

        self.logger.info(
            "Initialized BacktrackingLM with engine using config: "
            "%s", backtracking_config)

    def generate_until(self,
                       requests: list[instance.Instance],
                       disable_tqdm: bool = False) -> list[str]:
        self.logger.info("Received %d generation requests.", len(requests))

        results = []

        if disable_tqdm:
            iterator = requests
        else:
            iterator = tqdm.tqdm(requests, "Running generate_until requests:")

        for req in iterator:
            result = self._process_request(req)
            results.append(result)

        return results

    def _process_request(self, request: instance.Instance) -> str:
        args = request.args
        error_msg = "Generation Error"

        if args is None or len(args) == 0:
            self.logger.warning("Instance does not contain any arguments.")
            return error_msg

        context = args[0]
        if not isinstance(context, str):
            self.logger.warning("Context is not of type 'str'.")
            return error_msg

        stop_seq = None
        if len(args) > 1 and isinstance(args[1], dict):
            stop_seq = args[1].get("until", None)

        self.logger.debug("Generating for context: '%s...'", context[:50])
        try:
            token_ids = self.engine.generate(context, None, stop_seq)
            if token_ids is None:
                self.logger.warning("Engine returned no tokens.")
                return error_msg

            engine_prompt_input_ids = self.engine.tokenizer(
                context, return_tensors="pt").input_ids
            engine_prompt_token_length = engine_prompt_input_ids.shape[1]

            answer_token_ids = token_ids[0, engine_prompt_token_length:]

            decoded = self.tokenizer.decode(answer_token_ids,
                                            skip_special_tokens=True)

            self.logger.debug("Raw generated text (before stop seq): '%s...'",
                              decoded[:50])

            return decoded
        except Exception as e:
            self.logger.error(
                "Error during backtracking generation for context"
                " '%s...': %s",
                context[:50],
                e,
                exc_info=True)
            return error_msg
