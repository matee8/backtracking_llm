import logging

from lm_eval.api import instance, registry
from lm_eval.models import huggingface

from backtracking_llm.models import inference


@registry.register_model("backtracking_lm")
class BacktrackingLM(huggingface.HFLM):

    def __init__(self, pretrained: str,
                 backtracking_config: inference.BacktrackingInferenceConfig,
                 logger: logging.Logger, **kwargs) -> None:
        self.backtracking_config = backtracking_config
        self.logger = logger

        self.engine = inference.BacktrackingInferenceEngine(
            model_name=pretrained,
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

        for req in requests:
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
            token_ids = self.engine.generate(context, None)
            if token_ids is None:
                self.logger.warning("Engine returned no tokens.")
                return error_msg

            decoded = self.tokenizer.decode(
                token_ids[0], skip_special_tokens=True)[len(context):]

            self.logger.debug("Raw generated text (before stop seq): '%s...'",
                              decoded[:50])

            if stop_seq:
                decoded = self._apply_stop_sequences(decoded, stop_seq)

            return decoded.strip()
        except Exception as e:
            self.logger.error(
                "Error during backtracking generation for context"
                " '%s...': %s",
                context[:50],
                e,
                exc_info=True)
            return error_msg

    def _apply_stop_sequences(self, text: str,
                              stop_sequences: list[str]) -> str:
        for seq in stop_sequences:
            idx = text.find(seq)
            if idx != -1:
                self.logger.debug("Stopping at sequence: %s", seq)
                return text[:idx]

        return text
