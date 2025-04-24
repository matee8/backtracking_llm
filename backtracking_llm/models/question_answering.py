import logging
import typing

import torch

from backtracking_llm.models import inference


def cli_get_input() -> str:
    return input("You: ")


def cli_display_output(event: inference.GenerationEvent) -> None:
    if event.type == inference.GenerationEventType.TOKEN:
        print(event.data, end="", flush=True)
    elif event.type == inference.GenerationEventType.END:
        print()
    elif event.type == inference.GenerationEventType.ERROR:
        print(f"Error: {event.data}", flush=True)


class ChatError(RuntimeError):
    pass


class ChatSession:

    def __init__(
        self,
        engine: inference.InferenceEngine,
        logger: logging.Logger,
        input_fn: typing.Callable[[], str] = cli_get_input,
        output_fn: typing.Callable[[inference.GenerationEvent],
                                   None] = cli_display_output
    ) -> None:
        self.engine = engine
        self.tokenizer = engine.tokenizer
        self.logger = logger
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.chat_history: list[dict[str, str]] = []

    def run(self) -> None:
        self.logger.info("Starting interactive Question-Answering session.")
        self.logger.info("Type your questions below. Press Ctrl+C or Ctrl+D "
                         "(EOF) to exit.")

        try:
            while True:
                try:
                    user_input = self._get_user_input()
                    if user_input is None:
                        continue
                except (EOFError, KeyboardInterrupt):
                    self.logger.info("Input interrupted. Exiting chat loop.")
                    break

                self.chat_history.append({
                    "role": "user",
                    "content": user_input
                })

                try:
                    self._process_conversation_turn()
                except ChatError as e:
                    self.logger.error("Failed to process chat turn: %s",
                                      e,
                                      exc_info=True)

                    if self.chat_history:
                        self.chat_history.pop()

                    continue
                except inference.GenerationError as e:
                    self.logger.error("Model generation failed: %s",
                                      e,
                                      exc_info=True)

                    if self.chat_history:
                        self.chat_history.pop()

                    continue
                except Exception as e:
                    self.logger.error(
                        "An unexpected error occured during the "
                        "chat turn: %s",
                        e,
                        exc_info=True)

                    if self.chat_history:
                        self.chat_history.pop()

                    continue
        except (KeyboardInterrupt, EOFError):
            self.logger.info("Chat loop interrupted by keyboard interrupt. "
                             "Exiting.")
        except Exception as e:
            self.logger.error(
                "An unexpected critical error occured in the main"
                " chat loop: %s",
                e,
                exc_info=True)
            raise
        finally:
            self.logger.info("Chat session finished.")

    def _get_user_input(self) -> str | None:
        try:
            text = self.input_fn()

            if not text or not text.strip():
                self.logger.info("Empty input; please type a question.")
                return None

            return text.strip()
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception as e:
            msg = "Error reading user input"
            self.logger.error("%s: %s", msg, e, exc_info=True)
            raise ChatError(msg) from e

    def _process_conversation_turn(self) -> None:
        prompt_ids = self._prepare_prompt()

        generated_ids = self.engine.generate(prompt_ids, self.output_fn, None)
        if generated_ids is None:
            raise ChatError("Model generation did not return valid output.")

        answer = self._process_model_output(generated_ids,
                                            prompt_ids.shape[-1])

        self.chat_history.append({"role": "assistant", "content": answer})

    def _prepare_prompt(self) -> torch.Tensor:
        try:
            ids = self.tokenizer.apply_chat_template(
                self.chat_history,
                add_generation_prompt=True,
                return_tensors="pt")

            if not isinstance(ids, torch.Tensor) or ids.numel() == 0:
                raise ChatError(
                    "Applying chat template returned invalid "
                    "result. Check tokenizer configuration and chat"
                    " history format.")

            return ids.to(self.engine.device)  # type: ignore[reportReturnType]
        except Exception as e:
            msg = "Failed to apply chat template"
            self.logger.error("%s: %s", msg, e, exc_info=True)
            raise ChatError(msg) from e

    def _process_model_output(self, generated_ids: torch.Tensor,
                              prompt_len: int) -> str:
        if generated_ids.numel() <= prompt_len:
            raise ChatError(
                "Model generation finished, but no new tokens were "
                "generated")

        answer_ids = generated_ids[0, prompt_len:]
        try:
            text = (self.tokenizer.decode(answer_ids,
                                          skip_special_tokens=True).strip())
        except Exception as e:
            raise ChatError("Failed to decode the answer tokens") from e

        if not text:
            raise ChatError("Model generated empty or whitespace-only text "
                            "after decoding.")

        return text
