#!/usr/bin/env python3

import functools
import logging
import typing

import torch
import transformers

from backtracking_llm.models import inference


def run_qa_loop(
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer, logger: logging.Logger,
        max_length_per_turn: int, temperature: float, top_k: int,
        backtrack_every_n: int,
        backtracking_decision_function: functools.partial | None) -> None:
    logger.info("Starting interactive Question-Answering session.")
    logger.info("Model: %s", model.name_or_path)
    logger.info("Max length per turn: %d, Temperature: %.2f, Top-K: %d",
                max_length_per_turn, temperature, top_k)
    if backtracking_decision_function is not None:
        logger.info("Backtracking: enabled.")
    else:
        logger.info("Backtracking: disabled.")
    logger.info("Type your questions below, Press Ctrl+C to exit.")

    chat_history: list[dict[str, str]] = []

    try:
        while True:
            try:
                user_input = _get_user_input(logger)
                if user_input is None:
                    continue
            except (EOFError, KeyboardInterrupt):
                break

            chat_history.append({"role": "user", "content": user_input})

            formatted_prompt_ids = _prepare_prompt_ids(tokenizer, chat_history,
                                                       logger)
            if formatted_prompt_ids is None:
                chat_history.pop()
                continue

            try:
                num_prompt_tokens = formatted_prompt_ids.shape[-1]

                generated_ids = inference.run_inference_loop(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=formatted_prompt_ids,
                    max_answer_length=max_length_per_turn,
                    top_k=top_k,
                    logger=logger,
                    temperature=temperature,
                    backtrack_every_n=backtrack_every_n,
                    backtracking_decision_function=
                    backtracking_decision_function)

                if generated_ids is None:
                    chat_history.pop()
            except Exception as e:
                logger.error("An error occured during model inference: %s", e)

                if chat_history:
                    chat_history.pop()

                continue

            if generated_ids is None:
                logger.error("The model did not generate any answer.")

                if chat_history:
                    chat_history.pop()

                continue

            answer_text = _process_model_output(
                generated_ids=generated_ids,
                num_prompt_tokens=num_prompt_tokens,
                tokenizer=tokenizer,
                logger=logger)
            if answer_text is None:
                chat_history.pop()
                continue

            print("\n")

            chat_history.append({"role": "assistant", "content": answer_text})
    except KeyboardInterrupt:
        logger.info("QA loop interrupted by keyboard interrupt. Exiting.")
    except Exception as e:
        logger.error("An unexpected error occurred in the QA loop: %s",
                     e,
                     exc_info=True)
    finally:
        logger.info("QA session finished.")


def _get_user_input(logger: logging.Logger) -> str | None:
    try:
        user_input = input("You: ")

        if not user_input.strip():
            logger.info("Empty input received, skipping turn.")
            return None

        return user_input
    except (EOFError, KeyboardInterrupt):
        print("\n")
        logger.info("Input interrupted.")
        raise


def _prepare_prompt_ids(tokenizer: transformers.PreTrainedTokenizer,
                        chat_history: list[dict[str, str]],
                        logger: logging.Logger) -> torch.Tensor | None:
    try:
        formatted_prompt_ids = tokenizer.apply_chat_template(
            chat_history, add_generation_prompt=True, return_tensors="pt")

        if (not isinstance(formatted_prompt_ids, torch.Tensor)
                or formatted_prompt_ids.numel() == 0):
            logger.error("Failed to apply chat template. Check tokenizer "
                         "configuration.")
            return None
        return formatted_prompt_ids
    except Exception as e:
        logger.error("Failed to apply chat template: %s", e, exc_info=True)
        return None


def _process_model_output(generated_ids: torch.Tensor, num_prompt_tokens: int,
                          tokenizer: transformers.PreTrainedTokenizer,
                          logger: logging.Logger) -> str | None:
    if generated_ids.numel() <= num_prompt_tokens:
        logger.warning("Model did not generate any new tokens.")
        return None

    answer_ids = generated_ids[0, num_prompt_tokens:]

    try:
        answer_text = tokenizer.decode(answer_ids,
                                       skip_special_tokens=True).strip()

        if not answer_text:
            logger.warning("Model generated empty text after decoding.")
            return None

        return answer_text
    except Exception as e:
        logger.error("Failed to decode or print the answer: %s",
                     e,
                     exc_info=True)
        return None
