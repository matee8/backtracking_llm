#!/usr/bin/env python3

import functools
import logging
import typing

import torch
import transformers


def run_qa_loop(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer, logger: logging.Logger,
    max_length_per_turn: int, temperature: float, top_k: int,
    backtrack_every_n: int,
    backtracking_decision_function: typing.Optional[functools.partial]
) -> None:
    logger.info("Starting interactive Question-Answering session.")
    logger.info("Model: %s", model.name_or_path)
    logger.info("Max length per turn: %d, Temperature: %.2f, Top-K: %d",
                max_length_per_turn, temperature, top_k)
    if backtracking_decision_function is not None:
        logger.info("Backtracking: enabled.")
    else:
        logger.info("Backtracking: disabled.")
    logger.info("Type your questions below, Press Ctrl+C to exit.")

    chat_history: typing.List[typing.Dict[str, str]] = []

    try:
        while True:
            try:
                user_input: str = input("You: ")
            except (EOFError, KeyboardInterrupt):
                logger.info("Exiting QA loop due to keyboard interrupt.")
                break

            if not user_input.strip():
                logger.info("Empty input received, skipping turn.")
                continue

            chat_history.append({"role": "user", "content": user_input})

            try:
                formatted_prompt_ids = tokenizer.apply_chat_template(
                    chat_history,
                    add_generation_prompt=True,
                    return_tensors="pt")

                if not isinstance(
                        formatted_prompt_ids,
                        torch.Tensor) or formatted_prompt_ids.numel() == 0:
                    logger.error(
                        "Failed to apply chat template. Check tokenizer "
                        "configuration.")
                    if chat_history:
                        chat_history.pop()
                    continue
            except Exception as e:
                logger.error("Failed to apply chat template: %s",
                             e,
                             exc_info=True)
                if chat_history:
                    chat_history.pop()
                continue

            try:
                num_prompt_tokens: int = formatted_prompt_ids.shape[-1]

                generated_ids: typing.Optional[
                    torch.Tensor] = inference.run_inference_loop(
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
            except Exception as e:
                logger.error("An error occured during model inference: %s", e)

                if chat_history:
                    chat_history.pop()

                continue

            if generated_ids is not None and generated_ids.numel(
            ) > num_prompt_tokens:
                answer_ids: torch.Tensor = generated_ids[0, num_prompt_tokens:]

                try:
                    answer_text: str = tokenizer.decode(
                        answer_ids, skip_special_tokens=True)
                    answer_text = answer_text.strip()
                    print(f"Model: {answer_text}")

                    chat_history.append({
                        "role": "assistant",
                        "content": answer_text
                    })
                except Exception as e:
                    logger.error("Failed to decode or print the answer: %s",
                                 e,
                                 exc_info=True)
                    if chat_history and chat_history[-1]["role"] == "user":
                        chat_history.pop()
            elif generated_ids is not None:
                logger.warning("Model did not generate any new tokens.")
                if chat_history and chat_history[-1]["role"] == "user":
                    chat_history.pop()
            else:
                logger.warning("Inference did not return generated IDs.")
                if chat_history and chat_history[-1]["role"] == "user":
                    chat_history.pop()
    except KeyboardInterrupt:
        logger.info("QA loop interrupted by keyboard interrupt. Exiting.")
    except Exception as e:
        logger.error("An unexpected error occurred in the QA loop: %s",
                     e,
                     exc_info=True)
    finally:
        logger.info("QA session finished.")
