#!/usr/bin/env python3

import logging
import typing

import torch
import transformers

from backtracking_llm.models import inference


def run_qa_loop(model: transformers.PreTrainedModel,
                tokenizer: transformers.PreTrainedTokenizer,
                logger: logging.Logger, max_length_per_turn: int,
                temperature: float, top_k: int):
    logger.info("Starting interactive Question-Answering session.")
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
                formatted_prompt = tokenizer.apply_chat_template(
                    chat_history, tokenize=False, add_generation_prompt=True)

                if not isinstance(formatted_prompt, str):
                    logger.error("Failed to apply chat template.")
                    chat_history.pop()
                    continue

                logger.debug("Formatted prompt for the model: %s",
                             formatted_prompt)
            except Exception as e:
                logger.error("Failed to apply chat template: %s",
                             e,
                             exc_info=True)
                if chat_history:
                    chat_history.pop()
                continue

            try:
                prompt_input_ids: torch.Tensor = tokenizer(
                    formatted_prompt, return_tensors="pt").input_ids
                num_prompt_tokens: int = prompt_input_ids.shape[-1]

                generated_ids: typing.Optional[
                    torch.Tensor] = inference.run_inference_loop(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=formatted_prompt,
                        max_answer_length=max_length_per_turn,
                        top_k=top_k,
                        logger=logger,
                        temperature=temperature)
            except Exception as e:
                logger.error("Error during model inference: %s",
                             e,
                             exc_info=True)

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
                    if chat_history and chat_history[-1]:
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
