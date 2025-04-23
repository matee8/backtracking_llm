import json
import logging
import pathlib
import sys
import typing


def _extract_primary_score(results: dict[str, typing.Any],
                           task_name: str) -> float:
    if not results:
        raise ValueError("'results' cannot be None")

    if task_name not in results:
        raise ValueError(f"'results' has no '{task_name}' field")

    task_results = results[task_name]

    metric_keys = ["acc_norm,none", "acc,none", "exact_math,none"]

    for key in metric_keys:
        if key in task_results:
            return task_results[key]

    raise KeyError("No standard accuracy metric found in 'task_results'.")


def _save_json(data: typing.Any, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def _setup_logger() -> logging.Logger:
    level = logging.DEBUG
    log_format = "%(asctime)s  - %(name)s - %(levelname)s - %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=level,
                        format=log_format,
                        handlers=[handler],
                        force=True)
    return logging.getLogger(__name__)
