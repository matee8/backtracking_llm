"""Defines the configuration objects for the benchmarking pipeline."""

import dataclasses
from typing import Any, Dict, List, Optional, Union


@dataclasses.dataclass
class GenerationConfig:
    """Configuration for the token generation process.

    Attributes:
        max_new_tokens: The maximum number of new tokens to generate.
        temperature: The value used to modulate the next token probabilities.
            Higher values increase randomness.
        top_k: The number of highest probability vocabulary tokens to keep for
            top-k-filtering.
        backtrack_every_n: The frequency (in tokens) at which the decision
            operator is called.
    """
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 50
    backtrack_every_n: int = 1


@dataclasses.dataclass
class EvaluationConfig:
    """Configuration for an lm-evaluation-harness run.

    Attributes:
        tasks: A list of task names to run, compatible with lm-eval.
        num_fewshot: The number of few-shot examples to provide in the prompt.
        limit: The maximum number of samples to evaluate from the dataset. If
            None, the entire dataset is used.
        output_dir: The directory where evaluation results will be saved.
    """
    tasks: List[Union[str, Dict,
                      object]] = dataclasses.field(default_factory=List)
    num_fewshot: int = 0
    limit: Optional[int] = None
    output_dir: str = 'benchmark_results'


@dataclasses.dataclass
class HPOConfig:
    """Configuration for Hyperparameter Optimization.

    Attributes:
        n_trials: The number of optimization trials to run.
        search_space: A dictionary defining the search space for
            hyperparameters. The keys should correspond to the operator's
            `__init__` arguments. The values should be a list containing the
            [min, max] range for sampling.
    """
    n_trials: int
    search_space: Dict[str, List[Any]]


@dataclasses.dataclass
class BenchmarkingConfig:
    """The root configuration for a complete benchmarking pipeline.

    Attributes:
        model_name_or_path: The name or path of the Hugging Face model to use.
        device: The device to run the model on (e.g., 'cpu', 'cuda:0').
        operator_to_tune: The name of the decision Operator class to optimize.
            If None, only a baseline evaluation is run (if `run_baseline` is
            True).
        model_kwargs: A dictionary of keyword arguments to pass directly to the
            Hugging Face `from_pretrained` method.
        run_baseline: Whether to run a baseline evaluation without backtracking.
        generation: The configuration for the text generation process.
        evaluation: The configuration for the evaluation tasks.
        hpo: The optional configuration for hyperparameter optimization. If
            None, no HPO will be performed.
    """
    model_name_or_path: str
    model_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    device: str = 'cpu'
    operator_to_tune: Optional[str] = None
    run_baseline: bool = True
    generation: GenerationConfig = dataclasses.field(
        default_factory=GenerationConfig)
    evaluation: EvaluationConfig = dataclasses.field(
        default_factory=EvaluationConfig)
    hpo: Optional[HPOConfig] = None
