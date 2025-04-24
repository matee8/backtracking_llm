import dataclasses
import pathlib
import typing

from backtracking_llm.models import decision


@dataclasses.dataclass
class BenchmarkConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    task_names: list[str | dict | object] = (dataclasses.field(
        default_factory=lambda: ["hendrycks_math_algebra"]))
    fewshot: int = 8
    backtrack_every_n: int = 5
    batch_size: int = 1
    output_dir: pathlib.Path = pathlib.Path("benchmark_results")
    device: str = "cpu"
    skip_base: bool = False
    baseline_limit: int | None = None
    search_limit: int | None = 500
    final_limit: int | None = None
    decision_strategies: list[typing.Type[decision.BacktrackStrategy]] = (
        dataclasses.field(default_factory=lambda: [
            decision.ProbabilityThreshold,
            decision.EntropyThreshold,
            decision.ProbabilityMargin,
            decision.ProbabilityDrop,
            decision.ProbabilityTrend,
            decision.Repetition,
            decision.NGramOverlap,
            decision.LogitThreshold,
        ]))
    max_answer_length: int = 64
    top_k: int = 50
    temperature: float = 1.0
    bootstrap_iters: int = 1000
