import dataclasses
import pathlib
import typing

from backtracking_llm.models import decision


@dataclasses.dataclass(frozen=True)
class BenchmarkConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    task_name: str = "hendrycks_math_algebra"
    num_fewshot: int = 8
    limit_for_search: int = 500
    backtrack_every_n: int = 5
    batch_size: int = 1
    output_dir: pathlib.Path = pathlib.Path("benchmark_results")
    device: str = "cpu"
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
    backtracking_max_answer_length: int = 64
    backtracking_top_k: int = 50
    backtracking_temperature: float = 1.0
