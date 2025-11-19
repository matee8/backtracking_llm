# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2025-11-19

### Added

-   Full Reinforcement Learning pipeline for training custom backtracking policies (`backtracking-llm-train-rl`).
-   `RlPolicyOperator` for executing trained PPO policies during inference.
-   Gymnasium environment with LLM-as-a-Judge scoring and intermediate reward shaping.
-   `GenerationSession` class for fine-grained, stateful control over the generation loop.

### Fixed

-   Fixed critical state leakage in CLI chat sessions where operators retained history between turns.
-   Fixed "amnesia" bug in `NGramOverlap` where backtracking erased too much history.
-   Fixed greedy decoding to be truly deterministic when `temperature` is set to 0.

## [0.2.3] - 2025-10-19

### Fixed

-   Removed memory profiling from main benchmarking script.

## [0.2.2] - 2025-10-18

### Added

-   Added configuration option for passing keyword arguments to model loading.
-   Every hyperparameter search trial is saved in a CSV file.

### Fixed

-   Allowed code execution within benchmarks.
-   Best results are now correctly saved in a JSON file.

## [0.2.1] - 2025-10-07

### Fixed

-   Changed path of benchmarking script to actual path in the library.

## [0.2.0] - 2025-10-03

### Added

-   Introduced a configuration-driven benchmarking module for evaluating
    backtracking performance.
-   Integration with `lm-evaluation-harness` for running standardized NLP
    benchmarks.
-   Hyperparameter optimization using `optuna` to find the best `Operator`
    settings.
-   New CLI script for launching benchmark runs from a YAML configuration file.

### Fixed

-   Corrected the stop sequence handling to robustly and automatically stop on
    both the model's EOS token and user-defined sequences, preventing premature
    generation termination.

## [0.1.0] - 2025-09-04

### Added

- Initial public release.
- Core `Generator` class for text eneration with a novel backtracking mechanism.
- A suite of `Operator` classes for controlling backtracking logic.
- Stateless `ChatPipeline` for easy, multi-turn conversational interactions.
- Interactive `backtracking-llm` CLI for local-first chatting.

[Unreleased]: https://github.com/matee8/backtracking_llm/compare/v0.2.2...HEAD
[0.1.0]: https://github.com/matee8/backtracking_llm/releases/tag/v0.1.0
[0.2.0]: https://github.com/matee8/backtracking_llm/releases/tag/v0.2.0
[0.2.1]: https://github.com/matee8/backtracking_llm/releases/tag/v0.2.1
[0.2.2]: https://github.com/matee8/backtracking_llm/releases/tag/v0.2.2
[0.3.0]: https://github.com/matee8/backtracking_llm/releases/tag/v0.3.0
