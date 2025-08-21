"""Defines the core generation logic with backtracking capabilities."""

class Generation:
    """Orchestrates token-by-token text generation with a backtracking
    mechanism.

    This class wraps a `transformers` model and tokenizer, decoupling the
    generation logic from model loading and configuration. Its primary role is
    to execute a custom generation loop that can undo previous generation steps
    based on the logic provided by a given `Operator`.
    """
