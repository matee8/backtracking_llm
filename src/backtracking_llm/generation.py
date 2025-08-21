"""Defines the core generation logic with backtracking capabilities.

This module provides the `Generator` class, which is responsible for
orchestrating the token-by-token generation process. It interfaces with a
`transformers` model and uses a decision `Operator` to dynamically backtrack and
revise the generated sequence.
"""
