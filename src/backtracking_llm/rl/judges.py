"""Judge implementations for providing reward signals.

This module provides both a real OpenAI-based judge for production use and a
mock judge for testing without API dependencies.
"""

import logging
import re
from typing import Protocol, runtime_checkable

from openai import OpenAI
import tenacity

from backtracking_llm.rl.config import JudgeConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class Judge(Protocol):
    """Protocol for judge implementations that score generated text.

    All judges must implement this protocol to be compatible with the RL
    environment and training pipeline.
    """

    def score(self, text: str) -> float:
        """Score the quality of generated text.

        Args:
            text: The generated text to evaluate.

        Returns:
            A float score typically in range [1.0, 10.0]. Higher is better.
        """
        ...


class OpenAIJudge:
    """Judge implementation using OpenAI's API.

    This judge sends generated text to an OpenAI model with a carefully
    crafted prompt asking for a 1-10 quality rating. It includes retry
    logic for reliability and robust parsing of various response formats.

    Attributes:
        config: The JudgeConfig used for initialization.
        client: OpenAI client instance.
        _prompt_template: The formatted prompt template.
    """

    def __init__(self, config: JudgeConfig) -> None:
        """Initialize OpenAI judge.

        Args:
            config: Configuration for the judge, including model selection
                and API parameters.
        """
        self.config = config

        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=0,
        )

        self._prompt_template = self._default_prompt()

    @tenacity.retry(stop=tenacity.stop_after_attempt(3),
                    wait=tenacity.wait_exponential(multiplier=3, min=4, max=60),
                    reraise=True)
    def score(self, text: str) -> float:
        """Score text using OpenAI API with exponential backoff retry.

        This method is wrapped with tenacity retry logic to handle transient
        API failures. It will retry up to 3 times with exponential backoff.

        Args:
            text: Text to evaluate. Must be non-empty.

        Returns:
            Float score between 1.0 and 10.0.

        Raises:
            ValueError: If API response cannot be parsed into a numeric score.
            openai.APIError: If API call fails after all retries.
        """
        if len(text) == 0 or len(text.strip()) == 0:
            logger.warning(
                'Empty text provided to judge, returning minimum score.')
            return 0.0

        prompt = self._prompt_template.format(text=text)

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{
                'role':
                    'system',
                'content': ('You are a helpful assistant that evaluates'
                            ' text quality. Respond with only a numeric score.')
            }, {
                'role': 'user',
                'content': prompt
            }],
            temperature=0.0)

        content = response.choices[0].message.content
        return self._parse_score(content)

    def _default_prompt(self) -> str:
        """Return the default judge prompt template.

        Returns:
            A prompt template string with {text} placeholder.
        """
        return ('Please evaluate the following text on a scale of 1-10 '
                'for coherence, relevance, and grammatical correctness. '
                'Respond with only the numeric score.\n\n'
                'Text: {text}\n\n'
                'Score (1-10):')

    def _parse_score(self, response: str | None) -> float:
        """Parse score from API response.

        Handles various response formats:
        - '8.5' -> 8.5
        - 'Score: 7' -> 7.0
        - '8.5 out of 10' -> 8.5
        - 'The score is 9/10' -> 9.0

        Args:
            response: Raw API response string.

        Returns:
            Float score clamped to [1.0, 10.0] range.

        Raises:
            ValueError: If score cannot be parsed.
        """
        if not response:
            raise ValueError('Could not parse numeric score from no response.')

        cleaned = response.strip().lower()

        patterns = [
            r'(\d+(?:\.\d+)?)\s*/\s*10", r"(\d+(?:\.\d+)?)\s+out\s+of\s+10',
            r'(\d+(?:\.\d+)?)'
        ]

        for pattern in patterns:
            match = re.search(pattern, cleaned)
            if match:
                score = float(match.group(1))
                return min(10.0, max(1.0, score))

        raise ValueError('Could not parse numeric score from judge response: '
                         f'  "{response}".')


class MockJudge:
    """Deterministic mock judge for testing without API calls.

    Scores based on simple heuristics that are fast and deterministic:
        - Length scoring (penalizes very short and very long texts)
        - Repetition penalty (penalizes consecutive repeated words)
        - Baseline score of 5.0

    This judge is designed for unit tests and CI environments where
    API calls are expensive or unavailable.

    Attributes:
        base_score: Baseline score for average text.
        repetition_penalty: Penalty factor for repetitive text.
    """

    def __init__(self,
                 base_score: float = 5.0,
                 repetition_penalty: float = 2.0):
        """Initialize mock judge.

        Args:
            base_score: Baseline score for text with no major issues.
            repetition_penalty: Penalty multiplier for repetitive patterns.
        """
        self.base_score = base_score
        self.repetition_penalty = repetition_penalty

    def score(self, text: str) -> float:
        """Score text deterministically based on heuristics.

        Args:
            text: Text to evaluate.

        Returns:
            Score between 1.0 and 10.0.
        """
        if not text or not text.strip():
            return 1.0

        length = len(text)
        if length < 20:
            length_score = 2.0
        elif length > 1000:
            length_score = 3.0
        else:
            length_score = min(9.0, 3.0 + (length / 100.0))

        words = text.lower().split()
        max_repeat = 1
        current_repeat = 1

        for i in range(1, len(words)):
            if words[i] == words[i - 1]:
                current_repeat += 1
                max_repeat = max(max_repeat, current_repeat)
            else:
                current_repeat = 1

        repetition_factor = max(0.1, 1.0 - (max_repeat - 1) * 0.3)

        raw_score = (length_score + self.base_score) / 2.0 * repetition_factor
        return min(10.0, max(1.0, raw_score))
