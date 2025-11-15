# pylint: disable=missing-module-docstring

import logging
from unittest.mock import MagicMock, patch

from httpx import Request
import pytest
from openai import APIError

from backtracking_llm.rl.config import JudgeConfig
from backtracking_llm.rl.judges import Judge, MockJudge, OpenAIJudge

# pylint: disable=protected-access
# pylint: disable=missing-class-docstring


class TestJudgeProtocol:

    def test_mock_judge_implements_protocol(self, mock_judge):
        assert isinstance(mock_judge, Judge)

    def test_openai_judge_implements_protocol(self, sample_judge_config):
        try:
            judge = OpenAIJudge(sample_judge_config)
            assert isinstance(judge, Judge)
        except ImportError:
            pytest.skip('OpenAI dependencies not available')

    def test_protocol_signature_exists(self):
        assert hasattr(Judge, 'score')

        with pytest.raises(TypeError):
            Judge()  # type: ignore


class TestMockJudge:

    def test_default_initialization(self):
        judge = MockJudge()
        assert judge.base_score == 5.0
        assert judge.repetition_penalty == 2.0

    def test_custom_initialization(self):
        judge = MockJudge(base_score=7.0, repetition_penalty=1.5)
        assert judge.base_score == 7.0
        assert judge.repetition_penalty == 1.5

    def test_score_empty_text_returns_minimum(self, mock_judge):
        assert mock_judge.score('') == 1.0
        assert mock_judge.score('   ') == 1.0

    def test_score_very_short_text_low(self, mock_judge):
        score = mock_judge.score('Hi')
        assert 1.0 <= score < 4.0

    def test_score_very_long_text_moderate(self, mock_judge):
        long_text = 'word ' * 500
        score = mock_judge.score(long_text)
        assert 1.0 <= score < 6.0

    def test_score_repetitive_text_penalized(self, mock_judge):
        repetitive = 'the the the the the test text'
        normal = 'the test text is here for scoring'

        repetitive_score = mock_judge.score(repetitive)
        normal_score = mock_judge.score(normal)

        assert repetitive_score < normal_score

    def test_score_deterministic(self, mock_judge):
        text = 'This is a test text for scoring.'

        score1 = mock_judge.score(text)
        score2 = mock_judge.score(text)
        score3 = mock_judge.score(text)

        assert score1 == score2 == score3

    def test_score_range_valid(self, mock_judge):
        test_cases = [
            '',
            'short',
            'normal length test text',
            'word ' * 100,
            'repeated repeated repeated',
            'A well-structured sentence with good flow and variety.',
        ]

        for text in test_cases:
            score = mock_judge.score(text)
            assert (1.0 <= score <=
                    10.0), f'Score {score} out of bounds for: {text}'

    def test_repetition_penalty_calculated_correctly(self):
        judge = MockJudge(base_score=5.0, repetition_penalty=3.0)

        score = judge.score('word word word test')
        assert score < 3.0

    def test_length_scoring_interpolation(self, mock_judge):
        short_score = mock_judge.score('short')
        medium_score = mock_judge.score('medium length test text here')
        long_score = mock_judge.score('very long length test text is here')

        assert short_score < medium_score
        assert medium_score < long_score
        assert long_score < 10.0


class TestOpenAIJudge:

    def test_initialization_with_config(self, sample_judge_config):
        try:
            judge = OpenAIJudge(sample_judge_config)
            assert judge.config == sample_judge_config
        except ImportError:
            pytest.skip('OpenAI dependencies not available')

    def test_initialization_default_prompt(self):
        try:
            config = JudgeConfig(model='gpt-4', api_key='asd')
            judge = OpenAIJudge(config)
            assert '{text}' in judge._prompt_template
            assert '1-10' in judge._prompt_template
        except ImportError:
            pytest.skip('OpenAI dependencies not available')

    def test_score_empty_text_logs_warning_and_returns_min(
            self, sample_judge_config, caplog):
        try:
            judge = OpenAIJudge(sample_judge_config)

            score = judge.score('')
            assert score == 0.0
            assert 'Empty text provided' in caplog.text
        except ImportError:
            pytest.skip('OpenAI dependencies not available')

    def test_score_calls_api_and_parses_response(self, sample_judge_config):
        try:
            judge = OpenAIJudge(sample_judge_config)

            mock_response = MagicMock()
            mock_response.choices[0].message.content = '8.5'

            with patch.object(judge.client.chat.completions,
                              'create',
                              return_value=mock_response):
                score = judge.score('Test text for scoring')
                assert score == 8.5

                judge.client.chat.completions.create.assert_called_once()
                call_args = judge.client.chat.completions.create.call_args
                assert call_args.kwargs['model'] == 'gpt-3.5-turbo'
                assert call_args.kwargs['temperature'] == 0.0
        except ImportError:
            pytest.skip('OpenAI dependencies not available')

    def test_score_retries_on_failure(self, sample_judge_config):
        try:
            judge = OpenAIJudge(sample_judge_config)

            mock_response = MagicMock()
            mock_response.choices[0].message.content = '7.0'

            with patch.object(judge.client.chat.completions,
                              'create',
                              side_effect=[
                                  Exception('API Error'),
                                  Exception('API Error'), mock_response
                              ]):
                score = judge.score('Test text')
                assert score == 7.0
                assert judge.client.chat.completions.create.call_count == 3
        except ImportError:
            pytest.skip('OpenAI dependencies not available')

    def test_score_retry_exhaustion_raises(self, sample_judge_config):
        try:
            judge = OpenAIJudge(sample_judge_config)

            with patch.object(judge.client.chat.completions,
                              'create',
                              side_effect=Exception('Persistent API Error')):
                with pytest.raises(Exception, match='Persistent API Error'):
                    judge.score('Test text')
        except ImportError:
            pytest.skip('OpenAI dependencies not available')

    def test_parse_score_none(self):
        judge = OpenAIJudge(JudgeConfig(model='gpt-4', api_key='asd'))

        with pytest.raises(ValueError, match='Could not parse'):
            judge._parse_score(None)

    def test_parse_score_numeric_variants(self):
        try:
            judge = OpenAIJudge(JudgeConfig(model='gpt-4', api_key='asd'))

            assert judge._parse_score('8.5') == 8.5
            assert judge._parse_score('8') == 8.0
            assert judge._parse_score('10') == 10.0

            assert judge._parse_score('  7.5  ') == 7.5

            assert judge._parse_score('Score: 9.0') == 9.0
            assert judge._parse_score('The score is 6.5') == 6.5
        except ImportError:
            pytest.skip('OpenAI dependencies not available')

    def test_parse_score_clips_to_valid_range(self):
        try:
            judge = OpenAIJudge(JudgeConfig(model='gpt-4', api_key='asd'))

            assert judge._parse_score('15') == 10.0
            assert judge._parse_score('11.5') == 10.0
            assert judge._parse_score('0.5') == 1.0
        except ImportError:
            pytest.skip('OpenAI dependencies not available')

    def test_parse_score_out_of_10_format_variants(self):
        try:
            judge = OpenAIJudge(JudgeConfig(model='gpt-4', api_key='asd'))

            assert judge._parse_score('7 out of 10') == 7.0

            assert judge._parse_score('8.5 out of 10') == 8.5

            assert judge._parse_score('9/10') == 9.0
            assert judge._parse_score('7.5 / 10') == 7.5

            assert judge._parse_score('I give it 6 out of 10') == 6.0
        except ImportError:
            pytest.skip('OpenAI dependencies not available')

    def test_parse_score_invalid_raises_valueerror(self):
        try:
            judge = OpenAIJudge(JudgeConfig(model='gpt-4', api_key='asd'))

            invalid_responses = [
                'invalid response',
                'no numbers here!',
                'excellent'
                'score: ten',
            ]

            for response in invalid_responses:
                with pytest.raises(ValueError, match='Could not parse'):
                    judge._parse_score(response)
        except ImportError:
            pytest.skip('OpenAI dependencies not available')

    def test_parse_score_empty_raises(self):
        try:
            judge = OpenAIJudge(JudgeConfig(model='gpt-4', api_key='asd'))

            with pytest.raises(ValueError, match='Could not parse'):
                judge._parse_score('')
        except ImportError:
            pytest.skip('OpenAI dependencies not available')

    def test_default_prompt_includes_key_elements(self):
        judge = OpenAIJudge(JudgeConfig(model='gpt-4', api_key='asd'))
        prompt = judge._prompt_template

        assert '{text}' in prompt
        assert '1-10' in prompt or '1 to 10' in prompt
        assert 'score' in prompt.lower()

    def test_api_error_logging(self, sample_judge_config, caplog):
        try:
            judge = OpenAIJudge(sample_judge_config)

            with patch.object(
                    judge.client.chat.completions,
                    'create',
                    side_effect=APIError('Test error',
                                         request=Request(method='GET',
                                                         url='localhost'),
                                         body=None)), pytest.raises(APIError):
                with caplog.at_level(logging.ERROR):
                    judge.score('Test text')

                assert 'Judge API call failed' in caplog.text
        except ImportError:
            pytest.skip('OpenAI dependencies not available')
