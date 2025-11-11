# pylint: disable=missing-module-docstring

import pytest
import torch
from torch import Tensor

from backtracking_llm.decision import (Operator, EntropyThreshold,
                                       LogitThreshold, NGramOverlap, Never,
                                       ProbabilityDrop, ProbabilityMargin,
                                       ProbabilityThreshold, ProbabilityTrend,
                                       Repetition)

# pylint: disable=redefined-outer-name
# pylint: disable=missing-class-docstring
# pylint: disable=protected-access


@pytest.fixture
def base_logits() -> Tensor:
    return torch.tensor([-2.0, 2.5, -0.9, -0.4])


@pytest.fixture
def base_probabilities(base_logits: Tensor) -> Tensor:
    return torch.softmax(base_logits, dim=-1)


@pytest.fixture
def base_position() -> int:
    return 0


@pytest.fixture
def base_token() -> str:
    return 'hello'


@pytest.fixture
def low_margin_probabilities() -> Tensor:
    return torch.tensor([0.35, 0.32, 0.18, 0.15])


class TestNeverOperator:
    """Test suite for the Never operator."""

    def test_never_operator_always_returns_0(self, base_logits: Tensor,
                                             base_probabilities: Tensor,
                                             base_position: int,
                                             base_token: str) -> None:
        op = Never()
        result = op(base_logits, base_probabilities, base_position, base_token)

        assert result is 0


class TestProbabilityThreshold:
    """Test suite for the ProbabilityThreshold operator."""

    @pytest.mark.parametrize(('min_probability', 'expected'), [(0.5, 2),
                                                               (0.005, 0),
                                                               (0.0101, 0)])
    def test_backtrack_logic(self, base_logits: Tensor,
                             base_probabilities: Tensor, min_probability: float,
                             expected: int) -> None:
        op = ProbabilityThreshold(min_probability=min_probability,
                                  backtrack_count=2)
        result = op(base_logits, base_probabilities, 0, 'hello')
        assert result == expected

    @pytest.mark.parametrize('invalid_prob', [0.0, 1.0, -0.1, 1.1])
    def test_raises_error_for_invalid_prob(self, invalid_prob: float) -> None:
        with pytest.raises(ValueError,
                           match='`min_probability` must be between'):
            ProbabilityThreshold(min_probability=invalid_prob)


class TestEntropyThreshold:
    """Test suite for the EntropyThreshold operator."""

    @pytest.fixture
    def high_entropy_probabilities(self) -> Tensor:
        return torch.tensor([0.25, 0.25, 0.25, 0.25])

    @pytest.fixture
    def zero_entropy_probabilities(self) -> Tensor:
        return torch.tensor([1.0, 0.0, 0.0, 0.0])

    def test_triggers_backtrack_on_high_entropy(
            self, base_logits: Tensor,
            high_entropy_probabilities: Tensor) -> None:
        op = EntropyThreshold(max_entropy=1.0, backtrack_count=3)
        result = op(base_logits, high_entropy_probabilities, 0, 'hello')
        assert result == 3

    def test_no_backtrack_on_low_entropy(self, base_logits: Tensor,
                                         base_probabilities: Tensor) -> None:
        op = EntropyThreshold(max_entropy=1.0, backtrack_count=2)
        result = op(base_logits, base_probabilities, 0, 'hello')
        assert result == 0

    def test_no_backtrack_when_equal(self, base_logits: Tensor,
                                     base_probabilities: Tensor,
                                     base_position: int,
                                     base_token: str) -> None:
        entropy_val = -(base_probabilities * base_probabilities.log()).sum()
        op = EntropyThreshold(max_entropy=entropy_val.item(), backtrack_count=2)

        result = op(base_logits, base_probabilities, base_position, base_token)

        assert result == 0

    def test_handles_zero_probability_correctly(
            self, base_logits: Tensor,
            zero_entropy_probabilities: Tensor) -> None:
        op = EntropyThreshold(max_entropy=0.01)
        result = op(base_logits, zero_entropy_probabilities, 0, 'hello')
        assert result == 0

    @pytest.mark.parametrize('invalid_entropy', [-0.1, -100.0])
    def test_raises_error_for_invalid_entropy(self,
                                              invalid_entropy: float) -> None:
        with pytest.raises(ValueError,
                           match='`max_entropy` must be non-negative'):
            EntropyThreshold(max_entropy=invalid_entropy)


class TestProbabilityMargin:
    """Test suite for the ProbabilityMargin operator."""

    @pytest.fixture
    def low_margin_probabilities(self) -> Tensor:
        return torch.tensor([0.35, 0.32, 0.18, 0.15])

    @pytest.mark.parametrize(('min_margin', 'expected_result'), [(0.1, 2),
                                                                 (0.03, 0)])
    def test_backtrack_logic(self, base_logits: Tensor,
                             low_margin_probabilities: Tensor,
                             min_margin: float, expected_result: int) -> None:
        op = ProbabilityMargin(min_margin=min_margin, backtrack_count=2)
        result = op(base_logits, low_margin_probabilities, 0, 'hello')
        assert result == expected_result

    def test_handles_vocab_size_less_than_2(self, caplog,
                                            base_logits: Tensor) -> None:
        op = ProbabilityMargin()
        small_probs = torch.tensor([1.0])
        result = op(base_logits, small_probs, 0, 'hello')
        assert result == 0
        assert 'fewer than 2 elements' in caplog.text

    @pytest.mark.parametrize('invalid_margin', [-0.1, 1.1, -10.0])
    def test_raises_error_for_invalid_margin(self,
                                             invalid_margin: float) -> None:
        with pytest.raises(ValueError, match='`min_margin` must be between'):
            ProbabilityMargin(min_margin=invalid_margin)


class TestProbabilityDrop:
    """Test suite for the ProbabilityDrop operator."""

    def test_triggers_backtrack_on_sharp_drop(self,
                                              base_logits: Tensor) -> None:
        op = ProbabilityDrop(max_drop=0.5, backtrack_count=2)
        op(base_logits, torch.tensor([0.8, 0.1, 0.05, 0.05]), 0, 'hello')
        result = op(base_logits, torch.tensor([0.3, 0.3, 0.2, 0.2]), 0, 'hello')
        assert result == 2

    def test_no_backtrack_on_mild(self, base_logits: Tensor) -> None:
        op = ProbabilityDrop(max_drop=0.5)
        op(base_logits, torch.tensor([0.8, 0.1, 0.05, 0.05]), 0, 'hello')
        result = op(base_logits, torch.tensor([0.5, 0.2, 0.2, 0.1]), 0, 'hello')
        assert result == 0

    def test_no_backtrack_on_increase(self, base_logits: Tensor,
                                      base_token: str) -> None:
        op = ProbabilityDrop(max_drop=0.5)

        low_probability_distribution = torch.tensor([0.3, 0.3, 0.2, 0.2])
        high_probability_distribution = torch.tensor([0.8, 0.1, 0.05, 0.05])

        op(base_logits, low_probability_distribution, 0, base_token)

        result = op(base_logits, high_probability_distribution, 0, base_token)
        assert result == 0

    def test_no_drop_on_previous_zero(self, base_logits: Tensor,
                                      base_token: str) -> None:
        op = ProbabilityDrop(max_drop=0.5)

        zero_probability_distribution = torch.tensor([0.0, 0.5, 0.5, 0.0])
        next_probability_distribution = torch.tensor([0.1, 0.5, 0.4, 0.0])

        op(base_logits, zero_probability_distribution, 0, base_token)

        result = op(base_logits, next_probability_distribution, 0, base_token)
        assert result == 0

    @pytest.mark.parametrize('invalid_drop', [-0.1, 1.1, 10.0])
    def test_raises_error_for_invalid_drop(self, invalid_drop: float) -> None:
        with pytest.raises(ValueError, match='`max_drop` must be between'):
            ProbabilityDrop(max_drop=invalid_drop)


class TestProbabilityTrend:
    """Test suite for the ProbabilityTrend operator."""

    def test_triggers_backtrack_on_significant_drop(
            self, base_logits: Tensor) -> None:
        op = ProbabilityTrend(window_size=3,
                              drop_threshold=0.5,
                              backtrack_count=2)
        op(base_logits, torch.tensor([0.8, 0.2]), 0, 'hello')
        op(base_logits, torch.tensor([0.9, 0.1]), 0, 'hello')
        result = op(base_logits, torch.tensor([0.2, 0.8]), 0, 'hello')
        assert result == 2

    def test_does_not_backtrack_on_stable_trend(self,
                                                base_logits: Tensor) -> None:
        op = ProbabilityTrend(window_size=3, drop_threshold=0.5)
        op(base_logits, torch.tensor([0.8, 0.2]), 0, 'hello')
        op(base_logits, torch.tensor([0.7, 0.3]), 0, 'hello')
        result = op(base_logits, torch.tensor([0.9, 0.1]), 0, 'hello')
        assert result == 0

    def test_respects_warmup_period(self, base_logits: Tensor) -> None:
        op = ProbabilityTrend(window_size=5, drop_threshold=0.5)
        op(base_logits, torch.tensor([0.9, 0.1]), 0, 'hello')
        result = op(base_logits, torch.tensor([0.1, 0.9]), 0, 'hello')
        assert result == 0

    @pytest.mark.parametrize('invalid_size', [1, 0, -5])
    def test_raises_error_for_invalid_window_size(self,
                                                  invalid_size: int) -> None:
        with pytest.raises(ValueError,
                           match='`window_size` must be at least 2'):
            ProbabilityTrend(window_size=invalid_size)

    @pytest.mark.parametrize('invalid_threshold', [0.0, 1.0, -0.1, 1.1])
    def test_raises_error_for_invalid_drop_threshold(
            self, invalid_threshold: float) -> None:
        with pytest.raises(ValueError,
                           match='`drop_threshold` must be between'):
            ProbabilityTrend(drop_threshold=invalid_threshold)


class TestRepetition:
    """Test suite for the Repetition operator."""

    def test_triggers_backtrack_and_resets(self, base_logits: Tensor,
                                           base_probabilities: Tensor) -> None:
        op = Repetition(max_repetitions=2)
        assert op(base_logits, base_probabilities, 0, 'a') == 0
        assert op(base_logits, base_probabilities, 0, 'a') == 0
        assert op(base_logits, base_probabilities, 0, 'a') == 3
        assert op(base_logits, base_probabilities, 0, 'a') == 0

    def test_resets_on_different_token(self, base_logits: Tensor,
                                       base_probabilities: Tensor) -> None:
        op = Repetition(max_repetitions=2)
        op(base_logits, base_probabilities, 0, 'a')
        op(base_logits, base_probabilities, 0, 'a')
        result = op(base_logits, base_probabilities, 0, 'b')
        assert result == 0

    def test_no_backtrack_at_limit(self, base_logits: Tensor,
                                   base_probabilities: Tensor,
                                   base_position: int) -> None:
        op = Repetition(max_repetitions=3)

        op(base_logits, base_probabilities, base_position, '')
        op(base_logits, base_probabilities, base_position, '')
        result = op(base_logits, base_probabilities, base_position, '')

        assert result == 0

    def test_warmup_period(self, base_logits: Tensor,
                           base_probabilities: Tensor,
                           base_position: int) -> None:
        op = NGramOverlap(ngram_size=4)
        token_sequence = ['10', '20', '10']

        results = [
            op(base_logits, base_probabilities, base_position, token)
            for token in token_sequence
        ]

        assert results == [0, 0, 0]

    @pytest.mark.parametrize('invalid_n', [0, -1, -10])
    def test_raises_error_for_invalid_max_repetitions(self,
                                                      invalid_n: int) -> None:
        with pytest.raises(ValueError,
                           match='`max_repetitions` must be positive'):
            Repetition(max_repetitions=invalid_n)


class TestNGramOverlap:
    """Test suite for the NGramOverlap operator."""

    def test_triggers_backtrack_on_repeat(self, base_logits: Tensor,
                                          base_probabilities: Tensor) -> None:
        op = NGramOverlap(ngram_size=3, backtrack_count=2)
        token_sequence = ['1', '2', '3', '4', '1', '2', '3']
        results = [
            op(base_logits, base_probabilities, 0, token)
            for token in token_sequence
        ]
        assert results == [0, 0, 0, 0, 0, 0, 2]

    def test_does_not_backtrack_on_unique_ngrams(
            self, base_logits: Tensor, base_probabilities: Tensor) -> None:
        op = NGramOverlap(ngram_size=3)
        token_sequence = ['1', '2', '3', '4', '5', '6', '7']
        results = [
            op(base_logits, base_probabilities, 0, token)
            for token in token_sequence
        ]
        assert all(r == 0 for r in results)

    @pytest.mark.parametrize('invalid_n', [1, 0, -1])
    def test_raises_error_for_invalid_ngram_size(self, invalid_n: int) -> None:
        with pytest.raises(ValueError,
                           match='`ngram_size` must be greater than 1'):
            NGramOverlap(ngram_size=invalid_n)


class TestLogitThreshold:
    """Test suite for the LogitThreshold operator."""

    @pytest.mark.parametrize(('min_logit', 'expected'), [(-1.0, 2), (-5.0, 0),
                                                         (-2.0, 0)])
    def test_backtrack_logic(self, base_logits: Tensor,
                             base_probabilities: Tensor, min_logit: float,
                             expected: int) -> None:
        op = LogitThreshold(min_logit=min_logit, backtrack_count=2)
        result = op(base_logits, base_probabilities, 0, 'hello')
        assert result == expected


OPERATORS_WITH_INVALID_BACKTRACK = [
    (ProbabilityThreshold),
    (LogitThreshold),
    (EntropyThreshold),
    (ProbabilityMargin),
    (NGramOverlap),
    (ProbabilityDrop),
    (ProbabilityTrend),
]


@pytest.mark.parametrize('op_class', OPERATORS_WITH_INVALID_BACKTRACK)
@pytest.mark.parametrize('invalid_count', [0, -1, -10])
def test_init_raises_error_for_invalid_backtrack_count(
        op_class, invalid_count: int) -> None:
    with pytest.raises(ValueError, match='`backtrack_count` must be positive'):
        op_class(backtrack_count=invalid_count)


OPERATORS_WITH_OOB_HANDLING = [
    (ProbabilityThreshold),
    (LogitThreshold),
    (ProbabilityDrop),
    (ProbabilityTrend),
]


@pytest.mark.parametrize('op_class', OPERATORS_WITH_OOB_HANDLING)
def test_handles_out_of_bounds_position(caplog, op_class, base_logits: Tensor,
                                        base_probabilities: Tensor) -> None:
    op = op_class()
    result = op(base_logits, base_probabilities, 5, 'hello')
    assert result == 0
    assert 'out of bounds' in caplog.text


OPERATOR_TEST_CASES = [
    (ProbabilityThreshold, {
        'min_probability': 0.05,
        'backtrack_count': 1
    }, {
        'min_probability': 0.5,
        'backtrack_count': 2
    }),
    (LogitThreshold, {
        'min_logit': -20.0,
        'backtrack_count': 1
    }, {
        'min_logit': -10.0,
        'backtrack_count': 2
    }),
    (EntropyThreshold, {
        'max_entropy': 0.2,
        'backtrack_count': 2
    }, {
        'max_entropy': 1.5,
        'backtrack_count': 1
    }),
    (ProbabilityMargin, {
        'min_margin': 0.05,
        'backtrack_count': 1
    }, {
        'min_margin': 0.1,
        'backtrack_count': 2
    }),
    (Repetition, {
        'max_repetitions': 3
    }, {
        'max_repetitions': 5
    }),
    (NGramOverlap, {
        'ngram_size': 4,
        'backtrack_count': 1
    }, {
        'ngram_size': 3,
        'backtrack_count': 2
    }),
    (ProbabilityDrop, {
        'max_drop': 0.8,
        'backtrack_count': 1
    }, {
        'max_drop': 0.5,
        'backtrack_count': 2
    }),
    (ProbabilityTrend, {
        'window_size': 10,
        'drop_threshold': 0.5,
        'backtrack_count': 1
    }, {
        'window_size': 5,
        'drop_threshold': 0.8,
        'backtrack_count': 2
    }),
]


@pytest.mark.parametrize('op_class, default_args, different_args',
                         OPERATOR_TEST_CASES)
class TestDunderMethods:
    """A test suite for the common dunder methods of all operators."""

    # pylint: disable=unused-argument

    def test_repr_is_executable(self, op_class, default_args, different_args):
        op = op_class(**default_args)

        reconstructed_op = eval(repr(op))  # pylint: disable=eval-used

        assert op == reconstructed_op

    def test_equality(self, op_class, default_args, different_args):
        op_a = op_class(**default_args)
        op_b_equal = op_class(**default_args)
        op_c_different = op_class(**different_args)

        assert op_a == op_b_equal
        assert op_a != op_c_different
        assert op_a != 'a string'
        assert op_a is not None

    def test_hash_consistency(self, op_class, default_args, different_args):
        op_a = op_class(**default_args)
        op_b_equal = op_class(**default_args)

        assert hash(op_a) == hash(op_b_equal)

    def test_set_behavior(self, op_class, default_args, different_args):
        op_a = op_class(**default_args)
        op_b_equal = op_class(**default_args)
        op_c_different = op_class(**different_args)

        operator_set = {op_a, op_b_equal, op_c_different}

        assert len(operator_set) == 2
        assert op_a in operator_set
        assert op_c_different in operator_set


def test_operator_protocol_has_backtrack_method():
    assert hasattr(Operator, 'backtrack')


class TestProbabilityDropBacktrack:

    def test_backtrack_resets_last_probability(self):
        op = ProbabilityDrop(max_drop=0.5)
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
        probs = torch.tensor([0.1, 0.2, 0.3, 0.4])

        op(logits, probs, 3, 'test')
        assert (op._last_probability is not None and torch.equal(
            torch.Tensor([op._last_probability]), torch.Tensor([0.4])))

        op.backtrack(1)
        assert op._last_probability is None


class TestProbabilityTrendBacktrack:

    def test_backtrack_removes_from_history(self):
        op = ProbabilityTrend(window_size=5)
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
        probs = torch.tensor([0.1, 0.2, 0.3, 0.4])

        op(logits, probs, 0, 'a')
        op(logits, probs, 0, 'b')
        op(logits, probs, 0, 'c')
        assert len(op._history) == 3

        op.backtrack(2)
        assert len(op._history) == 1

        op.backtrack(1)
        assert len(op._history) == 0

        op.backtrack(10)
        assert len(op._history) == 0


class TestRepetitionBacktrack:

    def test_backtrack_resets_repeat_state(self):

        op = Repetition(max_repetitions=3)
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
        probs = torch.tensor([0.1, 0.2, 0.3, 0.4])

        op(logits, probs, 0, 'token')
        op(logits, probs, 0, 'token')
        assert op._repeat_count == 2
        assert op._last_token == 'token'

        op.backtrack(1)
        assert op._repeat_count == 0
        assert op._last_token is None


class TestNGramOverlapBacktrack:

    def test_backtrack_removes_ngrams(self):
        op = NGramOverlap(ngram_size=3, backtrack_count=1)
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
        probs = torch.tensor([0.1, 0.2, 0.3, 0.4])

        op(logits, probs, 0, 'a')
        op(logits, probs, 0, 'b')
        op(logits, probs, 0, 'c')

        assert len(op._seen_ngrams) == 1

        op.backtrack(1)

        assert len(op._window) == 2
        assert len(op._seen_ngrams) == 0


@pytest.mark.parametrize('operator', [
    EntropyThreshold,
    LogitThreshold,
    Never,
    ProbabilityMargin,
    ProbabilityThreshold,
])
def test_stateless_operator_backtrack(operator, base_logits, base_probabilities,
                                      base_position, base_token):
    op = operator()
    op(base_logits, base_probabilities, base_position, base_token)

    assert op.backtrack(2) is None
