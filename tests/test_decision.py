# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import logging

import pytest
import torch
from torch import Tensor

from backtracking_llm.decision import (
    PROBABILITIES,
    LOGITS,
    CHOSEN_OUTPUT,
    CHOSEN_INDEX,
    InvalidHyperparameterError,
    MissingContextDataError,
    Context,
    EntropyThreshold,
    LogitThreshold,
    NGramOverlap,
    ProbabilityDrop,
    ProbabilityMargin,
    ProbabilityThreshold,
    ProbabilityTrend,
    Repetition,
)


@pytest.fixture
def base_logits() -> Tensor:
    return torch.tensor([-2.0, 2.5, -0.9, -0.4])


@pytest.fixture
def base_probabilities(base_logits: Tensor) -> Tensor:
    return torch.softmax(base_logits, dim=-1)


def test_default_reset_do_not_fail():
    operator = ProbabilityThreshold()
    try:
        operator.reset()
    except Exception as e:
        pytest.fail(f"Default reset method raised an unexpected error: {e}")


def test_missing_key_raises_custom_error():
    operator = ProbabilityThreshold()
    context = Context()

    with pytest.raises(MissingContextDataError):
        operator.decide(context)


class TestProbabilityThreshold:

    @pytest.mark.parametrize(
        "minimum_probability, backtrack_count, should_raise",
        [
            (0.1, 1, False),
            (0.99, 10, False),
            (1.0, 1, True),
            (0.0, 1, True),
            (0.1, 0, True),
        ],
    )
    def test_init(self, minimum_probability, backtrack_count, should_raise):
        if should_raise:
            with pytest.raises(InvalidHyperparameterError):
                ProbabilityThreshold(minimum_probability, backtrack_count)
        else:
            delta = ProbabilityThreshold(minimum_probability, backtrack_count)
            assert delta.minimum_probability == minimum_probability
            assert delta.backtrack_count == backtrack_count

    def test_call_triggers_backtrack(self, base_probabilities):
        operator = ProbabilityThreshold(0.05, 3)
        context = Context.from_data({
            PROBABILITIES: base_probabilities,
            CHOSEN_INDEX: 0
        })

        result = operator.decide(context)

        assert result.should_backtrack is True
        assert result.steps_to_remove == 3

    def test_call_no_backtrack(self, base_probabilities):
        operator = ProbabilityThreshold(0.05, 3)
        context = Context.from_data({
            PROBABILITIES: base_probabilities,
            CHOSEN_INDEX: 1
        })

        result = operator.decide(context)

        assert result.should_backtrack is False

    def test_call_out_of_bounds_index(self, base_probabilities, caplog):
        operator = ProbabilityThreshold()
        context = Context.from_data({
            PROBABILITIES: base_probabilities,
            CHOSEN_INDEX: 99
        })

        with caplog.at_level(logging.WARNING):
            result = operator.decide(context)

        assert result.should_backtrack is False
        assert "out of bounds" in caplog.text


class TestEntropyThreshold:

    @pytest.mark.parametrize(
        "maximum_entropy, backtrack_count, should_raise",
        [
            (2.5, 2, False),
            (0.0, 1, True),
            (2.5, 0, True),
        ],
    )
    def test_init(self, maximum_entropy, backtrack_count, should_raise):
        if should_raise:
            with pytest.raises(InvalidHyperparameterError):
                EntropyThreshold(maximum_entropy, backtrack_count)
        else:
            delta = EntropyThreshold(maximum_entropy, backtrack_count)
            assert delta.maximum_entropy == maximum_entropy
            assert delta.backtrack_count == backtrack_count

    def test_call_triggers_backtrack(self):
        high_entropy_probabilities = torch.tensor([0.25, 0.25, 0.25, 0.25])
        operator = EntropyThreshold(1.0, 2)
        context = Context.from_data({
            PROBABILITIES: high_entropy_probabilities,
        })

        result = operator.decide(context)

        assert result.should_backtrack is True
        assert result.steps_to_remove == 2

    def test_call_no_backtrack(self, base_probabilities):
        operator = EntropyThreshold(1.0, 2)
        context = Context.from_data({PROBABILITIES: base_probabilities})

        result = operator.decide(context)

        assert result.should_backtrack is False


class TestProbabilityMargin:

    @pytest.mark.parametrize(
        "minimum_margin, backtrack_count, should_raise",
        [
            (0.5, 2, False),
            (-0.5, 1, True),
            (0.5, -1, True),
        ],
    )
    def test_init(self, minimum_margin, backtrack_count, should_raise):
        if should_raise:
            with pytest.raises(InvalidHyperparameterError):
                ProbabilityMargin(minimum_margin, backtrack_count)
        else:
            delta = ProbabilityMargin(minimum_margin, backtrack_count)
            assert delta.minimum_margin == minimum_margin
            assert delta.backtrack_count == backtrack_count

    def test_call_triggers_backtrack(self):
        probabilities_with_close_top_k = torch.tensor([0.4, 0.38, 0.12, 0.1])
        operator = ProbabilityMargin(0.05, 1)
        context = Context.from_data(
            {PROBABILITIES: probabilities_with_close_top_k})

        result = operator.decide(context)

        assert result.should_backtrack is True
        assert result.steps_to_remove == 1

    def test_call_no_backtrack(self, base_probabilities):
        operator = ProbabilityMargin(0.1, 1)
        context = Context.from_data({PROBABILITIES: base_probabilities})

        result = operator.decide(context)

        assert result.should_backtrack is False

    def test_call_small_vocab(self, caplog):
        small_vocab_probabilities = torch.tensor([1.0])
        operator = ProbabilityMargin()
        context = Context.from_data({PROBABILITIES: small_vocab_probabilities})

        with caplog.at_level(logging.WARNING):
            result = operator.decide(context)

        assert result.should_backtrack is False
        assert "< 2" in caplog.text


class TestProbabilityDrop:

    @pytest.mark.parametrize(
        "minimum_margin, backtrack_count, should_raise",
        [
            (2, 2, False),
            (0, 1, True),
            (1, -1, True),
        ],
    )
    def test_init(self, minimum_margin, backtrack_count, should_raise):
        if should_raise:
            with pytest.raises(InvalidHyperparameterError):
                ProbabilityDrop(minimum_margin, backtrack_count)
        else:
            operator = ProbabilityDrop(minimum_margin, backtrack_count)
            assert operator.minimum_margin == minimum_margin
            assert operator.backtrack_count == backtrack_count

    def test_call_no_backtrack_on_first_step(self, base_probabilities):
        operator = ProbabilityDrop()
        context = Context.from_data({
            PROBABILITIES: base_probabilities,
            CHOSEN_INDEX: 1
        })

        result = operator.decide(context)

        assert result.should_backtrack is False
        assert operator._last_probability is not None
        assert operator._last_probability == pytest.approx(
            base_probabilities[1].item())

    def test_call_triggers_backtrack_on_sharp_drop(self):
        operator = ProbabilityDrop(2.0, 2)
        high_probabilities = torch.tensor([0.1, 0.8, 1.0])
        context = Context.from_data({
            PROBABILITIES: high_probabilities,
            CHOSEN_INDEX: 1
        })

        operator.decide(context)

        assert operator._last_probability == pytest.approx(0.8)

        low_probabilities = torch.tensor([0.3, 0.4, 0.3])
        context = Context.from_data({
            PROBABILITIES: low_probabilities,
            CHOSEN_INDEX: 0
        })

        result = operator.decide(context)

        assert result.should_backtrack is True
        assert result.steps_to_remove == 2

    def test_state_resets_after_backtrack_trigger(self):
        operator = ProbabilityDrop(2.0, 1)
        context = Context.from_data({
            PROBABILITIES: torch.tensor([0.8, 0.2]),
            CHOSEN_INDEX: 0
        })

        operator.decide(context)

        assert operator._last_probability is not None

        context = Context.from_data({
            PROBABILITIES: torch.tensor([0.3, 0.7]),
            CHOSEN_INDEX: 0
        })
        operator.decide(context)

        assert operator._last_probability is None

    def test_call_no_backtrack_on_small_drop(self):
        operator = ProbabilityDrop(2.0, 1)
        context = Context.from_data({
            PROBABILITIES: torch.tensor([0.8, 0.2]),
            CHOSEN_INDEX: 0
        })

        operator.decide(context)

        assert operator._last_probability == pytest.approx(0.8)

        context = Context.from_data({
            PROBABILITIES: torch.tensor([0.5, 0.5]),
            CHOSEN_INDEX: 0
        })

        result = operator.decide(context)

        assert result.should_backtrack is False

        assert operator._last_probability == pytest.approx(0.5)

    def test_call_no_backtrack_on_increase(self):
        operator = ProbabilityDrop(2.0, 1)
        context = Context.from_data({
            PROBABILITIES: torch.tensor([0.3, 0.7]),
            CHOSEN_INDEX: 0
        })

        operator.decide(context)

        assert operator._last_probability == pytest.approx(0.3)

        context = Context.from_data({
            PROBABILITIES: torch.tensor([0.8, 0.2]),
            CHOSEN_INDEX: 0
        })

        result = operator.decide(context)

        assert result.should_backtrack is False
        assert operator._last_probability == pytest.approx(0.8)

    def test_call_out_of_bounds_index_resets_state(self, base_probabilities,
                                                   caplog):
        operator = ProbabilityDrop()
        context = Context.from_data({
            PROBABILITIES: base_probabilities,
            CHOSEN_INDEX: 0
        })

        operator.decide(context)

        assert operator._last_probability is not None

        context = Context.from_data({
            PROBABILITIES: base_probabilities,
            CHOSEN_INDEX: 99
        })

        with caplog.at_level(logging.WARNING):
            result = operator.decide(context)

        assert result.should_backtrack is False
        assert "out of bounds" in caplog.text
        assert operator._last_probability is None


class TestProbabilityTrend:

    @pytest.mark.parametrize(
        "window_size, minimum_margin, backtrack_count, should_raise",
        [
            (10, 0.5, 2, False),
            (1, 0.5, 2, True),
            (10, 1.5, 2, True),
            (10, 0.5, 0, True),
        ],
    )
    def test_init(self, window_size, minimum_margin, backtrack_count,
                  should_raise):
        if should_raise:
            with pytest.raises(InvalidHyperparameterError):
                ProbabilityTrend(window_size, minimum_margin, backtrack_count)
        else:
            operator = ProbabilityTrend(window_size, minimum_margin,
                                        backtrack_count)
            assert operator.window_size == window_size
            assert operator.minimum_margin == minimum_margin
            assert operator.backtrack_count == backtrack_count

    def test_call_state_management(self):
        operator = ProbabilityTrend(4, 0.5, 2)
        context = Context.from_data({
            PROBABILITIES: torch.tensor([0.9, 0.1]),
            CHOSEN_INDEX: 0
        })

        result = operator.decide(context)

        assert result.should_backtrack is False
        assert len(operator._history_window) == 1
        assert operator._history_window == pytest.approx([0.9])

        context = Context.from_data({
            PROBABILITIES: torch.tensor([0.8, 0.2]),
            CHOSEN_INDEX: 0
        })

        result = operator.decide(context)

        assert result.should_backtrack is False
        assert len(operator._history_window) == 2
        assert operator._history_window == pytest.approx([0.9, 0.8])

    def test_call_backtrack_on_drop(self):
        operator = ProbabilityTrend(2, 0.5, 1)
        context = Context.from_data({
            PROBABILITIES: torch.tensor([0.9, 0.1]),
            CHOSEN_INDEX: 0
        })

        operator.decide(context)

        context = Context.from_data({
            PROBABILITIES: torch.tensor([0.8, 0.2]),
            CHOSEN_INDEX: 0
        })

        operator.decide(context)

        assert len(operator._history_window) == 2
        assert operator._history_window == pytest.approx([0.9, 0.8])

        context = Context.from_data({
            PROBABILITIES: torch.tensor([0.1, 0.9]),
            CHOSEN_INDEX: 0
        })

        result = operator.decide(context)

        assert result.should_backtrack is True
        assert result.steps_to_remove == 1
        assert len(operator._history_window) == 2
        assert operator._history_window == pytest.approx([0.9, 0.8])

    def test_call_no_backtrack_on_increase(self):
        operator = ProbabilityTrend(2, 0.5, 1)
        context = Context.from_data({
            PROBABILITIES: torch.tensor([0.9, 0.1]),
            CHOSEN_INDEX: 0
        })

        operator.decide(context)

        context = Context.from_data({
            PROBABILITIES: torch.tensor([0.8, 0.2]),
            CHOSEN_INDEX: 0
        })

        operator.decide(context)

        assert len(operator._history_window) == 2
        assert operator._history_window == pytest.approx([0.9, 0.8])

        context = Context.from_data({
            PROBABILITIES: torch.tensor([0.1, 0.9]),
            CHOSEN_INDEX: 1
        })

        result = operator.decide(context)

        assert result.should_backtrack is False

    def test_call_no_backtrack_rolls_window(self):
        operator = ProbabilityTrend(2, 0.5, 1)
        context = Context.from_data({
            PROBABILITIES: torch.tensor([0.9, 0.1]),
            CHOSEN_INDEX: 0,
        })

        operator.decide(context)

        context = Context.from_data({
            PROBABILITIES: torch.tensor([0.8, 0.2]),
            CHOSEN_INDEX: 0,
        })

        operator.decide(context)

        assert len(operator._history_window) == 2
        assert operator._history_window == pytest.approx([0.9, 0.8])

        context = Context.from_data({
            PROBABILITIES: torch.tensor([0.1, 0.9]),
            CHOSEN_INDEX: 1,
        })

        operator.decide(context)

        assert len(operator._history_window) == 2
        assert operator._history_window == pytest.approx([0.8, 0.9])

    def test_reset_clears_window(self, base_probabilities):
        operator = ProbabilityTrend(2, 0.5, 1)
        context = Context.from_data({
            PROBABILITIES: base_probabilities,
            CHOSEN_INDEX: 0
        })

        operator.decide(context)
        operator.decide(context)

        assert len(operator._history_window) == 2

        operator.reset()

        assert len(operator._history_window) == 0


class TestRepetition:

    @pytest.mark.parametrize(
        "max_repeats, should_raise",
        [
            (3, False),
            (0, True),
        ],
    )
    def test_init(self, max_repeats, should_raise):
        if should_raise:
            with pytest.raises(InvalidHyperparameterError):
                Repetition(max_repeats)
        else:
            operator = Repetition(max_repeats)
            assert operator.maximum_repeats == max_repeats

    def test_call_state_management(self):
        operator = Repetition(3)
        context = Context.from_data({
            CHOSEN_OUTPUT: 1,
        })

        result = operator.decide(context)

        assert result.should_backtrack is False
        assert operator._last_chosen_output == 1
        assert operator._number_of_repeats == 1

        result = operator.decide(context)

        assert result.should_backtrack is False
        assert operator._last_chosen_output == 1
        assert operator._number_of_repeats == 2

    def test_call_repetition_triggers_backtrack(self):
        operator = Repetition(2)
        context = Context.from_data({CHOSEN_OUTPUT: 1})

        result = operator.decide(context)
        assert result.should_backtrack is False
        assert operator._number_of_repeats == 1

        result = operator.decide(context)

        assert result.should_backtrack is False
        assert operator._number_of_repeats == 2

        result = operator.decide(context)

        assert result.should_backtrack is True
        assert result.steps_to_remove == 3

        assert operator._number_of_repeats == 0

    def test_call_counter_resets_on_new_token(self):
        operator = Repetition(2)
        context = Context.from_data({CHOSEN_OUTPUT: 0})

        operator.decide(context)

        assert operator._number_of_repeats == 1
        assert operator._last_chosen_output == 0

        context = Context.from_data({CHOSEN_OUTPUT: 1})

        operator.decide(context)

        assert operator._number_of_repeats == 1
        assert operator._last_chosen_output == 1

    def test_reset_clears_window(self):
        operator = Repetition(2)
        context = Context.from_data({CHOSEN_OUTPUT: 0})

        operator.decide(context)
        operator.decide(context)

        assert operator._number_of_repeats == 2

        operator.reset()

        assert operator._number_of_repeats == 0


class TestNGramOverlap:

    @pytest.mark.parametrize(
        "ngram_size, should_raise",
        [
            (3, False),
            (0, True),
        ],
    )
    def test_init(self, ngram_size, should_raise):
        if should_raise:
            with pytest.raises(InvalidHyperparameterError):
                NGramOverlap(ngram_size)
        else:
            operator = NGramOverlap(ngram_size)
            assert operator.ngram_size == ngram_size

    def test_call_state_management(self):
        operator = NGramOverlap(2)
        outputs = [10, 20, 30]

        context = Context.from_data({CHOSEN_OUTPUT: outputs[0]})

        result = operator.decide(context)

        assert result.should_backtrack is False
        assert operator._current_ngram == pytest.approx([10])

        context = Context.from_data({CHOSEN_OUTPUT: outputs[1]})

        result = operator.decide(context)

        assert result.should_backtrack is False
        assert operator._current_ngram == pytest.approx([10, 20])

        context = Context.from_data({CHOSEN_OUTPUT: outputs[2]})

        result = operator.decide(context)

        assert result.should_backtrack is False
        assert list(operator._seen_ngrams) == pytest.approx([(20, 30),
                                                             (10, 20)])
        assert operator._current_ngram == pytest.approx([20, 30])

    def test_call_backtrack_on_ngram_repeat(self):
        operator = NGramOverlap(2)
        outputs = [10, 20, 30, 10, 20]

        context = Context.from_data({CHOSEN_OUTPUT: outputs[0]})

        result = operator.decide(context)

        assert result.should_backtrack is False
        assert len(operator._current_ngram) == 1

        context = Context.from_data({CHOSEN_OUTPUT: outputs[1]})

        result = operator.decide(context)

        assert result.should_backtrack is False
        assert len(operator._current_ngram) == 2

        context = Context.from_data({CHOSEN_OUTPUT: outputs[2]})

        result = operator.decide(context)

        assert result.should_backtrack is False
        assert len(operator._current_ngram) == 2

        context = Context.from_data({CHOSEN_OUTPUT: outputs[3]})

        result = operator.decide(context)

        assert result.should_backtrack is False
        assert len(operator._current_ngram) == 2

        context = Context.from_data({CHOSEN_OUTPUT: outputs[4]})

        result = operator.decide(context)

        assert result.should_backtrack is True
        assert result.steps_to_remove == 2

        assert len(operator._current_ngram) == 0

    def test_call_no_backtrack_on_unique_sequence(self):
        operator = NGramOverlap(3)
        outputs = [10, 20, 30, 40, 50, 60]

        for output in outputs:
            context = Context.from_data({CHOSEN_OUTPUT: output})

            result = operator.decide(context)

            assert result.should_backtrack is False


class TestLogitThreshold:

    @pytest.mark.parametrize(
        "minimum_logit, backtrack_count, should_raise",
        [
            (20, 2, False),
            (20, 0, True),
        ],
    )
    def test_init(self, minimum_logit, backtrack_count, should_raise):
        if should_raise:
            with pytest.raises(InvalidHyperparameterError):
                LogitThreshold(minimum_logit, backtrack_count)
        else:
            operator = LogitThreshold(minimum_logit, backtrack_count)
            assert operator.minimum_logit == minimum_logit
            assert operator.backtrack_count == backtrack_count

    def test_call_triggers_backtrack(self, base_logits):
        operator = LogitThreshold(1, 3)
        context = Context.from_data({LOGITS: base_logits, CHOSEN_INDEX: 0})

        result = operator.decide(context)

        assert result.should_backtrack is True
        assert result.steps_to_remove == 3

    def test_call_no_backtrack(self, base_logits):
        operator = LogitThreshold(-3, 3)
        context = Context.from_data({LOGITS: base_logits, CHOSEN_INDEX: 0})

        result = operator.decide(context)
        assert result.should_backtrack is False

    def test_call_out_of_bounds_index(self, base_logits, caplog):
        operator = LogitThreshold(1, 3)
        context = Context.from_data({LOGITS: base_logits, CHOSEN_INDEX: 99})

        with caplog.at_level(logging.WARNING):
            result = operator.decide(context)

        assert result.should_backtrack is False
        assert "out of bounds" in caplog.text
