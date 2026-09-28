"""Tests for eval/metrics.py (synthetic strings, no model downloads)."""

from __future__ import annotations

import pytest

from eval.metrics import corpus_bleu, distinct_2, rouge_scores, strategy_accuracy


def test_corpus_bleu_identical_is_high():
    hyps = ["the cat sat on the mat"]
    refs = ["the cat sat on the mat"]
    assert corpus_bleu(hyps, refs) > 90.0


def test_rouge_scores_identical_is_one():
    hyps = ["hello world"]
    refs = ["hello world"]
    scores = rouge_scores(hyps, refs)
    assert scores["rouge1"] == pytest.approx(1.0)


def test_distinct_2_no_repeats():
    assert distinct_2(["a b c d"]) == pytest.approx(1.0)


def test_distinct_2_all_repeats():
    assert distinct_2(["a a a a"]) == pytest.approx(1.0 / 3.0)


def test_strategy_accuracy():
    predicted = ["Question", "Reflection", "Affirmation"]
    gold = ["Question", "Affirmation", "Affirmation"]
    assert strategy_accuracy(predicted, gold) == pytest.approx(2.0 / 3.0)


def test_strategy_accuracy_ignores_missing_gold():
    predicted = ["Question", "Reflection"]
    gold = ["Question", None]
    assert strategy_accuracy(predicted, gold) == pytest.approx(1.0)


@pytest.mark.integration
def test_bertscore_smoke():
    from eval.metrics import bertscore_f1

    scores = bertscore_f1(["hello world"], ["hello world"])
    assert len(scores) == 1
    assert scores[0] > 0.9
