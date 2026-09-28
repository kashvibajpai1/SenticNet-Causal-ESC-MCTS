"""Evaluation metrics: BLEU, ROUGE, BERTScore, Distinct-2, strategy accuracy.

None of this existed before this pass — the paper reportedly reported these
metrics but no evaluation code existed anywhere in the repository.
"""

from __future__ import annotations

from collections.abc import Sequence

import sacrebleu
from rouge_score import rouge_scorer

_ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def corpus_bleu(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    bleu = sacrebleu.corpus_bleu(list(hypotheses), [list(references)])
    return float(bleu.score)


def sentence_bleu_scores(hypotheses: Sequence[str], references: Sequence[str]) -> list[float]:
    return [float(sacrebleu.sentence_bleu(h, [r]).score) for h, r in zip(hypotheses, references)]


def rouge_scores(hypotheses: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = max(len(hypotheses), 1)
    for h, r in zip(hypotheses, references):
        s = _ROUGE_SCORER.score(r, h)
        for k in totals:
            totals[k] += s[k].fmeasure
    return {k: v / n for k, v in totals.items()}


def per_example_rougeL(hypotheses: Sequence[str], references: Sequence[str]) -> list[float]:
    return [_ROUGE_SCORER.score(r, h)["rougeL"].fmeasure for h, r in zip(hypotheses, references)]


def bertscore_f1(
    hypotheses: Sequence[str],
    references: Sequence[str],
    *,
    model_type: str = "distilbert-base-uncased",
) -> list[float]:
    """BERTScore F1 per example. Uses distilbert (not the library default
    roberta-large) to keep CPU wall-clock reasonable."""
    from bert_score import score as bert_score_fn

    if not hypotheses:
        return []
    _, _, f1 = bert_score_fn(
        list(hypotheses),
        list(references),
        model_type=model_type,
        device="cpu",
        verbose=False,
    )
    return [float(x) for x in f1]


def distinct_2(hypotheses: Sequence[str]) -> float:
    bigrams: set[tuple[str, str]] = set()
    total = 0
    for h in hypotheses:
        toks = h.split()
        for i in range(len(toks) - 1):
            bigrams.add((toks[i], toks[i + 1]))
            total += 1
    return len(bigrams) / total if total > 0 else 0.0


def strategy_accuracy(predicted: Sequence[str | None], gold: Sequence[str | None]) -> float:
    pairs = [(p, g) for p, g in zip(predicted, gold) if g is not None]
    if not pairs:
        return 0.0
    correct = sum(1 for p, g in pairs if p == g)
    return correct / len(pairs)
