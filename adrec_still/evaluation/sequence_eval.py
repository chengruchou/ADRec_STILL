"""Exact full-catalog evaluation for sequence corpora (the ``dataset.pkl`` path).

Reuses the exact chunked ranking adapter: identical tie policy, float64
normalization and chunking as every other backend. The upstream protocol is
kept: validation inputs are the train sequences, test inputs train+validation;
one held-out target per user. All targets are real catalog items (the pkl
vocabulary covers every observed ID), so every row is ranking-eligible.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from adrec_still.data.sequences import SequenceCorpus, SequenceEvalDataset
from adrec_still.evaluation.exact import exact_ranking_contributions

RANKING_KEYS = tuple(
    f"{metric}@{k}" for metric in ("recall", "ndcg", "mrr") for k in (5, 10, 20)
) + ("poi_nll",)


def evaluate_sequence_corpus(
    model,
    corpus: SequenceCorpus,
    split: str,
    *,
    batch_size: int = 128,
    chunk_size: int = 16384,
    max_batches: int | None = None,
    progress=None,
) -> dict[str, Any]:
    """One streaming pass over a sequence split; count-weighted means."""
    if split == "validation":
        inputs, answers = corpus.train, corpus.validation
    elif split == "test":
        inputs = [train + val for train, val in zip(corpus.train, corpus.validation)]
        answers = corpus.test
    else:
        raise ValueError(f"unsupported sequence split: {split}")
    if not any(answers):
        raise ValueError(f"the corpus has no {split} answers")
    dataset = SequenceEvalDataset(inputs, answers, model.max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    started = time.perf_counter()
    totals = dict.fromkeys(RANKING_KEYS, 0.0)
    rows = 0
    batches = 0
    for history, answer_pad in loader:
        if max_batches is not None and batches >= max_batches:
            break
        targets = answer_pad[:, -1].long()
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                context = model.encode_sequences(history.to(model.device))
        finally:
            model.train(was_training)

        def score(ids: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return model.score_candidates(
                    context, ids.to(model.device).expand(len(context), -1)
                )

        known = targets >= 2
        observed = torch.ones_like(known)
        result = exact_ranking_contributions(
            score, targets, known, observed, model.number_of_pois,
            score_kind="logits", chunk_size=chunk_size,
        )
        eligible = result["eligible_mask"]
        count = int(eligible.sum())
        for key in RANKING_KEYS:
            totals[key] += float(result["values"][key][eligible].sum())
        rows += count
        batches += 1
        if progress is not None:
            progress(rows, len(dataset))
    summary: dict[str, Any] = {
        "split": split,
        "rows": rows,
        "eligible_rows": rows,
        "candidate_count": model.number_of_pois - 2,
        "evaluator_version": "exact-sequence-v1",
        "seconds": time.perf_counter() - started,
    }
    for key in RANKING_KEYS:
        summary[key] = totals[key] / rows if rows else None
    return summary
