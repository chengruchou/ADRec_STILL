import math

import pytest
import torch

from adrec_still.evaluation.exact import dense_exact_contributions, exact_ranking_contributions
from adrec_still.evaluation.ranking import ranking_contributions


@pytest.mark.parametrize("chunk", [1, 2, 7, 50])
@pytest.mark.parametrize("reverse", [False, True])
def test_chunked_equals_independent_dense_reference_including_ties(chunk, reverse):
    logits = torch.tensor(
        [
            [999.0, 999.0, 3.0, 3.0, 1.0, 3.0, -1.0],
            [999.0, 999.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [999.0, 999.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        ],
        dtype=torch.float64,
    )
    target = torch.tensor([5, 4, 1])
    known, observed = torch.tensor([True, True, False]), torch.ones(3, dtype=torch.bool)
    requested = []

    def score(ids):
        requested.extend(ids.tolist())
        return logits[:, ids]

    result = exact_ranking_contributions(
        score,
        target,
        known,
        observed,
        7,
        score_kind="logits",
        chunk_size=chunk,
        ks=(1, 2, 3, 5, 10, 20),
        candidate_order=torch.arange(6, 1, -1) if reverse else None,
    )
    assert sorted(requested) == list(range(2, 7))
    # Python lexicographic sort is independent of the streaming torch implementation.
    expected_ids = [sorted(range(2, 7), key=lambda poi: (-float(row[poi]), poi)) for row in logits]
    assert result["top_ids"].tolist() == expected_ids
    for i in range(2):
        rank = expected_ids[i].index(int(target[i])) + 1
        for k in (1, 2, 3, 5, 10, 20):
            assert result["values"][f"recall@{k}"][i] == (rank <= k)
            assert float(result["values"][f"ndcg@{k}"][i]) == pytest.approx(
                1 / math.log2(rank + 1) if rank <= k else 0
            )
        nll = torch.logsumexp(logits[i, 2:], 0) - logits[i, target[i]]
        assert result["values"]["poi_nll"][i] == pytest.approx(nll)
    assert all(values[2] == 0 for values in result["values"].values())
    assert result["resource_cost"]["peak_score_elements"] <= 3 * min(chunk, 5)


def test_no_ties_agrees_with_existing_v02_and_shared_globalpop_vector():
    logits = torch.tensor([1e9, 1e9, 0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    targets = torch.tensor([2, 3, 4, 5, 1, 0])
    observed = targets != 0
    known = targets >= 2
    old = ranking_contributions(logits, targets, known, observed)
    new = exact_ranking_contributions(
        lambda ids: logits[ids], targets, known, observed, 6, score_kind="logits", chunk_size=2
    )
    for key in old["values"]:
        torch.testing.assert_close(new["values"][key], old["values"][key])
    assert new["random_floor"] == old["random_floor"]


@pytest.mark.parametrize("kind", ["logits", "scores", "probabilities", "log_probabilities"])
def test_empty_eligible_set_never_calls_scorer(kind):
    def fail(ids):
        raise AssertionError("scorer should not be called")

    result = exact_ranking_contributions(
        fail,
        torch.tensor([1, 0]),
        torch.tensor([False, False]),
        torch.tensor([True, False]),
        10,
        score_kind=kind,
    )
    assert not result["eligible_mask"].any()
    assert result["resource_cost"]["callback_calls"] == 0


def test_zero_probability_is_infinite_nll_and_score_only_is_unsupported():
    target = torch.tensor([2])
    yes = torch.tensor([True])
    for values, kind in (
        ([999.0, 999.0, 0.0, 1.0], "probabilities"),
        ([999.0, 999.0, -math.inf, 0.0], "log_probabilities"),
    ):
        result = dense_exact_contributions(
            torch.tensor([values]), target, yes, yes, score_kind=kind
        )
        assert torch.isposinf(result["values"]["poi_nll"]).all()
    result = dense_exact_contributions(torch.zeros(1, 4), target, yes, yes, score_kind="scores")
    assert not result["poi_nll_supported"] and result["poi_nll_reason"]


@pytest.mark.parametrize(
    "values,kind",
    [
        ([0.2, 0.2], "probabilities"),
        ([-1.0, -1.0], "log_probabilities"),
        ([math.nan, 0.0], "logits"),
        ([math.inf, 0.0], "logits"),
        ([-math.inf, -math.inf], "logits"),
    ],
)
def test_invalid_probability_forecasts_are_not_silently_repaired(values, kind):
    with pytest.raises(ValueError):
        dense_exact_contributions(
            torch.tensor([[0.0, 0.0, *values]]),
            torch.tensor([2]),
            torch.tensor([True]),
            torch.tensor([True]),
            score_kind=kind,
        )


def test_candidate_permutation_cannot_omit_or_duplicate_ids():
    with pytest.raises(ValueError, match="permutation"):
        exact_ranking_contributions(
            lambda ids: torch.zeros(1, len(ids)),
            torch.tensor([2]),
            torch.tensor([True]),
            torch.tensor([True]),
            5,
            score_kind="logits",
            candidate_order=torch.tensor([2, 2, 4]),
        )
