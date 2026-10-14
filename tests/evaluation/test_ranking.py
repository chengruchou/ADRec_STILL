import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from adrec_still.evaluation.ranking import (
    analytical_random_floors,
    full_catalog_candidate_ids,
    ranking_metrics,
)

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "evaluation_v02_cases.json"


def test_ranking_metrics_match_hand_calculation_and_mask_censoring():
    logits = torch.tensor(
        [
            [0.0, 100.0, 3.0, 2.0, 1.0],
            [0.0, 100.0, 1.0, 3.0, 2.0],
            [9.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    targets = torch.tensor([2, 4, 0])
    observed = torch.tensor([True, True, False])
    known = torch.tensor([True, True, False])
    metrics = ranking_metrics(
        logits,
        targets,
        target_in_train_catalog=known,
        observed_mask=observed,
        ks=(1, 2),
    )
    assert metrics["sample_count"] == 2
    assert metrics["cold_start_target_count"] == 0
    assert metrics["ranking_coverage"] == 1.0
    assert metrics["recall@1"] == pytest.approx(0.5)
    assert metrics["recall@2"] == pytest.approx(1.0)
    assert metrics["ndcg@2"] == pytest.approx((1.0 + 1.0 / torch.log2(torch.tensor(3.0))) / 2)
    assert metrics["mrr@1"] == pytest.approx(0.5)
    assert metrics["mrr@2"] == pytest.approx(0.75)
    assert full_catalog_candidate_ids(4).tolist() == [2, 3]


def test_no_observed_target_returns_json_safe_null_metrics():
    result = ranking_metrics(
        torch.zeros(1, 4),
        torch.tensor([0]),
        target_in_train_catalog=torch.tensor([False]),
        observed_mask=torch.tensor([False]),
    )
    assert result["sample_count"] == 0
    assert result["recall@5"] is None
    assert result["ndcg@5"] is None
    assert result["mrr@5"] is None
    assert result["poi_nll"] is None
    assert result["candidate_count"] == 2
    assert result["random_floor"]["recall@5"] == 1.0


def test_distinct_unseen_pois_collapsed_to_unknown_are_not_counted_as_correct():
    # These rows represent two distinct raw POIs after both are encoded as unknown_id=1.
    logits = torch.tensor([[0.0, 100.0, 0.0], [0.0, 100.0, 0.0]])
    metrics = ranking_metrics(
        logits,
        torch.tensor([1, 1]),
        target_in_train_catalog=torch.tensor([False, False]),
        observed_mask=torch.tensor([True, True]),
        ks=(1,),
    )
    assert metrics["sample_count"] == 0
    assert metrics["observed_target_count"] == 2
    assert metrics["cold_start_target_count"] == 2
    assert metrics["ranking_coverage"] == 0.0
    assert metrics["recall@1"] is None
    assert metrics["ndcg@1"] is None
    assert metrics["mrr@1"] is None
    assert metrics["poi_nll"] is None


@pytest.mark.parametrize(
    "case_name",
    ["perfect_oracle", "rank_2", "all_wrong", "cold_start", "censored"],
)
def test_v02_ranking_sanity_fixture(case_name):
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert fixture["schema_version"] == "still-evaluation-v0.2"
    case = fixture["cases"][case_name]
    metrics = ranking_metrics(
        torch.tensor(case["logits"], dtype=torch.float64),
        torch.tensor(case["targets"]),
        target_in_train_catalog=torch.tensor(case["target_in_train_catalog"]),
        observed_mask=torch.tensor(case["observed_mask"]),
        ks=tuple(fixture["ks"]),
    )
    for key, expected in case["expected_ranking"].items():
        if expected is None:
            assert metrics[key] is None
        else:
            assert metrics[key] == pytest.approx(expected)


def test_cold_start_and_censored_rows_are_excluded_from_exact_nll():
    logits = torch.tensor(
        [
            [20.0, 30.0, 3.0, 2.0],
            [0.0, 100.0, 0.0, 0.0],
            [100.0, 0.0, 0.0, 0.0],
        ]
    )
    metrics = ranking_metrics(
        logits,
        torch.tensor([2, 1, 0]),
        target_in_train_catalog=torch.tensor([True, False, False]),
        observed_mask=torch.tensor([True, True, False]),
        ks=(1, 2),
    )
    expected = F.cross_entropy(
        torch.tensor([[-torch.inf, -torch.inf, 3.0, 2.0]]), torch.tensor([2])
    )
    assert metrics["sample_count"] == 1
    assert metrics["observed_target_count"] == 2
    assert metrics["cold_start_target_count"] == 1
    assert metrics["poi_nll"] == pytest.approx(float(expected))


def test_analytical_random_floors_match_hand_calculation_and_clip_k():
    floors = analytical_random_floors(candidate_count=4, ks=(1, 2, 5))
    assert floors["recall@1"] == pytest.approx(1 / 4)
    assert floors["mrr@2"] == pytest.approx((1.0 + 1.0 / 2.0) / 4)
    assert floors["ndcg@2"] == pytest.approx((1.0 + 1.0 / math.log2(3.0)) / 4)
    assert floors["recall@5"] == 1.0
    assert floors["mrr@5"] == pytest.approx(sum(1.0 / rank for rank in range(1, 5)) / 4)


@pytest.mark.parametrize("candidate_count", [0, -1])
def test_analytical_random_floors_reject_non_positive_candidate_count(candidate_count):
    with pytest.raises(ValueError, match="candidate_count"):
        analytical_random_floors(candidate_count, ks=(5,))
