"""#104 follow-up: the per-horizon breakdown keeps each metric's own population.

Ranking, joint NLL and time NLL have their own eligible rows (ranking and joint:
observed known targets; time: observed rows with a positive residual plus censored
rows with positive exposure); the rules are rebuilt here from the batch fields. With every batch scored once and bucketed by
``query_s_sec``, each horizon's counts and means must still equal a by-hand
computation over that horizon's rows for each metric separately, and the counts
must equal those of a pass filtered to that horizon.
"""

from __future__ import annotations

import pytest
import torch

from adrec_still.data.collate import collate_canonical_samples
from adrec_still.data.dataset import LandmarkDataset
from adrec_still.training.evaluate import EvaluationOptions, evaluate_split, time_nll_rows
from adrec_still.training.families import load_model
from adrec_still.training.service import run

from .conftest import small_spec

MODELS = {
    "still_joint": {"embedding_dim": 8, "hidden_dim": 8},
    "still_independent": {"embedding_dim": 8, "hidden_dim": 8},
    "external_joint_lnm": {"mark_embedding_dim": 4, "context_dim": 8},
}


@pytest.mark.parametrize("name", sorted(MODELS))
def test_breakdown_keeps_each_metrics_population_per_horizon(canonical_data, tmp_path, name):
    overrides = {"training.batch_size": 16, "training.negative_samples": 4, **MODELS[name]}
    run(small_spec(canonical_data, tmp_path / name, name, epochs=1, evaluate_test=False,
                   overrides=overrides))
    model, _ = load_model(tmp_path / name / "best.pt")
    model.is_fitted = True
    model.eval()  # the by-hand pass must match evaluate_split's eval mode (no dropout)
    landmarks = canonical_data / "landmark_samples.parquet"

    # By hand: every test row in one batch, per-row values and masks per metric.
    dataset = LandmarkDataset(landmarks, split="test")
    batch = collate_canonical_samples([dataset[i] for i in range(len(dataset))],
                                      max_sequence_length=20)
    with torch.no_grad():
        joint = model.score_joint(batch)
        time_values, helper_time_eligible = time_nll_rows(model.predict(batch), batch)
    joint_values = -joint["log_joint"].detach().cpu().to(torch.float64)

    # The eligibility rules, built from the batch fields alone (not from the code
    # under test), so a regression in the helpers' masks fails here too.
    observed = batch["observed_mask"].bool()
    known = batch["target_in_train_catalog"].bool()
    residual = batch["residual_time_sec"].to(torch.float64)
    censor = batch["censor_time_sec"].to(torch.float64)
    ranking_eligible = observed & known
    joint_eligible = observed & known
    time_eligible = (observed & residual.gt(0)) | (~observed & censor.gt(0))
    assert torch.equal(joint["eligible"].detach().cpu().bool(), joint_eligible)
    assert torch.equal(helper_time_eligible, time_eligible)
    # Ranking and joint share one population; time also takes censored rows with
    # exposure (and drops observed rows without a positive residual).
    assert (time_eligible & ~joint_eligible).any()
    horizons = batch["query_s_sec"]

    result = evaluate_split(model, landmarks, split="test", breakdown=True,
                            options=EvaluationOptions(batch_size=8, max_sequence_length=20))
    distinct = torch.unique(horizons).tolist()
    assert len(distinct) >= 2
    assert sorted(int(h) for h in result["by_horizon"]) == sorted(int(h) for h in distinct)

    populations_differ = False
    for h in distinct:
        rows = horizons.eq(h)
        bucket = result["by_horizon"][str(int(h))]
        assert bucket["rows"] == int(rows.sum())
        assert bucket["eligible_rows"] == int((rows & ranking_eligible).sum())
        for metric, values, eligible in (("joint_nll", joint_values, joint_eligible),
                                         ("time_nll", time_values, time_eligible)):
            mask = rows & eligible
            assert bucket[f"{metric}_rows"] == int(mask.sum()), (h, metric)
            if mask.any():
                assert bucket[metric] == pytest.approx(float(values[mask].mean()), rel=1e-6), (h, metric)
            else:
                assert bucket[metric] is None
        if len({bucket["eligible_rows"], bucket["joint_nll_rows"], bucket["time_nll_rows"]}) > 1:
            populations_differ = True
        # The same counts as a pass restricted to this horizon.
        filtered = evaluate_split(model, landmarks, split="test", horizon=int(h),
                                  options=EvaluationOptions(batch_size=8, max_sequence_length=20))
        for key in ("rows", "eligible_rows", "joint_nll_rows", "time_nll_rows"):
            assert filtered[key] == bucket[key], (h, key)
        for metric in ("joint_nll", "time_nll"):
            if bucket[metric] is None:
                assert filtered[metric] is None
            else:
                assert filtered[metric] == pytest.approx(bucket[metric], rel=1e-6), (h, metric)
    # Otherwise the test would not exercise distinct populations at all.
    assert populations_differ

    # Pooled figures are the sums of the per-horizon buckets for every metric, and each
    # pooled NLL is the per-horizon means weighted by that metric's own row counts.
    buckets = list(result["by_horizon"].values())
    for key in ("rows", "eligible_rows", "joint_nll_rows", "time_nll_rows"):
        assert result[key] == sum(b[key] for b in buckets), key
    for metric in ("joint_nll", "time_nll"):
        weighted = [(b[metric], b[f"{metric}_rows"]) for b in buckets if b[f"{metric}_rows"]]
        expected = sum(mean * n for mean, n in weighted) / sum(n for _, n in weighted)
        assert result[metric] == pytest.approx(expected, rel=1e-9), metric
        hand = joint_values if metric == "joint_nll" else time_values
        mask = joint_eligible if metric == "joint_nll" else time_eligible
        assert result[metric] == pytest.approx(float(hand[mask].mean()), rel=1e-6), metric
