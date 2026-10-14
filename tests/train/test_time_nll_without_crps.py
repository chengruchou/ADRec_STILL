"""Return-time NLL is computed from the models' distributions; CRPS is never called.

The regression guard, adapted to the port: CRPS was removed from adrec_still
entirely, so no class or module on the time path may expose a CRPS entry point,
and the remaining metric-adapter module surface is replaced with a function that
raises while training, per-epoch selection on ``time_nll``, the final evaluation
and the reported ``time_nll``/``joint_nll`` still complete. A hidden call anywhere
on the path would surface as that error.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from adrec_still.time import protocol as time_module
from adrec_still.training.evaluate import EvaluationOptions, evaluate_split, time_nll_rows
from adrec_still.training.families import load_model
from adrec_still.training.service import run

from .conftest import small_spec


class CrpsCalled(AssertionError):
    pass


def _forbid(*args, **kwargs):
    raise CrpsCalled("a CRPS routine was called on the training/evaluation path")


def _all_subclasses(cls):
    found = set()
    stack = [cls]
    while stack:
        current = stack.pop()
        for sub in current.__subclasses__():
            if sub not in found:
                found.add(sub)
                stack.append(sub)
    return found


@pytest.fixture
def crps_forbidden(monkeypatch):
    """CRPS no longer exists anywhere on the time path, and the metric-adapter
    module surface raises if the training/evaluation path were to call it."""
    import adrec_still.models.external.lnm_distributions
    import adrec_still.time.distributions  # noqa: F401

    assert not hasattr(time_module.ResidualTimeDistribution, "crps")
    assert not hasattr(time_module.ResidualTimeDistribution, "crps_unsupported_reason")
    for sub in _all_subclasses(time_module.ResidualTimeDistribution):
        assert "crps" not in vars(sub), f"{sub!r} still defines crps"
    assert not hasattr(time_module, "observed_crps_from_survival")
    monkeypatch.setattr(time_module, "lognormal_time_metrics", _forbid)
    monkeypatch.setattr(time_module, "residual_time_distribution_metrics", _forbid)
    monkeypatch.setattr(time_module, "score_time_joint_rows", _forbid)
    return monkeypatch


def test_the_guard_itself_bites(crps_forbidden):
    from adrec_still.models.still.independent import MixtureResidualDistribution
    from adrec_still.time.distributions import LognormalMixtureParameters

    parameters = LognormalMixtureParameters(
        log_weights=torch.full((1, 2), math.log(0.5)), mu=torch.zeros(1, 2),
        sigma=torch.ones(1, 2),
    )
    # CRPS is gone from the distribution API altogether in this port.
    with pytest.raises(AttributeError):
        MixtureResidualDistribution(parameters).crps(torch.ones(1))
    with pytest.raises(CrpsCalled):
        time_module.residual_time_distribution_metrics(
            MixtureResidualDistribution(parameters), torch.ones(1), torch.zeros(1),
            torch.ones(1, dtype=torch.bool),
        )


@pytest.mark.parametrize("model", ["still_joint", "external_joint_lnm"])
def test_time_nll_is_reported_and_selectable_without_any_crps(canonical_data, tmp_path,
                                                              crps_forbidden, model):
    overrides = {"training.batch_size": 16, "training.negative_samples": 4}
    if model == "external_joint_lnm":
        overrides.update({"mark_embedding_dim": 4, "context_dim": 8})
    else:
        overrides.update({"embedding_dim": 8, "hidden_dim": 8})
    output = tmp_path / model
    summary = run(small_spec(canonical_data, output, model, epochs=2, monitor="time_nll",
                             validation_horizon=86400, overrides=overrides))
    assert summary["status"] == "completed"
    assert summary["training"]["selection"]["metric"] == "time_nll"
    assert summary["training"]["selection"]["direction"] == "min"
    best = summary["training"]["best"]
    assert best["metric"] == "time_nll" and math.isfinite(best["value"])
    curve = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    for row in curve:
        assert math.isfinite(row["validation"]["time_nll"])
        assert row["validation"]["time_nll_rows"] > 0
        assert math.isfinite(row["validation"]["joint_nll"])
        assert "crps" not in json.dumps(row)
    test = summary["evaluation"]["test"]
    assert math.isfinite(test["time_nll"]) and test["time_nll_rows"] > 0
    assert math.isfinite(test["joint_nll"]) and test["joint_nll_rows"] > 0
    assert math.isfinite(test["poi_nll"])
    assert "crps" not in json.dumps(summary)
    # Censored rows take part in time_nll (survival term); joint_nll is observed-only.
    assert test["time_nll_rows"] > test["joint_nll_rows"]
    reloaded, _ = load_model(output / "best.pt")
    assert reloaded.resolved_config()["name"] == model


def test_time_nll_matches_the_distribution_by_hand(canonical_data, crps_forbidden):
    """The per-row values are exactly ``-log_density`` on observed rows and
    ``-log_survival`` on censored rows of the distribution ``predict`` returns."""
    from adrec_still.data.collate import collate_canonical_samples
    from adrec_still.data.dataset import LandmarkDataset
    from adrec_still.training import checkpoint as ckpt
    from adrec_still.training.defaults import default_model_config
    from adrec_still.training.families import build_model

    config = {**default_model_config("still_joint"), "embedding_dim": 8, "hidden_dim": 8}
    model = build_model(config, ckpt.number_of_pois(canonical_data / "poi_catalog.parquet"))
    model.is_fitted = True
    dataset = LandmarkDataset(canonical_data / "landmark_samples.parquet", split="test")
    batch = collate_canonical_samples([dataset[i] for i in range(len(dataset))],
                                      max_sequence_length=20)
    prediction = model.predict(batch)
    nll, eligible = time_nll_rows(prediction, batch)
    observed = batch["observed_mask"].bool()
    assert eligible.sum() > 0 and (eligible & ~observed).sum() > 0
    distribution = prediction.time_distribution
    residual = batch["residual_time_sec"].to(torch.float64)
    censor = batch["censor_time_sec"].to(torch.float64)
    expected_observed = -distribution.log_density(torch.where(observed, residual, torch.ones_like(residual)))
    expected_censored = -distribution.log_survival(torch.where(~observed, censor, torch.zeros_like(censor)))
    assert torch.allclose(nll[observed & eligible], expected_observed[observed & eligible])
    assert torch.allclose(nll[~observed & eligible], expected_censored[~observed & eligible])

    result = evaluate_split(
        model, canonical_data / "landmark_samples.parquet", split="test",
        options=EvaluationOptions(batch_size=32, max_sequence_length=20),
    )
    assert result["time_nll"] == pytest.approx(float(nll[eligible].mean()), rel=1e-9)
    assert result["time_nll_rows"] == int(eligible.sum())


def test_models_without_a_time_distribution_report_no_time_nll(canonical_data, tmp_path,
                                                                crps_forbidden):
    summary = run(small_spec(canonical_data, tmp_path / "gru", "gru4rec_h", epochs=1))
    assert "time_nll" not in summary["evaluation"]["test"]
    assert "crps" not in json.dumps(summary)
    with pytest.raises(TypeError, match="time_nll cannot be monitored"):
        run(small_spec(canonical_data, tmp_path / "gru2", "gru4rec_h", epochs=1,
                       monitor="time_nll"))
