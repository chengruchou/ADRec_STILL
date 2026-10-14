"""Tests for the minimal shared-contract extensions this workstream proposes.

These exercise files owned by Student A (evaluation) and Student B (baseline trainer /
harness); the extensions are additive and flagged for owner review in the PR.
"""

import math

import pytest
import torch

from adrec_still.evaluation.evaluator import evaluate
from adrec_still.time.protocol import (
    LogNormalPrediction,
    ResidualTimeDistribution,
    lognormal_time_metrics,
    residual_time_distribution_metrics,
)
from adrec_still.models.baselines.harness import _aggregate_time
from adrec_still.models.baselines.trainer import validate_training_config


class EqualWeightLognormalMixture(ResidualTimeDistribution):
    """Self-contained protocol double: an equal-weight log-normal mixture.

    This PR only proposes the adapter contract; the real model-side mixture
    distributions live in the follow-up models PR. The double computes exact mixture
    log-density and log-survival directly so the metric plumbing is tested against
    hand-checkable closed forms.
    """

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        self.mu = mu
        self.sigma = sigma
        self.log_weight = -math.log(mu.shape[1])

    def batch_size(self) -> int:
        return self.mu.shape[0]

    def log_density(self, residual_time_sec: torch.Tensor) -> torch.Tensor:
        time64 = residual_time_sec.to(torch.float64)
        safe = torch.where(time64 > 0, time64, torch.ones_like(time64))
        z = (torch.log(safe).unsqueeze(1) - self.mu) / self.sigma
        per_component = (
            -0.5 * z**2
            - torch.log(safe).unsqueeze(1)
            - torch.log(self.sigma)
            - 0.5 * math.log(2.0 * math.pi)
        )
        value = torch.logsumexp(self.log_weight + per_component, dim=1)
        return torch.where(time64 > 0, value, torch.full_like(value, -math.inf))

    def log_survival(self, time_sec: torch.Tensor) -> torch.Tensor:
        time64 = time_sec.to(torch.float64)
        safe = torch.where(time64 > 0, time64, torch.ones_like(time64))
        z = (torch.log(safe).unsqueeze(1) - self.mu) / self.sigma
        value = torch.logsumexp(self.log_weight + torch.special.log_ndtr(-z), dim=1)
        return torch.where(time64 > 0, value, torch.zeros_like(value))


def make_mixture(batch: int, components: int = 2) -> EqualWeightLognormalMixture:
    mu = torch.linspace(
        math.log(3600.0), math.log(86400.0), components, dtype=torch.float64
    ).repeat(batch, 1)
    sigma = torch.full((batch, components), 1.1, dtype=torch.float64)
    return EqualWeightLognormalMixture(mu=mu, sigma=sigma)


def test_mixture_metrics_match_hand_nll():
    distribution = make_mixture(batch=3)
    residual = torch.tensor([3600.0, 7200.0, 0.0], dtype=torch.float64)
    censor = torch.tensor([0.0, 0.0, 5400.0], dtype=torch.float64)
    observed = torch.tensor([True, True, False])
    metrics = residual_time_distribution_metrics(distribution, residual, censor, observed)
    expected_observed = -distribution.log_density(residual)[:2]
    expected_censored = -distribution.log_survival(censor)[2:]
    expected_nll = torch.cat([expected_observed, expected_censored]).mean().item()
    assert metrics["return_time_nll"] == pytest.approx(expected_nll, rel=1e-12)
    # The generic adapter path tags its version; CRPS keys no longer exist.
    assert metrics["time_adapter_version"] == "adrec-still-residual-time-adapter-v1"
    assert not any("crps" in key for key in metrics)
    assert metrics["observed_time_nll"] == pytest.approx(expected_observed.mean().item())
    assert metrics["censored_time_nll"] == pytest.approx(expected_censored.mean().item())
    assert metrics["observed_count"] == 2 and metrics["censored_count"] == 1


def test_single_component_mixture_nll_matches_the_accepted_lognormal_path():
    """Cross-check: a K=1 mixture reproduces the accepted LogNormalPrediction NLL."""
    distribution = make_mixture(batch=2, components=1)
    prediction = LogNormalPrediction(mu=distribution.mu[:, 0], sigma=distribution.sigma[:, 0])
    residual = torch.tensor([3600.0, 0.0], dtype=torch.float64)
    censor = torch.tensor([0.0, 9000.0], dtype=torch.float64)
    observed = torch.tensor([True, False])
    legacy = lognormal_time_metrics(prediction, residual, censor, observed)
    generic = residual_time_distribution_metrics(distribution, residual, censor, observed)
    assert generic["return_time_nll"] == pytest.approx(legacy["return_time_nll"], rel=1e-9)


def make_evaluation_batch() -> dict[str, torch.Tensor]:
    return {
        "next_poi": torch.tensor([3, 4]),
        "target_in_train_catalog": torch.tensor([True, True]),
        "observed_mask": torch.tensor([True, True]),
        "residual_time_sec": torch.tensor([3600.0, 7200.0], dtype=torch.float64),
        "censor_time_sec": torch.zeros(2, dtype=torch.float64),
        "query_s_sec": torch.tensor([0, 0]),
    }


def test_evaluate_routes_the_generic_time_distribution_and_rejects_ambiguity():
    batch = make_evaluation_batch()
    logits = torch.log_softmax(torch.randn(2, 10), dim=1)
    distribution = make_mixture(batch=2)
    result = evaluate(logits, batch, time_distribution=distribution)
    assert result["time"]["time_adapter_version"] == "adrec-still-residual-time-adapter-v1"
    prediction = LogNormalPrediction(mu=torch.zeros(2), sigma=torch.ones(2))
    with pytest.raises(ValueError):
        evaluate(logits, batch, time_prediction=prediction, time_distribution=distribution)


def test_evaluate_keeps_published_positional_parameter_semantics():
    """Regression: the adapter parameter must not shift the published positions.

    ``evaluate(logits, batch, time_prediction, split, cohorts)`` was the accepted
    call shape before this extension; each positional form must land in the same
    parameter and produce the same outputs as before.
    """
    torch.manual_seed(0)
    batch = make_evaluation_batch()
    logits = torch.log_softmax(torch.randn(2, 10), dim=1)

    # evaluate(logits, batch, None, "validation"): split lands in split.
    result = evaluate(logits, batch, None, "validation")
    assert result["split"] == "validation"
    assert "time" not in result
    assert result["sample_counts_by_cohort"] == {"observed": 2, "censored": 0}

    # evaluate(logits, batch, prediction, "validation"): third positional is still
    # the accepted single-LogNormal path and produces its exact metrics.
    prediction = LogNormalPrediction(
        mu=torch.full((2,), 8.0, dtype=torch.float64),
        sigma=torch.ones(2, dtype=torch.float64),
    )
    with_time = evaluate(logits, batch, prediction, "validation")
    assert with_time["split"] == "validation"
    expected_time = lognormal_time_metrics(
        prediction,
        batch["residual_time_sec"],
        batch["censor_time_sec"],
        batch["observed_mask"],
    )
    assert with_time["time"]["return_time_nll"] == pytest.approx(
        expected_time["return_time_nll"]
    )

    # evaluate(logits, batch, None, None, cohorts): fifth positional is cohorts and
    # must fill the cohort counts, never the split.
    cohort_result = evaluate(logits, batch, None, None, ["known", "known"])
    assert cohort_result["split"] is None
    assert cohort_result["sample_counts_by_cohort"] == {"known": 2}

    # The original keyword form and the new keyword-only adapter compose with both.
    keyword_result = evaluate(
        logits, batch, time_prediction=prediction, split="test", cohorts=["a", "b"]
    )
    assert keyword_result["split"] == "test"
    assert keyword_result["sample_counts_by_cohort"] == {"a": 1, "b": 1}
    adapter_result = evaluate(
        logits,
        batch,
        None,
        "validation",
        ["known", "known"],
        time_distribution=make_mixture(batch=2),
    )
    assert adapter_result["split"] == "validation"
    assert adapter_result["sample_counts_by_cohort"] == {"known": 2}
    assert adapter_result["time"]["time_adapter_version"] == "adrec-still-residual-time-adapter-v1"


def test_time_aggregation_reports_split_nll_populations_without_crps():
    """The ported harness aggregation: pooled NLL is sample-weighted, the observed and
    censored populations keep separate means, and no CRPS key is fabricated.
    (Replaces the donor's CRPS-aggregation regression test; CRPS was not ported.)"""
    first = {
        "sample_count": 4,
        "observed_count": 3,
        "censored_count": 1,
        "return_time_nll": 2.0,
        "observed_time_nll_sum": 6.0,
        "censored_time_nll_sum": 2.0,
    }
    second = {
        "sample_count": 2,
        "observed_count": 1,
        "censored_count": 1,
        "return_time_nll": 5.0,
        "observed_time_nll_sum": 7.0,
        "censored_time_nll_sum": 3.0,
    }
    aggregated = _aggregate_time([first, second])
    assert aggregated["sample_count"] == 6
    assert aggregated["observed_count"] == 4 and aggregated["censored_count"] == 2
    assert aggregated["return_time_nll"] == pytest.approx((2.0 * 4 + 5.0 * 2) / 6)
    assert aggregated["observed_time_nll"] == pytest.approx((6.0 + 7.0) / 4)
    assert aggregated["censored_time_nll"] == pytest.approx((2.0 + 3.0) / 2)
    assert not any("crps" in key for key in aggregated)


def test_trainer_hook_shape_contract_and_selection_mode_validation():
    from adrec_still.models.baselines.trainer import per_row_sampled_losses

    class BadHookModel:
        number_of_pois = 10

        def per_row_sampled_training_losses(self, kept, negative_samples, generator):
            del kept, negative_samples, generator
            return torch.zeros(1)  # wrong shape: not one value per kept row

    kept = {"next_poi": torch.tensor([3, 4, 5])}
    with pytest.raises(ValueError):
        per_row_sampled_losses(
            BadHookModel(), kept, negative_samples=2, generator=torch.Generator()
        )

    training = {
        "landmark_weighting": "per_gap",
        "optimizer": "adam",
        "learning_rate": 0.01,
        "batch_size": 8,
        "max_epochs": 1,
        "early_stopping_patience": 1,
        "early_stopping_metric": "ndcg@10",
        "negative_samples": 2,
        "early_stopping_mode": "sideways",
    }
    with pytest.raises(ValueError):
        validate_training_config(training)
    # The direction belongs to the metric: a contradicting one is refused, the
    # matching one (or none at all) passes.
    training["early_stopping_mode"] = "min"
    with pytest.raises(ValueError, match="contradicts"):
        validate_training_config(training)
    training["early_stopping_mode"] = "max"
    validate_training_config(training)
    del training["early_stopping_mode"]
    validate_training_config(training)


def test_baseline_prediction_rejects_mismatched_distribution_batch():
    from adrec_still.models.baselines.base import BaselinePrediction

    prediction = BaselinePrediction(
        poi_logits=torch.log_softmax(torch.randn(2, 10), dim=1),
        context=torch.zeros(2, 4),
        time_distribution=make_mixture(batch=3),
    )
    with pytest.raises(ValueError):
        prediction.validate(10)
