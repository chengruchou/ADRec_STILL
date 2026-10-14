"""Truncated log-normal sampling and the forecast's survivor/stratum arithmetic.

Statistical checks use one fixed generator seed and loose tolerances pinned well
above the observed Monte-Carlo noise; the forecast-path checks run a tiny
untrained ``adrec_still`` model over a two-user hand-built event memory whose
expected branch priors are computed independently in the test.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
import torch

from adrec_still.data.history import FullHistoryIndex
from adrec_still.models.adrec_still.forecast import (
    _truncated_lognormal_sample,
    log1mexp,
)
from adrec_still.models.adrec_still.memory import memory_features
from adrec_still.models.adrec_still.model import AdrecStillModel
from adrec_still.time.distributions import LognormalMixtureParameters

THRESHOLD_S = 24.0 * 3600.0  # the model default cell_time_threshold_hours=24


# ---------------------------------------------------------------- raw sampling


def test_single_component_truncated_samples_follow_the_analytic_law():
    rows = 4000
    mu, sigma, lower_value = math.log(3600.0), 0.8, 5400.0
    parameters = LognormalMixtureParameters(
        log_weights=torch.zeros((rows, 1), dtype=torch.float64),
        mu=torch.full((rows, 1), mu, dtype=torch.float64),
        sigma=torch.full((rows, 1), sigma, dtype=torch.float64),
    )
    lower = torch.full((rows,), lower_value, dtype=torch.float64)
    generator = torch.Generator().manual_seed(20251014)
    samples = _truncated_lognormal_sample(parameters, lower, None, generator)

    assert bool((samples > lower_value).all())
    # Truncated quantiles: S(T_q) = (1 - q) * S(L)  =>  T_q = exp(mu + sigma * z_q).
    z_lower = (math.log(lower_value) - mu) / sigma
    survival_at_lower = 0.5 * math.erfc(z_lower / math.sqrt(2.0))
    for quantile in (0.25, 0.5, 0.75):
        target_survival = (1.0 - quantile) * survival_at_lower
        z = -float(torch.special.ndtri(torch.tensor(target_survival, dtype=torch.float64)))
        analytic = math.exp(mu + sigma * z)
        empirical = float(samples.quantile(quantile))
        assert empirical == pytest.approx(analytic, rel=0.05)


def test_component_weights_are_survivor_reweighted_not_the_prior():
    # The prior puts 99.9% on a component whose mass is entirely below the lower
    # bound. Prior-weighted sampling would pick it ~99.9% of the time (samples
    # near the bound, ~1e3..1e5); survivor reweighting must pick the surviving
    # far component (median 1e7) essentially always.
    rows = 2000
    parameters = LognormalMixtureParameters(
        log_weights=torch.log(
            torch.tensor([[0.999, 0.001]], dtype=torch.float64)
        ).expand(rows, 2),
        mu=torch.tensor(
            [[math.log(10.0), math.log(1.0e7)]], dtype=torch.float64
        ).expand(rows, 2),
        sigma=torch.full((rows, 2), 0.3, dtype=torch.float64),
    )
    lower = torch.full((rows,), 1.0e3, dtype=torch.float64)
    generator = torch.Generator().manual_seed(7)
    samples = _truncated_lognormal_sample(parameters, lower, None, generator)
    assert float((samples > 1.0e6).double().mean()) > 0.99


def test_log1mexp_matches_the_reference_on_both_branches():
    # bound = log(0.5): values above it take the expm1 branch, below it log1p.
    values = torch.tensor([-1e-3, -0.1, -0.5, -0.7, -5.0, -30.0], dtype=torch.float64)
    reference = torch.log(1.0 - torch.exp(values))
    assert torch.allclose(log1mexp(values), reference, atol=1e-12, rtol=0.0)


def test_survivor_weights_at_zero_elapsed_equal_the_prior():
    parameters = LognormalMixtureParameters(
        log_weights=torch.log(torch.tensor([[0.2, 0.8], [0.6, 0.4]], dtype=torch.float64)),
        mu=torch.tensor([[1.0, 8.0], [3.0, 5.0]], dtype=torch.float64),
        sigma=torch.tensor([[0.5, 1.5], [1.0, 2.0]], dtype=torch.float64),
    )
    zeros = torch.zeros(2, dtype=torch.float64)
    assert torch.equal(
        parameters.survivor_log_weights(zeros), parameters.normalized_log_weights()
    )


# ------------------------------------------------------------- forecast path


def _fixture_index(tmp_path) -> FullHistoryIndex:
    """Two users over a two-item catalog: one saw everything, one saw only pA."""
    rows = []

    def event(user, raw, poi, time_s, source):
        rows.append(
            {
                "dataset": "ds",
                "raw_user_id": user,
                "user_id": 2,
                "raw_poi_id": raw,
                "poi_id": poi,
                "event_time_utc_s": time_s,
                "latitude": 0.0,
                "longitude": 0.0,
                "event_index": len(rows),
                "source_row_id": source,
            }
        )

    event("full", "pA", 2, 1000, 0)
    event("full", "pB", 3, 2000, 1)
    event("full", "pA", 2, 3000, 2)
    event("partial", "pA", 2, 1000, 10)
    event("partial", "pA", 2, 2000, 11)
    events_path = tmp_path / "events.parquet"
    pd.DataFrame(rows).to_parquet(events_path, index=False)
    catalog = pd.DataFrame(
        [
            {"poi_id": 2, "raw_poi_id": "pA", "latitude": 0.0, "longitude": 0.0,
             "train_count": 3, "first_train_time_s": 0, "last_train_time_s": 100},
            {"poi_id": 3, "raw_poi_id": "pB", "latitude": 0.0, "longitude": 0.05,
             "train_count": 1, "first_train_time_s": 0, "last_train_time_s": 100},
        ]
    )
    catalog_path = tmp_path / "poi_catalog.parquet"
    catalog.to_parquet(catalog_path, index=False)
    return FullHistoryIndex(events_path, catalog_path, train_cutoff_s=10**9)


def _fixture_model(index: FullHistoryIndex) -> AdrecStillModel:
    torch.manual_seed(11)
    model = AdrecStillModel(
        "adrec_still", number_of_pois=4, hidden_size=8, mixture_components=2,
        diffusion_steps=2, time_samples=1, diffusion_samples=1,
    )
    model.attach_memory(index)
    model.is_fitted = True
    model.eval()
    return model


def _batch(gap_ids, histories, last_timestamps, query_s):
    history = torch.tensor(histories, dtype=torch.long)
    return {
        "history_poi": history,
        "history_timestamp": torch.zeros_like(history),
        "history_mask": history.gt(0),
        "last_timestamp": torch.tensor(last_timestamps, dtype=torch.long),
        "query_s_sec": torch.tensor(query_s, dtype=torch.long),
        "gap_id": list(gap_ids),
    }


def test_zero_elapsed_catalog_repeat_probability_is_the_support_restricted_prior(tmp_path):
    index = _fixture_index(tmp_path)
    model = _fixture_model(index)
    batch = _batch(
        ["ds:000000000002:censored", "ds:000000000011:censored"],
        [[2, 3, 2], [0, 2, 2]],
        [3000, 2000],
        [0, 0],
    )
    forecast = model.prepare_forecast(batch, chunk_size=4)
    observed = forecast.extras["repeat_probability_catalog"]

    with torch.no_grad():
        context = model.base_context(batch)
        log_prior = model.branch_log_prior(context).to(torch.float64)
        memory = memory_features(index, batch, home_policy=model.home_policy_eval)
        masked = log_prior + model.catalog_support_log_mask(memory)
        expected = (masked - torch.logsumexp(masked, dim=1, keepdim=True)).exp()[:, 0]

    assert torch.allclose(observed, expected, atol=1e-9, rtol=0.0)
    # The "full" user has seen the whole catalog: the explore branch has no
    # support, so the catalog-view repeat probability is exactly one.
    assert float(observed[0]) == pytest.approx(1.0, abs=1e-9)
    # The "partial" user keeps both branches: the probability is the plain
    # softmax of the branch prior, strictly inside (0, 1).
    assert 0.0 < float(observed[1]) < 1.0
    assert float(observed[1]) == pytest.approx(
        float(torch.softmax(log_prior[1], dim=0)[0]), abs=1e-9
    )


def test_stratum_weights_sum_to_one_and_early_mass_vanishes_past_the_threshold(tmp_path):
    index = _fixture_index(tmp_path)
    model = _fixture_model(index)
    assert model.cell_time_threshold_s == THRESHOLD_S
    batch = _batch(
        ["ds:000000000011:censored", "ds:000000000011:censored"],
        [[0, 2, 2], [0, 2, 2]],
        [2000, 2000],
        [0, int(2 * THRESHOLD_S)],
    )
    weights = model.prepare_forecast(batch, chunk_size=4).extras["stratum_weights"]
    assert weights.shape == (2, 2, 2)  # [row, branch, early/late]
    assert bool((weights >= 0).all())
    totals = weights.sum(dim=(1, 2))
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-6, rtol=0.0)
    # s = 0 < threshold: the early stratum carries real mass.
    assert float(weights[0, :, 0].sum()) > 0.01
    # s >= threshold: the early stratum is exactly empty, late carries it all.
    assert torch.equal(weights[1, :, 0], torch.zeros(2, dtype=weights.dtype))
    assert float(weights[1, :, 1].sum()) == pytest.approx(1.0, abs=1e-6)
