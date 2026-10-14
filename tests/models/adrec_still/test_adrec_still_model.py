"""ADRec-STILL model contract on a real-pipeline canonical benchmark.

The data is the shared Gowalla-shaped synthetic fixture preprocessed once
through the committed pipeline (the same path real runs use), so the batches
carry genuine ``gap_id`` anchors, an out-of-vocabulary test target population,
and censored final-gap rows. The model stays untrained (fixed init seed): every
property checked here is structural, not learned.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from adrec_still.data.collate import collate_canonical_samples
from adrec_still.data.dataset import LandmarkDataset
from adrec_still.data.history import FullHistoryIndex, train_cutoff_from_split_manifest
from adrec_still.evaluation.ranking import full_catalog_candidate_ids
from adrec_still.models.adrec_still.memory import memory_features
from adrec_still.models.adrec_still.model import MASKED, AdrecStillModel

REPOSITORY_ROOT = Path(__file__).parents[3]
_BASELINE_CONFTEST = REPOSITORY_ROOT / "tests" / "models" / "baselines" / "conftest.py"


@pytest.fixture(scope="module")
def benchmark(tmp_path_factory):
    """Raw fixture -> canonical artifacts -> memory index + categorized test rows."""
    from adrec_still.training.inputs import resolve_data

    writer = runpy.run_path(str(_BASELINE_CONFTEST))["write_synthetic_raw"]
    root = tmp_path_factory.mktemp("adrec_still_model")
    raw = writer(root / "loc-gowalla_totalCheckins.txt")
    source = resolve_data(raw, cache_dir=root / "cache")
    cutoff = train_cutoff_from_split_manifest(source.split_manifest_path)
    index = FullHistoryIndex(source.events_path, source.catalog_path,
                             train_cutoff_s=cutoff)
    dataset = LandmarkDataset(source.landmark_path, split="test")
    samples = [dataset[i] for i in range(len(dataset))]
    observed_known = [s for s in samples
                      if s.event_observed and s.target_in_train_catalog]
    observed_oov = [s for s in samples
                    if s.event_observed and not s.target_in_train_catalog]
    censored = [s for s in samples if not s.event_observed]
    assert observed_known and observed_oov and censored
    return {
        "source": source,
        "cutoff": cutoff,
        "index": index,
        "observed_known": observed_known,
        "observed_oov": observed_oov,
        "censored": censored,
        "root": root,
    }


@pytest.fixture()
def model(benchmark) -> AdrecStillModel:
    torch.manual_seed(3)
    built = AdrecStillModel(
        "adrec_still", number_of_pois=benchmark["source"].number_of_pois,
        hidden_size=8, mixture_components=2, diffusion_steps=2,
        time_samples=1, diffusion_samples=1,
    )
    built.attach_memory(benchmark["index"])
    built.is_fitted = True
    built.eval()
    return built


def _mixed_batch(benchmark):
    return collate_canonical_samples(
        [benchmark["observed_known"][0], benchmark["observed_known"][1],
         benchmark["observed_oov"][0], benchmark["censored"][0]]
    )


def _losses(model, batch):
    """One deterministic loss pass: the diffusion core draws its timesteps from
    numpy's global stream and its noise from torch's, so both are pinned."""
    torch.manual_seed(1234)
    np.random.seed(1234)
    generator = torch.Generator().manual_seed(99)
    return model.per_row_survival_training_losses(batch, 3, generator).detach()


def test_survival_losses_are_finite_for_every_row_kind(benchmark, model):
    batch = _mixed_batch(benchmark)
    losses = _losses(model, batch)
    assert losses.shape == (4,)
    assert bool(torch.isfinite(losses).all())
    # Identical seeds reproduce the pass bit for bit.
    assert torch.equal(losses, _losses(model, batch))


def test_censored_rows_lose_nothing_from_their_target_padding(benchmark, model):
    batch = _mixed_batch(benchmark)
    reference = _losses(model, batch)
    mutated = {
        key: (value.clone() if isinstance(value, torch.Tensor) else list(value))
        for key, value in batch.items()
    }
    assert not bool(mutated["observed_mask"][3])
    mutated["next_poi"][3] = 2
    mutated["residual_time_sec"][3] = 123456.0
    assert torch.equal(reference, _losses(model, mutated))


def test_branch_supports_are_disjoint_and_cover_the_catalog(benchmark, model):
    batch = collate_canonical_samples(benchmark["observed_known"][:2])
    memory = memory_features(benchmark["index"], batch,
                             home_policy=model.home_policy_eval)
    candidates = full_catalog_candidate_ids(model.number_of_pois)
    with torch.no_grad():
        encoded, _ = model.encode_history(batch)
        generated = model._generate_inference(
            encoded, batch, torch.zeros(2, dtype=torch.long), None,
            model._forecast_generator(batch),
        )
        reference_time = batch["last_timestamp"].to(torch.float64)
        ids = candidates.expand(2, -1)
        repeat = model.branch_candidate_scores(generated, ids, memory, 0, reference_time)
        explore = model.branch_candidate_scores(generated, ids, memory, 1, reference_time)
    repeat_support = repeat > MASKED / 2
    explore_support = explore > MASKED / 2
    assert not bool((repeat_support & explore_support).any())
    assert bool((repeat_support | explore_support).all())
    # The repeat support is exactly the full-history seen set.
    for row in range(2):
        seen = set(memory.seen_poi[row][memory.seen_mask[row]].tolist())
        assert set(candidates[repeat_support[row]].tolist()) == seen


def test_forecast_scores_normalize_over_the_frozen_catalog(benchmark, model):
    batch = _mixed_batch(benchmark)
    forecast = model.prepare_forecast(batch, chunk_size=model.number_of_pois)
    scores = forecast.score(full_catalog_candidate_ids(model.number_of_pois))
    totals = scores.exp().sum(dim=1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5, rtol=0.0)


def test_forecast_is_chunk_size_invariant(benchmark, model):
    batch = _mixed_batch(benchmark)
    candidates = full_catalog_candidate_ids(model.number_of_pois)
    chunked = model.prepare_forecast(batch, chunk_size=3).score(candidates)
    full = model.prepare_forecast(
        batch, chunk_size=model.number_of_pois
    ).score(candidates)
    assert torch.allclose(chunked, full, atol=1e-9, rtol=0.0)


def test_forecast_never_reads_events_after_the_anchor(benchmark, model):
    """Rebuilding the memory with a changed post-anchor event must not move a
    single score: the history, seen set and home before the anchor are intact."""
    source = benchmark["source"]
    events = pd.read_parquet(source.events_path)
    ordered = events.sort_values(["raw_user_id", "event_time_utc_s", "source_row_id"],
                                 kind="mergesort")
    last_source_by_user = {
        user: int(frame["source_row_id"].iloc[-1])
        for user, frame in ordered.groupby("raw_user_id", sort=False)
    }
    user_by_source = dict(zip(events["source_row_id"].astype(int),
                              events["raw_user_id"]))

    def final_event_is_after_the_gap(sample):
        anchor, target = FullHistoryIndex.parse_gap_id(sample.gap_id)
        last = last_source_by_user[user_by_source[anchor]]
        return last not in (anchor, target)

    rows = [s for s in benchmark["observed_known"] if final_event_is_after_the_gap(s)]
    assert len(rows) >= 2
    batch = collate_canonical_samples(rows[:2])
    before = model.prepare_forecast(batch, chunk_size=model.number_of_pois)
    candidates = full_catalog_candidate_ids(model.number_of_pois)
    reference = before.score(candidates)

    # Change the POI of the first row's user's final event (strictly after both
    # anchors) and rebuild the full-history index from the rewritten events.
    anchor, _ = FullHistoryIndex.parse_gap_id(rows[0].gap_id)
    changed_source = last_source_by_user[user_by_source[anchor]]
    changed_row = events["source_row_id"].eq(changed_source)
    catalog = pd.read_parquet(source.catalog_path)
    original_poi = int(events.loc[changed_row, "poi_id"].iloc[0])
    replacement = catalog.loc[catalog["poi_id"].ne(original_poi)].iloc[0]
    rewritten = events.copy()
    rewritten.loc[changed_row, "raw_poi_id"] = replacement["raw_poi_id"]
    rewritten.loc[changed_row, "poi_id"] = int(replacement["poi_id"])
    rewritten_path = benchmark["root"] / "events_post_anchor_changed.parquet"
    rewritten.to_parquet(rewritten_path, index=False)
    changed_index = FullHistoryIndex(rewritten_path, source.catalog_path,
                                     train_cutoff_s=benchmark["cutoff"])

    # The rewrite is visible to the index (the user's final seen set moved) ...
    censored_gap = f"gowalla:{changed_source:012d}:censored"
    assert (
        benchmark["index"].seen_state(censored_gap).counts.tolist()
        != changed_index.seen_state(censored_gap).counts.tolist()
    )
    # ... yet the pre-anchor queries score identically.
    model.attach_memory(changed_index)
    try:
        after = model.prepare_forecast(batch, chunk_size=model.number_of_pois)
        assert torch.equal(reference, after.score(candidates))
    finally:
        model.attach_memory(benchmark["index"])


def test_predict_returns_normalized_log_probabilities_and_a_conditioned_time_law(
    benchmark, model,
):
    batch = collate_canonical_samples(
        [benchmark["observed_known"][0], benchmark["censored"][0]]
    )
    prediction = model.predict(batch)
    prediction.validate(model.number_of_pois)
    assert prediction.poi_logits.shape == (2, model.number_of_pois)
    # Reserved IDs stay at the masked floor; the catalog rows are one
    # distribution each.
    assert bool((prediction.poi_logits[:, :2] <= MASKED / 2).all())
    log_normalizers = torch.logsumexp(prediction.poi_logits[:, 2:].to(torch.float64),
                                      dim=1)
    assert torch.allclose(log_normalizers, torch.zeros_like(log_normalizers),
                          atol=1e-5, rtol=0.0)
    # Survival-conditioned residual law: log S(0 | s) == 0 exactly.
    survival_at_zero = prediction.time_distribution.log_survival(torch.zeros(2))
    assert torch.equal(survival_at_zero, torch.zeros(2, dtype=torch.float64))
    assert prediction.extras is not None
    assert "repeat_probability_catalog" in prediction.extras
