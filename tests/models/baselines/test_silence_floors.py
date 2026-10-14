"""Matched controls for empirical, destination-dependent silence floor variants."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from adrec_still.data.collate import collate_canonical_samples
from adrec_still.data.landmarks import LANDMARK_SCHEMA, LANDMARK_SECONDS
from adrec_still.data.schema import CanonicalSample
from adrec_still.models.baselines.harness import (
    build_fitted_model,
    resolve_experiment_config,
    run_baseline_experiment,
)
from adrec_still.models.baselines.popularity import GlobalPop
from adrec_still.models.baselines.silence import (
    SILENCE_FLOOR_BASES,
    SilenceAwareFloor,
    fit_destination_survival,
)

REPOSITORY_ROOT = Path(__file__).parents[3]
WIDTH = 8
SIX_HOURS = 21600


def landmark(index, target=2, duration=3600, **overrides):
    result = {
        "sample_id": f"gap{index}:q0",
        "gap_id": f"gap{index}",
        "user_id": 2,
        "split": "train",
        "history_poi": [2],
        "history_timestamp": [1000000],
        "last_timestamp": 1000000,
        "query_s_sec": 0,
        "event_observed": True,
        "target_in_train_catalog": True,
        "residual_time_sec": duration,
        "next_poi": target,
        "censor_time_sec": None,
    }
    return {**result, **overrides}


def write_landmarks(path, rows):
    pq.write_table(pa.Table.from_pylist(rows, schema=LANDMARK_SCHEMA), path)


@pytest.fixture
def processed(tmp_path):
    rows = [landmark(index) for index in range(100)]
    rows += [landmark(index + 100, target=3, duration=43200) for index in range(10)]
    write_landmarks(tmp_path / "landmark_samples.parquet", rows)
    pq.write_table(
        pa.table({"poi_id": list(range(2, WIDTH)), "train_count": [100, 10, 1, 1, 1, 1]}),
        tmp_path / "poi_catalog.parquet",
    )
    return tmp_path


def query(horizons=(0, SIX_HOURS), history=(2,)):
    return collate_canonical_samples([
        CanonicalSample(
            user_id=2, history_poi=list(history), history_timestamp=[1000000] * len(history),
            last_timestamp=1000000, query_s_sec=horizon, event_observed=True,
            target_in_train_catalog=True, residual_time_sec=1000, next_poi=3,
        )
        for horizon in horizons
    ])


def fitted(name, processed):
    return build_fitted_model(
        {"name": name, "max_history_length": 10, "silence_alpha": 1.0}, processed
    )


@pytest.mark.parametrize("name", SILENCE_FLOOR_BASES)
def test_all_four_match_original_at_zero_and_change_top_rank_with_silence(name, processed):
    model = fitted(name, processed)
    batch = query()
    base = model.base_model.predict(batch)
    prediction = model.predict(batch)
    assert torch.equal(prediction.poi_logits[0], base.poi_logits[0])
    assert base.poi_logits.argmax(dim=1).tolist() == [2, 2]
    assert prediction.poi_logits.argmax(dim=1).tolist() == [2, 3]
    probabilities = prediction.poi_logits.exp()
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))
    assert torch.equal(probabilities[:, :2], torch.zeros(2, 2))
    assert prediction.time_prediction is None and prediction.time_distribution is None
    assert model.prediction_stats()["silence_adjusted_row_count"] == 1


@pytest.mark.parametrize("name", SILENCE_FLOOR_BASES)
def test_query_future_labels_and_calendar_fields_cannot_change_prediction(name, processed):
    model = fitted(name, processed)
    batch = query()
    expected = model.predict(batch).poi_logits
    poisoned = {
        key: value.clone() if isinstance(value, torch.Tensor) else list(value)
        for key, value in batch.items()
    }
    poisoned["next_poi"].fill_(7)
    poisoned["observed_mask"].fill_(False)
    poisoned["censored_mask"].fill_(True)
    poisoned["target_in_train_catalog"].fill_(False)
    poisoned["residual_time_sec"].fill_(10**9)
    poisoned["censor_time_sec"].fill_(10**9)
    poisoned["last_timestamp"].fill_(10**9)
    poisoned["history_timestamp"].fill_(10**9)
    assert torch.equal(model.predict(poisoned).poi_logits, expected)


def test_fit_ignores_heldout_censored_unknown_and_repeated_nonzero_landmarks(tmp_path):
    clean = [landmark(0, duration=SIX_HOURS), landmark(1, target=3, duration=SIX_HOURS + 1)]
    contaminated = clean + [
        landmark(0, query_s_sec=SIX_HOURS, duration=1),
        landmark(0, query_s_sec=43200, duration=1),
        landmark(2, target=7, duration=10**9, split="validation"),
        landmark(3, target=7, duration=10**9, split="test"),
        landmark(4, target=None, duration=None, event_observed=False,
                 target_in_train_catalog=False, censor_time_sec=10**9),
        landmark(5, target=1, target_in_train_catalog=False),
    ]
    clean_path, other_path = tmp_path / "clean.parquet", tmp_path / "other.parquet"
    write_landmarks(clean_path, clean)
    write_landmarks(other_path, contaminated)
    expected = fit_destination_survival(clean_path, WIDTH, batch_size=1)
    actual = fit_destination_survival(other_path, WIDTH, batch_size=2)
    assert torch.equal(actual.survivor_counts, expected.survivor_counts)
    assert actual.gap_count == 2
    assert actual.survivor_counts[1, 2] == 0  # T=s has already returned.
    assert actual.survivor_counts[1, 3] == 1  # Strict T>s eligibility.


def test_alpha_uses_pooled_fraction_and_unseen_destinations_get_pooled_factor(processed):
    model = fitted("global_pop_s", processed)
    pooled = 10 / 110
    expected_seen = (0 + pooled) / (100 + 1)
    assert model._log_survival[1, 2].exp().item() == pytest.approx(expected_seen)
    assert model._log_survival[1, 7].exp().item() == pytest.approx(pooled)
    # A destination supported by one long gap is shrunk, not assigned certainty.
    rows = [landmark(0), landmark(1, target=3, duration=SIX_HOURS + 1)]
    write_landmarks(processed / "landmark_samples.parquet", rows)
    model = fitted("global_pop_s", processed)
    assert model._log_survival[1, 3].exp().item() == pytest.approx(0.75)


@pytest.mark.parametrize("name", SILENCE_FLOOR_BASES)
def test_zero_pooled_support_returns_original_and_reports_fallback(name, processed):
    model = fitted(name, processed)
    batch = query(horizons=(0, 43200, 604800))
    expected = model.base_model.predict(batch).poi_logits
    assert torch.equal(model.predict(batch).poi_logits, expected)
    stats = model.prediction_stats()
    assert stats["unsupported_horizon_fallback_row_count"] == 2
    assert stats["unsupported_horizon_fallback_rate"] == pytest.approx(2 / 3)
    config = model.resolved_config()
    assert config["pooled_survivor_counts"]["43200"] == 0
    assert config["training_gap_count"] == 110
    assert config["silence_alpha"] == 1.0
    assert config["horizons_seconds"] == list(LANDMARK_SECONDS)
    assert "censor" in config["limitation"]
    json.dumps(config, allow_nan=False)
    model.reset_prediction_stats()
    assert model.prediction_stats()["predicted_row_count"] == 0


@pytest.mark.parametrize("horizon", [1, -1, 21601, 604801])
def test_unregistered_horizons_are_rejected(horizon, processed):
    batch = query(horizons=(0,))
    batch["query_s_sec"].fill_(horizon)
    with pytest.raises(ValueError, match="fitted horizons"):
        fitted("global_pop_s", processed).predict(batch)


@pytest.mark.parametrize("alpha", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_shrinkage_is_rejected(alpha):
    with pytest.raises(ValueError, match="finite and positive"):
        SilenceAwareFloor(GlobalPop(WIDTH), silence_alpha=alpha)


@pytest.mark.parametrize("duration", [None, 0, -1])
def test_invalid_observed_training_duration_is_rejected(duration, tmp_path):
    path = tmp_path / "invalid.parquet"
    write_landmarks(path, [landmark(0, duration=duration)])
    with pytest.raises(ValueError, match="positive duration"):
        fit_destination_survival(path, WIDTH)


@pytest.mark.parametrize("name", SILENCE_FLOOR_BASES)
def test_committed_configs_resolve_without_overwriting_original_results(name, monkeypatch):
    monkeypatch.chdir(REPOSITORY_ROOT / "tests" / "fixtures")
    resolved = resolve_experiment_config(f"configs/experiment/gowalla_{name}.yaml")
    assert resolved["model"]["name"] == name
    assert resolved["model"]["silence_alpha"] == 1.0
    assert resolved["experiment"]["seeds"] == [41]
    assert resolved["experiment"]["output_dir"] == f"runs/gowalla_{name}"


@pytest.mark.parametrize("name", SILENCE_FLOOR_BASES)
def test_harness_records_new_floor_fit_policy_and_prediction_stats(name, processed, monkeypatch):
    monkeypatch.chdir(REPOSITORY_ROOT / "tests" / "fixtures")
    monkeypatch.setattr("adrec_still.models.baselines.harness.git_worktree_dirty", lambda: True)
    dataset = processed / "dataset.yaml"
    dataset.write_text(yaml.safe_dump({
        "name": "fixture", "protocol_version": "v0.1", "processed_root": str(processed),
        "max_history_length": 10,
    }))
    model = processed / "model.yaml"
    model.write_text(yaml.safe_dump({
        "name": name, "family": "baseline", "max_history_length": 10, "silence_alpha": 1.0,
    }))
    experiment = processed / "experiment.yaml"
    experiment.write_text(yaml.safe_dump({
        "name": f"fixture_{name}", "dataset_config": str(dataset),
        "model_config": str(model), "seeds": [41], "output_dir": str(processed / "run"),
        "evaluation": {"splits": ["train"], "batch_size": 4, "max_sequence_length": 10},
    }))
    outputs = run_baseline_experiment(experiment, max_batches=1)
    record = json.loads(outputs["train_seed41"].read_text())
    assert record["model"]["name"] == name
    assert record["model"]["training_gap_count"] == 110
    assert record["model"]["base_model"]["name"] == SILENCE_FLOOR_BASES[name]
    assert record["prediction_stats"]["predicted_row_count"] == 4
    assert record["max_batches"] == 1 and record["git_dirty"] is True
