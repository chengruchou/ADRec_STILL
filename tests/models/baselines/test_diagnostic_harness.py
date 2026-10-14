"""End-to-end coverage for the Issue #14 diagnostic baselines on the shared harness."""

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from adrec_still.data.schema import PADDING_POI_ID, UNKNOWN_POI_ID
from adrec_still.models.baselines.harness import (
    resolve_experiment_config,
    run_baseline_experiment,
)
from adrec_still.models.baselines.markov import train_transition_counts

REPOSITORY_ROOT = Path(__file__).parents[3]
DIAGNOSTIC_EXPERIMENTS = {
    "user_history_pop": "gowalla_user_history_pop.yaml",
    "repeat_last": "gowalla_repeat_last.yaml",
    "first_order_markov": "gowalla_first_order_markov.yaml",
}


def diagnostic_experiment_path(root: Path, benchmark: dict[str, Path], model_name: str) -> Path:
    """Point one committed diagnostic experiment config at the synthetic benchmark."""
    committed = REPOSITORY_ROOT / "tests" / "fixtures" / "configs" / "experiment" / DIAGNOSTIC_EXPERIMENTS[model_name]
    experiment = yaml.safe_load(committed.read_text(encoding="utf-8"))
    experiment["dataset_config"] = str(benchmark["dataset_config"])
    model_config = yaml.safe_load(
        (REPOSITORY_ROOT / "tests" / "fixtures" / "configs" / "model" / f"{model_name}.yaml").read_text(
            encoding="utf-8"
        )
    )
    if model_name == "user_history_pop":
        # The synthetic catalog concentrates GlobalPop mass far more than real
        # Gowalla, so the committed max_history_length of 100 cannot satisfy the
        # frequency-first guard here; shorten it (and the matching evaluation
        # truncation) instead of weakening the guard.
        model_config["max_history_length"] = 10
        experiment["evaluation"]["max_sequence_length"] = 10
    model_config_path = root / f"model_{model_name}.yaml"
    model_config_path.write_text(yaml.safe_dump(model_config), encoding="utf-8")
    experiment["model_config"] = str(model_config_path)
    experiment["manifest"] = str(benchmark["manifest"])
    experiment["output_dir"] = str(root / f"results_{model_name}")
    experiment["evaluation"]["batch_size"] = 16
    path = root / f"experiment_{model_name}.yaml"
    path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    return path


def test_committed_diagnostic_experiment_configs_resolve(monkeypatch):
    monkeypatch.chdir(REPOSITORY_ROOT / "tests" / "fixtures")
    for model_name, experiment_name in DIAGNOSTIC_EXPERIMENTS.items():
        resolved = resolve_experiment_config(
            REPOSITORY_ROOT / "tests" / "fixtures" / "configs" / "experiment" / experiment_name
        )
        assert resolved["model"]["name"] == model_name
        assert resolved["model"]["family"] == "baseline"
        assert resolved["experiment"]["evaluation"]["splits"] == ["validation", "test"]


def test_transition_counts_use_exactly_one_transition_per_training_gap(synthetic_benchmark):
    counts, stats = train_transition_counts(synthetic_benchmark["landmarks"], split="train")
    table = pq.read_table(
        synthetic_benchmark["landmarks"],
        columns=[
            "split",
            "query_s_sec",
            "event_observed",
            "target_in_train_catalog",
            "history_poi",
        ],
    )
    expected_gaps = 0
    expected_transitions = 0
    later_landmarks = 0
    for split, horizon, observed, known, history in zip(
        table.column("split").to_pylist(),
        table.column("query_s_sec").to_pylist(),
        table.column("event_observed").to_pylist(),
        table.column("target_in_train_catalog").to_pylist(),
        table.column("history_poi").to_pylist(),
        strict=True,
    ):
        if split != "train" or not observed or not known:
            continue
        if horizon != 0:
            # The same gap seen from a later landmark: it must NOT add a transition.
            later_landmarks += 1
            continue
        expected_gaps += 1
        # The transition state is the last *observed* entry: a trailing UNKNOWN
        # forms no transition even when an older known entry exists.
        last_observed = next(
            (poi for poi in reversed(history) if poi != PADDING_POI_ID), PADDING_POI_ID
        )
        if last_observed not in (PADDING_POI_ID, UNKNOWN_POI_ID):
            expected_transitions += 1
    assert later_landmarks > 0, "fixture must contain multi-landmark gaps for this test"
    assert stats["training_gap_count"] == expected_gaps
    assert sum(counts.values()) == expected_transitions
    assert all(
        state not in (PADDING_POI_ID, UNKNOWN_POI_ID)
        and target not in (PADDING_POI_ID, UNKNOWN_POI_ID)
        for state, target in counts
    )


def test_transition_fitting_never_reads_validation_or_test_gaps(synthetic_benchmark):
    train_counts, train_stats = train_transition_counts(
        synthetic_benchmark["landmarks"], split="train"
    )
    validation_counts, validation_stats = train_transition_counts(
        synthetic_benchmark["landmarks"], split="validation"
    )
    assert train_stats["training_gap_count"] > validation_stats["training_gap_count"]
    assert sum(train_counts.values()) > sum(validation_counts.values())


@pytest.mark.parametrize("model_name", sorted(DIAGNOSTIC_EXPERIMENTS))
def test_end_to_end_diagnostic_baseline_run_records_stats(
    synthetic_benchmark, tmp_path, model_name
):
    experiment_path = diagnostic_experiment_path(tmp_path, synthetic_benchmark, model_name)
    written = run_baseline_experiment(experiment_path, splits=["validation"])
    record = json.loads(written["validation_seed41"].read_text(encoding="utf-8"))
    assert record["model"]["name"] == model_name
    assert record["git_dirty"] is False
    assert record["split_manifest_sha256"]

    assert record["metrics"]["schema_version"] == "still-evaluation-v0.2"
    ranking = record["metrics"]["ranking"]
    assert ranking["sample_count"] > 0
    assert 0.0 <= ranking["recall@5"] <= 1.0
    assert ranking["mrr@5"] <= ranking["recall@5"] + 1e-9
    assert ranking["candidate_count"] > 0
    assert ranking["random_floor"]["recall@5"] == pytest.approx(
        min(5, ranking["candidate_count"]) / ranking["candidate_count"]
    )
    assert ranking["poi_nll"] > 0.0

    stats = record["prediction_stats"]
    assert stats["predicted_row_count"] == record["metrics"]["evaluated_sample_count"]
    assert 0.0 <= stats["fallback_rate"] <= 1.0
    if model_name == "first_order_markov":
        assert 0.0 <= stats["transition_support_rate"] <= 1.0
        assert (
            stats["supported_row_count"] + stats["fallback_row_count"]
            == stats["predicted_row_count"]
        )


def test_diagnostic_baseline_metrics_are_identical_across_seeds(
    synthetic_benchmark, tmp_path
):
    experiment_path = diagnostic_experiment_path(
        tmp_path, synthetic_benchmark, "user_history_pop"
    )
    first = run_baseline_experiment(
        experiment_path, splits=["validation"], output_dir=tmp_path / "run_a"
    )
    second = run_baseline_experiment(
        experiment_path, splits=["validation"], seeds=[43], output_dir=tmp_path / "run_b"
    )
    first_record = json.loads(first["validation_seed41"].read_text(encoding="utf-8"))
    second_record = json.loads(second["validation_seed43"].read_text(encoding="utf-8"))
    assert first_record["metrics"] == second_record["metrics"]
    assert first_record["prediction_stats"] == second_record["prediction_stats"]
