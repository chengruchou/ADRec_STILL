import json
from pathlib import Path

import pytest
import torch
import yaml

from adrec_still.data.collate import collate_canonical_samples
from adrec_still.data.schema import CanonicalSample
from adrec_still.evaluation.evaluator import evaluate
from adrec_still.models.baselines.harness import (
    aggregate_evaluations,
    evaluate_split,
    resolve_experiment_config,
    run_baseline_experiment,
)
from adrec_still.models.baselines.popularity import (
    GlobalPop,
    poi_catalog_counts,
    train_catalog_size,
    train_target_counts,
)

REPOSITORY_ROOT = Path(__file__).parents[3]
DATASET_CONFIG = REPOSITORY_ROOT / "tests" / "fixtures" / "configs" / "dataset" / "gowalla.yaml"
MODEL_CONFIG = REPOSITORY_ROOT / "tests" / "fixtures" / "configs" / "model" / "global_pop.yaml"
EXPERIMENT_CONFIG = REPOSITORY_ROOT / "tests" / "fixtures" / "configs" / "experiment" / "gowalla_global_pop.yaml"

LANDMARK_SECONDS = {"0", "21600", "43200", "86400", "172800", "259200", "604800"}


def toy_sample(next_poi: int, query_s_sec: int = 0) -> CanonicalSample:
    timestamps = [1_577_836_800, 1_577_840_400, 1_577_844_000]
    return CanonicalSample(
        user_id=2,
        history_poi=[2, 3, 2],
        history_timestamp=timestamps,
        last_timestamp=timestamps[-1],
        query_s_sec=query_s_sec,
        event_observed=True,
        target_in_train_catalog=True,
        residual_time_sec=7200,
        next_poi=next_poi,
    )


def test_committed_experiment_config_resolves(monkeypatch):
    monkeypatch.chdir(REPOSITORY_ROOT / "tests" / "fixtures")
    resolved = resolve_experiment_config(EXPERIMENT_CONFIG)
    assert resolved["model"]["name"] == "global_pop"
    assert resolved["model"]["family"] == "baseline"
    assert resolved["dataset"]["name"] == "gowalla"
    assert resolved["experiment"]["evaluation"]["splits"] == ["validation", "test"]


def test_training_counts_use_training_gaps_only(synthetic_benchmark):
    target_counts = train_target_counts(synthetic_benchmark["landmarks"], split="train")
    catalog_counts = poi_catalog_counts(synthetic_benchmark["catalog"])
    assert target_counts
    assert set(target_counts) <= set(catalog_counts)
    assert 0 not in target_counts and 1 not in target_counts
    validation_counts = train_target_counts(synthetic_benchmark["landmarks"], split="validation")
    assert sum(target_counts.values()) > sum(validation_counts.values())


def test_end_to_end_global_pop_run_writes_reproducible_records(synthetic_benchmark):
    written = run_baseline_experiment(synthetic_benchmark["experiment"])
    assert written["resolved_config"].exists()
    resolved = json.loads(written["resolved_config"].read_text(encoding="utf-8"))
    assert resolved["split_manifest_sha256"]
    assert resolved["dataset"]["protocol_version"] == "v0.1"
    assert resolved["git_dirty"] is False

    record = json.loads(written["test_seed41"].read_text(encoding="utf-8"))
    assert record["seed"] == 41
    assert record["split"] == "test"
    assert record["model"]["name"] == "global_pop"
    assert record["model"]["parameter_count"] > 0
    assert record["split_manifest_sha256"] == resolved["split_manifest_sha256"]
    assert record["git_dirty"] is False

    metrics = record["metrics"]
    ranking = metrics["ranking"]
    assert metrics["evaluated_sample_count"] == metrics["split_sample_count"]
    assert ranking["sample_count"] > 0
    assert ranking["sample_count"] <= ranking["observed_target_count"]
    assert 0.0 <= ranking["ranking_coverage"] <= 1.0
    assert 0.0 <= ranking["recall@5"] <= 1.0
    assert ranking["ndcg@5"] <= ranking["recall@5"] + 1e-9
    assert ranking["mrr@5"] <= ranking["recall@5"] + 1e-9
    assert ranking["candidate_count"] > 0
    assert ranking["random_floor"]["recall@5"] == pytest.approx(
        min(5, ranking["candidate_count"]) / ranking["candidate_count"]
    )
    assert ranking["poi_nll"] > 0.0
    assert set(metrics["sample_counts_by_cohort"]) == {"censored", "observed"}
    assert set(metrics["by_horizon_seconds"]) <= LANDMARK_SECONDS
    horizon_total = sum(
        summary["sample_counts_by_horizon_seconds"][horizon]
        for horizon, summary in metrics["by_horizon_seconds"].items()
    )
    assert horizon_total == metrics["evaluated_sample_count"]


def test_full_runs_from_a_dirty_worktree_proceed_and_record_it(synthetic_benchmark, monkeypatch):
    """Git state is metadata, not a gate: a dirty worktree trains and says so."""
    monkeypatch.setattr("adrec_still.models.baselines.harness.git_worktree_dirty", lambda: True)
    written = run_baseline_experiment(
        synthetic_benchmark["experiment"],
        splits=["validation"],
        output_dir=synthetic_benchmark["root"] / "run_dirty_full",
    )
    record = json.loads(written["validation_seed41"].read_text(encoding="utf-8"))
    assert record["git_dirty"] is True
    assert record["max_batches"] is None
    monkeypatch.setattr("adrec_still.models.baselines.harness.git_worktree_dirty", lambda: None)
    written = run_baseline_experiment(
        synthetic_benchmark["experiment"],
        splits=["validation"],
        output_dir=synthetic_benchmark["root"] / "run_no_git",
    )
    record = json.loads(written["validation_seed41"].read_text(encoding="utf-8"))
    assert record["git_dirty"] is None


def test_repeated_runs_are_identical_across_seeds(synthetic_benchmark):
    first = run_baseline_experiment(
        synthetic_benchmark["experiment"],
        splits=["validation"],
        output_dir=synthetic_benchmark["root"] / "run_a",
    )
    second = run_baseline_experiment(
        synthetic_benchmark["experiment"],
        splits=["validation"],
        seeds=[43],
        output_dir=synthetic_benchmark["root"] / "run_b",
    )
    first_metrics = json.loads(first["validation_seed41"].read_text(encoding="utf-8"))["metrics"]
    second_metrics = json.loads(second["validation_seed43"].read_text(encoding="utf-8"))["metrics"]
    assert first_metrics == second_metrics


def test_poi_catalog_count_source_is_supported(synthetic_benchmark):
    experiment = yaml.safe_load(synthetic_benchmark["experiment"].read_text(encoding="utf-8"))
    model_config = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    model_config["count_source"] = "poi_catalog"
    model_path = synthetic_benchmark["root"] / "global_pop_catalog.yaml"
    model_path.write_text(yaml.safe_dump(model_config), encoding="utf-8")
    experiment["model_config"] = str(model_path)
    experiment_path = synthetic_benchmark["root"] / "experiment_catalog.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    written = run_baseline_experiment(
        experiment_path,
        splits=["validation"],
        output_dir=synthetic_benchmark["root"] / "run_catalog",
    )
    record = json.loads(written["validation_seed41"].read_text(encoding="utf-8"))
    assert record["model"]["count_source"] == "poi_catalog"
    assert record["metrics"]["ranking"]["sample_count"] > 0


def test_max_batches_truncates_a_smoke_run(synthetic_benchmark):
    counts = train_target_counts(synthetic_benchmark["landmarks"], split="train")
    model = GlobalPop(train_catalog_size(synthetic_benchmark["catalog"])).fit(counts)
    full = evaluate_split(model, synthetic_benchmark["landmarks"], split="test", batch_size=8)
    truncated = evaluate_split(
        model, synthetic_benchmark["landmarks"], split="test", batch_size=8, max_batches=1
    )
    assert truncated["split_sample_count"] == full["split_sample_count"]
    assert 0 < truncated["evaluated_sample_count"] <= 8
    assert truncated["evaluated_sample_count"] < full["evaluated_sample_count"]


def test_batch_aggregation_matches_one_whole_split_call():
    model = GlobalPop(8).fit({2: 60, 3: 30, 4: 8, 5: 2})
    samples = [toy_sample(2 + index % 4) for index in range(9)]
    whole = collate_canonical_samples(samples)
    whole_result = evaluate(model.predict(whole).poi_logits, whole, split="test")
    parts = []
    for start in (0, 3, 6):
        chunk = collate_canonical_samples(samples[start : start + 3])
        parts.append(evaluate(model.predict(chunk).poi_logits, chunk, split="test"))
    combined = aggregate_evaluations(parts, split="test")
    # Counts and the float64 ranking metrics recombine exactly. poi_nll comes from a
    # float32 cross-entropy, so a weighted recombination differs by about one ulp.
    for key, value in whole_result["ranking"].items():
        if value is None:
            assert combined["ranking"][key] is None
        elif key in {"candidate_count", "random_floor"}:
            assert combined["ranking"][key] == value
        elif key == "poi_nll":
            assert combined["ranking"][key] == pytest.approx(value, rel=1e-6)
        else:
            assert combined["ranking"][key] == pytest.approx(value, rel=1e-9, abs=1e-9)
    assert (
        combined["sample_counts_by_horizon_seconds"]
        == whole_result["sample_counts_by_horizon_seconds"]
    )
    assert combined["sample_counts_by_cohort"] == whole_result["sample_counts_by_cohort"]


def test_aggregation_rejects_empty_and_inconsistent_results():
    model = GlobalPop(8).fit({2: 1})
    batch = collate_canonical_samples([toy_sample(2)])
    result = evaluate(model.predict(batch).poi_logits, batch, split="test")
    other = json.loads(json.dumps(result))
    other["schema_version"] = "still-evaluation-v9.9"
    with pytest.raises(ValueError):
        aggregate_evaluations([])
    with pytest.raises(ValueError):
        aggregate_evaluations([result, other])

    different_candidates = json.loads(json.dumps(result))
    different_candidates["ranking"]["candidate_count"] += 1
    with pytest.raises(ValueError, match="candidate_count"):
        aggregate_evaluations([result, different_candidates])


def test_prediction_contract_is_enforced():
    model = GlobalPop(8).fit({2: 1})
    prediction = model.predict(collate_canonical_samples([toy_sample(2)]))
    prediction.validate(8)
    assert torch.isfinite(prediction.poi_logits).all()
    with pytest.raises(ValueError):
        prediction.validate(9)
