"""Shared-trainer integration tests on the tiny end-to-end canonical benchmark."""

import json
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from adrec_still.data.collate import collate_canonical_samples
from adrec_still.data.dataset import LandmarkDataset
from adrec_still.data.pipeline import run_preprocessing
from adrec_still.models.baselines.trainer import RowGroupLocalShuffleSampler, load_checkpoint
from adrec_still.training.service import RunSpec, run

REPOSITORY_ROOT = Path(__file__).parents[3]
DATASET_CONFIG = REPOSITORY_ROOT / "tests" / "fixtures" / "configs" / "dataset" / "gowalla.yaml"
MODEL_CONFIG = REPOSITORY_ROOT / "tests" / "fixtures" / "configs" / "model" / "gru4rec_h.yaml"
EXPERIMENT_CONFIG = REPOSITORY_ROOT / "tests" / "fixtures" / "configs" / "experiment" / "gowalla_gru4rec_h.yaml"

GAP_HOURS = (1, 7, 13, 25, 49, 73, 169, 2, 5, 30, 100, 200, 3)
POI_CYCLE = ("p000", "p000", "p000", "p001", "p000", "p001", "p002", "p000", "p001", "p003")
USER_COUNT = 8
EVENTS_PER_USER = 14
LATE_EVENT_INDEX = 11


def write_synthetic_raw(path: Path) -> Path:
    """The deterministic Gowalla-shaped fixture shared with the harness tests."""
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    lines = []
    for user in range(USER_COUNT):
        moment = start + timedelta(hours=user)
        for index in range(EVENTS_PER_USER):
            cold_start = user % 2 == 1 and index >= LATE_EVENT_INDEX
            poi = f"p_late_{user}_{index}" if cold_start else POI_CYCLE[index % len(POI_CYCLE)]
            lines.append(
                "\t".join(
                    [
                        f"u{user}",
                        moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        f"{25.0 + user * 0.1:.4f}",
                        f"{121.0 + index * 0.1:.4f}",
                        poi,
                    ]
                )
            )
            moment += timedelta(hours=GAP_HOURS[index % len(GAP_HOURS)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def synthetic_benchmark(tmp_path_factory) -> dict[str, Path]:
    """Tiny canonical benchmark plus a fast committed-shape GRU4Rec-H experiment."""
    root = tmp_path_factory.mktemp("gru4rec_h")
    raw = write_synthetic_raw(root / "loc-gowalla_totalCheckins.txt")
    processed = root / "processed"
    paths = run_preprocessing(
        config_path=DATASET_CONFIG,
        input_path=raw,
        output_dir=processed,
        manifest_path=root / "gowalla_v0.1.json",
        audit_path=root / "gowalla_audit.json",
    )
    dataset_config = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))
    dataset_config["processed_root"] = str(processed)
    dataset_config_path = root / "dataset.yaml"
    dataset_config_path.write_text(yaml.safe_dump(dataset_config), encoding="utf-8")

    model_config = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    model_config.update(
        {"embedding_dim": 16, "hidden_dim": 16, "num_layers": 1, "dropout": 0.0}
    )
    model_config["training"].update(
        {
            "batch_size": 32,
            "max_epochs": 2,
            "early_stopping_patience": 1,
            "negative_samples": 8,
            "validation_max_batches": 2,
            "device": "cpu",
        }
    )
    model_config_path = root / "model.yaml"
    model_config_path.write_text(yaml.safe_dump(model_config), encoding="utf-8")

    experiment = yaml.safe_load(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    # This fixture covers the legacy unregistered trainer; a real Gowalla
    # registration must never be silently reused for these synthetic identities.
    experiment.pop("selection_manifest", None)
    experiment["dataset_config"] = str(dataset_config_path)
    experiment["model_config"] = str(model_config_path)
    experiment["manifest"] = str(paths["manifest"])
    experiment["output_dir"] = str(root / "results")
    experiment["seeds"] = [41]
    experiment["evaluation"]["batch_size"] = 16
    experiment_path = root / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    return {
        "root": root,
        "landmarks": paths["landmarks"],
        "experiment": experiment_path,
    }


@pytest.fixture(scope="module")
def multi_group_landmarks(synthetic_benchmark) -> Path:
    """The synthetic landmark table rewritten into many small Parquet row groups."""
    table = pq.read_table(synthetic_benchmark["landmarks"])
    path = synthetic_benchmark["root"] / "landmarks_multigroup.parquet"
    pq.write_table(table, path, row_group_size=32)
    return path


def test_row_group_local_sampler_visits_each_row_exactly_once(multi_group_landmarks):
    dataset = LandmarkDataset(multi_group_landmarks, split="train")
    lengths = dataset.row_group_lengths()
    assert len(lengths) > 3
    assert sum(lengths) == len(dataset)

    sampler = RowGroupLocalShuffleSampler(lengths, seed=41)
    first_epoch = list(iter(sampler))
    second_epoch = list(iter(sampler))
    assert sorted(first_epoch) == list(range(len(dataset)))
    assert sorted(second_epoch) == list(range(len(dataset)))
    assert first_epoch != second_epoch
    assert list(iter(RowGroupLocalShuffleSampler(lengths, seed=41))) == first_epoch

    offsets = [sum(lengths[:index]) for index in range(len(lengths))]

    def owning_group(row_index: int) -> int:
        return max(index for index, offset in enumerate(offsets) if offset <= row_index)

    groups_in_order = [owning_group(index) for index in first_epoch]
    seen_groups = []
    for group in groups_in_order:
        if not seen_groups or seen_groups[-1] != group:
            seen_groups.append(group)
    assert len(seen_groups) == len(set(seen_groups)), "a row group was revisited mid-epoch"


def test_shuffled_training_epoch_reads_each_row_group_at_most_once(
    multi_group_landmarks, monkeypatch
):
    dataset = LandmarkDataset(multi_group_landmarks, split="train")
    read_calls: list[int] = []
    original_read = pq.ParquetFile.read_row_group

    def counting_read(self, row_group, *args, **kwargs):
        read_calls.append(row_group)
        return original_read(self, row_group, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", counting_read)
    loader = DataLoader(
        dataset,
        batch_size=16,
        sampler=RowGroupLocalShuffleSampler(dataset.row_group_lengths(), seed=41),
        collate_fn=partial(collate_canonical_samples, max_sequence_length=100),
    )
    visited_rows = sum(int(batch["query_s_sec"].shape[0]) for batch in loader)
    assert visited_rows == len(dataset)
    non_empty_groups = sum(1 for length in dataset.row_group_lengths() if length)
    assert len(read_calls) <= non_empty_groups


def test_committed_experiment_config_resolves(monkeypatch):
    from adrec_still.models.baselines.harness import resolve_experiment_config

    monkeypatch.chdir(REPOSITORY_ROOT / "tests" / "fixtures")
    resolved = resolve_experiment_config(EXPERIMENT_CONFIG)
    assert resolved["model"]["name"] == "gru4rec_h"
    assert resolved["model"]["training"]["landmark_weighting"] == "per_gap"
    assert resolved["model"]["training"]["optimizer"] == "adam"
    assert resolved["experiment"]["seeds"] == [41, 42, 43]
    assert resolved["experiment"]["evaluation"]["splits"] == ["validation", "test"]


def test_end_to_end_trained_run_writes_the_run_directory(synthetic_benchmark):
    output = synthetic_benchmark["root"] / "run_a"
    summary = run(RunSpec(
        data=None, output=output, config=synthetic_benchmark["experiment"], seed=41,
        device="cpu", progress="off",
    ))
    assert summary["status"] == "completed"
    assert summary["model"] == "gru4rec_h"
    assert summary["resolved_model"]["parameter_count"] > 0
    assert summary["resolved_model"]["context_dim"] == 16
    assert summary["seed"] == 41

    training = summary["training"]
    assert training["training"]["landmark_weighting"] == "per_gap"
    assert training["selection"]["metric"] == "ndcg@10"
    assert training["selection"]["horizon"] == 86400
    assert 1 <= training["best"]["epoch"] <= training["epochs_run"] <= 2
    assert training["best"]["value"] is not None

    test = summary["evaluation"]["test"]
    assert test["rows"] > 0 and test["candidate_count"] > 0
    assert "mrr@10" in test

    # Retained files: config, metrics, summary, and exactly two model files.
    names = sorted(path.name for path in output.iterdir() if path.is_file())
    assert names == ["best.pt", "config.json", "last.pt", "metrics.jsonl", "summary.json"]
    curve = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in curve] == list(range(1, training["epochs_run"] + 1))
    assert all(row["monitor_value"] is not None for row in curve)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["data"]["number_of_pois"] > 2
    assert set(config["git"]) >= {"commit", "dirty"}


def test_runs_outside_a_git_checkout_record_null_provenance(synthetic_benchmark, monkeypatch):
    monkeypatch.setattr(
        "adrec_still.training.service.git_metadata",
        lambda: {"commit": None, "dirty": None, "note": "not a Git checkout"},
    )
    output = synthetic_benchmark["root"] / "run_no_git"
    summary = run(RunSpec(
        data=None, output=output, config=synthetic_benchmark["experiment"], seed=41,
        device="cpu", progress="off", max_train_batches=1, epochs=1, evaluate_test=False,
    ))
    assert summary["status"] == "completed"
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["git"]["commit"] is None and config["git"]["dirty"] is None


def test_same_seed_reruns_are_identical_and_checkpoints_reload(synthetic_benchmark):
    def rerun(name):
        output = synthetic_benchmark["root"] / name
        summary = run(RunSpec(
            data=None, output=output, config=synthetic_benchmark["experiment"], seed=41,
            device="cpu", progress="off", evaluate_test=False,
        ))
        curve = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        return output, summary, curve

    first_output, first, first_curve = rerun("run_repro_a")
    second_output, second, second_curve = rerun("run_repro_b")
    assert [row["train_loss"] for row in first_curve] == [row["train_loss"] for row in second_curve]
    assert [row["monitor_value"] for row in first_curve] == [
        row["monitor_value"] for row in second_curve
    ]
    assert first["training"]["best"] == second["training"]["best"]

    reloaded = load_checkpoint(first_output / "best.pt")
    twin = load_checkpoint(second_output / "best.pt")
    assert all(
        torch.equal(value, twin.state_dict()[key]) for key, value in reloaded.state_dict().items()
    )
    dataset = LandmarkDataset(synthetic_benchmark["landmarks"], split="validation")
    batch = collate_canonical_samples([dataset[i] for i in range(8)], max_sequence_length=100)
    assert torch.equal(reloaded.predict(batch).poi_logits, twin.predict(batch).poi_logits)
