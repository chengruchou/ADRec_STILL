"""Shared-trainer smoke, checkpoint round-trip and statistics gates (#51)."""

import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from adrec_still.data.dataset import LandmarkDataset
from adrec_still.models.external.joint_lnm import ExternalJointLNM
from adrec_still.models.external.registry import (
    build_external_model,
    load_external_checkpoint,
)
from adrec_still.models.external.statistics import (
    LogTimeStatistics,
    fit_log_time_statistics,
)
from adrec_still.models.still.synthetic_benchmark import write_synthetic_benchmark
from adrec_still.training.checkpoint import save_weights
from adrec_still.training.families import load_model
from adrec_still.training.loop import TrainingPlan, train_model

TRAINING = {
    "training_profile": "observed-known-v0-proposed",
    "landmark_weighting": "per_gap",
    "optimizer": "adam",
    "learning_rate": 0.01,
    "batch_size": 8,
    "max_epochs": 2,
    "early_stopping_patience": 2,
    "early_stopping_metric": "ndcg@10",
    "early_stopping_mode": "max",
    "negative_samples": 4,
    "validation_max_batches": 1,
}


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("external-lnm-fixture")
    return write_synthetic_benchmark(root)


def test_shared_trainer_smoke_trains_and_checkpoints(fixture_root, tmp_path):
    landmark_path = fixture_root["landmark_path"]
    torch.manual_seed(41)
    model = ExternalJointLNM(
        fixture_root["number_of_pois"],
        mark_embedding_dim=8,
        context_dim=16,
        mixture_components=2,
    )
    plan = TrainingPlan(
        train_source=LandmarkDataset(landmark_path, split="train"),
        training=TRAINING,
        seed=41,
        max_sequence_length=8,
        validation_landmark_path=landmark_path,
        validation_batch_size=8,
        artifacts_dir=tmp_path / "artifacts",
        max_train_batches=4,
    )
    summary = train_model(model, plan)
    assert summary["epochs_run"] >= 1
    assert math.isfinite(summary["final_train_loss"])
    assert summary["selection"]["metric"] == "ndcg@10"
    assert (tmp_path / "artifacts" / "best.pt").is_file()
    # Reload through the external registry and score identically.
    reloaded = load_external_checkpoint(tmp_path / "artifacts" / "best.pt")
    twin, _payload = load_model(tmp_path / "artifacts" / "best.pt")
    assert twin.resolved_config() == model.resolved_config()
    assert reloaded.resolved_config() == model.resolved_config()
    dataset = LandmarkDataset(landmark_path, split="validation")
    from adrec_still.data.collate import collate_canonical_samples

    batch = collate_canonical_samples([dataset[i] for i in range(4)], max_sequence_length=8)
    assert torch.allclose(
        model.predict(batch).poi_logits, reloaded.predict(batch).poi_logits, atol=1e-7
    )


def test_checkpoint_restores_nondefault_configuration_and_statistics(tmp_path):
    torch.manual_seed(5)
    model = ExternalJointLNM(
        9, mark_embedding_dim=5, context_dim=11, num_layers=2, dropout=0.1,
        mixture_components=3,
    )
    model.is_fitted = True
    model.set_log_time_statistics(
        LogTimeStatistics(
            mean=7.5, std=0.25, source="train-unique-gaps", unique_gap_count=17,
            status="fitted",
        )
    )
    path = tmp_path / "best.pt"
    save_weights(path, model, family="external")
    reloaded = load_external_checkpoint(path)
    assert reloaded.resolved_config() == model.resolved_config()
    assert float(reloaded.log_time_mean) == 7.5
    assert float(reloaded.log_time_std) == 0.25
    assert reloaded.log_time_statistics_record["unique_gap_count"] == 17
    assert reloaded.mixture_components == 3 and reloaded.num_layers == 2


def _strip_population(path):
    """Rewrite a checkpoint as the pre-population-field format (legacy shape)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["model"]["log_time_statistics"]["population"]
    torch.save(payload, path)


def test_legacy_checkpoints_keep_their_statistics_provenance(tmp_path):
    """A checkpoint that predates the population field must not be relabeled: an
    identity declaration stays the fixture population, only the executable fit's
    own source maps to the canonical population, and an unknown source records an
    explicit unknown instead of a guess."""
    from adrec_still.models.external.statistics import (
        CANONICAL_TRAIN_POPULATION,
        IDENTITY_STATISTICS,
        LEGACY_UNKNOWN_POPULATION,
    )

    cases = [
        (IDENTITY_STATISTICS, IDENTITY_STATISTICS.population),
        (
            LogTimeStatistics(
                mean=7.5, std=0.25, source="train-unique-gaps", unique_gap_count=17,
                status="fitted",
            ),
            CANONICAL_TRAIN_POPULATION,
        ),
        (
            LogTimeStatistics(
                mean=0.0, std=1.0, source="train-unique-gaps", unique_gap_count=1,
                status="fallback_identity_insufficient_gaps",
            ),
            CANONICAL_TRAIN_POPULATION,
        ),
        (
            LogTimeStatistics(
                mean=6.0, std=0.5, source="preexisting-external-tool",
                unique_gap_count=9, status="fitted",
            ),
            LEGACY_UNKNOWN_POPULATION,
        ),
        # Internally inconsistent pairs this codebase could not have written are
        # an explicit unknown, never tie-broken into a named population.
        (
            LogTimeStatistics(
                mean=0.0, std=1.0, source="train-unique-gaps", unique_gap_count=0,
                status="identity_declared",
            ),
            LEGACY_UNKNOWN_POPULATION,
        ),
        (
            LogTimeStatistics(
                mean=5.0, std=0.75, source="identity", unique_gap_count=4,
                status="fitted",
            ),
            LEGACY_UNKNOWN_POPULATION,
        ),
    ]
    for index, (statistics, expected_population) in enumerate(cases):
        torch.manual_seed(index)
        model = ExternalJointLNM(9, mark_embedding_dim=4, context_dim=6)
        model.is_fitted = True
        model.set_log_time_statistics(statistics)
        path = tmp_path / f"legacy_{index}.pt"
        save_weights(path, model, family="external")
        _strip_population(path)
        reloaded = load_external_checkpoint(path)
        record = reloaded.log_time_statistics_record
        assert record["population"] == expected_population, (index, record)
        assert record["source"] == statistics.source
        assert record["status"] == statistics.status
        # Numeric behavior is untouched by the provenance inference.
        assert float(reloaded.log_time_mean) == statistics.mean
        assert float(reloaded.log_time_std) == statistics.std


def test_checkpoint_reloads_in_a_fresh_python_process(tmp_path, fixture_root):
    """Registration must not depend on import side effects already present in this
    test process: a brand-new interpreter importing only the registry must rebuild
    the model and predict."""

    torch.manual_seed(11)
    model = ExternalJointLNM(
        fixture_root["number_of_pois"], mark_embedding_dim=6, context_dim=10,
        mixture_components=3,
    )
    model.is_fitted = True
    path = tmp_path / "checkpoint.pt"
    save_weights(path, model, family="external")
    script = (
        "import torch\n"
        "from adrec_still.models.external.registry import load_external_checkpoint\n"
        f"model = load_external_checkpoint(r'{path}')\n"
        "assert model.mixture_components == 3 and model.context_dim == 10\n"
        "batch = {\n"
        "    'history_poi': torch.tensor([[0, 3, 4]]),\n"
        "    'history_timestamp': torch.tensor([[0, 1000, 2000]]),\n"
        "    'history_mask': torch.tensor([[False, True, True]]),\n"
        "    'last_timestamp': torch.tensor([2000]),\n"
        "    'query_s_sec': torch.tensor([0]),\n"
        "    'next_poi': torch.tensor([3]),\n"
        "    'observed_mask': torch.tensor([True]),\n"
        "    'censored_mask': torch.tensor([False]),\n"
        "    'target_in_train_catalog': torch.tensor([True]),\n"
        "    'residual_time_sec': torch.tensor([3600.0]),\n"
        "    'censor_time_sec': torch.tensor([0.0], dtype=torch.float64),\n"
        "}\n"
        "prediction = model.predict(batch)\n"
        "prediction.validate(model.number_of_pois)\n"
        "print('fresh-process-ok')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd="src",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "fresh-process-ok" in completed.stdout


def test_fit_log_time_statistics_deduplicates_gaps_train_only(fixture_root):
    landmark_path = fixture_root["landmark_path"]
    statistics = fit_log_time_statistics(landmark_path)
    assert statistics.status == "fitted"
    assert statistics.source == "train-unique-gaps"
    # Manual reference with explicit dedup over (user, anchor) on train only.
    import pyarrow.parquet as pq

    rows = pq.read_table(landmark_path).to_pylist()
    gaps = {}
    per_row_totals = []
    for row in rows:
        if row["split"] != "train" or not row["event_observed"]:
            continue
        total = float(row["query_s_sec"]) + float(row["residual_time_sec"])
        if total <= 0:
            continue
        gaps[(row["user_id"], row["last_timestamp"])] = total
        per_row_totals.append(total)
    assert statistics.unique_gap_count == len(gaps)
    # The fixture emits multiple query rows per gap, so a per-row fit would see a
    # different population — the dedup is load-bearing, not cosmetic.
    assert len(per_row_totals) > len(gaps)
    logs = [math.log(total) for total in gaps.values()]
    mean = sum(logs) / len(logs)
    assert statistics.mean == pytest.approx(mean, rel=1e-12)
    with pytest.raises(ValueError):
        fit_log_time_statistics(landmark_path, split="validation")


def test_existing_shared_clis_refuse_the_external_family(monkeypatch):
    """No guard bypass: the baseline harness and the baseline model builder do not
    accept the external family; it is built through its own registry only."""
    from adrec_still.models.baselines.harness import resolve_experiment_config
    from adrec_still.models.baselines.trainer import build_baseline_model

    # The committed configs are vendored under tests/fixtures; the harness resolves
    # the experiment's relative dataset/model references against the working directory.
    monkeypatch.chdir(Path(__file__).parents[2] / "fixtures")
    with pytest.raises(ValueError, match="family: baseline"):
        resolve_experiment_config("configs/experiment/gowalla_external_joint_lnm.yaml")
    with pytest.raises(NotImplementedError):
        build_baseline_model({"name": "external_joint_lnm", "family": "external"}, 9)


def test_build_external_model_family_and_name_checks():
    with pytest.raises(ValueError, match="family: external"):
        build_external_model({"name": "external_joint_lnm", "family": "still"}, 9)
    with pytest.raises(ValueError, match="unknown external model"):
        build_external_model({"name": "nope", "family": "external"}, 9)
    model = build_external_model(
        {
            "name": "external_joint_lnm",
            "family": "external",
            "mixture_components": 4,
            "context_dim": 12,
        },
        9,
    )
    assert model.mixture_components == 4 and model.context_dim == 12
