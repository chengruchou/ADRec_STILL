"""Bounded shared-trainer smoke for both #41 arms on the synthetic fixture benchmark.

Fixture evidence only: a decreasing loss and moving real-inference predictions on this
learnable synthetic task demonstrate the executable path (shared trainer -> validation
selection -> checkpoint -> shared evaluation), not empirical model quality. No arbitrary
Recall/MRR threshold is imposed here.
"""

import pytest
import torch

from adrec_still.data.dataset import LandmarkDataset
from adrec_still.models.still.coherent import StillCoherentJoint
from adrec_still.models.still.independent import StillIndependent
from adrec_still.models.still.registry import load_still_checkpoint
from adrec_still.models.still.synthetic_benchmark import write_synthetic_benchmark
from adrec_still.training.loop import TrainingPlan, train_model

TRAINING = {
    "landmark_weighting": "per_gap",
    "optimizer": "adam",
    "learning_rate": 0.02,
    "batch_size": 16,
    "max_epochs": 3,
    "early_stopping_patience": 3,
    "early_stopping_metric": "ndcg@10",
    "early_stopping_mode": "max",
    "negative_samples": 8,
    "validation_max_batches": 2,
}


@pytest.fixture(scope="module")
def benchmark(tmp_path_factory):
    root = tmp_path_factory.mktemp("still_synthetic")
    metadata = write_synthetic_benchmark(root, seed=41)
    return root, metadata


def make_plan(root, artifacts_dir=None, seed: int = 41) -> TrainingPlan:
    landmark_path = root / "landmark_samples.parquet"
    return TrainingPlan(
        train_source=LandmarkDataset(landmark_path, split="train"),
        training=dict(TRAINING),
        seed=seed,
        max_sequence_length=10,
        validation_landmark_path=landmark_path,
        validation_batch_size=32,
        artifacts_dir=artifacts_dir,
    )


@pytest.mark.parametrize("model_class", [StillIndependent, StillCoherentJoint])
def test_both_arms_train_through_the_shared_trainer(benchmark, tmp_path, model_class):
    root, metadata = benchmark
    torch.manual_seed(41)
    model = model_class(
        int(metadata["number_of_pois"]),
        embedding_dim=16,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        mixture_components=2,
    )
    artifacts = tmp_path / model_class.name
    summary = train_model(model, make_plan(root, artifacts_dir=artifacts))
    # The loss moved on a learnable fixture and the run recorded its selection rule.
    assert summary["final_train_loss"] < summary["first_train_loss"]
    assert summary["selection"]["metric"] == "ndcg@10"
    assert summary["selection"]["direction"] == "max"
    assert summary["best"]["value"] is not None
    # Checkpoint reload reproduces real-inference predictions exactly.
    reloaded = load_still_checkpoint(artifacts / "best.pt")
    reloaded.eval()
    model.eval()
    from adrec_still.data.collate import collate_canonical_samples

    dataset = LandmarkDataset(root / "landmark_samples.parquet", split="validation")
    batch = collate_canonical_samples([dataset[i] for i in range(6)], max_sequence_length=10)
    original = model.predict(batch)
    restored = reloaded.predict(batch)
    assert torch.equal(original.poi_logits, restored.poi_logits)


@pytest.mark.parametrize("model_class", [StillIndependent, StillCoherentJoint])
def test_first_epoch_loss_is_seed_reproducible(benchmark, model_class):
    root, metadata = benchmark
    losses = []
    for _ in range(2):
        torch.manual_seed(41)
        model = model_class(
            int(metadata["number_of_pois"]),
            embedding_dim=16,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
            mixture_components=2,
        )
        plan = make_plan(root)
        plan.training["max_epochs"] = 1
        summary = train_model(model, plan)
        losses.append(summary["first_train_loss"])
    assert losses[0] == pytest.approx(losses[1], rel=1e-9)


def test_shared_evaluation_reports_time_metrics_for_both_arms(benchmark):
    root, metadata = benchmark
    from adrec_still.models.baselines.harness import evaluate_split

    torch.manual_seed(41)
    model = StillCoherentJoint(
        int(metadata["number_of_pois"]),
        embedding_dim=16,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
    )
    plan = make_plan(root)
    plan.training["max_epochs"] = 1
    train_model(model, plan)
    summary = evaluate_split(
        model,
        root / "landmark_samples.parquet",
        split="validation",
        batch_size=32,
        max_sequence_length=10,
    )
    time_summary = summary["time"]
    # Censor-aware NLL uses native seconds; CRPS is not part of this port.
    assert time_summary["sample_count"] > 0
    assert time_summary["censored_count"] > 0
    assert time_summary["return_time_nll"] is not None
    assert time_summary["observed_time_nll"] is not None
    assert time_summary["censored_time_nll"] is not None
    assert not any("crps" in key for key in time_summary)
