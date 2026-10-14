"""Device-boundary and eval-mode regressions for the #41 arms (F2).

The shared harness delivers CPU collate batches while the shared trainer may leave a
trained model on CUDA; every model entry point must move the fields it actually uses to
the model's parameter device without mutating the caller's batch. The CUDA tests below
are real-device regressions: they are skipped (and honestly reported as skipped) when
no CUDA device is available, never simulated with meta/mock tensors.
"""

import pytest
import torch

from adrec_still.models.baselines.harness import evaluate_split
from adrec_still.models.still.coherent import (
    VARIANT_FIXED_PI_TRUNCATED,
    StillCoherentJoint,
)
from adrec_still.models.still.independent import StillIndependent
from adrec_still.models.still.registry import load_still_checkpoint
from adrec_still.models.still.synthetic_benchmark import write_synthetic_benchmark
from adrec_still.training.loop import TrainingPlan, train_model

NUMBER_OF_POIS = 20

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def make_batch() -> dict[str, torch.Tensor]:
    history = torch.tensor([[0, 3, 4, 5], [0, 0, 6, 7], [2, 8, 9, 10]])
    mask = history.ne(0)
    timestamps = torch.where(mask, 1_600_000_000 + torch.arange(4) * 3600, 0)
    return {
        "user_id": torch.arange(1, 4),
        "history_poi": history,
        "history_timestamp": timestamps,
        "history_mask": mask,
        "last_timestamp": timestamps.max(dim=1).values,
        "query_s_sec": torch.tensor([0, 21600, 43200]),
        "event_observed": torch.tensor([True, True, False]),
        "observed_mask": torch.tensor([True, True, False]),
        "censored_mask": torch.tensor([False, False, True]),
        "target_in_train_catalog": torch.tensor([True, True, False]),
        "next_poi": torch.tensor([5, 7, 0]),
        "residual_time_sec": torch.tensor([3600.0, 7200.0, 0.0]),
        "censor_time_sec": torch.tensor([0.0, 0.0, 5400.0], dtype=torch.float64),
    }


def make_models() -> list[torch.nn.Module]:
    torch.manual_seed(11)
    models: list[torch.nn.Module] = [
        StillIndependent(
            NUMBER_OF_POIS, embedding_dim=16, hidden_dim=16, num_layers=1, dropout=0.5
        ),
        StillCoherentJoint(
            NUMBER_OF_POIS, embedding_dim=16, hidden_dim=16, num_layers=1, dropout=0.5
        ),
    ]
    for model in models:
        model.is_fitted = True
    return models


def snapshot_batch(batch: dict[str, torch.Tensor]) -> dict[str, tuple]:
    return {key: (value, value.dtype, value.clone()) for key, value in batch.items()}


def assert_batch_unchanged(batch: dict[str, torch.Tensor], snapshot: dict[str, tuple]):
    """The caller's dict must keep the same tensor objects, dtypes and values.

    On CPU a ``batch[k] = batch[k].to(device)`` rebinding is an identity no-op that
    no assertion can distinguish, so the device-rebinding class of mutation is only
    truly observable in the CUDA tests below (which re-run this check after every
    entry point); here the identity/dtype/value checks still catch in-place edits
    and dtype rebindings.
    """
    for key, (tensor, dtype, values) in snapshot.items():
        assert batch[key] is tensor, f"batch[{key!r}] was rebound"
        assert batch[key].dtype == dtype, f"batch[{key!r}] changed dtype"
        assert batch[key].device == values.device, f"batch[{key!r}] was moved"
        assert torch.equal(batch[key], values), f"batch[{key!r}] was edited in place"


def test_score_joint_outputs_share_one_device_and_batch_is_not_mutated():
    for model in make_models():
        model.eval()
        batch = make_batch()
        snapshot = snapshot_batch(batch)
        scores = model.score_joint(batch)
        assert {value.device for value in scores.values()} == {
            model.device
        }, "score_joint outputs must be co-located so the runner can index them"
        assert int(scores["log_joint"][scores["eligible"]].shape[0]) == 2
        assert_batch_unchanged(batch, snapshot)


def test_component_diagnostics_runs_in_eval_mode_and_restores_training():
    _, coherent = make_models()
    coherent.train()
    batch = make_batch()
    first = coherent.component_diagnostics(batch)
    second = coherent.component_diagnostics(batch)
    assert coherent.training is True, "caller's mode must be restored"
    assert first == second, "diagnostics must not consume fresh dropout randomness"


@requires_cuda
def test_cuda_model_accepts_cpu_collate_batches_for_predict_and_score_joint():
    for model in make_models():
        # A fresh batch per model: a mutation by one model must not hide behind
        # another model's pass over the same dict.
        batch = make_batch()
        snapshot = snapshot_batch(batch)
        model.to(torch.device("cuda"))
        model.eval()
        prediction = model.predict(batch)
        assert_batch_unchanged(batch, snapshot)
        prediction.validate(NUMBER_OF_POIS)
        assert prediction.poi_logits.device.type == "cpu", "shared outputs return CPU"
        if prediction.time_distribution is not None:
            value = prediction.time_distribution.log_survival(
                torch.tensor([0.0, 3600.0, 5400.0], dtype=torch.float64)
            )
            assert value.device.type == "cpu"
        scores = model.score_joint(batch)
        assert_batch_unchanged(batch, snapshot)
        eligible = scores["eligible"]
        assert torch.isfinite(scores["log_joint"][eligible]).all()
        diagnostics = getattr(model, "component_diagnostics", None)
        if diagnostics is not None:
            report = diagnostics(batch)
            assert_batch_unchanged(batch, snapshot)
            assert report["mixture_components"] == 2.0


@requires_cuda
def test_cuda_training_checkpoint_reload_and_shared_validation(tmp_path):
    metadata = write_synthetic_benchmark(tmp_path / "fixture")
    landmark_path = metadata["landmark_path"]
    from adrec_still.data.dataset import LandmarkDataset

    torch.manual_seed(41)
    model = StillCoherentJoint(
        metadata["number_of_pois"],
        embedding_dim=16,
        hidden_dim=16,
        num_layers=1,
        dropout=0.1,
    )
    plan = TrainingPlan(
        train_source=LandmarkDataset(landmark_path, split="train"),
        training={
            "landmark_weighting": "per_gap",
            "optimizer": "adam",
            "learning_rate": 0.01,
            "batch_size": 8,
            "max_epochs": 1,
            "early_stopping_patience": 1,
            "early_stopping_metric": "ndcg@10",
            "negative_samples": 4,
            "validation_max_batches": 1,
        },
        seed=41,
        validation_landmark_path=landmark_path,
        validation_batch_size=8,
        device=torch.device("cuda"),
        artifacts_dir=tmp_path / "artifacts",
        max_train_batches=2,
    )
    train_model(model, plan)
    assert model.device.type == "cuda", "the trained model stays on its device"
    # Shared evaluation streams CPU collate batches into the CUDA model.
    summary = evaluate_split(
        model, landmark_path, split="validation", batch_size=4, max_batches=2
    )
    assert summary["ranking"]["sample_count"] > 0
    # Reload and the same-checkpoint ablation must also accept CPU batches on CUDA;
    # the runner performs this same to-device move so the recorded variant-vs-ablation
    # comparison is device-matched.
    ablation = load_still_checkpoint(
        tmp_path / "artifacts" / "best.pt",
        variant_override=VARIANT_FIXED_PI_TRUNCATED,
    )
    ablation.to(torch.device("cuda"))
    batch = make_batch()
    ablation_scores = ablation.score_joint(batch)
    assert torch.isfinite(
        ablation_scores["log_joint"][ablation_scores["eligible"]]
    ).all()
