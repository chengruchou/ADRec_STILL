"""Cooperative stop-condition hook on the shared trainer/harness (#51 proposal).

The hook is opt-in: ``stop_condition=None`` (every existing caller) keeps the
previous behavior bit-for-bit, which the rest of the suite already pins. These
tests cover the new stop semantics: once the condition fires, no further
optimizer step, selection-validation batch or epoch starts; the unit of work
already running finishes safely and artifacts are kept.
"""


import torch

from adrec_still.data.dataset import LandmarkDataset
from adrec_still.models.baselines.gru4rec import GRU4RecH
from adrec_still.models.baselines.harness import evaluate_split
from adrec_still.models.still.synthetic_benchmark import write_synthetic_benchmark
from adrec_still.training.loop import STATUS_STOPPED, TrainingPlan, train_model

TRAINING = {
    "landmark_weighting": "per_gap",
    "optimizer": "adam",
    "learning_rate": 0.01,
    "batch_size": 4,
    "max_epochs": 5,
    "early_stopping_patience": 5,
    "early_stopping_metric": "ndcg@10",
    "negative_samples": 4,
    "validation_max_batches": 2,
}


def make_plan(landmark_path, tmp_path, stop_condition):
    return TrainingPlan(
        train_source=LandmarkDataset(landmark_path, split="train"),
        training=TRAINING,
        seed=41,
        max_sequence_length=8,
        validation_landmark_path=landmark_path,
        validation_batch_size=4,
        artifacts_dir=tmp_path / "artifacts",
        max_train_batches=2,
        stop_condition=stop_condition,
    )


def test_stop_after_first_step_runs_no_more_steps_or_selection(tmp_path):
    fixture = write_synthetic_benchmark(tmp_path / "tables")
    torch.manual_seed(41)
    model = GRU4RecH(
        fixture["number_of_pois"], embedding_dim=8, hidden_dim=8, num_layers=1, dropout=0.0
    )
    # The trainer polls at the epoch top and before each subsequent step:
    # returning True from the second poll on stops right after step 1.
    state = {"polls": 0}

    def stop() -> bool:
        state["polls"] += 1
        return state["polls"] > 1

    summary = train_model(model, make_plan(fixture["landmark_path"], tmp_path, stop))
    assert summary["status"] == STATUS_STOPPED
    assert summary["epochs_run"] == 0, "a partial epoch is not a completed epoch"
    assert summary["best"] is None, "no selection validation ran"
    # Nothing durable is written for a partial epoch: no last.pt, no metrics row.
    assert not (tmp_path / "artifacts" / "last.pt").exists()
    assert not (tmp_path / "artifacts" / "metrics.jsonl").exists()


def test_stop_before_any_work_yields_an_empty_but_valid_summary(tmp_path):
    fixture = write_synthetic_benchmark(tmp_path / "tables")
    torch.manual_seed(41)
    model = GRU4RecH(
        fixture["number_of_pois"], embedding_dim=8, hidden_dim=8, num_layers=1, dropout=0.0
    )
    summary = train_model(
        model, make_plan(fixture["landmark_path"], tmp_path, lambda: True)
    )
    assert summary["status"] == STATUS_STOPPED
    assert summary["epochs_run"] == 0
    assert summary["first_train_loss"] is None
    assert summary["final_train_loss"] is None


def test_evaluate_split_stop_condition_halts_between_batches(tmp_path):
    fixture = write_synthetic_benchmark(tmp_path / "tables")
    torch.manual_seed(41)
    model = GRU4RecH(
        fixture["number_of_pois"], embedding_dim=8, hidden_dim=8, num_layers=1, dropout=0.0
    )
    model.is_fitted = True
    # Fires after the first processed batch: that batch finishes, none follow.
    state = {"polls": 0}

    def stop() -> bool:
        state["polls"] += 1
        return state["polls"] >= 1

    summary = evaluate_split(
        model,
        fixture["landmark_path"],
        split="validation",
        batch_size=1,
        max_sequence_length=8,
        stop_condition=stop,
    )
    assert summary["stopped_early"] is True
    assert summary["batch_count"] == 1
    # Without the hook the key is absent and the whole split streams (one row per
    # batch; the row count comes from the fixture itself, not a hard-coded number).
    validation_rows = len(LandmarkDataset(fixture["landmark_path"], split="validation"))
    full = evaluate_split(
        model,
        fixture["landmark_path"],
        split="validation",
        batch_size=1,
        max_sequence_length=8,
    )
    assert "stopped_early" not in full
    assert full["evaluated_sample_count"] == validation_rows


def test_completed_streams_and_epochs_are_never_mislabeled_as_interrupted(tmp_path):
    """A pass that consumed its whole split or its max_batches prefix is complete:
    stopped_early must stay false however late the condition fires, and a complete
    epoch keeps its selection metric."""
    fixture = write_synthetic_benchmark(tmp_path / "tables")
    torch.manual_seed(41)
    model = GRU4RecH(
        fixture["number_of_pois"], embedding_dim=8, hidden_dim=8, num_layers=1, dropout=0.0
    )
    model.is_fitted = True
    # Whole split consumed (N one-row batches, N-1 polls): a condition that would
    # first fire on poll N is never reached.
    validation_rows = len(LandmarkDataset(fixture["landmark_path"], split="validation"))
    state = {"polls": 0}

    def late_stop() -> bool:
        state["polls"] += 1
        return state["polls"] >= validation_rows

    full = evaluate_split(
        model,
        fixture["landmark_path"],
        split="validation",
        batch_size=1,
        max_sequence_length=8,
        stop_condition=late_stop,
    )
    assert full["stopped_early"] is False
    assert full["evaluated_sample_count"] == validation_rows
    # max_batches prefix consumed with the condition already true: still complete.
    capped = evaluate_split(
        model,
        fixture["landmark_path"],
        split="validation",
        batch_size=1,
        max_sequence_length=8,
        max_batches=1,
        stop_condition=lambda: True,
    )
    assert capped["stopped_early"] is False
    assert capped["batch_count"] == 1
    # Trainer: the deadline crosses right after epoch 1's selection validation
    # completes (clock-like side-effect-free condition flipped by a wrapper);
    # the complete epoch keeps its metric and the stop lands at epoch 2's top.
    import adrec_still.training.loop as trainer_module

    torch.manual_seed(41)
    trainee = GRU4RecH(
        fixture["number_of_pois"], embedding_dim=8, hidden_dim=8, num_layers=1, dropout=0.0
    )
    flag = {"expired": False}
    real_evaluate_split = trainer_module.evaluate_split

    def evaluate_then_expire(*args, **kwargs):
        result = real_evaluate_split(*args, **kwargs)
        flag["expired"] = True
        return result

    trainer_module.evaluate_split = evaluate_then_expire
    try:
        plan = make_plan(fixture["landmark_path"], tmp_path, lambda: flag["expired"])
        plan.max_train_batches = 1
        summary = train_model(trainee, plan)
    finally:
        trainer_module.evaluate_split = real_evaluate_split
    assert summary["status"] == STATUS_STOPPED
    assert summary["epochs_run"] == 1
    assert summary["best"]["epoch"] == 1, "the complete epoch's selection metric is kept"
    assert summary["best"]["value"] is not None


def test_max_epochs_below_one_is_refused_loudly():
    import pytest

    from adrec_still.models.baselines.trainer import validate_training_config

    bad = dict(TRAINING, max_epochs=0)
    with pytest.raises(ValueError, match="max_epochs"):
        validate_training_config(bad)
