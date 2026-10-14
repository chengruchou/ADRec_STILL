"""Staged pretrain/warmup/finetune training on the bundled ml-100k pkl.

The runs are bounded (hidden 16, max_len 8, 4 diffusion steps, two 64-row
batches per epoch) so each one finishes in seconds. Selection behavior is
driven through scripted ``validate`` callables so the expected best values are
known exactly; nothing here asserts learned quality.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from adrec_still.data.sequences import load_dataset_pkl
from adrec_still.models.upstream.wrappers import AdRecRecommender
from adrec_still.training import checkpoint as ckpt
from adrec_still.training.loop import STATUS_COMPLETED, STATUS_INTERRUPTED
from adrec_still.training.stages import (
    STAGED_BEST_FORMAT,
    STAGED_LAST_FORMAT,
    StagedPlan,
    train_staged_model,
)

REPOSITORY_ROOT = Path(__file__).parents[2]
ML100K_PKL = REPOSITORY_ROOT / "datasets" / "data" / "ml-100k" / "dataset.pkl"

TRAINING = {
    "max_epochs": 2,          # finetune budget
    "pretrain_epochs": 1,
    "warmup_epochs": 1,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "early_stopping_patience": 5,
}


@pytest.fixture(scope="module")
def corpus():
    return load_dataset_pkl(ML100K_PKL)


def _build_model(corpus) -> AdRecRecommender:
    torch.manual_seed(5)
    np.random.seed(5)
    return AdRecRecommender(
        corpus.number_of_pois, hidden_size=16, max_len=8, diffusion_steps=4,
        ce_chunk_tokens=None,
    )


def _plan(corpus, artifacts, validate, *, training=TRAINING, resume_from=None):
    return StagedPlan(
        corpus=corpus, training=dict(training), seed=11, device=torch.device("cpu"),
        artifacts_dir=artifacts, data_identity=dict(corpus.identity),
        family="upstream", validate=validate, monitor="val_metric",
        monitor_direction="max", max_train_batches=2, resume_from=resume_from,
    )


def _metric_rows(artifacts: Path) -> list[dict]:
    return [json.loads(line)
            for line in (artifacts / "metrics.jsonl").read_text().splitlines()]


def test_stages_run_in_order_and_the_embedding_unfreezes_after_warmup(corpus, tmp_path):
    model = _build_model(corpus)
    embedding_trainable_at_validation: list[bool] = []
    values = iter([1.0, 2.0, 3.0])

    def validate(trained):
        embedding_trainable_at_validation.append(
            bool(trained.item_embedding.weight.requires_grad)
        )
        return next(values), {}

    summary = train_staged_model(model, _plan(corpus, tmp_path, validate))

    assert summary["status"] == STATUS_COMPLETED
    assert summary["stages"] == [["pretrain", 1], ["warmup", 1], ["finetune", 2]]
    assert [row["stage"] for row in _metric_rows(tmp_path)] == [
        "pretrain", "warmup", "finetune", "finetune"
    ]
    # Frozen through warmup's validation, trainable through both finetune ones.
    assert embedding_trainable_at_validation == [False, True, True]
    assert bool(model.item_embedding.weight.requires_grad)
    # The pretrain stage recorded its embedding hand-over, never a selection.
    pretrain_entries = [entry for entry in summary["stage_history"]
                        if entry["stage"] == "pretrain"]
    assert pretrain_entries and "embedding transfer" in pretrain_entries[0]["selection"]
    # best.pt is the final stage's best.
    best_payload = ckpt.load_payload(tmp_path / ckpt.BEST_NAME, STAGED_BEST_FORMAT)
    assert best_payload["selection"]["stage"] == "finetune"
    assert best_payload["selection"]["value"] == 3.0


def test_selection_scope_resets_at_the_stage_switch(corpus, tmp_path):
    model = _build_model(corpus)
    values = iter([0.9, 0.1, 0.2])
    summary = train_staged_model(
        model, _plan(corpus, tmp_path, lambda trained: (next(values), {}))
    )

    # The warmup epoch monitored 0.9 (direction max); a scope that leaked across
    # the stage switch would keep it as the comparison baseline and select
    # nothing in finetune. The real best must be finetune's own 0.2.
    assert summary["best"] == {
        "stage": "finetune", "stage_epoch": 2, "epoch": 4,
        "metric": "val_metric", "value": 0.2, "direction": "max",
    }
    best_payload = ckpt.load_payload(tmp_path / ckpt.BEST_NAME, STAGED_BEST_FORMAT)
    assert best_payload["selection"]["value"] == 0.2
    warmup_bests = [entry["best"] for entry in summary["stage_history"]
                    if entry["stage"] == "warmup" and entry.get("best")]
    assert warmup_bests and all(entry["value"] == 0.9 for entry in warmup_bests)
    assert all(entry["stage"] == "warmup" for entry in warmup_bests)


def test_resume_continues_into_finetune_without_redoing_pretrain(corpus, tmp_path):
    training = dict(TRAINING, max_epochs=1)

    # First segment: the interrupt lands at finetune's first validation, so the
    # newest complete checkpoint is the end of warmup.
    calls = {"count": 0}

    def interrupting_validate(_trained):
        calls["count"] += 1
        if calls["count"] == 2:
            raise KeyboardInterrupt
        return 0.9, {}

    first = train_staged_model(
        _build_model(corpus),
        _plan(corpus, tmp_path, interrupting_validate, training=training),
    )
    assert first["status"] == STATUS_INTERRUPTED
    assert [row["stage"] for row in _metric_rows(tmp_path)] == ["pretrain", "warmup"]
    last_payload = ckpt.load_payload(tmp_path / ckpt.LAST_NAME, STAGED_LAST_FORMAT)
    assert last_payload["stage"] == "warmup"
    assert last_payload["stage_epoch"] == 1
    assert last_payload["pretrain_done"] is True

    # Second segment resumes from last.pt and must go straight to finetune.
    resumed = train_staged_model(
        _build_model(corpus),
        _plan(corpus, tmp_path, lambda trained: (0.3, {}), training=training,
              resume_from=tmp_path / ckpt.LAST_NAME),
    )
    assert resumed["status"] == STATUS_COMPLETED
    stages_seen = [row["stage"] for row in _metric_rows(tmp_path)]
    assert stages_seen == ["pretrain", "warmup", "finetune"]
    assert stages_seen.count("pretrain") == 1  # never re-run
    assert resumed["epochs_run"] == 3
    assert resumed["best"]["stage"] == "finetune"
    assert resumed["best"]["value"] == 0.3
