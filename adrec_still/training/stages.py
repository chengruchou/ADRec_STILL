"""Stage-aware training for the upstream family (and the ADRec-STILL extension).

ADRec's recipe is three stages: **pretrain** (a position-free SASRec trains the
item-embedding table), **warmup** (the diffusion model trains around the frozen
pretrained table), **finetune** (the table unfreezes). Every budget is a plain
setting (``pretrain_epochs`` / ``warmup_epochs`` / ``max_epochs``); a bounded
smoke run may give each stage one epoch. Models without a pretrain stage run a
single ``train`` stage.

Selection is scoped per stage and explicitly reset at a stage switch: ``best.pt``
always holds the best checkpoint *of the current stage's scope*, so a pretrain
or warmup metric can never masquerade as the finetune best. Earlier stages'
bests are kept in the summary as ``stage_history``. ``last.pt`` carries the
stage, epoch, optimizer, scheduler and RNG needed to resume mid-run, including
mid-pretrain (the pretrain network's state rides along only while that stage is
active). A pretrained embedding is handed over in memory ??no ``pretrain.pth``
file is required for a new dataset.

Deviation from the fork recorded in MIGRATION.md: each stage builds its
optimizer over the parameters that are trainable *in that stage*, so a frozen
embedding is genuinely untouched during warmup (the fork's PCGrad wrapper set a
zero gradient on frozen parameters and Adam's weight decay still shrank them).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import optim
from torch.utils.data import DataLoader

from adrec_still.data.sequences import (
    SequenceCorpus,
    SequenceTrainDataset,
    expand_prefixes,
)
from adrec_still.models.upstream.pcgrad import PCGrad
from adrec_still.training import checkpoint as ckpt
from adrec_still.training.loop import (
    STATUS_COMPLETED,
    STATUS_INTERRUPTED,
    _improves,
)
from adrec_still.training.progress import Reporter

STAGED_LAST_FORMAT = "adrec-still-staged-last-v1"
STAGED_BEST_FORMAT = "adrec-still-staged-best-v1"
STAGED_TRAINER_VERSION = "adrec-still-staged-trainer-v1"

STAGE_PRETRAIN = "pretrain"
STAGE_WARMUP = "warmup"
STAGE_FINETUNE = "finetune"
STAGE_TRAIN = "train"


@dataclass
class StagedPlan:
    """Everything one staged run needs."""

    corpus: SequenceCorpus
    training: dict[str, Any]
    seed: int
    device: torch.device
    artifacts_dir: Path
    data_identity: dict[str, Any]
    family: str = "upstream"
    #: Called with the model; returns ``(value, details)`` for the monitor, or None
    #: when no validation is available.
    validate: Callable[[Any], tuple[float | None, dict[str, Any]]] | None = None
    monitor: str = "train_loss"
    monitor_direction: str = "min"
    reporter: Reporter = field(default_factory=Reporter)
    max_train_batches: int | None = None
    num_workers: int = 0
    resume_from: Path | None = None
    weights_from: Path | None = None
    pretrained_embedding: torch.Tensor | None = None


def stage_schedule(model, training: dict[str, Any]) -> list[tuple[str, int]]:
    """The run's stage names and epoch budgets."""
    max_epochs = int(training.get("max_epochs", 20))
    if getattr(model, "embedding_pretrain", False):
        return [
            (STAGE_PRETRAIN, int(training.get("pretrain_epochs", 10))),
            (STAGE_WARMUP, int(training.get("warmup_epochs", 5))),
            (STAGE_FINETUNE, max_epochs),
        ]
    return [(STAGE_TRAIN, max_epochs)]


def _training_view(model, corpus: SequenceCorpus, max_len: int) -> SequenceTrainDataset:
    if getattr(model, "training_view", "parallel") == "prefix_last":
        return SequenceTrainDataset(expand_prefixes(corpus.train, max_len), max_len,
                                    parallel=False)
    return SequenceTrainDataset(corpus.train, max_len, parallel=True)


def _build_pretrain_model(model):
    from adrec_still.models.upstream.wrappers import SASRecUpstreamRecommender

    return SASRecUpstreamRecommender(
        model.number_of_pois, hidden_size=model.hidden_size, max_len=model.max_len,
        dropout=model.dropout_rate, emb_dropout=model.emb_dropout_rate,
        ce_chunk_tokens=model.ce_chunk_tokens, pretrain_mode=True,
    )


def _stage_optimizer(network, training: dict[str, Any]):
    parameters = [p for p in network.parameters() if p.requires_grad]
    base = optim.Adam(
        parameters,
        lr=float(training.get("learning_rate", 0.001)),
        weight_decay=float(training.get("weight_decay", 1e-5)),
    )
    wrapped = PCGrad(base, pcgrad=bool(training.get("pcgrad", False)))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        base, T_max=int(training.get("scheduler_t_max", 500))
    )
    return wrapped, scheduler


def _epoch_pass(network, loader, optimizer, device, max_train_batches, reporter,
                stage: str, epoch: int, total_epochs: int):
    network.train()
    ce_total = mse_total = weight_total = 0.0
    batches = 0
    total_batches = len(loader)
    if max_train_batches is not None:
        total_batches = min(total_batches, max_train_batches)
    reporter.epoch_start(epoch, total_epochs, total_batches)
    for index, (history, targets) in enumerate(loader):
        if max_train_batches is not None and index >= max_train_batches:
            break
        history = history.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        objectives = network.training_objectives(history, targets)
        optimizer.pc_backward(objectives["objectives"])
        optimizer.step()
        rows = float(history.shape[0])
        ce_total += float(objectives["ce"].detach()) * rows
        mse_total += float(objectives["mse"].detach()) * rows
        weight_total += rows
        batches += 1
        reporter.step(index + 1, total_batches,
                      loss=f"{float(objectives['ce'].detach()):.4f}", stage=stage)
    if batches == 0:
        raise ValueError("a training epoch processed no batches")
    ce = ce_total / weight_total
    mse = mse_total / weight_total
    if math.isnan(ce) or math.isnan(mse) or math.isinf(ce) or math.isinf(mse):
        raise ValueError(f"non-finite stage {stage} epoch loss (ce={ce}, mse={mse})")
    return {"ce": ce, "mse": mse, "batches": batches}


def _write_metrics_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def train_staged_model(model, plan: StagedPlan) -> dict[str, Any]:
    """Run the stage schedule; returns the training summary."""
    reporter = plan.reporter
    training = dict(plan.training)
    device = plan.device
    artifacts = Path(plan.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    metrics_path = artifacts / "metrics.jsonl"
    stages = stage_schedule(model, training)
    eval_interval = max(1, int(training.get("eval_interval", 1)))
    patience = int(training.get("early_stopping_patience", 5))
    batch_size = int(training.get("batch_size", 256))
    max_len = model.max_len

    model.to(device)
    model.is_fitted = True

    resume_state: dict[str, Any] | None = None
    if plan.resume_from is not None:
        resume_state = ckpt.load_payload(plan.resume_from, STAGED_LAST_FORMAT)
        problems = ckpt.model_differences(resume_state["model"], model.resolved_config())
        problems += ckpt.data_differences(resume_state["data"], plan.data_identity)
        ckpt.require_compatible("resume", plan.resume_from, problems)
    elif plan.weights_from is not None:
        payload = ckpt.load_payload(plan.weights_from, STAGED_BEST_FORMAT)
        problems = ckpt.model_differences(payload["model"], model.resolved_config())
        problems += ckpt.data_differences(payload["data"], plan.data_identity)
        ckpt.require_compatible("weights initialization", plan.weights_from, problems)
        model.load_state_dict(payload["state_dict"])
        reporter.message(f"initialized weights from {plan.weights_from}; "
                         "optimizer and schedule start fresh")
    if resume_state is None and metrics_path.exists():
        metrics_path.unlink()

    loader_generator = torch.Generator().manual_seed(plan.seed)
    dataset = _training_view(model, plan.corpus, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=plan.num_workers, generator=loader_generator)

    status = STATUS_COMPLETED
    global_epoch = 0
    stage_history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    pretrain_network = None
    start_stage_index = 0
    start_stage_epoch = 0
    pretrain_done = plan.pretrained_embedding is not None

    if plan.pretrained_embedding is not None:
        model.load_embedding(plan.pretrained_embedding, freeze=True)
        reporter.message("loaded the supplied pretrained embedding (frozen for warmup)")
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state"])
        start_stage_index = int(resume_state["stage_index"])
        start_stage_epoch = int(resume_state["stage_epoch"])
        global_epoch = int(resume_state["global_epoch"])
        stage_history = list(resume_state.get("stage_history", []))
        pretrain_done = bool(resume_state.get("pretrain_done", False))
        best = resume_state.get("best")
        if best is not None:
            best_path = Path(plan.resume_from).with_name(ckpt.BEST_NAME)
            if best_path.is_file():
                best_payload = ckpt.load_payload(best_path, STAGED_BEST_FORMAT)
                best_state = best_payload["state_dict"]
        reporter.message(
            f"resuming from {plan.resume_from}: stage {stages[start_stage_index][0]} "
            f"after epoch {start_stage_epoch}"
        )

    def save_last(stage_index: int, stage_epoch: int, optimizer, scheduler) -> None:
        payload = {
            "format": STAGED_LAST_FORMAT,
            "trainer_version": STAGED_TRAINER_VERSION,
            "family": plan.family,
            "model": model.resolved_config(),
            "data": plan.data_identity,
            "training": training,
            "stages": [list(entry) for entry in stages],
            "stage_index": stage_index,
            "stage": stages[stage_index][0],
            "stage_epoch": stage_epoch,
            "global_epoch": global_epoch,
            "stage_history": stage_history,
            "pretrain_done": pretrain_done,
            "model_state": ckpt.cpu_state_dict(model),
            "pretrain_state": (
                ckpt.cpu_state_dict(pretrain_network) if pretrain_network is not None else None
            ),
            "optimizer_state": optimizer.optim.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best": best,
            "rng": ckpt.capture_rng(loader_generator, None, device),
            "seed": plan.seed,
            "monitor": plan.monitor,
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        ckpt.write_torch(artifacts / ckpt.LAST_NAME, payload)

    def save_best() -> None:
        payload = {
            "format": STAGED_BEST_FORMAT,
            "family": plan.family,
            "model": model.resolved_config(),
            "data": plan.data_identity,
            "state_dict": ckpt.cpu_state_dict(model),
            "selection": dict(best or {}),
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        ckpt.write_torch(artifacts / ckpt.BEST_NAME, payload)

    try:
        for stage_index in range(start_stage_index, len(stages)):
            stage_name, stage_budget = stages[stage_index]
            stage_start_epoch = start_stage_epoch if stage_index == start_stage_index else 0
            if stage_budget <= 0:
                raise ValueError(f"stage {stage_name} has a non-positive epoch budget")
            # ---------------------------------------------------------- pretrain
            if stage_name == STAGE_PRETRAIN:
                if pretrain_done:
                    reporter.message("pretrained embedding already in place; skipping pretrain")
                    continue
                if pretrain_network is None:
                    pretrain_network = _build_pretrain_model(model).to(device)
                    if resume_state is not None and resume_state.get("pretrain_state"):
                        pretrain_network.load_state_dict(resume_state["pretrain_state"])
                optimizer, scheduler = _stage_optimizer(pretrain_network, training)
                if resume_state is not None and stage_index == start_stage_index:
                    if resume_state.get("optimizer_state"):
                        optimizer.optim.load_state_dict(resume_state["optimizer_state"])
                    if resume_state.get("scheduler_state"):
                        scheduler.load_state_dict(resume_state["scheduler_state"])
                pretrain_loader = DataLoader(
                    SequenceTrainDataset(plan.corpus.train, max_len, parallel=True),
                    batch_size=batch_size, shuffle=True, num_workers=plan.num_workers,
                    generator=loader_generator,
                )
                final_pretrain_ce: float | None = None
                with reporter.phase(f"stage-{stage_name}"):
                    for stage_epoch in range(stage_start_epoch + 1, stage_budget + 1):
                        epoch_started = time.time()
                        result = _epoch_pass(
                            pretrain_network, pretrain_loader, optimizer, device,
                            plan.max_train_batches, reporter, stage_name, stage_epoch,
                            stage_budget,
                        )
                        scheduler.step()
                        global_epoch += 1
                        final_pretrain_ce = result["ce"]
                        _write_metrics_row(metrics_path, {
                            "stage": stage_name, "epoch": global_epoch,
                            "stage_epoch": stage_epoch, "train_ce": result["ce"],
                            "train_mse": result["mse"],
                            "monitor": "train_loss", "monitor_value": result["ce"],
                            "learning_rate": scheduler.get_last_lr()[0],
                            "epoch_seconds": time.time() - epoch_started,
                        })
                        reporter.epoch_end(stage_epoch, stage=stage_name,
                                           ce=f"{result['ce']:.4f}")
                        save_last(stage_index, stage_epoch, optimizer, scheduler)
                stage_history.append({
                    "stage": stage_name, "epochs": stage_budget,
                    "selection": "final epoch (embedding transfer; train-loss stage)",
                    "final_train_ce": final_pretrain_ce,
                })
                model.load_embedding(
                    pretrain_network.item_embedding.weight.detach().clone(), freeze=True,
                )
                pretrain_done = True
                pretrain_network = None
                reporter.message("pretrain complete; embedding transferred and frozen")
                continue

            # --------------------------------------------------- warmup/finetune
            if stage_name == STAGE_WARMUP:
                model.set_embedding_trainable(False)
            elif stage_name in (STAGE_FINETUNE, STAGE_TRAIN):
                model.set_embedding_trainable(True)
            optimizer, scheduler = _stage_optimizer(model, training)
            if (resume_state is not None and stage_index == start_stage_index
                    and stages[start_stage_index][0] == stage_name):
                if resume_state.get("optimizer_state"):
                    optimizer.optim.load_state_dict(resume_state["optimizer_state"])
                if resume_state.get("scheduler_state"):
                    scheduler.load_state_dict(resume_state["scheduler_state"])
            else:
                # Entering a new stage: the selection scope is explicitly reset.
                # The previous stage's best stays only as a record, never as the
                # comparison baseline for this stage. Stages that already wrote
                # their full end-of-stage record (with epochs) are not repeated;
                # this append covers only a resume landing past that record.
                previous_stage = stages[stage_index - 1][0]
                already_recorded = any(
                    entry.get("stage") == previous_stage and "epochs" in entry
                    for entry in stage_history
                )
                if best is not None and not already_recorded:
                    stage_history.append({"stage": previous_stage, "best": best})
                best = None
                best_state = None
            bad_epochs = 0
            # A resume may land on a stage whose budget is already exhausted (the
            # interrupt happened right after its last epoch); the loop below then
            # runs zero epochs and ``stage_epoch`` must still name the last
            # completed epoch for the stage-history record.
            stage_epoch = stage_start_epoch
            with reporter.phase(f"stage-{stage_name}"):
                for stage_epoch in range(stage_start_epoch + 1, stage_budget + 1):
                    epoch_started = time.time()
                    result = _epoch_pass(
                        model, loader, optimizer, device, plan.max_train_batches,
                        reporter, stage_name, stage_epoch, stage_budget,
                    )
                    scheduler.step()
                    global_epoch += 1
                    validation_detail: dict[str, Any] = {}
                    if plan.monitor == "train_loss" or plan.validate is None:
                        monitor_name = "train_loss"
                        monitor_value: float | None = result["ce"]
                        selectable = True
                    elif stage_epoch % eval_interval == 0:
                        monitor_value, validation_detail = plan.validate(model)
                        monitor_name = plan.monitor
                        selectable = True
                    else:
                        monitor_name, monitor_value, selectable = plan.monitor, None, False
                    improved = False
                    if selectable:
                        if monitor_value is None:
                            raise ValueError(
                                f"the validation pass returned no value for monitor "
                                f"{plan.monitor!r}; selection cannot proceed"
                            )
                        direction = (plan.monitor_direction
                                     if monitor_name == plan.monitor else "min")
                        best_value = best["value"] if best is not None else None
                        if _improves(float(monitor_value), best_value, direction):
                            improved = True
                            bad_epochs = 0
                            best = {
                                "stage": stage_name, "stage_epoch": stage_epoch,
                                "epoch": global_epoch, "metric": monitor_name,
                                "value": float(monitor_value), "direction": direction,
                            }
                            best_state = ckpt.cpu_state_dict(model)
                            save_best()
                        else:
                            bad_epochs += 1
                    _write_metrics_row(metrics_path, {
                        "stage": stage_name, "epoch": global_epoch,
                        "stage_epoch": stage_epoch, "train_ce": result["ce"],
                        "train_mse": result["mse"], "monitor": monitor_name,
                        "monitor_value": (
                            float(monitor_value) if monitor_value is not None else None
                        ),
                        "improved": improved,
                        "best_epoch": best["epoch"] if best else None,
                        "best_value": best["value"] if best else None,
                        "validation": {
                            key: value for key, value in validation_detail.items()
                            if isinstance(value, (int, float)) or value is None
                        },
                        "learning_rate": scheduler.get_last_lr()[0],
                        "epoch_seconds": time.time() - epoch_started,
                    })
                    reporter.epoch_end(stage_epoch, stage=stage_name,
                                       ce=f"{result['ce']:.4f}",
                                       monitor=monitor_value)
                    save_last(stage_index, stage_epoch, optimizer, scheduler)
                    if bad_epochs >= patience and stage_name in (STAGE_FINETUNE, STAGE_TRAIN):
                        reporter.message(
                            f"early stopping in stage {stage_name}: no {plan.monitor} "
                            f"improvement for {patience} evaluations"
                        )
                        break
            stage_history.append({"stage": stage_name, "epochs": stage_epoch,
                                  "best": best})
    except KeyboardInterrupt:
        status = STATUS_INTERRUPTED
        reporter.message(
            "interrupted; the most recent complete checkpoint is kept in last.pt"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoints = {
        name: str(artifacts / filename)
        for name, filename in (("last", ckpt.LAST_NAME), ("best", ckpt.BEST_NAME),
                               ("metrics", "metrics.jsonl"))
        if (artifacts / filename).is_file()
    }
    return {
        "trainer_version": STAGED_TRAINER_VERSION,
        "status": status,
        "family": plan.family,
        "seed": plan.seed,
        "device": str(device),
        "stages": [list(entry) for entry in stages],
        "stage_history": stage_history,
        "epochs_run": global_epoch,
        "selection": {"monitor": plan.monitor, "direction": plan.monitor_direction,
                      "scope": "per-stage; best.pt holds the final stage's best"},
        "best": best,
        "train_row_count": len(dataset),
        "checkpoints": checkpoints,
    }


def load_upstream_pretrain_pth(path: str | Path, number_of_pois: int,
                               hidden_size: int, *, id_shift: int = 1) -> torch.Tensor:
    """Explicit raw-ID remap of a fork ``pretrain.pth`` embedding into this vocabulary.

    The fork's table row ``i`` is raw item ``i`` (row 0 padding). This package's
    pkl vocabulary shifts items by ``id_shift`` (PAD=0, UNK=1 reserved), so row
    ``i`` maps to row ``i + id_shift``; the reserved rows keep fresh
    initialization. A table whose size disagrees with the target vocabulary is a
    different item universe and is refused ??weights are never silently matched
    by shape across datasets.
    """
    saved = torch.load(Path(path), map_location="cpu", weights_only=True)
    weight = saved["item_embedding.weight"] if isinstance(saved, dict) else saved
    if weight.shape[1] != hidden_size:
        raise ValueError(
            f"{path}: embedding width {weight.shape[1]} != model hidden size {hidden_size}"
        )
    expected_rows = number_of_pois - id_shift
    if weight.shape[0] != expected_rows:
        raise ValueError(
            f"{path}: embedding has {weight.shape[0]} rows but this vocabulary expects "
            f"{expected_rows} upstream rows (number_of_pois={number_of_pois}, "
            f"id_shift={id_shift}); refusing a cross-dataset weight transfer"
        )
    table = torch.empty((number_of_pois, hidden_size))
    torch.nn.init.normal_(table, std=0.1)
    table[0].zero_()
    # Upstream rows 1..N (raw items) land on rows 1+id_shift .. N+id_shift; the
    # reserved rows in between (UNK for id_shift == 1) keep fresh initialization.
    table[1 + id_shift:] = weight[1:]
    return table
