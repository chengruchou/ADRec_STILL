"""The shared epoch loop: train, validate in memory, keep ``last.pt`` and ``best.pt``.

Every neural backend (the encoder baselines, the STILL arms and the external LNM)
trains through :func:`train_model`. The objective, masks, weights and sampled
candidate policy come from :mod:`adrec_still.models.baselines.trainer`; this module owns
the loop, the per-epoch validation pass, checkpoint retention, resume and progress
reporting. It writes nothing but the two checkpoint files.
"""

from __future__ import annotations

import copy
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from adrec_still.data.collate import collate_canonical_samples
from adrec_still.data.training_views import EVENT_ONLY_TRAINING_PROFILE
from adrec_still.models.baselines.trainer import (
    TRAINER_VERSION,
    WEIGHTING_CONTRACTS,
    apply_precision_policy,
    build_training_sampler,
    event_only_training_source,
    resolve_training_settings,
    training_step,
    validate_event_only_profile,
)
from adrec_still.models.still.losses import SURVIVAL_TRAINING_PROFILE
from adrec_still.training import checkpoint as ckpt
from adrec_still.training.evaluate import (
    MONITOR_DIRECTIONS,
    EvaluationOptions,
    evaluate_split,
    monitor_key,
    split_horizons,
)
from adrec_still.training.progress import Reporter

METRICS_NAME = "metrics.jsonl"
STATUS_COMPLETED = "completed"
STATUS_INTERRUPTED = "interrupted"
STATUS_STOPPED = "stopped_by_condition"


@dataclass
class TrainingPlan:
    """Everything one seeded training run needs, resolved before any step runs."""

    train_source: Dataset
    training: dict[str, Any]
    seed: int
    family: str = "baseline"
    data: dict[str, Any] = field(default_factory=dict)
    max_sequence_length: int | None = 100
    validation_landmark_path: Path | None = None
    validation_batch_size: int = 128
    validation_chunk_size: int = 4096
    num_workers: int = 0
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    artifacts_dir: Path | None = None
    max_train_batches: int | None = None
    stop_condition: Any = None
    reporter: Reporter | None = None
    resume_from: Path | None = None
    weights_from: Path | None = None
    epoch_callback: Callable[[dict[str, Any]], None] | None = None


class SelectionUnavailable(ValueError):
    """The monitored metric has no population in the validation data."""


def _improves(value: float, best: float | None, direction: str) -> bool:
    if best is None:
        return True
    return value > best if direction == "max" else value < best


def _population_label(horizon: int | None) -> str:
    if horizon is None:
        return "all validation rows"
    return f"validation rows at query_s_sec={int(horizon)}"


def train_model(model, plan: TrainingPlan) -> dict[str, Any]:
    """Run the shared training loop and leave ``model`` fitted at its best epoch."""
    settings = resolve_training_settings(plan.training)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("the shared trainer only trains torch.nn.Module models")
    reporter = plan.reporter or Reporter()
    event_only = validate_event_only_profile(model, settings)
    if (
        not event_only
        and getattr(plan.train_source, "training_view", None) == EVENT_ONLY_TRAINING_PROFILE
    ):
        raise ValueError("an event-only training source requires the event-only model/profile")
    train_source = event_only_training_source(plan.train_source) if event_only else plan.train_source
    source_row_count = getattr(plan.train_source, "source_row_count", len(plan.train_source))
    if event_only and len(train_source) == 0:
        raise ValueError("event-only training has no eligible observed known-target s=0 rows")
    survival_profile = settings.get("training_profile") == SURVIVAL_TRAINING_PROFILE
    if survival_profile and not callable(
        getattr(model, "per_row_survival_training_losses", None)
    ):
        raise TypeError(
            f"{type(model).__name__} exposes no per_row_survival_training_losses, so it "
            f"cannot execute the {SURVIVAL_TRAINING_PROFILE!r} profile"
        )

    monitor = settings["early_stopping_metric"]
    direction = settings["early_stopping_mode"]
    horizon = settings["selection_horizon_sec"]
    requested_monitor = monitor
    selection_note = None
    if plan.validation_landmark_path is None and monitor != "train_loss":
        # No validation data: the only real scalar is the training loss. The policy
        # is applied and labeled, never a fabricated validation score.
        selection_note = (
            f"no validation split: {monitor} cannot be monitored; selecting on train_loss"
        )
        monitor, direction, horizon = "train_loss", "min", None
        reporter.message(selection_note)
    if monitor == "joint_nll" and not callable(getattr(model, "score_joint", None)):
        raise TypeError(
            f"{type(model).__name__} exposes no score_joint, so joint_nll cannot be monitored"
        )
    if monitor == "time_nll" and not callable(getattr(model, "score_joint", None)):
        raise TypeError(
            f"{type(model).__name__} has no residual-time distribution, so time_nll cannot "
            "be monitored"
        )
    validation_horizons: list[int] = []
    if plan.validation_landmark_path is not None and monitor != "train_loss":
        validation_horizons = split_horizons(plan.validation_landmark_path, "validation")
        if not validation_horizons:
            raise ValueError("the validation split has no rows")
        if horizon is not None and int(horizon) not in validation_horizons:
            raise SelectionUnavailable(
                f"the validation split has no rows at selection horizon {horizon}s "
                f"(present: {validation_horizons}); pass --validation-horizon all or one "
                "of those values"
            )
    metric_key = monitor_key(monitor, survival_profile=survival_profile)

    device = plan.device
    execution_settings = apply_precision_policy(settings, device)
    model.to(device)
    model.is_fitted = True
    sampler = build_training_sampler(train_source, int(plan.seed))
    sampler_identity = {
        "kind": type(sampler).__name__,
        "train_row_count": len(train_source),
        "row_group_lengths": getattr(sampler, "_lengths", None),
    }
    loader = DataLoader(
        train_source,
        batch_size=settings["batch_size"],
        sampler=sampler,
        num_workers=plan.num_workers,
        collate_fn=partial(
            collate_canonical_samples, max_sequence_length=plan.max_sequence_length
        ),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        betas=settings["betas"],
        eps=settings["eps"],
        amsgrad=settings["amsgrad"],
    )
    negative_generator = torch.Generator().manual_seed(int(plan.seed) + 1)
    model_config = model.resolved_config()
    family = getattr(model, "family", None) or plan.family
    environment = ckpt.environment_record(device)
    evaluation = EvaluationOptions(
        batch_size=plan.validation_batch_size,
        max_sequence_length=plan.max_sequence_length,
        num_workers=plan.num_workers,
        chunk_size=plan.validation_chunk_size,
        max_batches=settings["validation_max_batches"],
    )

    patience = settings["early_stopping_patience"]
    curve: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    historical_best: dict[str, Any] | None = None
    best_scope_start = 1
    epochs_run = 0
    early_stopped = False
    training_complete = False
    optimizer_updates = 0
    initial_optimizer_updates = 0
    next_epoch = 1
    resumed_from: dict[str, Any] | None = None
    initialized_from: dict[str, Any] | None = None
    rng_report: dict[str, Any] | None = None
    batches_per_epoch = len(loader)

    # ---- weights-only initialization ----------------------------------------
    if plan.weights_from is not None:
        payload = ckpt.load_payload(plan.weights_from, ckpt.BEST_FORMAT)
        problems = ckpt.model_differences(model_config, payload["model"])
        problems += ckpt.state_dict_differences(model.state_dict(), payload["state_dict"])
        if plan.data and payload.get("data"):
            problems += ckpt.data_differences(plan.data, payload["data"])
        ckpt.require_compatible("initialize weights", plan.weights_from, problems)
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        initialized_from = {
            "path": str(plan.weights_from),
            "selection": payload.get("selection"),
            "note": "weights only; optimizer, counters and RNG start fresh",
        }
        reporter.message(f"initialized weights from {plan.weights_from}")

    # ---- resume --------------------------------------------------------------
    if plan.resume_from is not None:
        if plan.weights_from is not None:
            raise ValueError("pass either --resume or --weights, not both")
        payload = ckpt.load_payload(plan.resume_from, ckpt.LAST_FORMAT)
        problems = ckpt.model_differences(model_config, payload["model"])
        problems += ckpt.state_dict_differences(model.state_dict(), payload["model_state"])
        if plan.data and payload.get("data"):
            problems += ckpt.data_differences(plan.data, payload["data"])
        problems += ckpt.recipe_differences(settings, payload["training"])
        ckpt.require_compatible("resume", plan.resume_from, problems)
        setting_changes = ckpt.recorded_differences(settings, payload["training"])
        model.load_state_dict(payload["model_state"])
        model.to(device)
        optimizer.load_state_dict(payload["optimizer_state"])
        # load_state_dict restores the source's param-group options along with the
        # accumulated state; this run's settings are what must be in effect, and what
        # the record and the next last.pt report.
        _apply_optimizer_settings(optimizer, settings)
        curve = copy.deepcopy(payload["curve"])
        epochs_run = int(payload["completed_epoch"])
        next_epoch = epochs_run + 1
        optimizer_updates = initial_optimizer_updates = int(payload["optimizer_updates"])
        early_stopped = bool(payload["early_stopped"])
        rng_report = ckpt.restore_rng(payload["rng"], negative_generator, sampler, device)
        if plan.num_workers:
            # Worker processes draw from their own streams, which no checkpoint holds;
            # they are seeded afresh and the report says so. Not a reason to refuse.
            rng_report["skipped"].append("dataloader_workers")
        saved_best = payload.get("best")
        best_path = Path(plan.resume_from).with_name(ckpt.BEST_NAME)
        if saved_best is not None and best_path.is_file():
            best_payload = ckpt.load_payload(best_path, ckpt.BEST_FORMAT)
            same_epoch = best_payload["selection"].get("epoch") == saved_best.get("epoch")
            compatible = not ckpt.state_dict_differences(
                model.state_dict(), best_payload["state_dict"]
            )
            if same_epoch and compatible:
                best = dict(saved_best)
                best_state = {k: v.clone() for k, v in best_payload["state_dict"].items()}
                best_scope_start = int(saved_best.get("scope_start_epoch", 1))
        if best is None and saved_best is not None:
            # The historical best weights are not available: the new best is chosen
            # within the resumed segment only, and says so. The old scalar is kept as a
            # record, never as a competitor the new segment silently loses to.
            historical_best = dict(saved_best)
            best_scope_start = next_epoch
            reporter.message(
                f"no {ckpt.BEST_NAME} beside {plan.resume_from}: best-checkpoint selection "
                f"restarts at epoch {next_epoch} (historical best epoch "
                f"{saved_best.get('epoch')} = {saved_best.get('value')} kept as a record)"
            )
        if (
            saved_best is not None
            and (saved_best.get("metric") != monitor or saved_best.get("horizon") != horizon)
        ):
            historical_best = historical_best or dict(saved_best)
            best, best_state, best_scope_start = None, None, next_epoch
            reporter.message(
                "the monitored metric changed since the checkpoint; best-checkpoint "
                f"selection restarts at epoch {next_epoch}"
            )
        # The stop state is re-decided under this run's settings, after the best scope
        # is settled: a restarted scope has no stop yet, and a larger patience or
        # budget reopens a run the source had stopped. A stop that still holds stands.
        if best is None:
            early_stopped = False
        else:
            early_stopped = epochs_run - int(best["epoch"]) >= patience
        training_complete = early_stopped or epochs_run >= settings["max_epochs"]
        if bool(payload["training_complete"]) and not training_complete:
            reporter.message(
                f"checkpoint had stopped at epoch {epochs_run} "
                f"(early_stopped={bool(payload['early_stopped'])}, max_epochs="
                f"{payload['training'].get('max_epochs')}); continuing under this run's "
                f"settings to {settings['max_epochs']} epochs, patience {patience}"
            )
        resumed_from = {
            "path": str(plan.resume_from),
            "completed_epoch": epochs_run,
            "optimizer_updates": optimizer_updates,
            "rng": rng_report,
            "setting_changes": setting_changes,
            "stop_state": {"source_early_stopped": bool(payload["early_stopped"]),
                           "early_stopped": early_stopped,
                           "training_complete": training_complete},
            "environment": payload.get("environment"),
        }
        if setting_changes:
            reporter.message("settings changed since the checkpoint: " + "; ".join(setting_changes))
        reporter.message(
            f"resumed from {plan.resume_from}: epoch {epochs_run} done, "
            f"{optimizer_updates} optimizer updates"
            + (f", RNG streams not restored: {rng_report['skipped']}"
               if rng_report and rng_report["skipped"] else "")
        )

    selection_record = {
        "metric": monitor,
        "requested_metric": requested_monitor,
        "metric_key": metric_key,
        "direction": direction,
        "horizon": horizon,
        "population": (
            "train loss (no validation selection)" if monitor == "train_loss"
            else _population_label(horizon)
        ),
        "validation_max_batches": settings["validation_max_batches"],
        "note": selection_note,
    }
    metrics_path = Path(plan.artifacts_dir) / METRICS_NAME if plan.artifacts_dir else None
    if metrics_path is not None and plan.resume_from is None and metrics_path.exists():
        metrics_path.unlink()

    def should_stop() -> bool:
        return plan.stop_condition is not None and bool(plan.stop_condition())

    def build_summary(status: str) -> dict[str, Any]:
        return {
            "trainer_version": TRAINER_VERSION,
            "status": status,
            "family": family,
            "seed": int(plan.seed),
            "device": str(device),
            "execution_settings": execution_settings,
            "training": {
                key: (list(value) if isinstance(value, tuple) else value)
                for key, value in settings.items()
            },
            "training_profile": settings["training_profile"],
            "training_view": EVENT_ONLY_TRAINING_PROFILE if event_only else "canonical-landmarks",
            "weighting_contract_version": WEIGHTING_CONTRACTS.get(settings["training_profile"]),
            "selection": selection_record,
            "train_row_count": len(train_source),
            "source_train_row_count": source_row_count,
            "prefilter_excluded_rows": source_row_count - len(train_source),
            "used_rows_total": sum(row["used_rows"] for row in curve),
            "excluded_rows_total": sum(row["excluded_rows"] for row in curve),
            "batches_per_epoch": batches_per_epoch,
            "max_train_batches": plan.max_train_batches,
            "epochs_run": epochs_run,
            "optimizer_updates": optimizer_updates,
            "optimizer_updates_this_segment": optimizer_updates - initial_optimizer_updates,
            "effective_optimizer": effective_optimizer_settings(optimizer),
            "early_stopped": early_stopped,
            "training_complete": training_complete,
            "best": best,
            "best_scope_start_epoch": best_scope_start,
            "historical_best": historical_best,
            "first_train_loss": curve[0]["train_loss"] if curve else None,
            "final_train_loss": curve[-1]["train_loss"] if curve else None,
            "resumed_from": resumed_from,
            "initialized_from": initialized_from,
            "checkpoints": _existing_checkpoints(plan.artifacts_dir, metrics_path),
        }

    def write_last(epoch: int) -> None:
        if plan.artifacts_dir is None:
            return
        payload = ckpt.last_payload(
            family=family, model_config=model_config, data=dict(plan.data),
            training={key: (list(v) if isinstance(v, tuple) else v) for key, v in settings.items()},
            model_state=ckpt.cpu_state_dict(model),
            optimizer_state=optimizer.state_dict(),
            rng=ckpt.capture_rng(negative_generator, sampler, device),
            completed_epoch=epoch, optimizer_updates=optimizer_updates, curve=curve,
            best=best, early_stopped=early_stopped, training_complete=training_complete,
            sampler_identity=sampler_identity, resumed_from=resumed_from,
            environment=environment,
        )
        ckpt.write_torch(Path(plan.artifacts_dir) / ckpt.LAST_NAME, payload)

    def write_best() -> None:
        if plan.artifacts_dir is None or best is None or best_state is None:
            return
        payload = ckpt.best_payload(
            family=family, model_config=model_config, data=dict(plan.data),
            state_dict=best_state, selection={**selection_record, **best},
            environment=environment,
        )
        ckpt.write_torch(Path(plan.artifacts_dir) / ckpt.BEST_NAME, payload)

    if resumed_from is not None and plan.artifacts_dir is not None:
        # The destination is self-contained from the first moment: the inherited
        # last state and, when it came along, the inherited best weights are
        # written here before any new epoch, so a later resume from this directory
        # (or none at all) never points at files that only existed elsewhere.
        with reporter.phase("checkpoint", epoch=epochs_run):
            write_best()
            write_last(epochs_run)

    status = STATUS_COMPLETED
    try:
        epoch_range = range(next_epoch, settings["max_epochs"] + 1) if not training_complete else ()
        for epoch in epoch_range:
            if should_stop():
                status = STATUS_STOPPED
                break
            epoch_started = time.perf_counter()
            model.train()
            loss_total = weight_total = 0.0
            used_rows = excluded_rows = 0
            contribution_counts: dict[str, int] = {}
            total_batches = (
                min(batches_per_epoch, plan.max_train_batches)
                if plan.max_train_batches is not None else batches_per_epoch
            )
            reporter.epoch_start(epoch, settings["max_epochs"], total_batches)
            stopped_in_epoch = False
            for batch_index, batch in enumerate(loader):
                if plan.max_train_batches is not None and batch_index >= plan.max_train_batches:
                    break
                if batch_index > 0 and should_stop():
                    stopped_in_epoch = True
                    break
                step = training_step(model, batch, settings, negative_generator, device)
                excluded_rows += step["excluded_rows"]
                for name, value in step.get("contribution_counts", {}).items():
                    contribution_counts[name] = contribution_counts.get(name, 0) + int(value)
                if step["loss"] is not None:
                    optimizer.zero_grad()
                    step["loss"].backward()
                    optimizer.step()
                    optimizer_updates += 1
                    used_rows += step["used_rows"]
                    batch_weight = float(step["weights"].sum())
                    loss_total += float(step["loss"].detach()) * batch_weight
                    weight_total += batch_weight
                reporter.step(
                    batch_index + 1, total_batches,
                    loss=loss_total / weight_total if weight_total else None,
                    lr=settings["learning_rate"], updates=optimizer_updates,
                )
            if stopped_in_epoch:
                status = STATUS_STOPPED
                break
            if used_rows == 0:
                raise ValueError("an entire training epoch produced no loss-eligible rows")
            epoch_loss = loss_total / weight_total
            if not math.isfinite(epoch_loss):
                raise ValueError(f"training loss is not finite at epoch {epoch}")
            train_seconds = time.perf_counter() - epoch_started

            # ---- selection ----------------------------------------------------
            validation: dict[str, Any] | None = None
            if monitor == "train_loss":
                value: float | None = epoch_loss
            else:
                if should_stop():
                    status = STATUS_STOPPED
                    break
                with reporter.phase("validation", epoch=epoch):
                    validation = evaluate_split(
                        model, plan.validation_landmark_path, split="validation",
                        horizon=horizon, options=evaluation,
                        time_nll=metric_key == "time_nll",
                        stop_condition=plan.stop_condition,
                        progress=lambda done, total, epoch=epoch: reporter.progress(
                            "validation", done, total, epoch=epoch
                        ),
                    )
                if validation.get("stopped"):
                    status = STATUS_STOPPED
                    break
                value = validation.get(metric_key)
                if value is None:
                    raise SelectionUnavailable(
                        f"validation produced no {metric_key} value at epoch {epoch} "
                        f"({selection_record['population']}: {validation['rows']} rows, "
                        f"{validation.get('eligible_rows', 0)} eligible); pick a horizon with "
                        "eligible rows (--validation-horizon) or monitor train_loss"
                    )
            epochs_run = epoch
            improved = _improves(value, best["value"] if best else None, direction)
            if improved:
                best = {
                    "epoch": epoch, "metric": monitor, "value": float(value),
                    "direction": direction, "horizon": horizon,
                    "scope_start_epoch": best_scope_start,
                }
                best_state = ckpt.cpu_state_dict(model)
            elif epoch - best["epoch"] >= patience:
                early_stopped = True
            training_complete = early_stopped or epoch == settings["max_epochs"]
            row = {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "monitor": monitor,
                "monitor_value": float(value),
                "improved": improved,
                "best_epoch": best["epoch"],
                "best_value": best["value"],
                "used_rows": used_rows,
                "excluded_rows": excluded_rows,
                "contribution_counts": contribution_counts,
                "optimizer_updates": optimizer_updates,
                "learning_rate": settings["learning_rate"],
                "train_seconds": train_seconds,
                "validation": _scalars(validation) if validation is not None else None,
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
            curve.append(row)
            reporter.validation(
                epoch, **_headline(validation, metric_key, value),
                improved=improved, best_epoch=best["epoch"],
            )
            with reporter.phase("checkpoint", epoch=epoch):
                if improved:
                    write_best()
                write_last(epoch)
            reporter.epoch_end(
                epoch, train_loss=epoch_loss,
                **({} if metric_key == "train_loss" else {metric_key: value}),
            )
            if metrics_path is not None:
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            if plan.epoch_callback is not None:
                plan.epoch_callback(row)
            if early_stopped:
                reporter.message(
                    f"early stop: no improvement for {patience} epochs "
                    f"(best epoch {best['epoch']}, {monitor} = {best['value']:.6g})"
                )
                break
    except KeyboardInterrupt:
        status = STATUS_INTERRUPTED
        reporter.message(
            f"interrupted; {ckpt.LAST_NAME} holds epoch {epochs_run}"
            + (f", {ckpt.BEST_NAME} holds epoch {best['epoch']}" if best else "")
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return build_summary(status)


#: Optimizer options this trainer supports, in the order Adam takes them.
OPTIMIZER_OPTION_KEYS = ("learning_rate", "weight_decay", "betas", "eps", "amsgrad")


def _apply_optimizer_settings(optimizer: torch.optim.Optimizer, settings: dict[str, Any]) -> None:
    """Put this run's optimizer options on every param group, keeping the state.

    Adam reads the AMSGrad running maximum (``max_exp_avg_sq``) from a parameter's
    state as soon as the option is on, and a checkpoint written without AMSGrad holds
    no such tensor: the first step after a false->true change would fail with a
    ``KeyError``. The maximum is seeded from the current second moment for every
    parameter that already has state (the maximum over a history that starts now);
    the accumulated step counts and moments are kept. A true->false change needs
    nothing: the tensor is ignored and stays in the state.
    """
    amsgrad = bool(settings["amsgrad"])
    for group in optimizer.param_groups:
        group["lr"] = float(settings["learning_rate"])
        group["weight_decay"] = float(settings["weight_decay"])
        group["betas"] = tuple(float(beta) for beta in settings["betas"])
        group["eps"] = float(settings["eps"])
        group["amsgrad"] = amsgrad
        if not amsgrad:
            continue
        for param in group["params"]:
            state = optimizer.state.get(param)
            if state and "exp_avg_sq" in state and "max_exp_avg_sq" not in state:
                state["max_exp_avg_sq"] = state["exp_avg_sq"].detach().clone()


def effective_optimizer_settings(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """What the optimizer will actually use on its next step (first param group)."""
    group = optimizer.param_groups[0]
    return {
        "learning_rate": float(group["lr"]),
        "weight_decay": float(group["weight_decay"]),
        "betas": [float(beta) for beta in group["betas"]],
        "eps": float(group["eps"]),
        "amsgrad": bool(group["amsgrad"]),
    }


def _existing_checkpoints(artifacts_dir, metrics_path) -> dict[str, str | None]:
    """Paths of the files actually on disk; a name is never reported for an absent file."""
    if artifacts_dir is None:
        return {"last": None, "best": None, "metrics": None}
    root = Path(artifacts_dir)
    return {
        "last": str(root / ckpt.LAST_NAME) if (root / ckpt.LAST_NAME).is_file() else None,
        "best": str(root / ckpt.BEST_NAME) if (root / ckpt.BEST_NAME).is_file() else None,
        "metrics": str(metrics_path) if metrics_path is not None and metrics_path.is_file() else None,
    }


def _headline(validation: dict[str, Any] | None, metric_key: str, value: float) -> dict[str, Any]:
    """What the terminal shows per epoch: the monitor first, then the usual headline."""
    if validation is None:
        return {metric_key: value}
    shown = {metric_key: value}
    for key in ("ndcg@10", "recall@10", "poi_nll", "joint_nll", "rows", "eligible_rows"):
        if key in validation and key not in shown:
            shown[key] = validation[key]
    return shown


def _scalars(evaluation: dict[str, Any]) -> dict[str, Any]:
    """The per-epoch metrics worth keeping in the curve: numbers and counts only."""
    keep = {}
    for key, value in evaluation.items():
        if key in {"by_horizon", "evaluator_version", "tie_policy", "score_source",
                   "score_kind", "split", "horizon", "stopped", "candidate_count"}:
            continue
        if isinstance(value, (int, float)) or value is None:
            keep[key] = value
    return keep


__all__ = [
    "MONITOR_DIRECTIONS",
    "STATUS_COMPLETED",
    "STATUS_INTERRUPTED",
    "STATUS_STOPPED",
    "SelectionUnavailable",
    "TrainingPlan",
    "train_model",
]
