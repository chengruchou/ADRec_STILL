"""One training run: resolve inputs, fit or train, evaluate, write the run directory.

Output layout::

    <output>/config.json      what this run resolved to (data, model, training, device)
    <output>/metrics.jsonl    one row per completed epoch (neural runs)
    <output>/summary.json     status, best epoch, evaluation, timings
    <output>/last.pt          resumable state after the last completed epoch (neural)
    <output>/best.pt          selected weights (neural)
    <output>/cache/           preprocessed raw data, survival view (when needed)
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from adrec_still.data.training_views import EVENT_ONLY_TRAINING_PROFILE
from adrec_still.models.baselines.harness import set_global_seed
from adrec_still.models.baselines.trainer import SELECTION_METRICS, resolve_device
from adrec_still.models.still.losses import SURVIVAL_TRAINING_PROFILE
from adrec_still.training import checkpoint as ckpt
from adrec_still.training import families
from adrec_still.training.defaults import EVALUATION_DEFAULTS, backend, default_model_config
from adrec_still.training.evaluate import EvaluationOptions, evaluate_split
from adrec_still.training.inputs import DataSource, resolve_data
from adrec_still.training.loop import STATUS_COMPLETED, TrainingPlan, train_model
from adrec_still.training.progress import Reporter, TerminalReporter

SERVICE_VERSION = "still-train-v1.0"
NEURAL_SELECTION_DEFAULT = "val_loss"
NO_VALIDATION_SELECTION = "train_loss"


@dataclass
class RunSpec:
    """Everything a run needs, as the CLI (or a caller) states it."""

    data: Path | None
    output: Path
    model: str | None = None
    config: Path | None = None
    model_config: Path | None = None
    overrides: dict[str, Any] = field(default_factory=dict)
    adapter: str | None = None
    poi_file: Path | None = None
    cache_dir: Path | None = None
    workers: int | None = None
    row_limit: int | None = None
    seed: int = 41
    device: str | None = None
    epochs: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    patience: int | None = None
    monitor: str | None = None
    validation_horizon: int | str | None = None
    validation_max_batches: int | None = None
    max_train_batches: int | None = None
    num_workers: int | None = None
    eval_batch_size: int | None = None
    max_sequence_length: int | None = None
    chunk_size: int | None = None
    resume: Path | None = None
    weights: Path | None = None
    evaluate_test: bool = True
    evaluate_validation: bool = False
    progress: str = "auto"
    progress_interval: float = 5.0

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        return {key: (str(value) if isinstance(value, Path) else value)
                for key, value in record.items()}


class RunFailed(RuntimeError):
    """A run could not proceed; the message is meant for the terminal."""


# --------------------------------------------------------------- configuration


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RunFailed(f"{path} does not hold a mapping")
    return loaded


def _apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise RunFailed(f"override {key}: {part} is not a mapping")
        target[parts[-1]] = value


def config_base(config_path: Path) -> Path:
    """The directory a config's relative paths are resolved against.

    Committed experiment files name repo-relative paths (``configs/model/x.yaml``,
    ``data/processed/gowalla``), so the base is the repository root that contains
    the config (the nearest ancestor with ``pyproject.toml``); a config outside any
    repository resolves against its own directory. The caller's working directory
    never takes part, so the same file loads the same settings from anywhere.
    """
    directory = Path(config_path).resolve().parent
    for candidate in (directory, *directory.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return directory


def resolve_config_path(reference: str | Path, config_path: Path) -> Path:
    """A path named inside ``config_path``: absolute as is, else against the config base."""
    path = Path(reference)
    if path.is_absolute():
        return path
    base = config_base(config_path)
    candidate = base / path
    if candidate.exists():
        return candidate
    sibling = Path(config_path).resolve().parent / path
    if sibling.exists():
        return sibling
    return candidate


def resolve_model_config(spec: RunSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    """The model config this run trains, plus a record of where it came from."""
    source: dict[str, Any] = {}
    if spec.config is not None:
        experiment = _load_yaml(spec.config)
        model_path = experiment.get("model_config")
        if not model_path:
            raise RunFailed(f"{spec.config} names no model_config")
        model_path = resolve_config_path(model_path, spec.config)
        config = _load_yaml(model_path)
        source = {"experiment_config": str(spec.config), "model_config": str(model_path),
                  "experiment": {k: v for k, v in experiment.items()
                                 if k in ("name", "evaluation", "seeds", "device", "output_dir")}}
        dataset_path = experiment.get("dataset_config")
        if dataset_path:
            dataset_path = resolve_config_path(dataset_path, spec.config)
            if dataset_path.exists():
                dataset = _load_yaml(dataset_path)
                source["dataset_config"] = str(dataset_path)
                if dataset.get("processed_root"):
                    source["processed_root"] = resolve_config_path(
                        dataset["processed_root"], dataset_path
                    )
        if spec.model is not None and spec.model != config.get("name"):
            raise RunFailed(
                f"--model {spec.model} disagrees with {model_path} (name {config.get('name')})"
            )
    elif spec.model_config is not None:
        config = _load_yaml(spec.model_config)
        source = {"model_config": str(spec.model_config)}
        if spec.model is not None and spec.model != config.get("name"):
            raise RunFailed(
                f"--model {spec.model} disagrees with {spec.model_config} "
                f"(name {config.get('name')})"
            )
    else:
        if spec.model is None:
            raise RunFailed("pass --model NAME, --model-config FILE or --config FILE")
        config = default_model_config(spec.model)
        source = {"defaults": "built-in"}
    name = config.get("name")
    if not name:
        raise RunFailed("the model config has no name")
    info = backend(name)
    config.setdefault("family", info["family"])
    defaults = default_model_config(name)
    for key, value in defaults.items():
        if key == "training":
            training = dict(value)
            training.update(config.get("training") or {})
            config["training"] = training
        else:
            config.setdefault(key, value)
    if spec.overrides:
        _apply_overrides(config, spec.overrides)
        source["overrides"] = dict(spec.overrides)
    config.pop("formal_recipe", None)
    return config, source


def resolve_evaluation(spec: RunSpec, source: dict[str, Any], data: DataSource) -> EvaluationOptions:
    block = dict(EVALUATION_DEFAULTS)
    block.update((source.get("experiment") or {}).get("evaluation") or {})
    block["max_sequence_length"] = data.max_history_length or block["max_sequence_length"]
    if spec.eval_batch_size is not None:
        block["batch_size"] = spec.eval_batch_size
    if spec.num_workers is not None:
        block["num_workers"] = spec.num_workers
    if spec.max_sequence_length is not None:
        block["max_sequence_length"] = spec.max_sequence_length
    if spec.chunk_size is not None:
        block["chunk_size"] = spec.chunk_size
    return EvaluationOptions(
        batch_size=int(block["batch_size"]),
        max_sequence_length=int(block["max_sequence_length"]),
        num_workers=int(block["num_workers"]),
        chunk_size=int(block["chunk_size"]),
    )


def resolve_training(
    spec: RunSpec, config: dict[str, Any], data: DataSource
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The training block this run executes, plus the selection policy record."""
    training = dict(config.get("training") or {})
    if spec.epochs is not None:
        training["max_epochs"] = int(spec.epochs)
    if spec.batch_size is not None:
        training["batch_size"] = int(spec.batch_size)
    if spec.learning_rate is not None:
        training["learning_rate"] = float(spec.learning_rate)
    if spec.patience is not None:
        training["early_stopping_patience"] = int(spec.patience)
    if spec.validation_max_batches is not None:
        training["validation_max_batches"] = int(spec.validation_max_batches)
    if spec.validation_horizon is not None:
        training["selection_horizon_sec"] = (
            None if str(spec.validation_horizon).lower() == "all" else int(spec.validation_horizon)
        )
    policy: dict[str, Any] = {}
    has_validation = data.has_split("validation")
    monitor = spec.monitor or training.get("early_stopping_metric")
    if not has_validation:
        if monitor not in (None, NO_VALIDATION_SELECTION):
            policy["note"] = (
                f"no validation split in the data: {monitor} cannot be monitored, selecting "
                "on train_loss instead"
            )
        monitor = NO_VALIDATION_SELECTION
        policy["label"] = "train-loss selection (no validation split)"
    elif monitor is None:
        monitor = NEURAL_SELECTION_DEFAULT
    if monitor not in SELECTION_METRICS:
        raise RunFailed(
            f"--monitor {monitor!r} is not a selectable metric; choose one of "
            f"{sorted(SELECTION_METRICS)}"
        )
    training["early_stopping_metric"] = monitor
    training.pop("early_stopping_mode", None)
    if monitor == NO_VALIDATION_SELECTION:
        training["selection_horizon_sec"] = None
    profile = training.get("training_profile")
    policy.update({
        "monitor": monitor,
        "direction": SELECTION_METRICS[monitor],
        "resolves_to": (
            "joint_nll" if monitor == "val_loss" and profile == SURVIVAL_TRAINING_PROFILE
            else "poi_nll" if monitor == "val_loss" else monitor
        ),
        "horizon_seconds": training.get("selection_horizon_sec"),
    })
    return training, policy


def git_metadata() -> dict[str, Any]:
    """Commit and dirty state when available; ``None`` values otherwise, never a failure."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True, timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
            text=True, check=True, timeout=10,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "note": "not a Git checkout or git unavailable"}
    return {"commit": commit, "dirty": dirty}


# ---------------------------------------------------------------------- run


def run(spec: RunSpec) -> dict[str, Any]:
    """Execute one run and return its summary (also written to ``summary.json``)."""
    reporter: Reporter = (
        Reporter() if spec.progress == "off"
        else TerminalReporter(mode=spec.progress, interval=spec.progress_interval)
    )
    started = time.time()
    output = Path(spec.output)
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(spec.cache_dir) if spec.cache_dir is not None else output / "cache"
    with ckpt.RunLock(output):
        config, source = resolve_model_config(spec)
        info = backend(config["name"])
        data_path = spec.data if spec.data is not None else source.get("processed_root")
        if data_path is None:
            raise RunFailed("pass --data (a canonical directory or a raw check-in file)")
        data = resolve_data(
            data_path, adapter=spec.adapter, poi_file=spec.poi_file, cache_dir=cache_dir,
            workers=spec.workers, row_limit=spec.row_limit, reporter=reporter,
        )
        evaluation = resolve_evaluation(spec, source, data)
        device = resolve_device(spec.device or (config.get("training") or {}).get("device"))
        set_global_seed(int(spec.seed))
        record: dict[str, Any] = {
            "service_version": SERVICE_VERSION,
            "model": config["name"],
            "family": info["family"],
            "kind": info["kind"],
            "seed": int(spec.seed),
            "device": str(device),
            "data": data.record(),
            "model_config": config,
            "config_source": source,
            "evaluation": asdict(evaluation),
            "spec": spec.as_record(),
            "git": git_metadata(),
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        }
        if info["kind"] == "floor":
            summary = _run_floor(spec, config, data, evaluation, reporter, record, output)
        else:
            summary = _run_neural(
                spec, config, data, evaluation, device, reporter, record, output, cache_dir,
            )
        summary["seconds"] = time.time() - started
        summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        summary = jsonable(summary)
        ckpt.atomic_json(output / "summary.json", summary)
        reporter.message(f"wrote {output / 'summary.json'} (status {summary['status']})")
        return summary


def jsonable(value: Any) -> Any:
    """Paths become strings, tuples become lists; everything else is left to json."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def _write_config(output: Path, record: dict[str, Any]) -> None:
    ckpt.atomic_json(output / "config.json", jsonable(record))


def _evaluate(model, data: DataSource, split: str, evaluation: EvaluationOptions,
              reporter: Reporter) -> dict[str, Any]:
    with reporter.phase(f"evaluate-{split}"):
        result = evaluate_split(
            model, data.landmark_path, split=split, horizon=None, breakdown=True,
            options=evaluation,
            progress=lambda done, total: reporter.progress(f"evaluate-{split}", done, total),
        )
    headline = {key: result.get(key) for key in ("rows", "eligible_rows", "ndcg@10",
                                                    "recall@10", "poi_nll", "joint_nll",
                                                    "time_nll")
                if result.get(key) is not None}
    reporter.message(f"[{split}] " + " ".join(f"{k} {v:.6g}" if isinstance(v, float) else f"{k} {v}"
                                              for k, v in headline.items()))
    return result


def _run_floor(spec, config, data, evaluation, reporter, record, output) -> dict[str, Any]:
    record["training"] = None
    record["selection"] = {"label": "deterministic fit; no checkpoint selection"}
    _write_config(output, record)
    fit_started = time.time()
    model = families.fit_floor(config, data, reporter=reporter)
    summary: dict[str, Any] = {
        "service_version": SERVICE_VERSION,
        "status": STATUS_COMPLETED,
        "model": config["name"],
        "family": record["family"],
        "kind": "floor",
        "seed": int(spec.seed),
        "fit_seconds": time.time() - fit_started,
        "resolved_model": model.resolved_config(),
        "parameter_count": model.parameter_count(),
        "checkpoints": None,
        "evaluation": {},
    }
    for split, wanted in (("validation", spec.evaluate_validation or True),
                          ("test", spec.evaluate_test)):
        if wanted and data.has_split(split):
            if hasattr(model, "reset_prediction_stats"):
                model.reset_prediction_stats()
            summary["evaluation"][split] = _evaluate(
                model, data, split, evaluation, reporter
            )
            if hasattr(model, "prediction_stats"):
                summary["evaluation"][split]["prediction_stats"] = model.prediction_stats()
    return summary


def _run_neural(spec, config, data, evaluation, device, reporter, record, output,
                cache_dir) -> dict[str, Any]:
    training, policy = resolve_training(spec, config, data)
    record["training"] = training
    record["selection"] = policy
    train_source, train_record = families.prepare_training_source(
        training, data, cache_dir, reporter=reporter,
    )
    model = families.build_model(config, data.number_of_pois)
    if record["family"] == "external":
        checkpoint = spec.resume or spec.weights
        if checkpoint is not None:
            # The weights being loaded were trained under the checkpoint's own
            # normalization statistics; those are restored with them. The new data's
            # statistics are a fit provenance, not a loading condition.
            payload = ckpt.load_payload(
                checkpoint, ckpt.LAST_FORMAT if spec.resume else ckpt.BEST_FORMAT
            )
            families.restore_external_statistics(
                model, (payload.get("model") or {}).get("log_time_statistics")
            )
            train_record["log_time_statistics"] = {
                **dict(model.log_time_statistics_record),
                "restored_from": str(checkpoint),
            }
            reporter.message(f"restored log-time statistics from {checkpoint}")
        else:
            train_record["log_time_statistics"] = families.fit_external_statistics(
                model, config, data, train_record, cache_dir, reporter=reporter,
            )
    record["training_source"] = train_record
    record["resolved_model"] = model.resolved_config()
    record["parameter_count"] = model.parameter_count()
    _write_config(output, record)
    if policy.get("note"):
        reporter.message(policy["note"])
    plan = TrainingPlan(
        train_source=train_source,
        training=training,
        seed=int(spec.seed),
        family=record["family"],
        data=dict(data.identity),
        max_sequence_length=evaluation.max_sequence_length,
        validation_landmark_path=data.landmark_path if data.has_split("validation") else None,
        validation_batch_size=evaluation.batch_size,
        validation_chunk_size=evaluation.chunk_size,
        num_workers=evaluation.num_workers,
        device=device,
        artifacts_dir=output,
        max_train_batches=spec.max_train_batches,
        reporter=reporter,
        resume_from=Path(spec.resume) if spec.resume else None,
        weights_from=Path(spec.weights) if spec.weights else None,
    )
    reporter.message(
        f"training {config['name']} on {device} | seed {spec.seed} | "
        f"{len(train_source)} train rows | monitor {policy['monitor']} "
        f"({policy['direction']}, {policy['resolves_to']}) | "
        f"horizon {policy['horizon_seconds'] if policy['horizon_seconds'] is not None else 'all'}"
    )
    training_summary = train_model(model, plan)
    summary: dict[str, Any] = {
        "service_version": SERVICE_VERSION,
        "status": training_summary["status"],
        "model": config["name"],
        "family": record["family"],
        "kind": "neural",
        "seed": int(spec.seed),
        "selection_policy": policy,
        "training": training_summary,
        "resolved_model": model.resolved_config(),
        "parameter_count": model.parameter_count(),
        "checkpoints": training_summary["checkpoints"],
        "evaluation": {},
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
    }
    if training_summary["status"] != STATUS_COMPLETED:
        reporter.message(f"training ended with status {training_summary['status']}; "
                         "skipping evaluation")
        return summary
    if training_summary["best"] is None:
        return summary
    if spec.evaluate_validation and data.has_split("validation"):
        summary["evaluation"]["validation"] = _evaluate(
            model, data, "validation", evaluation, reporter,
        )
    if spec.evaluate_test and data.has_split("test"):
        summary["evaluation"]["test"] = _evaluate(
            model, data, "test", evaluation, reporter,
        )
    elif spec.evaluate_test:
        summary["evaluation"]["test"] = None
        reporter.message("no test split in the data; training is complete without one")
    return summary


def profile_label(training: dict[str, Any]) -> str:
    profile = training.get("training_profile")
    if profile == SURVIVAL_TRAINING_PROFILE:
        return "survival"
    if profile == EVENT_ONLY_TRAINING_PROFILE:
        return "event-only"
    return "observed-known"


def experiment_defaults(config_path: Path) -> dict[str, Any]:
    """Seeds, output directory and processed root a legacy experiment YAML declares."""
    experiment = _load_yaml(config_path)
    out: dict[str, Any] = {
        "seeds": [int(seed) for seed in experiment.get("seeds", [41])],
        "output_dir": Path(experiment["output_dir"]) if experiment.get("output_dir") else None,
        "processed_root": None,
    }
    dataset_path = experiment.get("dataset_config")
    if dataset_path:
        dataset_path = resolve_config_path(dataset_path, config_path)
        if dataset_path.exists():
            root = _load_yaml(dataset_path).get("processed_root")
            out["processed_root"] = resolve_config_path(root, dataset_path) if root else None
    return out
