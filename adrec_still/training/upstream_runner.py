"""Service glue for the upstream family: canonical and ``dataset.pkl`` runs.

Canonical runs train on per-user pre-cutoff sequences (one supervised pass per
gap) and are selected/evaluated on the shared landmark evaluator, so their
numbers live in the same tables as every other backend. ``dataset.pkl`` runs
keep the fork's own protocol (per-user leave-one-out validation/test) scored by
the exact chunked ranker. The bundled pkl files carry no timestamps, so any
silence/time/cell request against them fails with a named missing field instead
of fabricated values.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from adrec_still.data.sequences import (
    SequenceCorpus,
    canonical_train_sequences,
    load_dataset_pkl,
)
from adrec_still.evaluation.sequence_eval import evaluate_sequence_corpus
from adrec_still.models.upstream.wrappers import build_upstream_model
from adrec_still.training import checkpoint as ckpt
from adrec_still.training.evaluate import EvaluationOptions, evaluate_split
from adrec_still.training.inputs import DataSource
from adrec_still.training.loop import STATUS_COMPLETED
from adrec_still.training.progress import Reporter
from adrec_still.training.stages import (
    StagedPlan,
    load_upstream_pretrain_pth,
    train_staged_model,
)

PKL_SPLITS = ("train", "validation", "test")


class PklDataSource:
    """A loaded ``dataset.pkl`` behind the small surface the service needs."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.corpus: SequenceCorpus = load_dataset_pkl(self.path)
        self.number_of_pois = self.corpus.number_of_pois
        self.identity = dict(self.corpus.identity)
        self.max_history_length: int | None = None  # the model's max_len governs
        self.splits = [
            split for split, answers in (
                ("train", self.corpus.train), ("validation", self.corpus.validation),
                ("test", self.corpus.test),
            ) if any(answers)
        ]

    def has_split(self, split: str) -> bool:
        return split in self.splits

    def record(self) -> dict[str, Any]:
        return {
            "kind": "dataset_pkl",
            "path": str(self.path),
            "number_of_pois": self.number_of_pois,
            "splits": self.splits,
            "identity": self.identity,
            "timestamps": False,
            "note": "no timestamps: silence/time/cell semantics are unavailable",
        }


def _resolve_pretrained_embedding(config: dict[str, Any], model) -> Any:
    path = config.get("pretrained_embedding_path")
    if not path:
        return None
    return load_upstream_pretrain_pth(path, model.number_of_pois, model.hidden_size)


def _canonical_validator(data: DataSource, evaluation: EvaluationOptions,
                         training: dict[str, Any], metric_key: str):
    horizon = training.get("selection_horizon_sec")
    max_batches = training.get("validation_max_batches")

    def validate(model):
        result = evaluate_split(
            model, data.landmark_path, split="validation", horizon=horizon,
            options=EvaluationOptions(
                batch_size=evaluation.batch_size,
                max_sequence_length=evaluation.max_sequence_length,
                num_workers=evaluation.num_workers,
                chunk_size=evaluation.chunk_size,
                max_batches=max_batches,
            ),
            time_nll=False,
        )
        return result.get(metric_key), result

    return validate


def _pkl_validator(source: PklDataSource, evaluation: EvaluationOptions, metric_key: str,
                   max_batches: int | None):
    def validate(model):
        result = evaluate_sequence_corpus(
            model, source.corpus, "validation", batch_size=evaluation.batch_size,
            chunk_size=evaluation.chunk_size, max_batches=max_batches,
        )
        return result.get(metric_key), result

    return validate


def run_upstream(
    spec, config: dict[str, Any], data, evaluation: EvaluationOptions, device,
    reporter: Reporter, record: dict[str, Any], output: Path, training: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Train + evaluate one upstream-family model on canonical or pkl data."""
    is_pkl = isinstance(data, PklDataSource)
    model = build_upstream_model(config, data.number_of_pois)
    record["training"] = training
    record["selection"] = policy
    record["resolved_model"] = model.resolved_config()
    record["parameter_count"] = model.parameter_count()
    monitor = training["early_stopping_metric"]
    metric_key = "poi_nll" if monitor == "val_loss" else monitor

    if is_pkl:
        corpus = data.corpus
        data_identity = {"number_of_pois": data.number_of_pois,
                         "catalog_fingerprint": None,
                         "source": data.identity}
        validator = (
            _pkl_validator(data, evaluation, metric_key,
                           training.get("validation_max_batches"))
            if data.has_split("validation") and monitor != "train_loss" else None
        )
    else:
        if data.events_path is None or data.split_manifest_path is None:
            raise FileNotFoundError(
                "upstream sequence training needs events.parquet and "
                f"split_manifest.parquet beside {data.landmark_path}; pass the raw "
                "check-in file to --data or point it at a complete canonical directory"
            )
        with reporter.phase("sequence-view"):
            corpus = canonical_train_sequences(
                data.events_path, data.split_manifest_path, data.number_of_pois
            )
        record["training_source"] = {"training_view": "pre-cutoff user sequences",
                                     **corpus.identity}
        data_identity = dict(data.identity)
        validator = (
            _canonical_validator(data, evaluation, training, metric_key)
            if data.has_split("validation") and monitor != "train_loss" else None
        )

    from adrec_still.models.baselines.trainer import SELECTION_METRICS

    plan = StagedPlan(
        corpus=corpus,
        training=training,
        seed=int(spec.seed),
        device=device,
        artifacts_dir=output,
        data_identity=data_identity,
        family="upstream",
        validate=validator,
        monitor=monitor if validator is not None else "train_loss",
        monitor_direction=SELECTION_METRICS.get(monitor, "min"),
        reporter=reporter,
        max_train_batches=spec.max_train_batches,
        num_workers=evaluation.num_workers,
        resume_from=Path(spec.resume) if spec.resume else None,
        weights_from=Path(spec.weights) if spec.weights else None,
        pretrained_embedding=_resolve_pretrained_embedding(config, model),
    )
    from adrec_still.training.service import _write_config  # late: avoid import cycle

    _write_config(output, record)
    reporter.message(
        f"training {config['name']} on {device} | seed {spec.seed} | "
        f"{len(corpus.train)} train sequences | monitor {plan.monitor} "
        f"({plan.monitor_direction})"
    )
    training_summary = train_staged_model(model, plan)

    summary: dict[str, Any] = {
        "status": training_summary["status"],
        "model": config["name"],
        "family": "upstream",
        "kind": "neural",
        "seed": int(spec.seed),
        "selection_policy": policy,
        "training": training_summary,
        "resolved_model": model.resolved_config(),
        "parameter_count": model.parameter_count(),
        "checkpoints": training_summary["checkpoints"],
        "evaluation": {},
    }
    if training_summary["status"] != STATUS_COMPLETED:
        reporter.message(f"training ended with status {training_summary['status']}; "
                         "skipping evaluation")
        return summary

    def final_evaluate(split: str) -> dict[str, Any]:
        with reporter.phase(f"evaluate-{split}"):
            if is_pkl:
                return evaluate_sequence_corpus(
                    model, data.corpus, split, batch_size=evaluation.batch_size,
                    chunk_size=evaluation.chunk_size,
                )
            return evaluate_split(
                model, data.landmark_path, split=split, horizon=None, breakdown=True,
                options=evaluation, time_nll=False,
                progress=lambda done, total: reporter.progress(
                    f"evaluate-{split}", done, total
                ),
            )

    if spec.evaluate_validation and data.has_split("validation"):
        summary["evaluation"]["validation"] = final_evaluate("validation")
    if spec.evaluate_test and data.has_split("test"):
        summary["evaluation"]["test"] = final_evaluate("test")
    elif spec.evaluate_test:
        summary["evaluation"]["test"] = None
        reporter.message("no test split in the data; training is complete without one")
    return summary


def pkl_time_semantics_error(requested: str) -> ValueError:
    return ValueError(
        f"{requested} needs event timestamps, and dataset.pkl files carry none "
        "(only per-user item orderings); use a raw check-in file or a canonical "
        "events directory for time/silence/cell semantics"
    )


def jsonable_duration(started: float) -> float:
    return time.time() - started
