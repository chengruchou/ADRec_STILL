"""Config-driven run harness shared by the deterministic baselines.

The harness owns nothing metric-specific: it streams the shared
:class:`adrec_still.data.dataset.LandmarkDataset`, collates with the shared collator,
calls :func:`adrec_still.evaluation.evaluator.evaluate` once per (batch, query
horizon) group, and combines those results.

Combination is a weighted mean: ``Recall@K``, ``NDCG@K``, ``MRR@K`` and the POI
NLL are per-sample means, so weighting each call by its ranking ``sample_count``
reproduces a single whole-split call without re-implementing any metric. Fixed
candidate metadata and analytical random floors are copied only after verifying
that every call uses the same candidate contract. Counts and the float64 ranking
metrics recombine exactly; the POI NLL comes from a float32 cross-entropy, so it
matches a whole-split call to float32 rounding (about 1e-7 relative).
"""

from __future__ import annotations

import json
import random
import subprocess
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from adrec_still.data.collate import collate_canonical_samples
from adrec_still.data.dataset import LandmarkDataset
from adrec_still.evaluation.evaluator import evaluate, write_evaluation_json
from adrec_still.models.baselines.base import BASELINE_INTERFACE_VERSION, BaselineModel
from adrec_still.models.baselines.markov import (
    FirstOrderMarkov,
    MarkovTrainingSource,
    train_transition_counts,
)
from adrec_still.models.baselines.popularity import (
    GlobalPop,
    poi_catalog_counts,
    train_catalog_size,
    train_target_counts,
)
from adrec_still.models.baselines.repeat_last import RepeatLast
from adrec_still.models.baselines.user_history_pop import UserHistoryPop

RUN_RECORD_SCHEMA_VERSION = "still-baseline-run-v0.1"
RANKING_COUNT_KEYS = ("sample_count", "observed_target_count", "cold_start_target_count")
RANKING_STATIC_KEYS = ("candidate_count", "random_floor")
TIME_COUNT_KEYS = ("sample_count", "observed_count", "censored_count")


def set_global_seed(seed: int) -> None:
    """Seed every source of randomness used by the baseline harness."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_commit_sha() -> str | None:
    """Return the current commit, or None outside a Git checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def git_worktree_dirty() -> bool | None:
    """Return True when tracked files carry uncommitted changes, None outside Git.

    Untracked files are ignored: run outputs written under ``results/`` must not
    poison the provenance of the code that produced them.
    """
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(completed.stdout.strip())


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def resolve_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment config together with the dataset and model files it names."""
    experiment_path = Path(path)
    experiment = _load_yaml(experiment_path)
    for key in ("name", "dataset_config", "model_config", "evaluation", "seeds", "output_dir"):
        if key not in experiment:
            raise KeyError(f"experiment config is missing required key: {key}")
    dataset = _load_yaml(experiment["dataset_config"])
    model = _load_yaml(experiment["model_config"])
    if model.get("family") != "baseline":
        raise ValueError("the baseline harness only runs configs with family: baseline")
    return {
        "experiment": experiment,
        "dataset": dataset,
        "model": model,
        "paths": {
            "experiment_config": str(experiment_path),
            "dataset_config": str(experiment["dataset_config"]),
            "model_config": str(experiment["model_config"]),
        },
    }


def _sum_counters(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def _weighted_mean(total: float, weight: int) -> float | None:
    return total / weight if weight else None


def aggregate_evaluations(
    results: list[dict[str, Any]], split: str | None = None
) -> dict[str, Any]:
    """Combine per-batch shared-evaluator results into one exact split summary."""
    if not results:
        raise ValueError("cannot aggregate an empty list of evaluation results")
    versions = {result["schema_version"] for result in results}
    if len(versions) != 1:
        raise ValueError("evaluation results use different schema versions")
    ranking_keys = list(results[0]["ranking"])
    mean_keys = [
        key
        for key in ranking_keys
        if key not in RANKING_COUNT_KEYS
        and key not in RANKING_STATIC_KEYS
        and key != "ranking_coverage"
    ]
    static_values = {key: results[0]["ranking"][key] for key in RANKING_STATIC_KEYS}
    counts = dict.fromkeys(RANKING_COUNT_KEYS, 0)
    totals = dict.fromkeys(mean_keys, 0.0)
    horizons: dict[str, int] = {}
    cohorts: dict[str, int] = {}
    for result in results:
        ranking = result["ranking"]
        if list(ranking) != ranking_keys:
            raise ValueError("evaluation results expose different ranking keys")
        for key in RANKING_STATIC_KEYS:
            if ranking[key] != static_values[key]:
                raise ValueError(f"evaluation results use different ranking {key}")
        for key in RANKING_COUNT_KEYS:
            counts[key] += int(ranking[key])
        weight = int(ranking["sample_count"])
        if weight:
            for key in mean_keys:
                value = ranking[key]
                if value is None:
                    raise ValueError(f"metric {key} is missing for a non-empty ranking batch")
                totals[key] += float(value) * weight
        _sum_counters(horizons, result["sample_counts_by_horizon_seconds"])
        _sum_counters(cohorts, result["sample_counts_by_cohort"])

    ranking_summary: dict[str, Any] = {key: counts[key] for key in RANKING_COUNT_KEYS}
    ranking_summary["ranking_coverage"] = _weighted_mean(
        float(counts["sample_count"]), counts["observed_target_count"]
    )
    ranking_summary.update(static_values)
    for key in mean_keys:
        ranking_summary[key] = _weighted_mean(totals[key], counts["sample_count"])

    summary: dict[str, Any] = {
        "schema_version": results[0]["schema_version"],
        "split": split if split is not None else results[0]["split"],
        "batch_count": len(results),
        "ranking": ranking_summary,
        "sample_counts_by_horizon_seconds": dict(
            sorted(horizons.items(), key=lambda item: int(item[0]))
        ),
        "sample_counts_by_cohort": dict(sorted(cohorts.items())),
    }
    time_results = [result["time"] for result in results if result.get("time") is not None]
    if time_results:
        summary["time"] = _aggregate_time(time_results)
    return summary


def _aggregate_time(time_results: list[dict[str, Any]]) -> dict[str, Any]:
    counts = dict.fromkeys(TIME_COUNT_KEYS, 0)
    nll_total = 0.0
    observed_sum = 0.0
    censored_sum = 0.0
    for time_result in time_results:
        for key in TIME_COUNT_KEYS:
            counts[key] += int(time_result[key])
        nll_total += float(time_result["return_time_nll"]) * int(time_result["sample_count"])
        observed_sum += float(time_result.get("observed_time_nll_sum") or 0.0)
        censored_sum += float(time_result.get("censored_time_nll_sum") or 0.0)
    summary: dict[str, Any] = dict(counts)
    summary["return_time_nll"] = _weighted_mean(nll_total, counts["sample_count"])
    summary["observed_time_nll"] = _weighted_mean(observed_sum, counts["observed_count"])
    summary["censored_time_nll"] = _weighted_mean(censored_sum, counts["censored_count"])
    return summary


def evaluate_split(
    model: BaselineModel,
    landmark_path: str | Path,
    split: str,
    batch_size: int = 128,
    max_sequence_length: int | None = 100,
    num_workers: int = 0,
    max_batches: int | None = None,
    stop_condition: Any = None,
) -> dict[str, Any]:
    """Stream one split through the shared evaluator and report overall and per-horizon metrics.

    ``stop_condition`` is an optional cooperative-deadline hook (proposed for the
    #51 bounded smoke; needs Student A review as a harness extension): checked
    before each *subsequent* batch starts, so the batch already running finishes
    and no further batch starts once it fires. ``stopped_early: true`` is
    reported only when a pending batch was actually skipped — a stream that
    consumed its whole split or its ``max_batches`` prefix is complete and is
    never labeled interrupted, no matter when the condition fires. Callers must
    treat a ``stopped_early`` summary as covering a partial cohort. ``None``
    (the default) preserves the existing behavior for every current caller
    exactly.
    """
    dataset = LandmarkDataset(landmark_path, split=split)
    if not len(dataset):
        raise ValueError(f"split has no landmark samples: {split}")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial(collate_canonical_samples, max_sequence_length=max_sequence_length),
    )
    by_horizon: dict[int, list[dict[str, Any]]] = {}
    checked_output_contract = False
    evaluated_samples = 0
    stopped_early = False
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        # Cooperative deadline: polled only when another batch is about to start,
        # so a completed stream is never mislabeled as interrupted.
        if stop_condition is not None and batch_index > 0 and stop_condition():
            stopped_early = True
            break
        for horizon in torch.unique(batch["query_s_sec"]).tolist():
            rows = batch["query_s_sec"].eq(horizon)
            sub_batch = {key: value[rows] for key, value in batch.items()}
            prediction = model.predict(sub_batch)
            prediction.validate(model.number_of_pois, check_finite=not checked_output_contract)
            checked_output_contract = True
            by_horizon.setdefault(int(horizon), []).append(
                evaluate(
                    prediction.poi_logits,
                    sub_batch,
                    time_prediction=prediction.time_prediction,
                    time_distribution=getattr(prediction, "time_distribution", None),
                    split=split,
                )
            )
        evaluated_samples += int(batch["query_s_sec"].shape[0])
    if not by_horizon:
        raise ValueError(f"no batches were evaluated for split: {split}")
    flattened = [result for results in by_horizon.values() for result in results]
    summary = aggregate_evaluations(flattened, split=split)
    summary["evaluated_sample_count"] = evaluated_samples
    summary["split_sample_count"] = len(dataset)
    summary["by_horizon_seconds"] = {
        str(horizon): aggregate_evaluations(results, split=split)
        for horizon, results in sorted(by_horizon.items())
    }
    if stop_condition is not None:
        summary["stopped_early"] = stopped_early
    return summary


def build_fitted_model(
    model_config: dict[str, Any], processed_root: Path, reporter: Any = None,
    landmark_seconds: tuple[int, ...] | None = None,
) -> BaselineModel:
    """Instantiate and fit the configured baseline from training data only.

    ``reporter`` (a :class:`adrec_still.training.progress.Reporter`) sees each real fit
    phase: counting training targets, transition counts, destination survival.

    Extension point: register further deterministic baselines here so that they
    share one fitting, evaluation, and reporting path.
    """
    from contextlib import nullcontext

    name = model_config["name"]
    from adrec_still.models.baselines.silence import SILENCE_FLOOR_BASES, SilenceAwareFloor

    def phase(label: str):
        return reporter.phase(label) if reporter is not None else nullcontext()

    if name in SILENCE_FLOOR_BASES:
        base_config = {**model_config, "name": SILENCE_FLOOR_BASES[name]}
        base_model = build_fitted_model(base_config, processed_root, reporter)
        with phase(f"fit destination survival ({name})"):
            floor = SilenceAwareFloor(
                base_model, silence_alpha=float(model_config.get("silence_alpha", 1.0)),
                **({"landmark_seconds": tuple(landmark_seconds)} if landmark_seconds else {}),
            )
            return floor.fit(processed_root / "landmark_samples.parquet")
    supported = (GlobalPop.name, UserHistoryPop.name, RepeatLast.name, FirstOrderMarkov.name)
    if name not in supported:
        raise NotImplementedError(f"no baseline harness support for model: {name}")
    catalog_path = processed_root / "poi_catalog.parquet"
    landmark_path = processed_root / "landmark_samples.parquet"
    number_of_pois = train_catalog_size(catalog_path)
    count_source = model_config.get("count_source", "train_targets")
    smoothing = float(model_config.get("smoothing", 1.0))
    with phase(f"count {count_source} ({name})"):
        if count_source == "poi_catalog":
            counts = poi_catalog_counts(catalog_path)
        else:
            counts = train_target_counts(landmark_path, split="train")
    if not counts:
        raise ValueError("the training count source produced no POI counts")
    if name == GlobalPop.name:
        model: BaselineModel = GlobalPop(
            number_of_pois=number_of_pois, smoothing=smoothing, count_source=count_source
        )
        return model.fit(counts)
    if name == UserHistoryPop.name:
        model = UserHistoryPop(
            number_of_pois=number_of_pois,
            history_weight=float(model_config.get("history_weight", 0.9)),
            max_history_length=int(model_config.get("max_history_length", 100)),
            smoothing=smoothing,
            count_source=count_source,
        )
        return model.fit(counts)
    if name == RepeatLast.name:
        model = RepeatLast(
            number_of_pois=number_of_pois,
            repeat_weight=float(model_config.get("repeat_weight", 0.5)),
            smoothing=smoothing,
            count_source=count_source,
        )
        return model.fit(counts)
    with phase(f"count train transitions ({name})"):
        transition_counts, fit_stats = train_transition_counts(landmark_path, split="train")
    model = FirstOrderMarkov(
        number_of_pois=number_of_pois,
        transition_weight=float(model_config.get("transition_weight", 0.9)),
        smoothing=smoothing,
        count_source=count_source,
    )
    return model.fit(
        MarkovTrainingSource(
            transition_counts=transition_counts,
            global_counts=counts,
            fit_stats=fit_stats,
        )
    )


def run_baseline_experiment(
    config_path: str | Path,
    splits: list[str] | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    max_batches: int | None = None,
    seeds: list[int] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Run one committed baseline experiment and write reproducible run records.

    Full runs (``max_batches is None``) are refused from a dirty worktree so
    that the recorded ``git_commit`` always reproduces the reported result;
    smoke runs are allowed but carry ``git_dirty`` in their records.
    """
    resolved = resolve_experiment_config(config_path)
    experiment = resolved["experiment"]
    dataset_config = resolved["dataset"]
    model_config = resolved["model"]
    evaluation = experiment["evaluation"]

    processed_root = Path(dataset_config["processed_root"])
    landmark_path = processed_root / "landmark_samples.parquet"
    if not landmark_path.exists():
        raise FileNotFoundError(
            f"missing canonical landmark table: {landmark_path}; run the preprocessing pipeline"
        )
    selected_splits = list(splits or evaluation["splits"])
    selected_seeds = list(seeds or experiment["seeds"])
    selected_batch_size = int(batch_size or evaluation["batch_size"])
    selected_workers = int(
        evaluation.get("num_workers", 0) if num_workers is None else num_workers
    )
    max_sequence_length = evaluation.get(
        "max_sequence_length", dataset_config["max_history_length"]
    )

    git_dirty = git_worktree_dirty()

    destination = Path(output_dir or experiment["output_dir"])
    destination.mkdir(parents=True, exist_ok=True)

    manifest_path = experiment.get("manifest")
    manifest: dict[str, Any] = {}
    if manifest_path and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    resolved_record = {
        "schema_version": RUN_RECORD_SCHEMA_VERSION,
        "baseline_interface_version": BASELINE_INTERFACE_VERSION,
        "experiment": experiment,
        "dataset": {
            "name": dataset_config["name"],
            "protocol_version": dataset_config["protocol_version"],
            "processed_root": str(processed_root),
        },
        "model_config": model_config,
        "config_paths": resolved["paths"],
        "git_commit": git_commit_sha(),
        "git_dirty": git_dirty,
        "split_manifest_sha256": manifest.get("split_manifest_sha256"),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "evaluation": {
            "splits": selected_splits,
            "seeds": selected_seeds,
            "batch_size": selected_batch_size,
            "num_workers": selected_workers,
            "max_sequence_length": max_sequence_length,
            "max_batches": max_batches,
        },
    }
    written: dict[str, Path] = {}
    resolved_path = destination / "resolved_config.json"
    resolved_path.write_text(
        json.dumps(resolved_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    written["resolved_config"] = resolved_path

    for seed in selected_seeds:
        set_global_seed(int(seed))
        model = build_fitted_model(model_config, processed_root)
        for split in selected_splits:
            if hasattr(model, "reset_prediction_stats"):
                model.reset_prediction_stats()
            metrics = evaluate_split(
                model,
                landmark_path,
                split=split,
                batch_size=selected_batch_size,
                max_sequence_length=max_sequence_length,
                num_workers=selected_workers,
                max_batches=max_batches,
            )
            record = {
                "schema_version": RUN_RECORD_SCHEMA_VERSION,
                "baseline_interface_version": BASELINE_INTERFACE_VERSION,
                "experiment_name": experiment["name"],
                "dataset": resolved_record["dataset"],
                "git_commit": resolved_record["git_commit"],
                "git_dirty": resolved_record["git_dirty"],
                "split_manifest_sha256": resolved_record["split_manifest_sha256"],
                "model": model.resolved_config(),
                "seed": int(seed),
                "split": split,
                "max_batches": max_batches,
                "metrics": metrics,
            }
            if hasattr(model, "prediction_stats"):
                record["prediction_stats"] = model.prediction_stats()
            output_path = destination / f"{split}_seed{int(seed)}.json"
            write_evaluation_json(record, output_path)
            written[f"{split}_seed{int(seed)}"] = output_path
    return written
