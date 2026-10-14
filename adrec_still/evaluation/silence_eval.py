"""Silence-aware evaluation: one streaming pass per split with §-style breakdowns.

For every model exposing the shared scorer surface this adds, on top of exact
full-catalog ranking: repeat/explore ranking splits (full-history raw-ID
labels), event-weighted and user-macro repeat rates (raw all-target and
known-catalog populations reported separately), repeat-probability quality for
models that forecast it, censor-aware time NLL, and the four-silence-cell
metrics with their simple comparisons. Every mean carries its count; rows a
metric cannot honestly cover are counted in a reason table, never filled.

Diffusion-based item probabilities are Monte-Carlo estimates: their sampling
configuration rides along as ``mc`` metadata and the known-item mark NLL is
reported as ``poi_nll_mc`` for those models.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from adrec_still.data.collate import collate_canonical_samples, select_batch_rows
from adrec_still.data.dataset import LandmarkDataset
from adrec_still.data.history import FullHistoryIndex
from adrec_still.evaluation.cells import (
    CellLabels,
    CellMetricAccumulator,
    CellPrior,
    LogisticCellClassifier,
    cell_labels,
    elapsed_features,
    history_elapsed_features,
)
from adrec_still.evaluation.exact import exact_ranking_contributions
from adrec_still.evaluation.scoring import INFERENCE_FIELDS, exact_model_candidate_factory
from adrec_still.models.adrec_still.memory import memory_labels
from adrec_still.time.protocol import residual_time_distribution_metrics
from adrec_still.training.evaluate import EvaluationOptions

RANKING_KEYS = tuple(
    f"{metric}@{k}" for metric in ("recall", "ndcg", "mrr") for k in (5, 10, 20)
) + ("poi_nll",)


class _Bucket:
    def __init__(self) -> None:
        self.count = 0
        self.sums = dict.fromkeys(RANKING_KEYS, 0.0)

    def add(self, values: dict[str, torch.Tensor], rows: torch.Tensor) -> None:
        kept = int(rows.sum())
        if not kept:
            return
        self.count += kept
        for key in RANKING_KEYS:
            self.sums[key] += float(values[key][rows].sum())

    def result(self) -> dict[str, Any]:
        out: dict[str, Any] = {"count": self.count}
        for key in RANKING_KEYS:
            out[key] = self.sums[key] / self.count if self.count else None
        return out


class _ProbabilityQuality:
    """Log loss / Brier / 10-bin reliability for a forecast probability."""

    def __init__(self) -> None:
        self.count = 0
        self.log_loss_sum = 0.0
        self.brier_sum = 0.0
        self.bin_counts = torch.zeros(10, dtype=torch.long)
        self.bin_probability = torch.zeros(10, dtype=torch.float64)
        self.bin_outcome = torch.zeros(10, dtype=torch.float64)

    def add(self, probability: torch.Tensor, outcome: torch.Tensor) -> None:
        probability = probability.to(torch.float64).clamp(1e-12, 1 - 1e-12)
        outcome = outcome.to(torch.float64)
        self.count += int(outcome.shape[0])
        self.log_loss_sum += float(-(
            outcome * probability.log() + (1 - outcome) * (1 - probability).log()
        ).sum())
        self.brier_sum += float(((probability - outcome) ** 2).sum())
        bins = (probability * 10).floor().clamp(max=9).long()
        self.bin_counts += torch.bincount(bins, minlength=10)
        self.bin_probability += torch.bincount(bins, weights=probability, minlength=10)
        self.bin_outcome += torch.bincount(bins, weights=outcome, minlength=10)

    def result(self) -> dict[str, Any]:
        if not self.count:
            return {"count": 0}
        counts = self.bin_counts.to(torch.float64)
        return {
            "count": self.count,
            "log_loss": self.log_loss_sum / self.count,
            "brier": self.brier_sum / self.count,
            "reliability": [
                {
                    "bin": index,
                    "count": int(self.bin_counts[index]),
                    "mean_probability": (
                        float(self.bin_probability[index] / counts[index])
                        if self.bin_counts[index] else None
                    ),
                    "observed_rate": (
                        float(self.bin_outcome[index] / counts[index])
                        if self.bin_counts[index] else None
                    ),
                }
                for index in range(10)
            ],
        }


def fit_cell_baselines(
    index: FullHistoryIndex,
    landmark_path,
    *,
    cell_radius_km: float,
    cell_time_threshold_hours: float,
    batch_size: int = 256,
    max_rows: int = 50_000,
    seed: int = 41,
) -> dict[str, Any]:
    """Fit the prior / elapsed-only / history+elapsed comparison classifiers.

    Trained on train-split rows with honest labels (causal homes). These are
    labeled comparisons; none of them is the joint model's own readout.
    """
    dataset = LandmarkDataset(landmark_path, split="train")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_canonical_samples)
    threshold_s = cell_time_threshold_hours * 3600.0
    feature_rows: list[torch.Tensor] = []
    rich_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    for batch in loader:
        labels = cell_labels(
            index, batch, cell_radius_km=cell_radius_km,
            cell_time_threshold_hours=cell_time_threshold_hours, home_policy="causal",
        )
        if not bool(labels.available.any()):
            continue
        kept = select_batch_rows(batch, labels.available)
        feature_rows.append(elapsed_features(kept["query_s_sec"], threshold_s))
        rich_rows.append(history_elapsed_features(kept, threshold_s))
        label_rows.append(labels.cell[labels.available])
        if sum(int(block.shape[0]) for block in label_rows) >= max_rows:
            break
    if not label_rows:
        return {}
    labels = torch.cat(label_rows)[:max_rows]
    elapsed = torch.cat(feature_rows)[:max_rows]
    rich = torch.cat(rich_rows)[:max_rows]
    counts = torch.bincount(labels, minlength=4)
    return {
        "cell_prior": CellPrior(counts),
        "elapsed_only": LogisticCellClassifier("elapsed_only", elapsed.shape[1]).fit(
            elapsed, labels, seed=seed,
        ),
        "history_elapsed": LogisticCellClassifier(
            "history_elapsed", rich.shape[1]
        ).fit(rich, labels, seed=seed),
        "training_rows": int(labels.shape[0]),
    }


def evaluate_silence_split(
    model,
    landmark_path,
    split: str,
    *,
    index: FullHistoryIndex,
    options: EvaluationOptions | None = None,
    cell_radius_km: float = 10.0,
    cell_time_threshold_hours: float = 24.0,
    home_policy: str = "frozen",
    horizons_seconds: list[int] | None = None,
    cell_baselines: dict[str, Any] | None = None,
    max_batches: int | None = None,
    time_dense_catalog_limit: int = 200_000,
    progress=None,
) -> dict[str, Any]:
    options = options or EvaluationOptions()
    dataset = LandmarkDataset(landmark_path, split=split)
    if not len(dataset):
        raise ValueError(f"split {split} has no rows")
    loader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False,
                        num_workers=options.num_workers,
                        collate_fn=collate_canonical_samples)
    started = time.perf_counter()
    threshold_s = cell_time_threshold_hours * 3600.0
    horizon_filter = {int(h) for h in horizons_seconds} if horizons_seconds else None

    has_forecast = callable(getattr(model, "prepare_forecast", None))
    factory = None if has_forecast else exact_model_candidate_factory(
        model, chunk_size=options.chunk_size
    )

    buckets = {"overall": _Bucket(), "repeat": _Bucket(), "explore": _Bucket(),
               "unlabeled": _Bucket()}
    by_horizon: dict[int, _Bucket] = {}
    repeat_counts = {"observed_all": 0, "repeat_all": 0,
                     "observed_known": 0, "repeat_known": 0}
    user_rates: dict[int, list[int]] = {}
    repeat_quality = {"all": _ProbabilityQuality(), "catalog": _ProbabilityQuality()}
    time_sums = {"observed_nll": 0.0, "observed_count": 0,
                 "censored_nll": 0.0, "censored_count": 0}
    time_supported = True
    time_reason: str | None = None
    cell_metrics = CellMetricAccumulator()
    baseline_metrics = {
        name: CellMetricAccumulator()
        for name in (cell_baselines or {}) if name != "training_rows"
    }
    mc_metadata: dict[str, Any] | None = None
    batches = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if horizon_filter is not None:
            rows = torch.tensor(
                [int(value) in horizon_filter for value in batch["query_s_sec"].tolist()]
            )
            if not bool(rows.any()):
                continue
            batch = select_batch_rows(batch, rows)
        batches += 1
        features = {
            key: (batch[key].clone() if isinstance(batch[key], torch.Tensor)
                  else list(batch[key]))
            for key in INFERENCE_FIELDS if key in batch
        }
        extras: dict[str, Any] = {}
        time_distribution = None
        cell_probabilities = None
        cell_available = torch.zeros(len(batch["gap_id"]), dtype=torch.bool)
        unresolved_mass = None
        if has_forecast:
            forecast = model.prepare_forecast(features, chunk_size=options.chunk_size)
            scorer = forecast.score
            extras = forecast.extras
            time_distribution = forecast.time_distribution
            readout = forecast.cell_readout(cell_radius_km=cell_radius_km)
            cell_probabilities = readout["probabilities"]
            cell_available = readout["available"]
            unresolved_mass = readout.get("unresolved_mass")
            if mc_metadata is None:
                mc_metadata = {
                    key: extras.get(key)
                    for key in ("estimator", "time_samples", "diffusion_samples",
                                "inference_steps", "mc_sample_count")
                }
        else:
            scorer = factory(features)
            if callable(getattr(model, "predict", None)) and time_supported:
                if model.number_of_pois <= time_dense_catalog_limit:
                    prediction = model.predict(batch)
                    time_distribution = getattr(prediction, "time_distribution", None)
                    if time_distribution is None and getattr(
                        prediction, "time_prediction", None
                    ) is not None:
                        from adrec_still.time.protocol import lognormal_time_metrics

                        block = lognormal_time_metrics(
                            prediction.time_prediction,
                            batch["residual_time_sec"], batch["censor_time_sec"],
                            batch["observed_mask"],
                        )
                        time_sums["observed_nll"] += block["observed_time_nll_sum"]
                        time_sums["observed_count"] += block["observed_count"]
                        time_sums["censored_nll"] += block["censored_time_nll_sum"]
                        time_sums["censored_count"] += block["censored_count"]
                    extras = getattr(prediction, "extras", None) or {}
                else:
                    time_supported = False
                    time_reason = (
                        "the model exposes time only through dense predict(), and the "
                        f"catalog ({model.number_of_pois}) exceeds the dense limit"
                    )

        score_kind = getattr(model, "exact_score_kind", "logits")
        contributions = exact_ranking_contributions(
            scorer, batch["next_poi"], batch["target_in_train_catalog"],
            batch["observed_mask"], model.number_of_pois,
            score_kind=score_kind, chunk_size=options.chunk_size,
        )
        values = contributions["values"]
        eligible = contributions["eligible_mask"]

        labels = memory_labels(index, batch)
        repeat = labels.repeat_label
        observed = batch["observed_mask"].bool()
        known = batch["target_in_train_catalog"].bool()
        labeled = repeat >= 0

        buckets["overall"].add(values, eligible)
        buckets["repeat"].add(values, eligible & (repeat == 1))
        buckets["explore"].add(values, eligible & (repeat == 0))
        buckets["unlabeled"].add(values, eligible & ~labeled)
        for horizon in torch.unique(batch["query_s_sec"]).tolist():
            bucket = by_horizon.setdefault(int(horizon), _Bucket())
            bucket.add(values, eligible & batch["query_s_sec"].eq(horizon))

        observed_labeled = observed & labeled
        repeat_counts["observed_all"] += int(observed_labeled.sum())
        repeat_counts["repeat_all"] += int((observed_labeled & (repeat == 1)).sum())
        observed_known_labeled = observed_labeled & known
        repeat_counts["observed_known"] += int(observed_known_labeled.sum())
        repeat_counts["repeat_known"] += int(
            (observed_known_labeled & (repeat == 1)).sum()
        )
        for user, flag, is_observed in zip(batch["user_id"].tolist(),
                                           repeat.tolist(),
                                           observed_labeled.tolist()):
            if is_observed:
                user_rates.setdefault(int(user), []).append(int(flag == 1))

        if "repeat_probability_all" in extras:
            rows = observed_labeled
            if bool(rows.any()):
                repeat_quality["all"].add(
                    extras["repeat_probability_all"][rows], (repeat[rows] == 1)
                )
        if "repeat_probability_catalog" in extras:
            rows = observed_known_labeled
            if bool(rows.any()):
                repeat_quality["catalog"].add(
                    extras["repeat_probability_catalog"][rows], (repeat[rows] == 1)
                )

        if time_distribution is not None:
            block = residual_time_distribution_metrics(
                time_distribution, batch["residual_time_sec"],
                batch["censor_time_sec"], batch["observed_mask"],
            )
            time_sums["observed_nll"] += block["observed_time_nll_sum"]
            time_sums["observed_count"] += block["observed_count"]
            time_sums["censored_nll"] += block["censored_time_nll_sum"]
            time_sums["censored_count"] += block["censored_count"]

        label_block: CellLabels = cell_labels(
            index, batch, cell_radius_km=cell_radius_km,
            cell_time_threshold_hours=cell_time_threshold_hours,
            home_policy=home_policy,
        )
        if cell_probabilities is not None:
            cell_metrics.add(cell_probabilities, cell_available, label_block,
                             batch["query_s_sec"], threshold_s,
                             unresolved_mass=unresolved_mass)
        for name, accumulator in baseline_metrics.items():
            classifier = (cell_baselines or {}).get(name)
            if classifier is None:
                continue
            baseline_features = (
                elapsed_features(batch["query_s_sec"], threshold_s)
                if name in ("cell_prior", "elapsed_only")
                else history_elapsed_features(batch, threshold_s)
            )
            probabilities = classifier.predict(baseline_features)
            accumulator.add(probabilities,
                            torch.ones(len(batch["gap_id"]), dtype=torch.bool),
                            label_block, batch["query_s_sec"], threshold_s)
        if progress is not None:
            progress(batch_index + 1, len(loader))

    if not batches:
        raise ValueError(f"no rows matched the requested horizons in split {split}")

    macro = [sum(flags) / len(flags) for flags in user_rates.values() if flags]
    result: dict[str, Any] = {
        "split": split,
        "ranking": {name: bucket.result() for name, bucket in buckets.items()},
        "ranking_by_horizon_seconds": {
            str(horizon): bucket.result()
            for horizon, bucket in sorted(by_horizon.items())
        },
        "repeat": {
            "event_rate_all_targets": (
                repeat_counts["repeat_all"] / repeat_counts["observed_all"]
                if repeat_counts["observed_all"] else None
            ),
            "event_rate_known_catalog": (
                repeat_counts["repeat_known"] / repeat_counts["observed_known"]
                if repeat_counts["observed_known"] else None
            ),
            "user_macro_rate": sum(macro) / len(macro) if macro else None,
            "user_count": len(macro),
            **repeat_counts,
            "probability_all": repeat_quality["all"].result(),
            "probability_catalog": repeat_quality["catalog"].result(),
        },
        "time": (
            {
                "observed_time_nll": (
                    time_sums["observed_nll"] / time_sums["observed_count"]
                    if time_sums["observed_count"] else None
                ),
                "observed_count": time_sums["observed_count"],
                "censored_time_nll": (
                    time_sums["censored_nll"] / time_sums["censored_count"]
                    if time_sums["censored_count"] else None
                ),
                "censored_count": time_sums["censored_count"],
            }
            if time_sums["observed_count"] or time_sums["censored_count"]
            else {"available": False,
                  "reason": time_reason or "the model exposes no time distribution"}
        ),
        "cells": {
            "config": {
                "cell_radius_km": cell_radius_km,
                "cell_time_threshold_hours": cell_time_threshold_hours,
                "home_policy": home_policy,
                "calibration": "engineering starting point; not a confirmed setting",
            },
            "model": cell_metrics.result(),
            "baselines": {name: accumulator.result()
                          for name, accumulator in baseline_metrics.items()},
        },
        "seconds": time.perf_counter() - started,
    }
    if mc_metadata:
        result["mc"] = mc_metadata
        overall = result["ranking"]["overall"]
        overall["poi_nll_mc"] = overall.pop("poi_nll")
    return result
