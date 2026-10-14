"""Four silence cells: observable outcome labels, metrics, and simple baselines.

The two axes keep the proposal's definitions:

- **local / displaced** — whether the *next* POI lies within ``cell_radius_km``
  of the user's activity center ("home": most-visited geocoded known POI of the
  causal prefix in training, of the train-cutoff-frozen prefix in evaluation);
- **early / late** — whether the *total* gap ``T`` exceeds the time threshold
  (never the residual ``R``).

Cell names: ``local_early``, ``displaced_early``, ``local_late``,
``displaced_late``. Repeat/explore is a different axis and is cross-tabulated,
never substituted. These are observable outcome labels: nothing here claims to
identify why a user was silent. Rows without geo support, home, an observed
known target, or target coordinates get no label — they are counted, not filled.

``cell_radius_km=10`` and ``cell_time_threshold_hours=24`` are engineering
starting points awaiting research calibration (docs/MODELING.md); they are
configuration, not confirmed settings, and must never be tuned on test labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from adrec_still.data.history import FullHistoryIndex
from adrec_still.models.adrec_still.forecast import CELL_NAMES, haversine_km

DEFAULT_CELL_RADIUS_KM = 10.0
DEFAULT_CELL_TIME_THRESHOLD_HOURS = 24.0


@dataclass
class CellLabels:
    """Per-row labels; ``available`` marks rows with a complete, honest label."""

    cell: torch.Tensor        # int64 [B] in [0, 4); meaningless where unavailable
    available: torch.Tensor   # bool [B]
    early: torch.Tensor       # bool [B] (T <= threshold), meaningless if no T
    reasons: dict[str, int]   # why rows were excluded, by count


def cell_labels(
    index: FullHistoryIndex,
    batch: dict[str, Any],
    *,
    cell_radius_km: float,
    cell_time_threshold_hours: float,
    home_policy: str = "frozen",
) -> CellLabels:
    gap_ids = batch["gap_id"]
    observed = batch["observed_mask"].bool().cpu()
    known = batch["target_in_train_catalog"].bool().cpu()
    rows = len(gap_ids)
    cell = torch.zeros(rows, dtype=torch.long)
    available = torch.zeros(rows, dtype=torch.bool)
    early = torch.zeros(rows, dtype=torch.bool)
    threshold_s = cell_time_threshold_hours * 3600.0
    reasons = {"censored": 0, "no_gap_id": 0, "oov_target": 0, "no_home": 0,
               "target_not_geocoded": 0, "labeled": 0}
    for row in range(rows):
        if gap_ids[row] is None:
            reasons["no_gap_id"] += 1
            continue
        if not bool(observed[row]):
            reasons["censored"] += 1
            continue
        duration = index.gap_duration_s(gap_ids[row])
        if duration is None:
            reasons["censored"] += 1
            continue
        early_label = duration <= threshold_s
        early[row] = early_label
        if not bool(known[row]):
            reasons["oov_target"] += 1
            continue
        home = index.home(gap_ids[row], policy=home_policy)
        if home is None and home_policy == "frozen":
            home = index.home(gap_ids[row], policy="causal")
        if home is None:
            reasons["no_home"] += 1
            continue
        target_poi = index.target_poi(gap_ids[row])
        coordinates = index.poi_coordinates(int(target_poi)) if target_poi else None
        if coordinates is None:
            reasons["target_not_geocoded"] += 1
            continue
        distance = haversine_km(
            torch.tensor([home[0]]), torch.tensor([home[1]]),
            torch.tensor([coordinates[0]]), torch.tensor([coordinates[1]]),
        )[0]
        local = bool(distance <= cell_radius_km)
        cell[row] = (0 if local else 1) + (0 if early_label else 2)
        available[row] = True
        reasons["labeled"] += 1
    return CellLabels(cell=cell, available=available, early=early, reasons=reasons)


class CellMetricAccumulator:
    """Count-weighted cell metrics, split by s < threshold vs s >= threshold.

    A query already past the threshold has its early cells pinned at zero by
    construction; its (easy) accuracy is reported separately and never mixed
    into the nontrivial-population headline.
    """

    def __init__(self) -> None:
        self.populations = {
            "nontrivial": _CellSums(),   # s < threshold: early cells still possible
            "late_determined": _CellSums(),
        }
        self.reasons: dict[str, int] = {}
        self.unresolved_mass_sum = 0.0
        self.unresolved_rows = 0

    def add(self, probabilities: torch.Tensor, prediction_available: torch.Tensor,
            labels: CellLabels, query_s_sec: torch.Tensor,
            threshold_s: float, unresolved_mass: torch.Tensor | None = None) -> None:
        for key, count in labels.reasons.items():
            self.reasons[key] = self.reasons.get(key, 0) + count
        usable = labels.available & prediction_available.cpu()
        if unresolved_mass is not None:
            self.unresolved_mass_sum += float(unresolved_mass[usable].sum())
            self.unresolved_rows += int(usable.sum())
        nontrivial = query_s_sec.cpu().to(torch.float64) < threshold_s
        for name, member in (("nontrivial", nontrivial), ("late_determined", ~nontrivial)):
            rows = usable & member
            if not bool(rows.any()):
                continue
            self.populations[name].add(probabilities[rows], labels.cell[rows])

    def result(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "cells": list(CELL_NAMES),
            "label_reasons": dict(sorted(self.reasons.items())),
            "mean_unresolved_spatial_mass": (
                self.unresolved_mass_sum / self.unresolved_rows
                if self.unresolved_rows else None
            ),
        }
        for name, sums in self.populations.items():
            out[name] = sums.result()
        return out


class _CellSums:
    def __init__(self) -> None:
        self.count = 0
        self.nll_sum = 0.0
        self.brier_sum = 0.0
        self.confusion = torch.zeros((4, 4), dtype=torch.long)

    def add(self, probabilities: torch.Tensor, labels: torch.Tensor) -> None:
        probabilities = probabilities.to(torch.float64)
        count = int(labels.shape[0])
        self.count += count
        picked = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
        self.nll_sum += float(-picked.clamp_min(1e-300).log().sum())
        one_hot = torch.zeros_like(probabilities)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        self.brier_sum += float(((probabilities - one_hot) ** 2).sum())
        predicted = probabilities.argmax(dim=1)
        for true_label, predicted_label in zip(labels.tolist(), predicted.tolist()):
            self.confusion[true_label, predicted_label] += 1

    def result(self) -> dict[str, Any]:
        if not self.count:
            return {"count": 0}
        confusion = self.confusion
        per_class_f1 = []
        for cell in range(4):
            true_positive = float(confusion[cell, cell])
            precision_denominator = float(confusion[:, cell].sum())
            recall_denominator = float(confusion[cell, :].sum())
            if precision_denominator == 0 or recall_denominator == 0:
                per_class_f1.append(0.0 if recall_denominator else None)
                continue
            precision = true_positive / precision_denominator
            recall = true_positive / recall_denominator
            per_class_f1.append(
                0.0 if precision + recall == 0
                else 2 * precision * recall / (precision + recall)
            )
        present = [value for value in per_class_f1 if value is not None]
        return {
            "count": self.count,
            "nll": self.nll_sum / self.count,
            "brier": self.brier_sum / self.count,
            "accuracy": float(confusion.diagonal().sum()) / self.count,
            "macro_f1": sum(present) / len(present) if present else None,
            "per_class_f1": per_class_f1,
            "confusion_matrix": confusion.tolist(),
        }


# ------------------------------------------------------------- simple baselines


class CellPrior:
    """Train-frequency prior over the four cells (comparison, not the model)."""

    name = "cell_prior"

    def __init__(self, counts: torch.Tensor) -> None:
        self.probabilities = (counts.to(torch.float64) + 1.0) / float(counts.sum() + 4)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return self.probabilities.unsqueeze(0).expand(features.shape[0], 4)


class LogisticCellClassifier:
    """Tiny multinomial logistic baseline over explicit features."""

    def __init__(self, name: str, feature_dim: int) -> None:
        self.name = name
        self.linear = torch.nn.Linear(feature_dim, 4).double()

    def fit(self, features: torch.Tensor, labels: torch.Tensor, *, epochs: int = 200,
            learning_rate: float = 0.1, seed: int = 41) -> LogisticCellClassifier:
        torch.manual_seed(seed)
        optimizer = torch.optim.Adam(self.linear.parameters(), lr=learning_rate)
        features = features.to(torch.float64)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(self.linear(features), labels)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.softmax(self.linear(features.to(torch.float64)), dim=1)


def elapsed_features(query_s_sec: torch.Tensor, threshold_s: float) -> torch.Tensor:
    elapsed = query_s_sec.to(torch.float64)
    return torch.stack(
        [torch.log1p(elapsed / 3600.0), (elapsed >= threshold_s).to(torch.float64)],
        dim=1,
    )


def history_elapsed_features(batch: dict[str, Any], threshold_s: float) -> torch.Tensor:
    elapsed_part = elapsed_features(batch["query_s_sec"], threshold_s)
    lengths = batch["history_mask"].to(torch.float64).sum(dim=1, keepdim=True)
    anchor = batch["last_timestamp"].to(torch.float64)
    day = (anchor % 86400.0) / 86400.0 * 2 * math.pi
    clock = torch.stack([day.sin(), day.cos()], dim=1)
    return torch.cat([elapsed_part, torch.log1p(lengths), clock], dim=1)
