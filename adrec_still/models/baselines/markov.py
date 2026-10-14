"""FirstOrderMarkov: training-only last-POI transition baseline (B1).

Transition counts ``P(next_poi = j | last_poi = i)`` are fitted from completed
**training** gaps only, and each gap contributes exactly one transition: only
the ``query_s_sec == 0`` landmark of a gap is read, mirroring GlobalPop's
``train_targets`` count source, so a gap is never re-counted once per eligible
horizon. The transition state is the *last observed* POI of that landmark's
history: unlike RepeatLast's last-known lookup, a trailing ``UNKNOWN_POI_ID``
is never skipped in favor of an older known entry, so such a gap forms no
transition.

At prediction time a row whose last observed POI has fitted transitions
receives the committed interpolation smoothing

``p = transition_weight * transition_row + (1 - transition_weight) * global_pop``

while a row whose last observed POI is unknown (Issue #14's
``unknown last POI -> GlobalPop`` contract), missing, or an unseen transition
state falls back to the pure GlobalPop distribution. Both the fallback rate
and the transition-support rate are reported per split.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from adrec_still.data.schema import PADDING_POI_ID, UNKNOWN_POI_ID
from adrec_still.models.baselines.base import (
    BaselineModel,
    BaselinePrediction,
    masked_log_probabilities,
    validate_batch,
)
from adrec_still.models.baselines.history import last_observed_poi
from adrec_still.models.baselines.popularity import (
    BASE_LANDMARK_SECONDS,
    DEFAULT_SMOOTHING,
    GlobalPop,
)

DEFAULT_TRANSITION_WEIGHT = 0.9

_LOG_FLOOR = 1.0e-30

RESERVED_POI_IDS = (PADDING_POI_ID, UNKNOWN_POI_ID)


def _last_observed(history: list[int]) -> int:
    """Return the last observed entry, which may be ``UNKNOWN_POI_ID``."""
    for poi_id in reversed(history):
        if poi_id != PADDING_POI_ID:
            return int(poi_id)
    return PADDING_POI_ID


def train_transition_counts(
    landmark_path: str | Path,
    split: str = "train",
    base_landmark_seconds: int = BASE_LANDMARK_SECONDS,
    batch_size: int = 50_000,
) -> tuple[dict[tuple[int, int], int], dict[str, int]]:
    """Count one last-POI transition per completed training gap.

    Only the ``query_s_sec == base_landmark_seconds`` landmark of each observed,
    in-catalog gap is read, so multiple eligible landmarks of the same gap never
    inflate a transition. The state is the gap's last *observed* POI: a trailing
    ``UNKNOWN_POI_ID`` forms no transition (it is counted separately) rather
    than reaching back to an older known entry. Returns the counts together
    with fit-time coverage diagnostics.
    """
    parquet = pq.ParquetFile(Path(landmark_path))
    columns = [
        "split",
        "query_s_sec",
        "event_observed",
        "target_in_train_catalog",
        "next_poi",
        "history_poi",
    ]
    counts: dict[tuple[int, int], int] = {}
    gap_count = 0
    gaps_without_observed_last = 0
    gaps_with_unknown_last = 0
    for record_batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        selected = pc.and_(
            pc.equal(record_batch.column("split"), split),
            pc.equal(record_batch.column("query_s_sec"), base_landmark_seconds),
        )
        selected = pc.and_(selected, record_batch.column("event_observed"))
        selected = pc.and_(selected, record_batch.column("target_in_train_catalog"))
        targets = pc.filter(record_batch.column("next_poi"), selected).to_pylist()
        histories = pc.filter(record_batch.column("history_poi"), selected).to_pylist()
        for target, history in zip(targets, histories, strict=True):
            gap_count += 1
            state = _last_observed(history)
            if state == PADDING_POI_ID:
                gaps_without_observed_last += 1
                continue
            if state == UNKNOWN_POI_ID:
                gaps_with_unknown_last += 1
                continue
            key = (state, int(target))
            counts[key] = counts.get(key, 0) + 1
    stats = {
        "training_gap_count": gap_count,
        "training_gaps_without_observed_last": gaps_without_observed_last,
        "training_gaps_with_unknown_last": gaps_with_unknown_last,
    }
    return counts, stats


@dataclass(frozen=True)
class MarkovTrainingSource:
    """Training-only inputs for :class:`FirstOrderMarkov`."""

    transition_counts: Mapping[tuple[int, int], int]
    global_counts: Mapping[int, int]
    fit_stats: Mapping[str, int] | None = None


class FirstOrderMarkov(BaselineModel):
    """Last-POI transition distribution with a GlobalPop fallback."""

    name = "first_order_markov"
    family = "baseline"

    def __init__(
        self,
        number_of_pois: int,
        transition_weight: float = DEFAULT_TRANSITION_WEIGHT,
        smoothing: float = DEFAULT_SMOOTHING,
        count_source: str = "train_targets",
    ) -> None:
        super().__init__(number_of_pois)
        if not 0.0 < transition_weight < 1.0:
            raise ValueError("transition_weight must be strictly between 0 and 1")
        self.transition_weight = float(transition_weight)
        self._fallback = GlobalPop(
            number_of_pois, smoothing=smoothing, count_source=count_source
        )
        self._fallback_probabilities = torch.empty(0)
        self._transitions: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._transition_entry_count = 0
        self._fit_stats: dict[str, int] = {}
        self.reset_prediction_stats()

    @property
    def context_dim(self) -> int:
        """FirstOrderMarkov exports no learned history representation."""
        return 0

    def fit(self, training_source: MarkovTrainingSource) -> FirstOrderMarkov:
        """Fit per-state transition rows and the GlobalPop fallback."""
        if not isinstance(training_source, MarkovTrainingSource):
            raise TypeError("FirstOrderMarkov.fit expects a MarkovTrainingSource")
        self._fallback.fit(training_source.global_counts)
        self._fallback_probabilities = self._fallback.log_probabilities.exp()

        by_state: dict[int, dict[int, int]] = {}
        for (state, target), count in training_source.transition_counts.items():
            state, target = int(state), int(target)
            for poi_id in (state, target):
                if poi_id in RESERVED_POI_IDS:
                    raise ValueError("reserved POI IDs cannot appear in transitions")
                if poi_id < 0 or poi_id >= self.number_of_pois:
                    raise ValueError(f"POI ID outside the training vocabulary: {poi_id}")
            if int(count) <= 0:
                raise ValueError("transition counts must be positive")
            by_state.setdefault(state, {})[target] = int(count)

        self._transitions = {}
        self._transition_entry_count = 0
        for state, row in by_state.items():
            targets = torch.tensor(sorted(row), dtype=torch.long)
            row_counts = torch.tensor(
                [row[target] for target in sorted(row)], dtype=torch.float64
            )
            conditional = (row_counts / row_counts.sum()).to(torch.float32)
            self._transitions[state] = (targets, conditional)
            self._transition_entry_count += len(row)
        self._fit_stats = dict(training_source.fit_stats or {})
        self.is_fitted = True
        return self

    def predict(self, batch: dict[str, torch.Tensor]) -> BaselinePrediction:
        """Interpolate each supported row's transition distribution with GlobalPop."""
        self.require_fitted()
        batch_size = validate_batch(batch)
        last_poi = last_observed_poi(batch["history_poi"], batch["history_mask"])
        if last_poi.numel() and int(last_poi.max()) >= self.number_of_pois:
            raise ValueError("history POI ID outside the training vocabulary")

        probabilities = (
            self._fallback_probabilities.unsqueeze(0).expand(batch_size, -1).clone()
        )
        supported_rows = 0
        for row_index in range(batch_size):
            state = int(last_poi[row_index])
            if state in RESERVED_POI_IDS or state not in self._transitions:
                continue
            supported_rows += 1
            targets, conditional = self._transitions[state]
            probabilities[row_index] *= 1.0 - self.transition_weight
            probabilities[row_index, targets] += self.transition_weight * conditional

        self._predicted_rows += batch_size
        self._supported_rows += supported_rows
        self._fallback_rows += batch_size - supported_rows
        return BaselinePrediction(
            poi_logits=masked_log_probabilities(
                torch.log(probabilities.clamp_min(_LOG_FLOOR))
            ),
            context=torch.zeros((batch_size, self.context_dim), dtype=torch.float32),
        )

    def reset_prediction_stats(self) -> None:
        """Zero the per-split fallback and support counters."""
        self._predicted_rows = 0
        self._fallback_rows = 0
        self._supported_rows = 0

    def prediction_stats(self) -> dict[str, Any]:
        """Report transition support and GlobalPop fallback frequency."""
        return {
            "predicted_row_count": self._predicted_rows,
            "supported_row_count": self._supported_rows,
            "fallback_row_count": self._fallback_rows,
            "transition_support_rate": (
                self._supported_rows / self._predicted_rows if self._predicted_rows else None
            ),
            "fallback_rate": (
                self._fallback_rows / self._predicted_rows if self._predicted_rows else None
            ),
        }

    def parameter_count(self) -> int:
        """Stored transition entries plus the fallback catalog scores."""
        return self._transition_entry_count + self._fallback.parameter_count()

    def resolved_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "interface_version": self.interface_version,
            "number_of_pois": self.number_of_pois,
            "transition_weight": self.transition_weight,
            "smoothing_policy": "interpolation with the GlobalPop fallback",
            "fitted_state_count": len(self._transitions),
            "fitted_transition_count": self._transition_entry_count,
            "fit_stats": self._fit_stats,
            "fallback": self._fallback.resolved_config(),
            "context_dim": self.context_dim,
            "parameter_count": self.parameter_count(),
        }
