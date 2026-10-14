"""GlobalPop: the non-personalized training-frequency floor (baseline B0).

GlobalPop assigns every query the same next-POI distribution, estimated from
training data only. Two count sources are supported:

``train_targets`` (default)
    Observed, in-catalog next-POI targets of completed **training** gaps. One
    gap contributes exactly one count because only the ``query_s_sec == 0``
    landmark is read; every other landmark of the same gap repeats the same
    target and would otherwise weight a POI by how many horizons its gap
    happens to be eligible for.

``poi_catalog``
    The ``train_count`` column of the canonical POI catalog, i.e. raw
    training check-in frequency rather than next-event frequency.

Both sources are derived from the frozen training split, so neither leaks
validation or test information.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
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

COUNT_SOURCES = ("train_targets", "poi_catalog")
DEFAULT_SMOOTHING = 1.0
BASE_LANDMARK_SECONDS = 0


def train_target_counts(
    landmark_path: str | Path,
    split: str = "train",
    base_landmark_seconds: int = BASE_LANDMARK_SECONDS,
    batch_size: int = 200_000,
) -> dict[int, int]:
    """Count observed in-catalog next-POI targets once per completed training gap."""
    parquet = pq.ParquetFile(Path(landmark_path))
    columns = ["split", "query_s_sec", "event_observed", "target_in_train_catalog", "next_poi"]
    counts: dict[int, int] = {}
    for record_batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        selected = pc.and_(
            pc.equal(record_batch.column("split"), split),
            pc.equal(record_batch.column("query_s_sec"), base_landmark_seconds),
        )
        selected = pc.and_(selected, record_batch.column("event_observed"))
        selected = pc.and_(selected, record_batch.column("target_in_train_catalog"))
        targets = pc.filter(record_batch.column("next_poi"), selected).to_numpy(
            zero_copy_only=False
        )
        if not len(targets):
            continue
        unique, occurrences = np.unique(targets.astype(np.int64), return_counts=True)
        for poi_id, occurrence in zip(unique.tolist(), occurrences.tolist(), strict=True):
            counts[int(poi_id)] = counts.get(int(poi_id), 0) + int(occurrence)
    return counts


def poi_catalog_counts(catalog_path: str | Path) -> dict[int, int]:
    """Read training check-in frequency from the canonical POI catalog."""
    table = pq.read_table(Path(catalog_path), columns=["poi_id", "train_count"])
    poi_ids = table.column("poi_id").to_pylist()
    train_counts = table.column("train_count").to_pylist()
    return {
        int(poi_id): int(count)
        for poi_id, count in zip(poi_ids, train_counts, strict=True)
        if int(count) > 0
    }


def train_catalog_size(catalog_path: str | Path) -> int:
    """Return the POI vocabulary width, including the two reserved IDs."""
    table = pq.read_table(Path(catalog_path), columns=["poi_id"])
    if not table.num_rows:
        raise ValueError("the POI catalog is empty")
    return int(pc.max(table.column("poi_id")).as_py()) + 1


class GlobalPop(BaselineModel):
    """Non-personalized frequency floor over the frozen training POI catalog."""

    name = "global_pop"
    family = "baseline"

    def __init__(
        self,
        number_of_pois: int,
        smoothing: float = DEFAULT_SMOOTHING,
        count_source: str = "train_targets",
    ) -> None:
        super().__init__(number_of_pois)
        if smoothing <= 0:
            raise ValueError("smoothing must be positive so unseen catalog POIs stay finite")
        if count_source not in COUNT_SOURCES:
            raise ValueError(f"count_source must be one of {COUNT_SOURCES}")
        self.smoothing = float(smoothing)
        self.count_source = count_source
        self._log_probabilities = torch.empty((1, 0))
        self._fitted_poi_count = 0
        self._total_count = 0

    @property
    def context_dim(self) -> int:
        """GlobalPop exports no history representation."""
        return 0

    @property
    def log_probabilities(self) -> torch.Tensor:
        """Fitted ``[number_of_pois]`` log-probability vector."""
        self.require_fitted()
        return self._log_probabilities[0]

    def fit(self, training_source: Mapping[int, int]) -> GlobalPop:
        """Fit from a POI-ID to training-count mapping."""
        if not isinstance(training_source, Mapping):
            raise TypeError("GlobalPop.fit expects a mapping of POI ID to training count")
        scores = torch.full((1, self.number_of_pois), self.smoothing, dtype=torch.float64)
        total = 0
        for poi_id, count in training_source.items():
            poi_id = int(poi_id)
            count = int(count)
            if poi_id in (PADDING_POI_ID, UNKNOWN_POI_ID):
                raise ValueError("reserved POI IDs cannot carry training counts")
            if poi_id < 0 or poi_id >= self.number_of_pois:
                raise ValueError(f"POI ID outside the training vocabulary: {poi_id}")
            if count < 0:
                raise ValueError("training counts must be non-negative")
            scores[0, poi_id] += count
            total += count
        self._log_probabilities = masked_log_probabilities(torch.log(scores)).to(torch.float32)
        self._fitted_poi_count = len(training_source)
        self._total_count = total
        self.is_fitted = True
        return self

    def predict(self, batch: dict[str, torch.Tensor]) -> BaselinePrediction:
        """Broadcast the fitted distribution to every row of the batch."""
        self.require_fitted()
        batch_size = validate_batch(batch)
        return BaselinePrediction(
            poi_logits=self._log_probabilities.expand(batch_size, self.number_of_pois),
            context=torch.zeros((batch_size, self.context_dim), dtype=torch.float32),
        )

    def parameter_count(self) -> int:
        """Number of stored catalog scores; these are counts, not gradient parameters."""
        return max(self.number_of_pois - 2, 0)

    def resolved_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "interface_version": self.interface_version,
            "number_of_pois": self.number_of_pois,
            "smoothing": self.smoothing,
            "count_source": self.count_source,
            "context_dim": self.context_dim,
            "fitted_poi_count": self._fitted_poi_count,
            "total_training_count": self._total_count,
            "parameter_count": self.parameter_count(),
        }
