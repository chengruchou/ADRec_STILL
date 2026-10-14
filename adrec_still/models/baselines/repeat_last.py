"""RepeatLast: rank the most recent known historical POI first (B1).

For every query with a known last POI the model emits

``p = repeat_weight * onehot(last_poi) + (1 - repeat_weight) * global_pop``

which is a normalized distribution whose top-1 is always the last known POI
(``repeat_weight >= 0.5`` strictly dominates any single GlobalPop probability)
while the remaining ranks follow the GlobalPop fallback order. Rows without a
known last POI fall back to the pure GlobalPop distribution and are counted in
the fallback rate. Padding and unknown history entries are never candidates
for the repeated POI.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from adrec_still.data.schema import PADDING_POI_ID
from adrec_still.models.baselines.base import (
    BaselineModel,
    BaselinePrediction,
    masked_log_probabilities,
    validate_batch,
)
from adrec_still.models.baselines.history import last_known_poi
from adrec_still.models.baselines.popularity import DEFAULT_SMOOTHING, GlobalPop

DEFAULT_REPEAT_WEIGHT = 0.5

_LOG_FLOOR = 1.0e-30


class RepeatLast(BaselineModel):
    """Repetition baseline with a GlobalPop fallback tail."""

    name = "repeat_last"
    family = "baseline"

    def __init__(
        self,
        number_of_pois: int,
        repeat_weight: float = DEFAULT_REPEAT_WEIGHT,
        smoothing: float = DEFAULT_SMOOTHING,
        count_source: str = "train_targets",
    ) -> None:
        super().__init__(number_of_pois)
        if not 0.5 <= repeat_weight < 1.0:
            raise ValueError(
                "repeat_weight must be in [0.5, 1) so the last POI always ranks first"
            )
        self.repeat_weight = float(repeat_weight)
        self._fallback = GlobalPop(
            number_of_pois, smoothing=smoothing, count_source=count_source
        )
        self._fallback_probabilities = torch.empty(0)
        self.reset_prediction_stats()

    @property
    def context_dim(self) -> int:
        """RepeatLast exports no learned history representation."""
        return 0

    def fit(self, training_source: Mapping[int, int]) -> RepeatLast:
        """Fit the GlobalPop fallback from training counts only."""
        self._fallback.fit(training_source)
        self._fallback_probabilities = self._fallback.log_probabilities.exp()
        self.is_fitted = True
        return self

    def predict(self, batch: dict[str, torch.Tensor]) -> BaselinePrediction:
        """Boost each row's last known POI on top of the GlobalPop tail."""
        self.require_fitted()
        batch_size = validate_batch(batch)
        last_poi = last_known_poi(batch["history_poi"], batch["history_mask"])
        if last_poi.numel() and int(last_poi.max()) >= self.number_of_pois:
            raise ValueError("history POI ID outside the training vocabulary")
        has_last = last_poi.ne(PADDING_POI_ID)

        probabilities = (
            self._fallback_probabilities.unsqueeze(0).expand(batch_size, -1).clone()
        )
        probabilities[has_last] *= 1.0 - self.repeat_weight
        rows = has_last.nonzero(as_tuple=True)[0]
        probabilities[rows, last_poi[rows]] += self.repeat_weight

        self._predicted_rows += batch_size
        self._fallback_rows += int((~has_last).sum())
        return BaselinePrediction(
            poi_logits=masked_log_probabilities(
                torch.log(probabilities.clamp_min(_LOG_FLOOR))
            ),
            context=torch.zeros((batch_size, self.context_dim), dtype=torch.float32),
        )

    def reset_prediction_stats(self) -> None:
        """Zero the per-split fallback counters."""
        self._predicted_rows = 0
        self._fallback_rows = 0

    def prediction_stats(self) -> dict[str, Any]:
        """Report how often the model fell back to pure GlobalPop."""
        return {
            "predicted_row_count": self._predicted_rows,
            "fallback_row_count": self._fallback_rows,
            "fallback_rate": (
                self._fallback_rows / self._predicted_rows if self._predicted_rows else None
            ),
        }

    def parameter_count(self) -> int:
        """Stored fallback catalog scores; the repeated POI is per-query."""
        return self._fallback.parameter_count()

    def resolved_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "interface_version": self.interface_version,
            "number_of_pois": self.number_of_pois,
            "repeat_weight": self.repeat_weight,
            "fallback": self._fallback.resolved_config(),
            "context_dim": self.context_dim,
            "parameter_count": self.parameter_count(),
        }
