"""UserHistoryPop: rank POIs by their frequency inside the query history (B1).

For every query the model mixes two normalized distributions estimated without
any validation or test information:

``p = history_weight * history_frequency + (1 - history_weight) * global_pop``

``history_frequency`` is the per-row relative frequency of each known POI in
the query history; ``global_pop`` is the shared GlobalPop fallback fitted on
training counts. Because one in-history occurrence contributes at least
``history_weight / max_history_length`` probability while any out-of-history
POI keeps at most ``(1 - history_weight) * max(global_pop)``, the fitted guard
below makes the ranking provably follow in-history frequency first, with ties
broken by the GlobalPop score. Rows without any known historical POI fall back
to the pure GlobalPop distribution and are counted in the fallback rate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from adrec_still.models.baselines.base import (
    BaselineModel,
    BaselinePrediction,
    masked_log_probabilities,
    validate_batch,
)
from adrec_still.models.baselines.history import known_history_mask
from adrec_still.models.baselines.popularity import DEFAULT_SMOOTHING, GlobalPop

DEFAULT_HISTORY_WEIGHT = 0.9
DEFAULT_MAX_HISTORY_LENGTH = 100

#: Floor applied before the elementwise log so reserved IDs stay finite; the
#: shared masking then pushes them to exactly zero probability.
_LOG_FLOOR = 1.0e-30


class UserHistoryPop(BaselineModel):
    """Personalized in-history frequency ranking with a GlobalPop fallback."""

    name = "user_history_pop"
    family = "baseline"

    def __init__(
        self,
        number_of_pois: int,
        history_weight: float = DEFAULT_HISTORY_WEIGHT,
        max_history_length: int = DEFAULT_MAX_HISTORY_LENGTH,
        smoothing: float = DEFAULT_SMOOTHING,
        count_source: str = "train_targets",
    ) -> None:
        super().__init__(number_of_pois)
        if not 0.0 < history_weight < 1.0:
            raise ValueError("history_weight must be strictly between 0 and 1")
        if max_history_length <= 0:
            raise ValueError("max_history_length must be positive")
        self.history_weight = float(history_weight)
        self.max_history_length = int(max_history_length)
        self._fallback = GlobalPop(
            number_of_pois, smoothing=smoothing, count_source=count_source
        )
        self._fallback_probabilities = torch.empty(0)
        self.reset_prediction_stats()

    @property
    def context_dim(self) -> int:
        """UserHistoryPop exports no learned history representation."""
        return 0

    def fit(self, training_source: Mapping[int, int]) -> UserHistoryPop:
        """Fit the GlobalPop fallback and verify the frequency-first guard."""
        self._fallback.fit(training_source)
        self._fallback_probabilities = self._fallback.log_probabilities.exp()
        smallest_history_step = self.history_weight / self.max_history_length
        largest_global_step = (1.0 - self.history_weight) * float(
            self._fallback_probabilities.max()
        )
        if smallest_history_step <= largest_global_step:
            raise ValueError(
                "history_weight is too small for max_history_length: an unvisited "
                "popular POI could outrank a visited one; raise history_weight or "
                "lower max_history_length"
            )
        self.is_fitted = True
        return self

    def predict(self, batch: dict[str, torch.Tensor]) -> BaselinePrediction:
        """Mix per-row history frequencies with the GlobalPop fallback."""
        self.require_fitted()
        batch_size = validate_batch(batch)
        history_poi = batch["history_poi"]
        if history_poi.shape[1] > self.max_history_length:
            raise ValueError(
                "batch history is longer than the configured max_history_length; "
                "the frequency-first guard would no longer hold"
            )
        if history_poi.numel() and int(history_poi.max()) >= self.number_of_pois:
            raise ValueError("history POI ID outside the training vocabulary")
        valid = known_history_mask(history_poi, batch["history_mask"])
        totals = valid.sum(dim=1)
        has_history = totals.gt(0)

        probabilities = (
            self._fallback_probabilities.unsqueeze(0).expand(batch_size, -1).clone()
        )
        probabilities[has_history] *= 1.0 - self.history_weight
        per_entry = torch.where(
            valid & has_history.unsqueeze(1),
            self.history_weight / totals.clamp_min(1).to(probabilities.dtype).unsqueeze(1),
            torch.zeros((), dtype=probabilities.dtype),
        )
        probabilities.scatter_add_(1, history_poi.clamp_min(0), per_entry)

        self._predicted_rows += batch_size
        self._fallback_rows += int((~has_history).sum())
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
        """Stored fallback catalog scores; the history counts are per-query."""
        return self._fallback.parameter_count()

    def resolved_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "interface_version": self.interface_version,
            "number_of_pois": self.number_of_pois,
            "history_weight": self.history_weight,
            "max_history_length": self.max_history_length,
            "tie_breaking": "in-history frequency, then GlobalPop score",
            "fallback": self._fallback.resolved_config(),
            "context_dim": self.context_dim,
            "parameter_count": self.parameter_count(),
        }
