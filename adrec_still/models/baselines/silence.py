"""Empirical silence controls for the four existing deterministic floors.

For destination j, completed training gaps estimate S_j(s) = P(T > s | j),
shrunk toward the pooled completed-gap fraction with ``silence_alpha`` prior
gaps. Predictions are proportional to the original floor probability times
S_j(s). This is a destination-only empirical adjustment, not a censor-corrected
survival model or a time-distribution head. In particular, Markov-S does not
estimate a separate duration distribution for each transition state.

The original smoothed RepeatLast distribution has a GlobalPop tail, so its
silence-adjusted top destination can change without introducing arbitrary
decay or changing its original mixture weight. No universal direction of
change in repeat probability is assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from adrec_still.data.landmarks import LANDMARK_SECONDS
from adrec_still.models.baselines.base import (
    BaselineModel,
    BaselinePrediction,
    masked_log_probabilities,
)

SILENCE_FLOOR_BASES = {
    f"{name}_s": name
    for name in ("global_pop", "user_history_pop", "repeat_last", "first_order_markov")
}
PROFILE_VERSION = "still-completed-gap-destination-survival-v1"


@dataclass(frozen=True)
class DestinationSurvivalProfile:
    """Counts from one training base landmark per observed known-target gap.

    ``targets``/``durations`` (the raw completed-gap pairs) are retained only in
    ``step`` horizon-support mode, where the same estimator answers arbitrary
    non-negative queries as an empirical step survivor; at the fitted grid the
    step values coincide with ``survivor_counts`` by construction.
    """

    survivor_counts: torch.Tensor
    gap_count: int
    landmark_seconds: tuple[int, ...] = LANDMARK_SECONDS
    targets: torch.Tensor | None = None
    durations: torch.Tensor | None = None


def fit_destination_survival(
    landmark_path: str | Path,
    number_of_pois: int,
    *,
    landmark_seconds: tuple[int, ...] = LANDMARK_SECONDS,
    batch_size: int = 200_000,
    keep_durations: bool = False,
) -> DestinationSurvivalProfile:
    """Read training s=0 gaps only; a gap ending exactly at s does not survive.

    Canonical inputs provide exactly one s=0 row per gap; fitting relies on
    that existing uniqueness contract instead of retaining millions of IDs.
    Censored and out-of-catalog rows provide no destination labels and are
    excluded, not assigned fabricated targets.
    """
    columns = [
        "split", "query_s_sec", "event_observed",
        "target_in_train_catalog", "next_poi", "residual_time_sec",
    ]
    schedule = tuple(int(value) for value in landmark_seconds)
    if not schedule or schedule[0] != 0 or list(schedule) != sorted(set(schedule)):
        raise ValueError("landmark_seconds must be distinct, ascending and start at 0")
    counts = torch.zeros((len(schedule), number_of_pois), dtype=torch.int64)
    gap_count = 0
    kept_targets: list[torch.Tensor] = []
    kept_durations: list[torch.Tensor] = []
    for batch in pq.ParquetFile(Path(landmark_path)).iter_batches(
        batch_size=batch_size, columns=columns
    ):
        selected = pc.and_(
            pc.equal(batch.column("split"), "train"),
            pc.equal(batch.column("query_s_sec"), 0),
        )
        selected = pc.and_(selected, batch.column("event_observed"))
        selected = pc.and_(selected, batch.column("target_in_train_catalog"))
        eligible = batch.filter(selected)
        targets = eligible.column("next_poi").to_pylist()
        durations = eligible.column("residual_time_sec").to_pylist()
        for target, duration in zip(targets, durations, strict=True):
            if target is None or not 2 <= target < number_of_pois:
                raise ValueError("training survival target is outside the known catalog")
            if duration is None or duration <= 0:
                raise ValueError("observed training base gaps require positive duration")
        if not targets:
            continue
        gap_count += len(targets)
        target_tensor = torch.tensor(targets, dtype=torch.long)
        duration_tensor = torch.tensor(durations, dtype=torch.int64)
        if keep_durations:
            kept_targets.append(target_tensor)
            kept_durations.append(duration_tensor)
        for index, horizon in enumerate(schedule):
            surviving = target_tensor[duration_tensor > horizon]
            counts[index].scatter_add_(
                0, surviving, torch.ones_like(surviving, dtype=torch.int64)
            )
    if not gap_count:
        raise ValueError("no observed known-target training base gaps for silence fitting")
    return DestinationSurvivalProfile(
        counts, gap_count, schedule,
        targets=torch.cat(kept_targets) if keep_durations else None,
        durations=torch.cat(kept_durations) if keep_durations else None,
    )


class SilenceAwareFloor(BaselineModel):
    """Wrap an unchanged fitted floor with a registered-horizon survival factor."""

    def __init__(
        self, base_model: BaselineModel, silence_alpha: float = 1.0,
        landmark_seconds: tuple[int, ...] = LANDMARK_SECONDS,
        horizon_support: str = "grid",
    ) -> None:
        if f"{base_model.name}_s" not in SILENCE_FLOOR_BASES:
            raise ValueError("silence controls require one of the four original floors")
        if not math.isfinite(silence_alpha) or silence_alpha <= 0:
            raise ValueError("silence_alpha must be finite and positive")
        if horizon_support not in ("grid", "step"):
            raise ValueError("horizon_support must be 'grid' (reject off-grid queries) "
                             "or 'step' (empirical step survivor at any s)")
        super().__init__(base_model.number_of_pois)
        self.name = f"{base_model.name}_s"
        self.base_model = base_model
        self.silence_alpha = float(silence_alpha)
        self.landmark_seconds = tuple(int(value) for value in landmark_seconds)
        #: "grid" preserves the donor contract: off-grid queries are rejected,
        #: never silently rounded. "step" answers any non-negative s with the
        #: same estimator's empirical step survivor; grid values are identical.
        self.horizon_support = horizon_support
        self.profile: DestinationSurvivalProfile | None = None
        self._log_survival = torch.empty(0)
        self._pooled_survivors: list[int] = []
        self.reset_prediction_stats()

    def fit(self, training_source: str | Path) -> SilenceAwareFloor:
        """Fit only empirical duration counts; retain the original fitted floor."""
        self.base_model.require_fitted()
        self.profile = fit_destination_survival(
            training_source, self.number_of_pois, landmark_seconds=self.landmark_seconds,
            keep_durations=self.horizon_support == "step",
        )
        counts = self.profile.survivor_counts.to(torch.float64)
        pooled = counts.sum(dim=1) / self.profile.gap_count
        fractions = (counts + self.silence_alpha * pooled[:, None]) / (
            counts[0] + self.silence_alpha
        )
        # Unsupported horizons are never scored with these factors. Store a
        # neutral vector rather than inventing positive survival beyond support.
        fractions[pooled == 0] = 1.0
        self._log_survival = fractions.log().to(torch.float32)
        self._pooled_survivors = self.profile.survivor_counts.sum(dim=1).tolist()
        self.is_fitted = True
        self.reset_prediction_stats()
        return self

    def predict(self, batch: dict[str, torch.Tensor]) -> BaselinePrediction:
        """Read history and elapsed silence only; current query labels are unused."""
        self.require_fitted()
        horizons = batch["query_s_sec"]
        if horizons.ndim != 1 or horizons.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        ):
            raise ValueError("query_s_sec must be a one-dimensional integer tensor")
        requested = horizons.unique().tolist()
        off_grid = [h for h in requested if h not in self.landmark_seconds]
        if off_grid and self.horizon_support == "grid":
            raise ValueError(
                f"query_s_sec must use the fitted horizons {self.landmark_seconds} "
                "(horizon_support='step' answers arbitrary non-negative queries)"
            )
        if any(horizon < 0 for horizon in requested):
            raise ValueError("query_s_sec must be non-negative")
        base = self.base_model.predict(batch)
        logits = base.poi_logits.clone()
        for horizon in requested:
            rows = horizons == horizon
            row_count = int(rows.sum())
            self._rows_by_horizon[str(horizon)] = (
                self._rows_by_horizon.get(str(horizon), 0) + row_count
            )
            if horizon == 0:
                continue  # Preserve every original score exactly at s=0.
            log_survival, pooled_survivors = self._horizon_adjustment(int(horizon))
            if pooled_survivors == 0:
                self._unsupported_rows += row_count
                continue  # No extrapolated positive tail beyond observed support.
            logits[rows] = masked_log_probabilities(logits[rows] + log_survival)
            self._adjusted_rows += row_count
        self._predicted_rows += len(horizons)
        return BaselinePrediction(poi_logits=logits, context=base.context)

    def _horizon_adjustment(self, horizon: int) -> tuple[torch.Tensor, int]:
        """(log survival-fraction vector, pooled survivor count) for one horizon.

        Grid horizons read the fitted table. In ``step`` mode any other horizon
        is answered by the same estimator on the retained completed-gap
        durations — an empirical step survivor, never a rounded grid lookup —
        and reproduces the table exactly at the grid points.
        """
        if horizon in self.landmark_seconds:
            index = self.landmark_seconds.index(horizon)
            return self._log_survival[index], self._pooled_survivors[index]
        assert self.profile is not None and self.profile.durations is not None
        surviving = self.profile.durations > horizon
        counts = torch.zeros(self.number_of_pois, dtype=torch.float64)
        picked = self.profile.targets[surviving]
        counts.scatter_add_(0, picked, torch.ones_like(picked, dtype=torch.float64))
        pooled_survivors = int(surviving.sum())
        pooled = pooled_survivors / self.profile.gap_count
        base_counts = self.profile.survivor_counts[0].to(torch.float64)
        fractions = (counts + self.silence_alpha * pooled) / (
            base_counts + self.silence_alpha
        )
        if pooled_survivors == 0:
            fractions.fill_(1.0)
        return fractions.log().to(torch.float32), pooled_survivors

    def reset_prediction_stats(self) -> None:
        self._predicted_rows = 0
        self._adjusted_rows = 0
        self._unsupported_rows = 0
        self._rows_by_horizon = {str(horizon): 0 for horizon in self.landmark_seconds}
        if hasattr(self.base_model, "reset_prediction_stats"):
            self.base_model.reset_prediction_stats()

    def prediction_stats(self) -> dict[str, Any]:
        result = {
            "predicted_row_count": self._predicted_rows,
            "silence_adjusted_row_count": self._adjusted_rows,
            "unsupported_horizon_fallback_row_count": self._unsupported_rows,
            "unsupported_horizon_fallback_rate": (
                self._unsupported_rows / self._predicted_rows if self._predicted_rows else None
            ),
            "rows_by_horizon_seconds": dict(self._rows_by_horizon),
        }
        if hasattr(self.base_model, "prediction_stats"):
            result["base_prediction_stats"] = self.base_model.prediction_stats()
        return result

    def parameter_count(self) -> int:
        return self.base_model.parameter_count() + (
            self.profile.survivor_counts.numel() if self.profile is not None else 0
        )

    def resolved_config(self) -> dict[str, Any]:
        self.require_fitted()
        assert self.profile is not None
        return {
            "name": self.name,
            "family": self.family,
            "interface_version": self.interface_version,
            "number_of_pois": self.number_of_pois,
            "base_model": self.base_model.resolved_config(),
            "profile_version": PROFILE_VERSION,
            "silence_alpha": self.silence_alpha,
            "horizons_seconds": list(self.landmark_seconds),
            "fit_population": "train s=0, observed completed gaps, known target, one per gap",
            "uniqueness_contract": "canonical inputs contain exactly one s=0 row per gap",
            "training_gap_count": self.profile.gap_count,
            "fitted_destination_count": int((self.profile.survivor_counts[0] > 0).sum()),
            "pooled_survivor_counts": dict(zip(
                map(str, self.landmark_seconds), self._pooled_survivors, strict=True
            )),
            "survival_boundary": "duration_sec > query_s_sec",
            "smoothing_policy": "alpha pseudo-gaps with pooled completed-gap survival",
            "unseen_destination_policy": "pooled survival fraction",
            "unsupported_horizon_policy": "original floor when pooled survivors are zero",
            "horizon_support": self.horizon_support,
            "off_grid_horizon_policy": (
                "reject" if self.horizon_support == "grid"
                else "empirical step survivor from the same estimator (no rounding)"
            ),
            "zero_horizon_policy": "return original scores exactly",
            "limitation": "observed-completed-gap estimate; no censor correction or time head",
            "prediction_stats": self.prediction_stats(),
            "context_dim": self.context_dim,
            "parameter_count": self.parameter_count(),
        }
