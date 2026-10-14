"""Shared deterministic-baseline interface.

Every Student B baseline consumes the collated canonical batch produced by
:func:`adrec_still.data.collate.collate_canonical_samples` and returns normalized
next-POI log-probabilities over the frozen training vocabulary, so that one
shared evaluator can score every model without model-specific metric code.

Reserved IDs are never candidates: padding (``0``) and the shared unknown POI
(``1``) are pushed to a finite floor before normalization, which keeps the
returned tensor finite while making their probability exactly zero.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from adrec_still.data.schema import PADDING_POI_ID, UNKNOWN_POI_ID
from adrec_still.time.protocol import LogNormalPrediction, ResidualTimeDistribution

BASELINE_INTERFACE_VERSION = "still-baseline-v0.1"

#: Finite floor used instead of ``-inf`` so model output stays finite and JSON-safe.
MASKED_LOGIT = -1.0e9

#: Batch fields every baseline may rely on.
REQUIRED_BATCH_FIELDS = (
    "history_poi",
    "history_timestamp",
    "history_mask",
    "last_timestamp",
    "query_s_sec",
    "next_poi",
    "observed_mask",
    "censored_mask",
    "target_in_train_catalog",
)


def validate_batch(batch: dict[str, torch.Tensor]) -> int:
    """Check the shared batch contract and return the batch size."""
    missing = [field for field in REQUIRED_BATCH_FIELDS if field not in batch]
    if missing:
        raise KeyError(f"batch is missing shared canonical fields: {sorted(missing)}")
    if batch["history_poi"].ndim != 2:
        raise ValueError("history_poi must be [batch, sequence]")
    if batch["history_mask"].shape != batch["history_poi"].shape:
        raise ValueError("history_mask must have the same shape as history_poi")
    batch_size = int(batch["history_poi"].shape[0])
    for field in ("query_s_sec", "next_poi", "observed_mask", "target_in_train_catalog"):
        if batch[field].shape != (batch_size,):
            raise ValueError(f"{field} must have shape [batch]")
    return batch_size


def masked_log_probabilities(
    scores: torch.Tensor,
    padding_id: int = PADDING_POI_ID,
    unknown_id: int = UNKNOWN_POI_ID,
) -> torch.Tensor:
    """Normalize scores over the train catalog only, keeping every value finite."""
    if scores.ndim != 2:
        raise ValueError("scores must be [rows, number_of_pois]")
    if padding_id == unknown_id:
        raise ValueError("padding_id and unknown_id must be distinct")
    masked = scores.clone()
    for reserved_id in (padding_id, unknown_id):
        if 0 <= reserved_id < masked.shape[1]:
            masked[:, reserved_id] = MASKED_LOGIT
    log_probabilities = torch.log_softmax(masked, dim=1)
    if not torch.isfinite(log_probabilities).all():
        raise ValueError("normalized POI log-probabilities must be finite")
    return log_probabilities


@dataclass(frozen=True)
class BaselinePrediction:
    """Shared model output consumed by the evaluator.

    ``time_prediction`` carries the accepted single-LogNormal path;
    ``time_distribution`` carries the proposed generic residual-time adapter
    (``adrec_still.time.protocol.ResidualTimeDistribution``, pending #39 review) for
    mixture/survival-conditioned models that a single LogNormal cannot represent.
    """

    poi_logits: torch.Tensor
    context: torch.Tensor
    time_prediction: LogNormalPrediction | None = None
    time_distribution: ResidualTimeDistribution | None = None
    #: Optional forecast extras (repeat probabilities, MC metadata); consumers
    #: read it through ``getattr(prediction, "extras", None)`` and must ignore
    #: keys they do not know.
    extras: dict[str, Any] | None = None

    def validate(self, number_of_pois: int, check_finite: bool = True) -> None:
        """Check the published output contract.

        ``check_finite`` materializes broadcast views, so callers that stream
        millions of rows may check finiteness on the first batch only.
        """
        if self.poi_logits.ndim != 2:
            raise ValueError("poi_logits must be [batch, number_of_pois]")
        if self.poi_logits.shape[1] != number_of_pois:
            raise ValueError("poi_logits must cover the full training POI vocabulary")
        batch_size = int(self.poi_logits.shape[0])
        if self.context.ndim != 2 or self.context.shape[0] != batch_size:
            raise ValueError("context must be [batch, representation_dim]")
        if check_finite and not torch.isfinite(self.poi_logits).all():
            raise ValueError("poi_logits must be finite")
        if self.time_prediction is not None and self.time_distribution is not None:
            raise ValueError("provide either time_prediction or time_distribution, not both")
        if self.time_prediction is not None:
            self.time_prediction.validate()
            if self.time_prediction.mu.shape != (batch_size,):
                raise ValueError("time-distribution parameters must have shape [batch]")
        if self.time_distribution is not None and self.time_distribution.batch_size() != batch_size:
            raise ValueError("time_distribution must describe exactly the predicted rows")


class BaselineModel(ABC):
    """Minimal contract shared by deterministic and gradient-trained baselines.

    Gradient-trained baselines may additionally inherit :class:`torch.nn.Module`;
    the contract below is what the shared harness and evaluator depend on.
    """

    name: str = "baseline"
    family: str = "baseline"
    interface_version: str = BASELINE_INTERFACE_VERSION

    def __init__(self, number_of_pois: int) -> None:
        if number_of_pois <= UNKNOWN_POI_ID:
            raise ValueError("number_of_pois must include at least one train-catalog POI")
        self.number_of_pois = int(number_of_pois)
        self.is_fitted = False

    @property
    def context_dim(self) -> int:
        """Width of the exported history representation."""
        return 0

    @abstractmethod
    def fit(self, training_source: Any) -> BaselineModel:
        """Fit on training data only and return ``self``."""

    @abstractmethod
    def predict(self, batch: dict[str, torch.Tensor]) -> BaselinePrediction:
        """Return normalized POI log-probabilities for one collated batch."""

    @abstractmethod
    def resolved_config(self) -> dict[str, Any]:
        """Return every non-trivial setting needed to reproduce the fitted model."""

    def parameter_count(self) -> int:
        """Number of learned or counted values stored by the model."""
        return 0

    def require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before prediction")
