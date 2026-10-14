"""Training-only population identities; evaluation keeps the canonical landmarks."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from adrec_still.data.schema import UNKNOWN_POI_ID, CanonicalSample

EVENT_ONLY_TRAINING_PROFILE = "still-event-only-observed-known-v1"
EVENT_ONLY_WEIGHTING_CONTRACT = "still-event-gap-weighting-v1"


def event_only_sample_eligible(sample: CanonicalSample) -> bool:
    """Keep one existing, positive-duration, known-target s=0 row per gap.

    This does not reconstruct transitions from event sequences: zero-duration
    transitions absent from the canonical landmark population stay absent.
    """
    return bool(
        sample.query_s_sec == 0
        and sample.event_observed
        and sample.target_in_train_catalog
        and sample.next_poi is not None
        and sample.next_poi > UNKNOWN_POI_ID
        and sample.residual_time_sec is not None
        and sample.residual_time_sec > 0
    )


def event_only_training_mask(
    batch: Mapping[str, torch.Tensor], number_of_pois: int | None = None,
) -> torch.Tensor:
    """Defensive loss mask for callers supplying an unfiltered collated batch."""
    residual = batch["residual_time_sec"]
    target = batch["next_poi"]
    keep = (
        batch["query_s_sec"].eq(0)
        & batch["event_observed"].bool()
        & batch["observed_mask"].bool()
        & batch["target_in_train_catalog"].bool()
        & target.gt(UNKNOWN_POI_ID)
        & residual.gt(0)
        & torch.isfinite(residual)
    )
    if number_of_pois is not None:
        keep &= target.lt(number_of_pois)
    return keep
