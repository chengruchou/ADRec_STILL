"""Runtime weights for the frozen completed-gap training contract."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import torch

from adrec_still.data.landmarks import LANDMARK_SECONDS

WEIGHTING_CONTRACT_VERSION = "still-landmark-weighting-v0.1"
SURVIVAL_WEIGHTING_CONTRACT_VERSION = "still-survival-gap-weighting-v1"


def _validate_schedule(landmark_seconds: tuple[int, ...]) -> tuple[int, ...]:
    if (
        not landmark_seconds
        or any(
            isinstance(s, bool) or not isinstance(s, Integral) or s < 0 for s in landmark_seconds
        )
        or len(set(landmark_seconds)) != len(landmark_seconds)
    ):
        raise ValueError("landmark_seconds must contain unique non-negative integer seconds")
    return tuple(sorted(landmark_seconds))


def landmark_weights(
    batch: Mapping[str, torch.Tensor],
    mode: str = "per_gap",
    landmark_seconds: tuple[int, ...] = LANDMARK_SECONDS,
) -> torch.Tensor:
    """Return a float64 [batch] tensor on the query tensor's device.

    Only completed, observed gaps are supported (including unknown targets).
    Pass the complete landmark schedule used for training, even for shuffled or
    partial batches. All eligible rows must be visited once per training epoch
    for each gap's weights to sum to one; batch-local counting is never used.
    Censored rows are rejected in both modes, before any loss-specific masking.
    """
    if mode not in {"per_gap", "per_landmark"}:
        raise ValueError("mode must be per_gap or per_landmark")
    landmark_seconds = _validate_schedule(landmark_seconds)
    query = batch["query_s_sec"]
    residual = batch["residual_time_sec"]
    observed = batch["event_observed"]
    if query.ndim != 1 or residual.shape != query.shape or observed.shape != query.shape:
        raise ValueError("query, residual, and event_observed must have shape [batch]")
    if residual.device != query.device or observed.device != query.device:
        raise ValueError("weighting tensors must share a device")
    if observed.dtype != torch.bool or not observed.all():
        raise ValueError("landmark weighting requires observed completed gaps; censoring rejected")
    for name, values in (("query_s_sec", query), ("residual_time_sec", residual)):
        if values.dtype == torch.bool or values.is_complex():
            raise ValueError(f"{name} must contain finite integer seconds")
        if not torch.isfinite(values).all() or (values != values.trunc()).any():
            raise ValueError(f"{name} must contain finite integer seconds")
    query = query.to(torch.float64)
    residual = residual.to(torch.float64)
    if (query < 0).any() or (residual <= 0).any():
        raise ValueError("observed rows require non-negative query and positive residual time")
    schedule = torch.tensor(sorted(landmark_seconds), dtype=torch.float64, device=query.device)
    if not torch.isin(query, schedule).all():
        raise ValueError("query_s_sec is outside the training landmark schedule")
    duration = query + residual
    if not torch.isfinite(duration).all():
        raise ValueError("completed-gap duration must be finite")
    if mode == "per_landmark":
        return torch.ones_like(query)
    # Strict eligibility is delta_t > s, including at an exact horizon boundary.
    counts = (duration[:, None] > schedule).sum(dim=1)
    return counts.to(torch.float64).reciprocal()


def survival_landmark_weights(
    batch: Mapping[str, torch.Tensor],
    mode: str = "per_gap",
    landmark_seconds: tuple[int, ...] = LANDMARK_SECONDS,
) -> torch.Tensor:
    """Weights for the complete #39 observed plus administrative-censor profile.

    Each row's full observable interval is reconstructed as ``query + residual``
    for completed gaps or ``query + censor exposure`` for unfinished gaps. Counting
    against the registered full schedule makes weights independent of batch order,
    partial batches and loss masks. Positive follow-up censors are required.
    """
    if mode not in {"per_gap", "per_landmark"}:
        raise ValueError("mode must be per_gap or per_landmark")
    schedule_values = _validate_schedule(landmark_seconds)
    query = batch["query_s_sec"]
    residual = batch["residual_time_sec"]
    censor = batch["censor_time_sec"]
    observed = batch["event_observed"]
    if any(values.shape != query.shape for values in (residual, censor, observed)) or query.ndim != 1:
        raise ValueError("survival weighting tensors must have shape [batch]")
    if any(values.device != query.device for values in (residual, censor, observed)):
        raise ValueError("survival weighting tensors must share a device")
    if observed.dtype != torch.bool:
        raise ValueError("event_observed must be boolean")
    for name, values in (
        ("query_s_sec", query),
        ("residual_time_sec", residual),
        ("censor_time_sec", censor),
    ):
        if values.dtype == torch.bool or values.is_complex():
            raise ValueError(f"{name} must contain finite integer seconds")
        if not torch.isfinite(values).all() or (values != values.trunc()).any():
            raise ValueError(f"{name} must contain finite integer seconds")
    query64 = query.to(torch.float64)
    residual64 = residual.to(torch.float64)
    censor64 = censor.to(torch.float64)
    if (query64 < 0).any():
        raise ValueError("query_s_sec must be non-negative")
    if (residual64[observed] <= 0).any() or (censor64[observed] != 0).any():
        raise ValueError("observed rows require positive residual and zero censor exposure")
    if (censor64[~observed] <= 0).any() or (residual64[~observed] != 0).any():
        raise ValueError("censored rows require positive follow-up and zero residual padding")
    schedule = torch.tensor(schedule_values, dtype=torch.float64, device=query.device)
    if not torch.isin(query64, schedule).all():
        raise ValueError("query_s_sec is outside the training landmark schedule")
    duration = query64 + torch.where(observed, residual64, censor64)
    if mode == "per_landmark":
        return torch.ones_like(query64)
    eligible_counts = (duration[:, None] > schedule).sum(dim=1)
    if (eligible_counts <= 0).any():
        raise ValueError("every survival row must belong to a positive eligible schedule")
    return eligible_counts.to(torch.float64).reciprocal()
