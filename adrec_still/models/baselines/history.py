"""Shared history-inspection helpers for the diagnostic baselines.

All diagnostic baselines read the collated ``history_poi`` / ``history_mask``
pair with histories left-padded in chronological order, so the most recent
entry of a row is its rightmost valid position. They differ in how the shared
unknown POI (``1``) is treated: UserHistoryPop and RepeatLast skip it while
searching backwards (:func:`last_known_poi`), whereas FirstOrderMarkov keys its
transition state on the actual last observation (:func:`last_observed_poi`) so
a trailing unknown POI triggers the GlobalPop fallback instead of silently
reaching past it to an older known entry. Padding (``0``) is never usable
history evidence for any model.
"""

from __future__ import annotations

import torch

from adrec_still.data.schema import PADDING_POI_ID, UNKNOWN_POI_ID


def known_history_mask(
    history_poi: torch.Tensor, history_mask: torch.Tensor
) -> torch.Tensor:
    """Mark history positions that carry a usable train-catalog POI."""
    if history_poi.shape != history_mask.shape:
        raise ValueError("history_poi and history_mask must have the same shape")
    return (
        history_mask.bool()
        & history_poi.ne(PADDING_POI_ID)
        & history_poi.ne(UNKNOWN_POI_ID)
    )


def _rightmost_valid_entry(
    history_poi: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Return each row's rightmost valid entry, or ``PADDING_POI_ID`` when none exists."""
    batch_size, sequence_length = history_poi.shape
    if sequence_length == 0:
        return torch.full((batch_size,), PADDING_POI_ID, dtype=torch.long)
    positions = torch.arange(sequence_length, device=history_poi.device)
    rightmost = torch.where(valid, positions, torch.full_like(positions, -1)).max(dim=1).values
    gathered = history_poi.gather(1, rightmost.clamp_min(0).unsqueeze(1)).squeeze(1)
    return torch.where(rightmost.ge(0), gathered, torch.full_like(gathered, PADDING_POI_ID))


def last_known_poi(history_poi: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
    """Return the most recent known POI per row, or ``PADDING_POI_ID`` when none exists.

    Histories are left-padded and chronological, so the answer is the rightmost
    position whose entry is neither padding nor the shared unknown POI.
    """
    return _rightmost_valid_entry(history_poi, known_history_mask(history_poi, history_mask))


def last_observed_poi(
    history_poi: torch.Tensor, history_mask: torch.Tensor
) -> torch.Tensor:
    """Return the most recent observed entry per row, or ``PADDING_POI_ID`` when none exists.

    Unlike :func:`last_known_poi`, the shared unknown POI is NOT skipped: a row
    whose actual last observation is ``UNKNOWN_POI_ID`` returns it as-is, so
    callers that key behavior on the true last state (FirstOrderMarkov's
    transition lookup) fall back instead of using an older known entry.
    """
    if history_poi.shape != history_mask.shape:
        raise ValueError("history_poi and history_mask must have the same shape")
    valid = history_mask.bool() & history_poi.ne(PADDING_POI_ID)
    return _rightmost_valid_entry(history_poi, valid)
