"""Batch-side bridge between collated canonical rows and the full event memory.

For every row the bridge resolves, through its causal ``gap_id`` anchor:

- the padded seen-set tensors the decoders read (known POI IDs with full-history
  frequency and last-visit time);
- the branch label of the gap's target (raw-ID repeat/explore) for training and
  evaluation — label-side, returned separately from the features;
- the home (activity center) coordinates under the requested policy.

Rows whose source carries no ``gap_id`` cannot be resolved; models that need the
full memory refuse them with a named remedy instead of silently falling back to
window-limited semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from adrec_still.data.history import FullHistoryIndex


@dataclass
class MemoryFeatures:
    """Prefix-side features (causal; safe to condition forecasts on)."""

    seen_poi: torch.Tensor        # int64 [B, M] padded with 0
    seen_count: torch.Tensor      # float32 [B, M]
    seen_last_s: torch.Tensor     # int64 [B, M]
    seen_mask: torch.Tensor       # bool [B, M]
    seen_sizes: torch.Tensor      # int64 [B]
    home_lat: torch.Tensor        # float64 [B] (NaN when unavailable)
    home_lon: torch.Tensor        # float64 [B]
    home_available: torch.Tensor  # bool [B]


@dataclass
class MemoryLabels:
    """Label-side information (never enters a forecast)."""

    repeat_label: torch.Tensor           # int8 [B]: 1 repeat, 0 explore, -1 unavailable
    target_poi: torch.Tensor             # int64 [B]: frozen-vocab target (0 when none)
    gap_duration_s: torch.Tensor         # int64 [B]: total gap T (-1 when none)


def require_gap_ids(batch: dict) -> list[str]:
    gap_ids = batch.get("gap_id")
    if gap_ids is None or any(value is None for value in gap_ids):
        raise ValueError(
            "this model needs full-history repeat semantics, which require each row's "
            "gap_id anchor; the supplied data carries none. Preprocess from a raw "
            "check-in file or point --data at a canonical directory whose "
            "landmark_samples.parquet has the gap_id column (window-limited "
            "'window_repeat' analysis is a separate, explicitly named mode)"
        )
    return list(gap_ids)


def memory_features(index: FullHistoryIndex, batch: dict, *,
                    home_policy: str = "causal") -> MemoryFeatures:
    gap_ids = require_gap_ids(batch)
    states = [index.seen_state(gap_id) for gap_id in gap_ids]
    batch_size = len(states)
    width = max(1, max(state.size for state in states))
    seen_poi = torch.zeros((batch_size, width), dtype=torch.long)
    seen_count = torch.zeros((batch_size, width), dtype=torch.float32)
    seen_last = torch.zeros((batch_size, width), dtype=torch.long)
    seen_mask = torch.zeros((batch_size, width), dtype=torch.bool)
    sizes = torch.zeros(batch_size, dtype=torch.long)
    home_lat = torch.full((batch_size,), float("nan"), dtype=torch.float64)
    home_lon = torch.full((batch_size,), float("nan"), dtype=torch.float64)
    home_available = torch.zeros(batch_size, dtype=torch.bool)
    for row, (gap_id, state) in enumerate(zip(gap_ids, states)):
        m = state.size
        sizes[row] = m
        if m:
            seen_poi[row, :m] = torch.from_numpy(state.poi_ids)
            seen_count[row, :m] = torch.from_numpy(state.counts).float()
            seen_last[row, :m] = torch.from_numpy(state.last_visit_s)
            seen_mask[row, :m] = True
        home = index.home(gap_id, policy=home_policy)
        if home is None and home_policy == "frozen":
            # No pre-cutoff activity center: fall back to the causal prefix so a
            # post-cutoff-only user still has a feature-side home.
            home = index.home(gap_id, policy="causal")
        if home is not None:
            home_lat[row], home_lon[row] = home
            home_available[row] = True
    return MemoryFeatures(
        seen_poi=seen_poi, seen_count=seen_count, seen_last_s=seen_last,
        seen_mask=seen_mask, seen_sizes=sizes,
        home_lat=home_lat, home_lon=home_lon, home_available=home_available,
    )


def memory_labels(index: FullHistoryIndex, batch: dict) -> MemoryLabels:
    gap_ids = require_gap_ids(batch)
    batch_size = len(gap_ids)
    repeat = torch.full((batch_size,), -1, dtype=torch.int8)
    target = torch.zeros(batch_size, dtype=torch.long)
    duration = torch.full((batch_size,), -1, dtype=torch.long)
    for row, gap_id in enumerate(gap_ids):
        label = index.repeat_label(gap_id)
        if label is not None:
            repeat[row] = int(label)
        target_poi = index.target_poi(gap_id)
        if target_poi is not None:
            target[row] = target_poi
        gap_duration = index.gap_duration_s(gap_id)
        if gap_duration is not None:
            duration[row] = gap_duration
    return MemoryLabels(repeat_label=repeat, target_poi=target, gap_duration_s=duration)
