"""Canonical sample collation and padding masks."""

from __future__ import annotations

from typing import Any

import torch

from adrec_still.data.schema import PADDING_POI_ID, CanonicalSample


def move_batch(batch: dict[str, Any], device) -> dict[str, Any]:
    """Tensors to the device; non-tensor fields (gap_id strings) pass through."""
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def select_batch_rows(batch: dict[str, Any], keep: torch.Tensor) -> dict[str, Any]:
    """Row selection that keeps tensor and list fields aligned."""
    flags = keep.tolist()
    return {
        key: value[keep] if isinstance(value, torch.Tensor)
        else [item for item, flag in zip(value, flags) if flag]
        for key, value in batch.items()
    }


def slice_batch_rows(batch: dict[str, Any], start: int, stop: int | None) -> dict[str, Any]:
    """Contiguous row slice over tensor and list fields alike."""
    return {
        key: value[start:stop] for key, value in batch.items()
    }


def collate_canonical_samples(
    samples: list[CanonicalSample],
    max_sequence_length: int | None = None,
) -> dict[str, Any]:
    """Left-pad histories and create explicit observed/censored masks."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    for sample in samples:
        sample.validate()
    longest = max(len(sample.history_poi) for sample in samples)
    sequence_length = min(longest, max_sequence_length or longest)
    batch_size = len(samples)
    history_poi = torch.zeros((batch_size, sequence_length), dtype=torch.long)
    history_timestamp = torch.zeros((batch_size, sequence_length), dtype=torch.long)
    history_mask = torch.zeros((batch_size, sequence_length), dtype=torch.bool)

    for row_index, sample in enumerate(samples):
        poi = list(sample.history_poi)[-sequence_length:]
        timestamps = list(sample.history_timestamp)[-sequence_length:]
        start = sequence_length - len(poi)
        history_poi[row_index, start:] = torch.tensor(poi, dtype=torch.long)
        history_timestamp[row_index, start:] = torch.tensor(timestamps, dtype=torch.long)
        history_mask[row_index, start:] = True

    observed_mask = torch.tensor([sample.event_observed for sample in samples], dtype=torch.bool)
    target_in_train_catalog = torch.tensor(
        [sample.target_in_train_catalog for sample in samples], dtype=torch.bool
    )
    return {
        "user_id": torch.tensor([sample.user_id for sample in samples], dtype=torch.long),
        "history_poi": history_poi,
        "history_timestamp": history_timestamp,
        "history_mask": history_mask,
        "last_timestamp": torch.tensor(
            [sample.last_timestamp for sample in samples], dtype=torch.long
        ),
        "query_s_sec": torch.tensor([sample.query_s_sec for sample in samples], dtype=torch.long),
        "event_observed": observed_mask,
        "observed_mask": observed_mask,
        "censored_mask": ~observed_mask,
        "target_in_train_catalog": target_in_train_catalog,
        "next_poi": torch.tensor(
            [
                sample.next_poi if sample.next_poi is not None else PADDING_POI_ID
                for sample in samples
            ],
            dtype=torch.long,
        ),
        "residual_time_sec": torch.tensor(
            [
                sample.residual_time_sec if sample.residual_time_sec is not None else 0
                for sample in samples
            ],
            dtype=torch.float64,
        ),
        "censor_time_sec": torch.tensor(
            [
                sample.censor_time_sec if sample.censor_time_sec is not None else 0
                for sample in samples
            ],
            dtype=torch.float64,
        ),
        # Anchor identities (strings, not tensors): causal anchor-event names for
        # full-memory lookups; None entries mean the source carried no gap_id.
        "gap_id": [sample.gap_id for sample in samples],
    }
