"""Canonical sample collation and padding masks."""

from __future__ import annotations

from typing import Any

import torch

from adrec_still.data.schema import PADDING_POI_ID, CanonicalSample


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
    }
