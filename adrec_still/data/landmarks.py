"""Observed and right-censored landmark sample construction."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from adrec_still.data.schema import UNKNOWN_POI_ID
from adrec_still.data.split import make_gap_id, split_period_start_time_s

LANDMARK_HOURS = (0, 6, 12, 24, 48, 72, 168)
LANDMARK_SECONDS = tuple(hours * 3600 for hours in LANDMARK_HOURS)
LANDMARK_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("gap_id", pa.string()),
        ("user_id", pa.int64()),
        ("split", pa.string()),
        ("history_poi", pa.list_(pa.int64())),
        ("history_timestamp", pa.list_(pa.int64())),
        ("last_timestamp", pa.int64()),
        ("query_s_sec", pa.int64()),
        ("event_observed", pa.bool_()),
        ("target_in_train_catalog", pa.bool_()),
        ("residual_time_sec", pa.int64()),
        ("next_poi", pa.int64()),
        ("censor_time_sec", pa.int64()),
    ]
)


def iter_landmark_records(
    encoded_events: pd.DataFrame,
    split_gaps: pd.DataFrame,
    max_history_length: int | None = None,
    landmark_seconds: tuple[int, ...] = LANDMARK_SECONDS,
    dataset_name: str | None = None,
    dataset_end_s: int | None = None,
    test_period_start_s: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield observed landmarks and final-gap queries within the completed-gap test period."""
    if max_history_length is not None and max_history_length <= 0:
        raise ValueError("max_history_length must be positive when provided")
    if dataset_name is None:
        dataset_values = encoded_events["dataset"].unique().tolist()
        if len(dataset_values) != 1:
            raise ValueError("encoded events must contain exactly one dataset")
        dataset_name = str(dataset_values[0])
    gap_split = dict(zip(split_gaps["gap_id"], split_gaps["split"].astype(str), strict=True))
    if dataset_end_s is None:
        dataset_end_s = int(encoded_events["event_time_utc_s"].max())
    if test_period_start_s is None:
        test_period_start_s = split_period_start_time_s(split_gaps, "test")

    for _, user_events in encoded_events.groupby("raw_user_id", sort=False, observed=True):
        poi_values = user_events["poi_id"].to_numpy(dtype=np.int64)
        time_values = user_events["event_time_utc_s"].to_numpy(dtype=np.int64)
        source_values = user_events["source_row_id"].to_numpy(dtype=np.int64)
        user_id = int(user_events["user_id"].iloc[0])
        for index in range(len(user_events) - 1):
            gap_id = make_gap_id(
                int(source_values[index]),
                int(source_values[index + 1]),
                dataset_name=dataset_name,
            )
            split = gap_split[gap_id]
            delta_t_sec = int(time_values[index + 1] - time_values[index])
            history_start = (
                0 if max_history_length is None else max(0, index + 1 - max_history_length)
            )
            history_poi = poi_values[history_start : index + 1].tolist()
            history_time = time_values[history_start : index + 1].tolist()
            for query_s_sec in landmark_seconds:
                if delta_t_sec <= query_s_sec:
                    continue
                yield {
                    "sample_id": f"{gap_id}:q{query_s_sec:09d}",
                    "gap_id": gap_id,
                    "user_id": user_id,
                    "split": split,
                    "history_poi": history_poi,
                    "history_timestamp": history_time,
                    "last_timestamp": int(time_values[index]),
                    "query_s_sec": query_s_sec,
                    "event_observed": True,
                    "target_in_train_catalog": int(poi_values[index + 1]) != UNKNOWN_POI_ID,
                    "residual_time_sec": delta_t_sec - query_s_sec,
                    "next_poi": int(poi_values[index + 1]),
                    "censor_time_sec": None,
                }

        final_index = len(user_events) - 1
        available_followup_sec = dataset_end_s - int(time_values[final_index])
        history_start = (
            0 if max_history_length is None else max(0, final_index + 1 - max_history_length)
        )
        history_poi = poi_values[history_start : final_index + 1].tolist()
        history_time = time_values[history_start : final_index + 1].tolist()
        censor_gap_id = f"{dataset_name}:{int(source_values[final_index]):012d}:censored"
        for query_s_sec in landmark_seconds:
            if available_followup_sec < query_s_sec:
                continue
            query_time_s = int(time_values[final_index]) + query_s_sec
            if query_time_s < test_period_start_s:
                continue
            yield {
                "sample_id": f"{censor_gap_id}:q{query_s_sec:09d}",
                "gap_id": censor_gap_id,
                "user_id": user_id,
                "split": "test",
                "history_poi": history_poi,
                "history_timestamp": history_time,
                "last_timestamp": int(time_values[final_index]),
                "query_s_sec": query_s_sec,
                "event_observed": False,
                "target_in_train_catalog": False,
                "residual_time_sec": None,
                "next_poi": None,
                "censor_time_sec": available_followup_sec - query_s_sec,
            }


def write_landmarks(
    records: Iterator[dict[str, Any]],
    output_path: str | Path,
    batch_size: int = 100_000,
) -> int:
    """Stream landmark records to compressed Parquet with bounded memory."""
    output_path = Path(output_path)
    writer: pq.ParquetWriter | None = None
    buffer: list[dict[str, Any]] = []
    total = 0
    try:
        for record in records:
            buffer.append(record)
            if len(buffer) >= batch_size:
                table = pa.Table.from_pylist(buffer, schema=LANDMARK_SCHEMA)
                writer = writer or pq.ParquetWriter(
                    output_path, LANDMARK_SCHEMA, compression="zstd"
                )
                writer.write_table(table)
                total += len(buffer)
                buffer.clear()
        if buffer:
            table = pa.Table.from_pylist(buffer, schema=LANDMARK_SCHEMA)
            writer = writer or pq.ParquetWriter(output_path, LANDMARK_SCHEMA, compression="zstd")
            writer.write_table(table)
            total += len(buffer)
        if writer is None:
            pq.write_table(pa.Table.from_pylist([], schema=LANDMARK_SCHEMA), output_path)
    finally:
        if writer is not None:
            writer.close()
    return total
