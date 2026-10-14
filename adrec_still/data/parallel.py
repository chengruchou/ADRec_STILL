"""Deterministic multi-process execution for user-local preprocessing stages."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from adrec_still.data.landmarks import (
    LANDMARK_SCHEMA,
    iter_landmark_records,
    write_landmarks,
)
from adrec_still.data.split import build_base_gaps, split_period_start_time_s


@dataclass(frozen=True)
class _UserPartition:
    index: int
    events_path: Path
    user_ids: tuple[str, ...]


def _user_row_slices(events: pd.DataFrame, workers: int) -> list[slice]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if events.empty:
        return []
    user_values = events["raw_user_id"]
    block_start_mask = user_values.ne(user_values.shift(1)).fillna(True).to_numpy(dtype=bool)
    block_starts = np.flatnonzero(block_start_mask)
    block_users = user_values.iloc[block_starts]
    if block_users.duplicated().any():
        raise ValueError("events for each user must be contiguous before parallel processing")
    block_ends = np.concatenate([block_starts[1:], np.array([len(events)], dtype=np.int64)])
    user_groups = np.array_split(
        np.arange(len(block_starts), dtype=np.int64), min(workers, len(block_starts))
    )
    return [
        slice(int(block_starts[group[0]]), int(block_ends[group[-1]]))
        for group in user_groups
        if len(group)
    ]


def _write_user_partitions(
    events: pd.DataFrame,
    directory: Path,
    prefix: str,
    workers: int,
) -> list[_UserPartition]:
    partitions: list[_UserPartition] = []
    for index, row_slice in enumerate(_user_row_slices(events, workers)):
        frame = events.iloc[row_slice]
        path = directory / f"{prefix}-{index:05d}.parquet"
        frame.to_parquet(path, index=False, compression="zstd")
        partitions.append(
            _UserPartition(
                index=index,
                events_path=path,
                user_ids=tuple(frame["raw_user_id"].drop_duplicates().astype(str).tolist()),
            )
        )
    return partitions


def _build_gap_shard(job: tuple[int, str, str, str]) -> tuple[int, str]:
    index, events_path, output_path, dataset_name = job
    events = pd.read_parquet(events_path)
    gaps = build_base_gaps(events, dataset_name=dataset_name)
    gaps.to_parquet(output_path, index=False, compression="zstd")
    return index, output_path


def build_base_gaps_parallel(
    events: pd.DataFrame,
    dataset_name: str,
    workers: int,
    work_dir: str | Path,
) -> pd.DataFrame:
    """Build user-local gaps in parallel while preserving sequential user order."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    if workers == 1:
        return build_base_gaps(events, dataset_name=dataset_name)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="still-gaps-", dir=work_dir) as temporary:
        temporary_path = Path(temporary)
        partitions = _write_user_partitions(events, temporary_path, "events", workers)
        jobs = [
            (
                partition.index,
                str(partition.events_path),
                str(temporary_path / f"gaps-{partition.index:05d}.parquet"),
                dataset_name,
            )
            for partition in partitions
        ]
        with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
            shard_results = list(executor.map(_build_gap_shard, jobs))
        shard_paths = [Path(path) for _, path in sorted(shard_results)]
        combined = pd.concat(
            [pd.read_parquet(path) for path in shard_paths],
            ignore_index=True,
        )
        # Parquet round-trips can downgrade pandas 3's inferred ``str`` dtype
        # to object; restore it so the canonical split Parquet bytes match the
        # sequential path as well as its logical rows.
        combined["gap_id"] = combined["gap_id"].astype("str")
        return combined


def _write_landmark_shard(
    job: tuple[int, str, str, str, int, tuple[int, ...], str, int, int, int],
) -> tuple[int, str, int]:
    (
        index,
        events_path,
        gaps_path,
        output_path,
        max_history_length,
        landmark_seconds,
        dataset_name,
        dataset_end_s,
        test_period_start_s,
        batch_size,
    ) = job
    encoded_events = pd.read_parquet(events_path)
    split_gaps = pd.read_parquet(gaps_path)
    count = write_landmarks(
        iter_landmark_records(
            encoded_events,
            split_gaps,
            max_history_length=max_history_length,
            landmark_seconds=landmark_seconds,
            dataset_name=dataset_name,
            dataset_end_s=dataset_end_s,
            test_period_start_s=test_period_start_s,
        ),
        output_path,
        batch_size=batch_size,
    )
    return index, output_path, count


def _merge_landmark_shards(
    shard_paths: list[Path],
    output_path: Path,
    batch_size: int,
) -> int:
    writer = pq.ParquetWriter(output_path, LANDMARK_SCHEMA, compression="zstd")
    pending: pa.Table | None = None
    total = 0
    try:
        for shard_path in shard_paths:
            parquet_file = pq.ParquetFile(shard_path)
            for record_batch in parquet_file.iter_batches(batch_size=batch_size):
                table = pa.Table.from_batches([record_batch], schema=LANDMARK_SCHEMA)
                pending = table if pending is None else pa.concat_tables([pending, table])
                total += table.num_rows
                while pending.num_rows >= batch_size:
                    writer.write_table(pending.slice(0, batch_size).combine_chunks())
                    pending = pending.slice(batch_size)
                    if pending.num_rows == 0:
                        pending = None
                        break
        if pending is not None and pending.num_rows:
            writer.write_table(pending.combine_chunks())
    finally:
        writer.close()
    return total


def write_landmarks_parallel(
    encoded_events: pd.DataFrame,
    split_gaps: pd.DataFrame,
    output_path: str | Path,
    *,
    max_history_length: int,
    landmark_seconds: tuple[int, ...],
    dataset_name: str,
    workers: int,
    batch_size: int = 100_000,
) -> int:
    """Generate user-local landmarks in parallel and merge shards deterministically."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output_path = Path(output_path)
    if workers == 1:
        return write_landmarks(
            iter_landmark_records(
                encoded_events,
                split_gaps,
                max_history_length=max_history_length,
                landmark_seconds=landmark_seconds,
                dataset_name=dataset_name,
            ),
            output_path,
            batch_size=batch_size,
        )

    dataset_end_s = int(encoded_events["event_time_utc_s"].max())
    test_period_start_s = split_period_start_time_s(split_gaps, "test")
    with TemporaryDirectory(prefix="still-landmarks-", dir=output_path.parent) as temporary:
        temporary_path = Path(temporary)
        partitions = _write_user_partitions(
            encoded_events,
            temporary_path,
            "encoded-events",
            workers,
        )
        jobs = []
        for partition in partitions:
            partition_gaps = split_gaps.loc[split_gaps["raw_user_id"].isin(partition.user_ids)]
            gaps_path = temporary_path / f"split-gaps-{partition.index:05d}.parquet"
            partition_gaps.to_parquet(gaps_path, index=False, compression="zstd")
            jobs.append(
                (
                    partition.index,
                    str(partition.events_path),
                    str(gaps_path),
                    str(temporary_path / f"landmarks-{partition.index:05d}.parquet"),
                    max_history_length,
                    landmark_seconds,
                    dataset_name,
                    dataset_end_s,
                    test_period_start_s,
                    batch_size,
                )
            )
        with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
            shard_results = list(executor.map(_write_landmark_shard, jobs))
        shard_paths = [Path(path) for _, path, _ in sorted(shard_results)]
        return _merge_landmark_shards(shard_paths, output_path, batch_size)
