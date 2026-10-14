"""Leakage-safe base-gap construction and chronological splitting."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def make_gap_id(
    current_source_row: int,
    next_source_row: int,
    dataset_name: str = "gowalla",
) -> str:
    """Create a stable identifier from the two traceable raw row IDs."""
    return f"{dataset_name}:{current_source_row:012d}:{next_source_row:012d}"


def build_base_gaps(events: pd.DataFrame, dataset_name: str = "gowalla") -> pd.DataFrame:
    """Build consecutive completed gaps before landmark expansion."""
    grouped = events.groupby("raw_user_id", sort=False, observed=True)
    gaps = events[["raw_user_id", "raw_poi_id", "event_time_utc_s", "source_row_id"]].copy()
    gaps["next_poi_raw"] = grouped["raw_poi_id"].shift(-1)
    gaps["next_event_time_s"] = grouped["event_time_utc_s"].shift(-1)
    gaps["next_source_row_id"] = grouped["source_row_id"].shift(-1)
    gaps = gaps.loc[gaps["next_event_time_s"].notna()].copy()
    gaps["next_event_time_s"] = gaps["next_event_time_s"].astype("int64")
    gaps["next_source_row_id"] = gaps["next_source_row_id"].astype("int64")
    gaps["delta_t_sec"] = gaps["next_event_time_s"] - gaps["event_time_utc_s"]
    if (gaps["delta_t_sec"] < 0).any():
        raise ValueError("events are not chronological within a user")
    gaps["gap_id"] = [
        make_gap_id(current, next_row, dataset_name=dataset_name)
        for current, next_row in zip(
            gaps["source_row_id"].to_numpy(),
            gaps["next_source_row_id"].to_numpy(),
            strict=True,
        )
    ]
    return gaps.reset_index(drop=True)


def assign_chronological_splits(
    gaps: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.10,
) -> pd.DataFrame:
    """Split globally by next-event time with gap_id as deterministic tie-breaker."""
    if train_ratio <= 0 or validation_ratio < 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("split ratios must leave positive train and test partitions")
    ordered = gaps.sort_values(["next_event_time_s", "gap_id"], kind="mergesort").reset_index(
        drop=True
    )
    count = len(ordered)
    train_end = math.floor(count * train_ratio)
    validation_end = train_end + math.floor(count * validation_ratio)
    labels = np.full(count, "test", dtype=object)
    labels[:train_end] = "train"
    labels[train_end:validation_end] = "validation"
    ordered["split"] = pd.Categorical(
        labels, categories=["train", "validation", "test"], ordered=True
    )
    return ordered


def split_manifest_frame(split_gaps: pd.DataFrame) -> pd.DataFrame:
    """Select the versioned, persisted membership columns."""
    return split_gaps[
        [
            "gap_id",
            "raw_user_id",
            "source_row_id",
            "next_source_row_id",
            "event_time_utc_s",
            "next_event_time_s",
            "delta_t_sec",
            "split",
        ]
    ].copy()


def split_period_start_time_s(split_gaps: pd.DataFrame, split: str) -> int:
    """Return the first next-event timestamp assigned to a completed-gap split."""
    selected = split_gaps.loc[split_gaps["split"] == split, "next_event_time_s"]
    if selected.empty:
        raise ValueError(f"completed-gap split has no rows: {split}")
    return int(selected.min())
