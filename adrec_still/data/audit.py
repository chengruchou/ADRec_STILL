"""Gowalla audit summaries and file digests."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adrec_still.data.adapters.base import CleaningAudit
from adrec_still.data.schema import PADDING_POI_ID, UNKNOWN_POI_ID
from adrec_still.data.split import split_period_start_time_s


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_audit_report(
    events: pd.DataFrame,
    split_gaps: pd.DataFrame,
    cleaning: CleaningAudit,
    vocabulary_metadata: dict[str, int],
    split_manifest_hash: str,
    protocol_version: str,
    landmark_hours: tuple[int, ...],
    dataset_name: str = "gowalla",
    coverage_notes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the minimum machine-readable audit required by protocol v0.1."""
    train = split_gaps.loc[split_gaps["split"] == "train"]
    train_users = set(train["raw_user_id"])
    train_pois = set(train["raw_poi_id"]) | set(train["next_poi_raw"])
    by_split: dict[str, Any] = {}
    eligible: dict[str, dict[str, int]] = {}
    eligible_users: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        selected = split_gaps.loc[split_gaps["split"] == split]
        split_users = set(selected["raw_user_id"])
        split_pois = set(selected["raw_poi_id"]) | set(selected["next_poi_raw"])
        by_split[split] = {
            "completed_gaps": len(selected),
            "users": len(split_users),
            "pois": len(split_pois),
            "cold_start_users": len(split_users - train_users) if split != "train" else 0,
            "cold_start_pois": len(split_pois - train_pois) if split != "train" else 0,
        }
        eligible[split] = {
            str(hours): int((selected["delta_t_sec"] > hours * 3600).sum())
            for hours in landmark_hours
        }
        eligible_users[split] = {
            str(hours): int(
                selected.loc[selected["delta_t_sec"] > hours * 3600, "raw_user_id"].nunique()
            )
            for hours in landmark_hours
        }

    gap_values = split_gaps["delta_t_sec"].to_numpy(dtype=np.int64)
    levels = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    quantiles = np.quantile(gap_values, levels) if len(gap_values) else np.zeros(len(levels))
    timestamp_poi_counts = events.groupby(["raw_user_id", "event_time_utc_s"], observed=True)[
        "raw_poi_id"
    ].nunique()
    multi_poi_groups = timestamp_poi_counts[timestamp_poi_counts > 1]
    final_events = events.groupby("raw_user_id", sort=False, observed=True).tail(1)
    dataset_end_s = int(events["event_time_utc_s"].max())
    test_period_start_s = split_period_start_time_s(split_gaps, "test")
    final_event_times = final_events["event_time_utc_s"].to_numpy(dtype=np.int64)
    censored_counts: dict[str, int] = {}
    censored_query_min_s: int | None = None
    censored_query_max_s: int | None = None
    for hours in landmark_hours:
        query_times = final_event_times + hours * 3600
        eligible_query_times = query_times[
            (query_times >= test_period_start_s) & (query_times <= dataset_end_s)
        ]
        censored_counts[str(hours)] = len(eligible_query_times)
        if len(eligible_query_times):
            horizon_min = int(eligible_query_times.min())
            horizon_max = int(eligible_query_times.max())
            censored_query_min_s = (
                horizon_min
                if censored_query_min_s is None
                else min(censored_query_min_s, horizon_min)
            )
            censored_query_max_s = (
                horizon_max
                if censored_query_max_s is None
                else max(censored_query_max_s, horizon_max)
            )
    cleaning_counts = asdict(cleaning)
    coverage = dict(coverage_notes or {})
    coverage.setdefault("limitations", [])
    coverage.setdefault("registered_cell_protocols", {})
    coverage["missingness_counts"] = {
        key: value
        for key, value in cleaning_counts.items()
        if key.startswith(("missing_", "invalid_poi_metadata"))
    }
    return {
        "dataset": dataset_name,
        "protocol_version": protocol_version,
        "cleaning": cleaning_counts,
        "overall": {
            "events": len(events),
            "users": int(events["raw_user_id"].nunique()),
            "pois": int(events["raw_poi_id"].nunique()),
            "completed_gaps": len(split_gaps),
            "dataset_start_time_s": int(events["event_time_utc_s"].min()),
            "dataset_end_time_s": dataset_end_s,
        },
        "by_split": by_split,
        "gap_quantiles": {
            str(level): {"seconds": int(value), "hours": float(value / 3600)}
            for level, value in zip(levels, quantiles, strict=True)
        },
        "eligible_observed_samples_by_split_and_landmark_hours": eligible,
        "eligible_unique_users_by_split_and_landmark_hours": eligible_users,
        "censored_samples_by_landmark_hours": censored_counts,
        "censored_query_time_range_s": {
            "minimum": censored_query_min_s,
            "maximum": censored_query_max_s,
            "test_period_start": test_period_start_s,
        },
        "same_timestamp_multi_poi": {
            "user_timestamp_groups": len(multi_poi_groups),
            "events_in_groups": int(
                events.set_index(["raw_user_id", "event_time_utc_s"])
                .index.isin(multi_poi_groups.index)
                .sum()
            ),
        },
        "vocabulary": vocabulary_metadata,
        "coverage": coverage,
        "derived_cell_support_by_protocol": coverage["registered_cell_protocols"],
        "split_manifest_sha256": split_manifest_hash,
        "notes": {
            "unknown_id": UNKNOWN_POI_ID,
            "padding_id": PADDING_POI_ID,
            "censored_split": "test queries at or after the completed-gap test-period start",
            "censor_time_definition": "follow-up remaining after the query landmark",
            "ranking_target_policy": (
                "Recall, NDCG, and POI NLL use observed targets in the training POI catalog only"
            ),
        },
    }
