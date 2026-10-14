"""Leakage-safe training-only administrative-censor profile for Issue #39."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from adrec_still.data.audit import sha256_file
from adrec_still.data.landmarks import LANDMARK_SECONDS, write_landmarks
from adrec_still.data.schema import UNKNOWN_POI_ID
from adrec_still.data.split import make_gap_id

SURVIVAL_PROFILE_VERSION = "still-train-admin-censor-v1"


def _digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _training_projection(
    events: pd.DataFrame, train_gaps: pd.DataFrame, cutoff: int
) -> tuple[pd.DataFrame, int]:
    """Return the disclosed train-source union strictly before ``cutoff``.

    Membership is established before the strict time filter. Every event at the
    boundary second is excluded, including an endpoint named by a completed train gap.
    """
    source_ids = set(train_gaps["source_row_id"].astype(int)) | set(
        train_gaps["next_source_row_id"].astype(int)
    )
    source_union = events.loc[events["source_row_id"].isin(source_ids)].copy()
    if set(source_union["source_row_id"].astype(int)) != source_ids:
        raise ValueError("training split references missing event source rows")
    if source_union["source_row_id"].duplicated().any():
        raise ValueError("event source rows must be unique")
    boundary_count = int(source_union["event_time_utc_s"].eq(cutoff).sum())
    selected = source_union.loc[source_union["event_time_utc_s"] < cutoff].copy()
    if selected.empty:
        raise ValueError("strict training cutoff leaves no disclosed events")
    return selected.sort_values(
        ["raw_user_id", "event_time_utc_s", "source_row_id"], kind="mergesort"
    ).reset_index(drop=True), boundary_count


def build_survival_training_profile(
    events_path: str | Path,
    split_manifest_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    landmark_seconds: tuple[int, ...] = LANDMARK_SECONDS,
    max_history_length: int | None = None,
) -> dict:
    """Materialize the accepted training view and return its immutable manifest.

    The cutoff is the largest endpoint timestamp among train gaps. The disclosed
    train-source union is then filtered with strict ``event_time < cutoff``. Completed
    gaps are rebuilt from adjacent events in that visible per-user prefix; each user's
    terminal visible event contributes administrative-censor landmarks with strictly
    positive remaining follow-up. Events outside that prefix never enter features,
    targets, identities, or hashes.
    """
    schedule = tuple(sorted(landmark_seconds))
    if (
        not schedule
        or len(set(schedule)) != len(schedule)
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in schedule)
    ):
        raise ValueError("landmark_seconds must be distinct non-negative integers")
    if max_history_length is not None and max_history_length <= 0:
        raise ValueError("max_history_length must be positive when provided")

    event_columns = [
        "dataset",
        "raw_user_id",
        "user_id",
        "poi_id",
        "event_time_utc_s",
        "source_row_id",
    ]
    events = pd.read_parquet(events_path, columns=event_columns)
    gaps = pd.read_parquet(split_manifest_path)
    train = gaps.loc[gaps["split"].astype(str) == "train"].copy()
    if train.empty:
        raise ValueError("survival training profile requires completed train gaps")
    cutoff = int(train["next_event_time_s"].max())
    disclosed, boundary_event_count = _training_projection(events, train, cutoff)
    datasets = disclosed["dataset"].astype(str).unique().tolist()
    if len(datasets) != 1:
        raise ValueError("training events must belong to exactly one dataset")
    dataset = datasets[0]
    counts: Counter[str] = Counter()
    rebuilt_gaps: list[dict[str, Any]] = []

    def records() -> Iterator[dict[str, Any]]:
        for _, user_events in disclosed.groupby("raw_user_id", sort=False, observed=True):
            user_events = user_events.reset_index(drop=True)
            for index in range(len(user_events) - 1):
                event = user_events.iloc[index]
                target = user_events.iloc[index + 1]
                history_start = (
                    0
                    if max_history_length is None
                    else max(0, index + 1 - max_history_length)
                )
                history = user_events.iloc[history_start : index + 1]
                duration = int(target["event_time_utc_s"]) - int(event["event_time_utc_s"])
                if duration < 0:
                    raise ValueError("training events are not chronological within a user")
                gap_id = make_gap_id(
                    int(event["source_row_id"]), int(target["source_row_id"]), dataset
                )
                rebuilt_gaps.append(
                    {
                        "gap_id": gap_id,
                        "raw_user_id": str(event["raw_user_id"]),
                        "source_row_id": int(event["source_row_id"]),
                        "next_source_row_id": int(target["source_row_id"]),
                        "event_time_utc_s": int(event["event_time_utc_s"]),
                        "next_event_time_s": int(target["event_time_utc_s"]),
                        "delta_t_sec": duration,
                    }
                )
                for query in schedule:
                    if duration <= query:
                        continue
                    counts["observed_rows"] += 1
                    yield {
                        "sample_id": f"{gap_id}:q{query:09d}",
                        "gap_id": gap_id,
                        "user_id": int(event["user_id"]),
                        "split": "train",
                        "history_poi": history["poi_id"].astype(int).tolist(),
                        "history_timestamp": history["event_time_utc_s"].astype(int).tolist(),
                        "last_timestamp": int(event["event_time_utc_s"]),
                        "query_s_sec": query,
                        "event_observed": True,
                        "target_in_train_catalog": int(target["poi_id"]) != UNKNOWN_POI_ID,
                        "residual_time_sec": duration - query,
                        "next_poi": int(target["poi_id"]),
                        "censor_time_sec": None,
                    }

            final = user_events.iloc[-1]
            exposure = cutoff - int(final["event_time_utc_s"])
            if exposure <= 0:
                counts["zero_followup_gaps_excluded"] += 1
                continue
            gap_id = f"{dataset}:{int(final['source_row_id']):012d}:train-cutoff-{cutoff}"
            history_start = (
                0
                if max_history_length is None
                else max(0, len(user_events) - max_history_length)
            )
            history = user_events.iloc[history_start:]
            emitted = False
            for query in schedule:
                remaining = exposure - query
                if remaining == 0:
                    counts["zero_followup_landmarks_excluded"] += 1
                    continue
                if remaining < 0:
                    counts["outside_followup_landmarks_excluded"] += 1
                    continue
                emitted = True
                counts["censored_rows"] += 1
                yield {
                    "sample_id": f"{gap_id}:q{query:09d}",
                    "gap_id": gap_id,
                    "user_id": int(final["user_id"]),
                    "split": "train",
                    "history_poi": history["poi_id"].astype(int).tolist(),
                    "history_timestamp": history["event_time_utc_s"].astype(int).tolist(),
                    "last_timestamp": int(final["event_time_utc_s"]),
                    "query_s_sec": query,
                    "event_observed": False,
                    "target_in_train_catalog": False,
                    "residual_time_sec": None,
                    "next_poi": None,
                    "censor_time_sec": remaining,
                }
            counts["censored_gaps"] += int(emitted)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    row_count = write_landmarks(records(), output)
    if row_count != counts["observed_rows"] + counts["censored_rows"]:
        raise RuntimeError("survival profile row accounting mismatch")

    projection_records = disclosed[event_columns].to_dict(orient="records")
    rebuilt_gaps.sort(key=lambda row: row["gap_id"])
    metadata = {
        "version": SURVIVAL_PROFILE_VERSION,
        "dataset": dataset,
        "time_unit": "seconds",
        "split": "train",
        "cutoff_time_sec": cutoff,
        "cutoff_rule": "max train-gap endpoint; retain disclosed source rows only when t < cutoff",
        "censor_rule": "last strict-prefix user event to cutoff; require cutoff > anchor + query",
        "zero_followup_rule": "exclude landmarks with cutoff == anchor + query",
        "landmark_seconds": list(schedule),
        "max_history_length": max_history_length,
        "training_event_projection_hash": _digest(projection_records),
        "rebuilt_completed_gap_hash": _digest(rebuilt_gaps),
        "training_event_count": len(disclosed),
        "boundary_event_count": boundary_event_count,
        "completed_gap_count": len(rebuilt_gaps),
        "censored_gap_count": counts["censored_gaps"],
        "zero_followup_gap_count": counts["zero_followup_gaps_excluded"],
        "zero_followup_landmark_count": counts["zero_followup_landmarks_excluded"],
        "outside_followup_landmark_count": counts["outside_followup_landmarks_excluded"],
        "exclusion_reasons": {
            "boundary_events": "event_time_utc_s == cutoff_time_sec",
            "zero_followup_landmarks": "cutoff_time_sec == last_timestamp + query_s_sec",
            "outside_followup_landmarks": "cutoff_time_sec < last_timestamp + query_s_sec",
        },
        "observed_row_count": counts["observed_rows"],
        "censored_row_count": counts["censored_rows"],
        "row_count": row_count,
        "landmarks_sha256": sha256_file(output),
    }
    manifest = {**metadata, "profile_id": _digest(metadata)}
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest
