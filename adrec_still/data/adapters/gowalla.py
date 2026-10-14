"""Deterministic parser and cleaner for raw Gowalla check-ins."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from adrec_still.data.adapters.base import CleaningAudit

RAW_COLUMNS = (
    "raw_user_id",
    "timestamp_raw",
    "latitude_raw",
    "longitude_raw",
    "raw_poi_id",
)


def _read_raw(input_path: Path, row_limit: int | None = None) -> pd.DataFrame:
    return pd.read_csv(
        input_path,
        sep="\t",
        header=None,
        names=RAW_COLUMNS,
        dtype={column: "string[pyarrow]" for column in RAW_COLUMNS},
        keep_default_na=False,
        na_filter=False,
        nrows=row_limit,
    )


def parse_and_clean_events(
    input_path: str | Path,
    row_limit: int | None = None,
) -> tuple[pd.DataFrame, CleaningAudit]:
    """Parse, validate, exactly deduplicate, and deterministically sort events."""
    raw = _read_raw(Path(input_path), row_limit=row_limit)
    raw.insert(0, "source_row_id", np.arange(len(raw), dtype=np.int64))

    timestamps = pd.to_datetime(raw["timestamp_raw"], utc=True, errors="coerce")
    latitude = pd.to_numeric(raw["latitude_raw"], errors="coerce")
    longitude = pd.to_numeric(raw["longitude_raw"], errors="coerce")
    invalid_timestamp = timestamps.isna()
    invalid_coordinate = (
        latitude.isna()
        | longitude.isna()
        | ~latitude.between(-90.0, 90.0, inclusive="both")
        | ~longitude.between(-180.0, 180.0, inclusive="both")
    )
    invalid_identifier = raw["raw_user_id"].str.strip().eq("") | raw["raw_poi_id"].str.strip().eq(
        ""
    )

    valid_mask = ~(invalid_timestamp | invalid_coordinate | invalid_identifier)
    valid = raw.loc[valid_mask].copy()
    # Explicit seconds avoid pandas-version-dependent datetime storage units.
    valid["event_time_utc_s"] = timestamps.loc[valid_mask].dt.as_unit("s").astype("int64")
    valid["latitude"] = latitude.loc[valid_mask].astype("float64")
    valid["longitude"] = longitude.loc[valid_mask].astype("float64")

    duplicate_mask = valid.duplicated(subset=list(RAW_COLUMNS), keep="first")
    duplicate_rows = int(duplicate_mask.sum())
    valid = valid.loc[~duplicate_mask].copy()
    valid = valid.sort_values(
        ["raw_user_id", "event_time_utc_s", "source_row_id"], kind="mergesort"
    ).reset_index(drop=True)
    valid["event_index"] = valid.groupby("raw_user_id", sort=False).cumcount().astype("int32")

    events = valid[
        [
            "raw_user_id",
            "raw_poi_id",
            "event_time_utc_s",
            "latitude",
            "longitude",
            "event_index",
            "source_row_id",
        ]
    ]
    audit = CleaningAudit(
        raw_rows=len(raw),
        retained_rows=len(events),
        invalid_timestamp_rows=int(invalid_timestamp.sum()),
        invalid_coordinate_rows=int(invalid_coordinate.sum()),
        invalid_identifier_rows=int(invalid_identifier.sum()),
        duplicate_rows=duplicate_rows,
    )
    return events, audit
