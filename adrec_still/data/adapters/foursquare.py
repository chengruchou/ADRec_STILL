"""Deterministic adapters for the TIST2015 and TSMC2014 Foursquare datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from adrec_still.data.adapters.base import CleaningAudit

GLOBAL_CHECKIN_COLUMNS = (
    "raw_user_id",
    "raw_poi_id",
    "timestamp_raw",
    "timezone_offset_raw",
)
GLOBAL_POI_COLUMNS = (
    "raw_poi_id",
    "latitude_raw",
    "longitude_raw",
    "category_name_raw",
    "country_code_raw",
)
SUBSET_COLUMNS = (
    "raw_user_id",
    "raw_poi_id",
    "category_id_raw",
    "category_name_raw",
    "latitude_raw",
    "longitude_raw",
    "timezone_offset_raw",
    "timestamp_raw",
)
_MONTH_NUMBERS = {
    "Jan": "01",
    "Feb": "02",
    "Mar": "03",
    "Apr": "04",
    "May": "05",
    "Jun": "06",
    "Jul": "07",
    "Aug": "08",
    "Sep": "09",
    "Oct": "10",
    "Nov": "11",
    "Dec": "12",
}
_TIMESTAMP_PATTERN = (
    r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) "
    r"(?P<month>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) "
    r"(?P<day>\d{1,2}) (?P<clock>\d{2}:\d{2}:\d{2}) "
    r"(?P<offset>[+-]\d{4}) (?P<year>\d{4})$"
)


@dataclass(frozen=True)
class FoursquareCleaningAudit(CleaningAudit):
    """Canonical cleaning counts plus Foursquare coverage/missingness counts."""

    invalid_timezone_rows: int
    missing_poi_metadata_rows: int
    missing_category_id_rows: int
    missing_category_name_rows: int
    missing_country_code_rows: int
    poi_metadata_rows: int
    invalid_poi_metadata_rows: int
    duplicate_poi_metadata_rows: int
    poi_coordinate_conflict_count: int


def _read_tsv(
    input_path: Path,
    columns: tuple[str, ...],
    *,
    encoding: str,
    row_limit: int | None = None,
    chunksize: int | None = None,
) -> pd.DataFrame | pd.io.parsers.TextFileReader:
    return pd.read_csv(
        input_path,
        sep="\t",
        header=None,
        names=columns,
        dtype={column: "string[pyarrow]" for column in columns},
        keep_default_na=False,
        na_filter=False,
        encoding=encoding,
        nrows=row_limit,
        chunksize=chunksize,
    )


def _parse_utc_timestamps(values: pd.Series) -> pd.Series:
    """Parse the fixed English Foursquare timestamp format without locale dependence."""
    parts = values.str.strip().str.extract(_TIMESTAMP_PATTERN)
    normalized = (
        parts["year"]
        + "-"
        + parts["month"].map(_MONTH_NUMBERS)
        + "-"
        + parts["day"].str.zfill(2)
        + "T"
        + parts["clock"]
        + parts["offset"]
    )
    return pd.to_datetime(
        normalized,
        format="%Y-%m-%dT%H:%M:%S%z",
        utc=True,
        errors="coerce",
    )


def _invalid_timezone_mask(values: pd.Series) -> pd.Series:
    offsets = pd.to_numeric(values, errors="coerce")
    return (
        offsets.isna() | ~offsets.between(-14 * 60, 14 * 60, inclusive="both") | (offsets % 1 != 0)
    )


def _invalid_coordinate_mask(latitude: pd.Series, longitude: pd.Series) -> pd.Series:
    return (
        latitude.isna()
        | longitude.isna()
        | ~latitude.between(-90.0, 90.0, inclusive="both")
        | ~longitude.between(-180.0, 180.0, inclusive="both")
    )


def _finalize_events(valid: pd.DataFrame) -> pd.DataFrame:
    valid = valid.sort_values(
        ["raw_user_id", "event_time_utc_s", "source_row_id"], kind="mergesort"
    ).reset_index(drop=True)
    valid["event_index"] = valid.groupby("raw_user_id", sort=False).cumcount().astype("int32")
    return valid[
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


def _load_global_poi_metadata(
    poi_path: Path,
    needed_poi_ids: pd.Index,
    chunk_size: int = 500_000,
) -> tuple[pd.DataFrame, set[str], set[str], set[str], dict[str, int]]:
    selected_chunks: list[pd.DataFrame] = []
    seen_poi_ids: set[str] = set()
    category_poi_ids: set[str] = set()
    country_poi_ids: set[str] = set()
    raw_row_count = 0
    reader = _read_tsv(
        poi_path,
        GLOBAL_POI_COLUMNS,
        encoding="utf-8",
        chunksize=chunk_size,
    )
    if not isinstance(reader, pd.io.parsers.TextFileReader):
        raise TypeError("chunked POI parsing did not return a TextFileReader")
    for chunk in reader:
        raw_row_count += len(chunk)
        selected = chunk.loc[chunk["raw_poi_id"].isin(needed_poi_ids)].copy()
        if selected.empty:
            continue
        seen_poi_ids.update(selected["raw_poi_id"].tolist())
        category_poi_ids.update(
            selected.loc[selected["category_name_raw"].str.strip().ne(""), "raw_poi_id"].tolist()
        )
        country_poi_ids.update(
            selected.loc[selected["country_code_raw"].str.strip().ne(""), "raw_poi_id"].tolist()
        )
        selected_chunks.append(selected)

    if selected_chunks:
        metadata = pd.concat(selected_chunks, ignore_index=True)
    else:
        metadata = pd.DataFrame(columns=GLOBAL_POI_COLUMNS).astype("string[pyarrow]")
    latitude = pd.to_numeric(metadata["latitude_raw"], errors="coerce")
    longitude = pd.to_numeric(metadata["longitude_raw"], errors="coerce")
    invalid_identifier = metadata["raw_poi_id"].str.strip().eq("")
    invalid_coordinate = _invalid_coordinate_mask(latitude, longitude)
    valid_mask = ~(invalid_identifier | invalid_coordinate)
    valid = metadata.loc[valid_mask].copy()
    valid["latitude"] = latitude.loc[valid_mask].astype("float64")
    valid["longitude"] = longitude.loc[valid_mask].astype("float64")
    duplicate_mask = valid.duplicated(subset=list(GLOBAL_POI_COLUMNS), keep="first")
    duplicate_count = int(duplicate_mask.sum())
    valid = valid.loc[~duplicate_mask].copy()

    coordinate_counts = (
        valid.groupby(["raw_poi_id", "latitude", "longitude"], observed=True)
        .size()
        .rename("coordinate_count")
        .reset_index()
    )
    coordinate_variants = coordinate_counts.groupby("raw_poi_id", observed=True).size()
    coordinate_counts = coordinate_counts.sort_values(
        ["raw_poi_id", "coordinate_count", "latitude", "longitude"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    canonical = coordinate_counts.drop_duplicates("raw_poi_id", keep="first")
    canonical = canonical[["raw_poi_id", "latitude", "longitude"]]
    counts = {
        "poi_metadata_rows": raw_row_count,
        "invalid_poi_metadata_rows": int((invalid_identifier | invalid_coordinate).sum()),
        "duplicate_poi_metadata_rows": duplicate_count,
        "poi_coordinate_conflict_count": int((coordinate_variants > 1).sum()),
    }
    return canonical, seen_poi_ids, category_poi_ids, country_poi_ids, counts


def parse_and_clean_global_events(
    checkin_path: str | Path,
    poi_path: str | Path,
    row_limit: int | None = None,
) -> tuple[pd.DataFrame, FoursquareCleaningAudit]:
    """Parse TIST2015 check-ins and join deterministic coordinates from its POI table."""
    raw = _read_tsv(
        Path(checkin_path),
        GLOBAL_CHECKIN_COLUMNS,
        encoding="utf-8",
        row_limit=row_limit,
    )
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("non-chunked check-in parsing did not return a DataFrame")
    raw.insert(0, "source_row_id", np.arange(len(raw), dtype=np.int64))
    needed_poi_ids = pd.Index(raw["raw_poi_id"].drop_duplicates())
    (
        canonical_pois,
        seen_poi_ids,
        category_poi_ids,
        country_poi_ids,
        poi_counts,
    ) = _load_global_poi_metadata(Path(poi_path), needed_poi_ids)

    timestamps = _parse_utc_timestamps(raw["timestamp_raw"])
    invalid_timestamp = timestamps.isna()
    invalid_timezone = _invalid_timezone_mask(raw["timezone_offset_raw"])
    invalid_identifier = raw["raw_user_id"].str.strip().eq("") | raw["raw_poi_id"].str.strip().eq(
        ""
    )
    missing_poi_metadata = ~raw["raw_poi_id"].isin(seen_poi_ids)
    joined = raw.merge(
        canonical_pois, on="raw_poi_id", how="left", sort=False, validate="many_to_one"
    )
    invalid_coordinate = (
        joined["latitude"].isna() | joined["longitude"].isna()
    ) & ~missing_poi_metadata.to_numpy()
    missing_category_name = raw["raw_poi_id"].isin(seen_poi_ids) & ~raw["raw_poi_id"].isin(
        category_poi_ids
    )
    missing_country_code = raw["raw_poi_id"].isin(seen_poi_ids) & ~raw["raw_poi_id"].isin(
        country_poi_ids
    )

    valid_mask = ~(
        invalid_timestamp
        | invalid_identifier
        | missing_poi_metadata
        | invalid_coordinate.to_numpy()
    )
    valid = joined.loc[valid_mask].copy()
    valid["event_time_utc_s"] = timestamps.loc[valid_mask].dt.as_unit("s").astype("int64")
    duplicate_mask = valid.duplicated(subset=list(GLOBAL_CHECKIN_COLUMNS), keep="first")
    duplicate_rows = int(duplicate_mask.sum())
    valid = valid.loc[~duplicate_mask].copy()
    events = _finalize_events(valid)
    audit = FoursquareCleaningAudit(
        raw_rows=len(raw),
        retained_rows=len(events),
        invalid_timestamp_rows=int(invalid_timestamp.sum()),
        invalid_coordinate_rows=int(invalid_coordinate.sum()),
        invalid_identifier_rows=int(invalid_identifier.sum()),
        duplicate_rows=duplicate_rows,
        invalid_timezone_rows=int(invalid_timezone.sum()),
        missing_poi_metadata_rows=int(missing_poi_metadata.sum()),
        missing_category_id_rows=0,
        missing_category_name_rows=int(missing_category_name.sum()),
        missing_country_code_rows=int(missing_country_code.sum()),
        **poi_counts,
    )
    return events, audit


def parse_and_clean_subset_events(
    input_path: str | Path,
    row_limit: int | None = None,
) -> tuple[pd.DataFrame, FoursquareCleaningAudit]:
    """Parse one self-contained TSMC2014 NYC or TKY check-in file."""
    raw = _read_tsv(
        Path(input_path),
        SUBSET_COLUMNS,
        encoding="latin-1",
        row_limit=row_limit,
    )
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("non-chunked subset parsing did not return a DataFrame")
    raw.insert(0, "source_row_id", np.arange(len(raw), dtype=np.int64))
    timestamps = _parse_utc_timestamps(raw["timestamp_raw"])
    latitude = pd.to_numeric(raw["latitude_raw"], errors="coerce")
    longitude = pd.to_numeric(raw["longitude_raw"], errors="coerce")
    invalid_timestamp = timestamps.isna()
    invalid_coordinate = _invalid_coordinate_mask(latitude, longitude)
    invalid_timezone = _invalid_timezone_mask(raw["timezone_offset_raw"])
    invalid_identifier = raw["raw_user_id"].str.strip().eq("") | raw["raw_poi_id"].str.strip().eq(
        ""
    )
    missing_category_id = raw["category_id_raw"].str.strip().eq("")
    missing_category_name = raw["category_name_raw"].str.strip().eq("")

    valid_mask = ~(invalid_timestamp | invalid_coordinate | invalid_identifier)
    valid = raw.loc[valid_mask].copy()
    valid["event_time_utc_s"] = timestamps.loc[valid_mask].dt.as_unit("s").astype("int64")
    valid["latitude"] = latitude.loc[valid_mask].astype("float64")
    valid["longitude"] = longitude.loc[valid_mask].astype("float64")
    duplicate_mask = valid.duplicated(subset=list(SUBSET_COLUMNS), keep="first")
    duplicate_rows = int(duplicate_mask.sum())
    valid = valid.loc[~duplicate_mask].copy()
    coordinate_variants = (
        valid[["raw_poi_id", "latitude", "longitude"]]
        .drop_duplicates()
        .groupby("raw_poi_id", observed=True)
        .size()
    )
    events = _finalize_events(valid)
    audit = FoursquareCleaningAudit(
        raw_rows=len(raw),
        retained_rows=len(events),
        invalid_timestamp_rows=int(invalid_timestamp.sum()),
        invalid_coordinate_rows=int(invalid_coordinate.sum()),
        invalid_identifier_rows=int(invalid_identifier.sum()),
        duplicate_rows=duplicate_rows,
        invalid_timezone_rows=int(invalid_timezone.sum()),
        missing_poi_metadata_rows=0,
        missing_category_id_rows=int(missing_category_id.sum()),
        missing_category_name_rows=int(missing_category_name.sum()),
        missing_country_code_rows=0,
        poi_metadata_rows=0,
        invalid_poi_metadata_rows=0,
        duplicate_poi_metadata_rows=0,
        poi_coordinate_conflict_count=int((coordinate_variants > 1).sum()),
    )
    return events, audit
