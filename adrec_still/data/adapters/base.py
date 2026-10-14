"""Shared types for deterministic raw-dataset adapters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CleaningAudit:
    """Counts produced while validating and cleaning canonical event fields."""

    raw_rows: int
    retained_rows: int
    invalid_timestamp_rows: int
    invalid_coordinate_rows: int
    invalid_identifier_rows: int
    duplicate_rows: int
