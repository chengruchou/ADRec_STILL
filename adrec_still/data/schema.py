"""Canonical data structures shared by baselines and STILL."""

from collections.abc import Sequence
from dataclasses import dataclass

PADDING_POI_ID = 0
UNKNOWN_POI_ID = 1


@dataclass(frozen=True)
class CanonicalSample:
    user_id: int
    history_poi: Sequence[int]
    history_timestamp: Sequence[int]
    last_timestamp: int
    query_s_sec: int
    event_observed: bool
    target_in_train_catalog: bool
    residual_time_sec: int | None = None
    next_poi: int | None = None
    spatial_label: str | None = None
    temporal_label: str | None = None
    cell_label: str | None = None
    censor_time_sec: int | None = None

    def validate(self) -> None:
        if len(self.history_poi) != len(self.history_timestamp):
            raise ValueError("history_poi and history_timestamp must have equal length")
        if self.query_s_sec < 0:
            raise ValueError("query_s_sec must be non-negative")
        if self.event_observed:
            if self.residual_time_sec is None or self.residual_time_sec <= 0:
                raise ValueError("observed samples require positive residual_time_sec")
            if self.next_poi is None:
                raise ValueError("observed samples require next_poi")
            if self.next_poi == PADDING_POI_ID:
                raise ValueError("observed samples cannot target the padding POI")
            expected_known = self.next_poi != UNKNOWN_POI_ID
            if self.target_in_train_catalog != expected_known:
                raise ValueError(
                    "target_in_train_catalog must match the frozen training-vocabulary encoding"
                )
            if self.censor_time_sec is not None:
                raise ValueError("observed samples must not define censor_time_sec")
        else:
            if self.next_poi is not None or self.residual_time_sec is not None:
                raise ValueError("censored samples must not fabricate event targets")
            if self.cell_label is not None:
                raise ValueError("censored samples must not fabricate completed cell labels")
            if self.target_in_train_catalog:
                raise ValueError("censored samples cannot define a known next-POI target")
            if self.censor_time_sec is None or self.censor_time_sec < 0:
                raise ValueError("censored samples require censor_time_sec")
