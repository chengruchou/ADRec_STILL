"""Full per-user event memory, independent of the encoder's truncated window.

The landmark rows carry at most ``max_history_length`` events. Repeat semantics
must not depend on that window (a return to a POI last visited before the window
is still a repeat; two different raw IDs that both encode to UNK are not), so
this index keeps every user's complete event list from ``events.parquet`` —
raw IDs for labels, frozen-vocabulary IDs for features, timestamps for recency,
and canonical catalog coordinates for the activity-center ("home") axis.

Everything answered here is causal: a query names an anchor event (the
``gap_id`` the canonical rows already carry) and only events at positions up to
that anchor are read. The repeat *label* of a gap concerns the gap's target
event and is label-side information: models receive only prefix-side features;
evaluation and training-loss code read labels through separate methods.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SeenState:
    """Known-item memory of one prefix: unique IDs, counts, last-visit times."""

    poi_ids: np.ndarray        # int64 [m], frozen-vocabulary IDs >= 2
    counts: np.ndarray         # int64 [m]
    last_visit_s: np.ndarray   # int64 [m]

    @property
    def size(self) -> int:
        return int(self.poi_ids.shape[0])


class FullHistoryIndex:
    """Complete causal event memory with anchor lookups by ``gap_id``."""

    def __init__(self, events_path: str | Path, catalog_path: str | Path | None = None,
                 *, train_cutoff_s: int | None = None) -> None:
        events = pd.read_parquet(
            events_path,
            columns=["raw_user_id", "user_id", "raw_poi_id", "poi_id",
                     "event_time_utc_s", "source_row_id"],
        ).sort_values(["raw_user_id", "event_time_utc_s", "source_row_id"],
                      kind="mergesort")
        raw_codes, _ = pd.factorize(events["raw_poi_id"], sort=False)
        self._user_poi: list[np.ndarray] = []
        self._user_raw: list[np.ndarray] = []
        self._user_time: list[np.ndarray] = []
        self._user_is_repeat: list[np.ndarray] = []
        self._source_to_anchor: dict[int, tuple[int, int]] = {}
        self._source_to_raw: dict[int, int] = {}
        start = 0
        events = events.reset_index(drop=True)
        user_values = events["raw_user_id"].to_numpy()
        poi_values = events["poi_id"].to_numpy(dtype=np.int64)
        time_values = events["event_time_utc_s"].to_numpy(dtype=np.int64)
        source_values = events["source_row_id"].to_numpy(dtype=np.int64)
        boundaries = np.flatnonzero(user_values[1:] != user_values[:-1]) + 1
        for stop in [*boundaries.tolist(), len(events)]:
            user_index = len(self._user_poi)
            raw = raw_codes[start:stop]
            self._user_poi.append(poi_values[start:stop])
            self._user_raw.append(np.asarray(raw, dtype=np.int64))
            self._user_time.append(time_values[start:stop])
            seen: set[int] = set()
            is_repeat = np.zeros(stop - start, dtype=bool)
            for position, code in enumerate(raw):
                if code in seen:
                    is_repeat[position] = True
                else:
                    seen.add(int(code))
            self._user_is_repeat.append(is_repeat)
            for position, source in enumerate(source_values[start:stop]):
                self._source_to_anchor[int(source)] = (user_index, position)
                self._source_to_raw[int(source)] = int(raw[position])
            start = stop
        self.train_cutoff_s = train_cutoff_s
        self._coordinates: dict[int, tuple[float, float]] | None = None
        if catalog_path is not None and Path(catalog_path).is_file():
            catalog = pd.read_parquet(catalog_path)
            if {"poi_id", "latitude", "longitude"} <= set(catalog.columns):
                self._coordinates = {
                    int(row.poi_id): (float(row.latitude), float(row.longitude))
                    for row in catalog.itertuples()
                    if np.isfinite(row.latitude) and np.isfinite(row.longitude)
                }

    # ------------------------------------------------------------------ anchors

    @staticmethod
    def parse_gap_id(gap_id: str) -> tuple[int, int | None]:
        """``(anchor_source_row, target_source_row | None)`` from a gap identity."""
        parts = gap_id.split(":")
        if len(parts) < 3:
            raise ValueError(f"not a gap identity: {gap_id!r}")
        anchor = int(parts[-2])
        tail = parts[-1]
        target = int(tail) if tail.isdigit() else None
        return anchor, target

    def anchor(self, gap_id: str) -> tuple[int, int]:
        """(user_index, position) of the gap's anchor event."""
        source, _ = self.parse_gap_id(gap_id)
        try:
            return self._source_to_anchor[source]
        except KeyError:
            raise KeyError(
                f"anchor source row {source} (gap {gap_id!r}) is not in the events table; "
                "the landmark rows and events.parquet must come from the same preprocessing"
            ) from None

    # ------------------------------------------------------------------- labels

    def repeat_label(self, gap_id: str) -> bool | None:
        """Whether the gap's target raw ID was seen anywhere in the full prefix.

        Judged on raw IDs over the complete per-user past: a return from beyond
        the encoder window still counts, and two distinct out-of-vocabulary raw
        IDs never match through their shared UNK encoding. ``None`` for censored
        gaps (no target exists).
        """
        _, target_source = self.parse_gap_id(gap_id)
        if target_source is None:
            return None
        user, _position = self.anchor(gap_id)
        target_anchor = self._source_to_anchor.get(target_source)
        if target_anchor is None or target_anchor[0] != user:
            raise KeyError(f"target source row of gap {gap_id!r} is not the same user's event")
        return bool(self._user_is_repeat[user][target_anchor[1]])

    def target_poi(self, gap_id: str) -> int | None:
        """The target event's frozen-vocabulary POI ID (None for censored gaps)."""
        _, target_source = self.parse_gap_id(gap_id)
        if target_source is None:
            return None
        user, position = self._source_to_anchor[target_source]
        return int(self._user_poi[user][position])

    def gap_duration_s(self, gap_id: str) -> int | None:
        """Total gap T in seconds (None for censored gaps)."""
        _, target_source = self.parse_gap_id(gap_id)
        if target_source is None:
            return None
        user, position = self.anchor(gap_id)
        target_user, target_position = self._source_to_anchor[target_source]
        return int(self._user_time[target_user][target_position]
                   - self._user_time[user][position])

    # ----------------------------------------------------------------- features

    def seen_state(self, gap_id: str) -> SeenState:
        """Known-item memory of the anchor's full prefix (anchor event included)."""
        user, position = self.anchor(gap_id)
        prefix_poi = self._user_poi[user][: position + 1]
        prefix_time = self._user_time[user][: position + 1]
        unique, counts = np.unique(prefix_poi, return_counts=True)
        _reversed_unique, reversed_first = np.unique(prefix_poi[::-1], return_index=True)
        last_index = (len(prefix_poi) - 1) - reversed_first
        known = unique >= 2
        return SeenState(
            poi_ids=unique[known].astype(np.int64),
            counts=counts[known].astype(np.int64),
            last_visit_s=prefix_time[last_index[known]].astype(np.int64),
        )

    def home(self, gap_id: str, *, policy: str = "causal") -> tuple[float, float] | None:
        """Activity center of the anchor's causal prefix, or the frozen train-cutoff one.

        The center is the most-visited geocoded known POI (ties: smaller POI ID),
        computed from events at positions up to the anchor ("causal") or up to the
        last event strictly before the train cutoff ("frozen", the evaluation
        policy). ``None`` when no geocoded known POI exists in the chosen prefix —
        including a frozen query for a user with no pre-cutoff event, so callers
        can fall back to the causal prefix *explicitly* and count that fallback.
        """
        if self._coordinates is None:
            return None
        user, position = self.anchor(gap_id)
        if policy == "frozen":
            if self.train_cutoff_s is None:
                raise ValueError("the frozen home policy needs a train cutoff")
            frozen_stop = bisect.bisect_left(
                self._user_time[user].tolist(), int(self.train_cutoff_s)
            )
            if frozen_stop == 0:
                return None  # no pre-cutoff event: the caller owns the fallback
            position = min(position, frozen_stop - 1)
        elif policy != "causal":
            raise ValueError(f"unknown home policy: {policy!r}")
        prefix = self._user_poi[user][: position + 1]
        known = prefix[prefix >= 2]
        if not len(known):
            return None
        unique, counts = np.unique(known, return_counts=True)
        geocoded = np.array([int(poi) in self._coordinates for poi in unique])
        if not geocoded.any():
            return None
        unique, counts = unique[geocoded], counts[geocoded]
        order = np.lexsort((unique, -counts))  # count desc, then smaller POI ID
        return self._coordinates[int(unique[order[0]])]

    def poi_coordinates(self, poi_id: int) -> tuple[float, float] | None:
        if self._coordinates is None:
            return None
        return self._coordinates.get(int(poi_id))

    @property
    def has_coordinates(self) -> bool:
        return bool(self._coordinates)


def train_cutoff_from_split_manifest(split_manifest_path: str | Path) -> int:
    gaps = pd.read_parquet(split_manifest_path, columns=["split", "next_event_time_s"])
    train = gaps.loc[gaps["split"].astype(str) == "train", "next_event_time_s"]
    if train.empty:
        raise ValueError("the split manifest has no train gaps")
    return int(train.max())
