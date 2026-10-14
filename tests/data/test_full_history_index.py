"""Full per-user event memory: repeat semantics, seen-state, homes, gap identities.

The fixture is a hand-built canonical ``events.parquet``/``poi_catalog.parquet``
pair small enough that every expected answer is computed on paper:

- vocabulary: ``pX -> 2``, ``pY -> 3``, ``p4 -> 4`` (``p4`` has no coordinates);
  two distinct out-of-vocabulary raw IDs both encode to UNK (``1``);
- user A repeats ``pX`` three times before the probe anchor and returns to it
  after a long out-of-vocabulary stretch;
- users B-E each isolate one home rule (tie, frozen cutoff, fallback, geocoding).
"""

from __future__ import annotations

import pandas as pd
import pytest

from adrec_still.data.history import FullHistoryIndex

CUTOFF_S = 3500


def _gap(anchor: int, target: int) -> str:
    return f"ds:{anchor:012d}:{target:012d}"


def _write_fixture(tmp_path):
    rows = []

    def event(user, user_id, raw, poi, time_s, source):
        rows.append(
            {
                "dataset": "ds",
                "raw_user_id": user,
                "user_id": user_id,
                "raw_poi_id": raw,
                "poi_id": poi,
                "event_time_utc_s": time_s,
                "latitude": 0.0,
                "longitude": 0.0,
                "event_index": len(rows),
                "source_row_id": source,
            }
        )

    # user A: three pX visits, one pY, then an out-of-vocabulary stretch, then pX again.
    event("uA", 2, "pX", 2, 1000, 0)
    event("uA", 2, "pY", 3, 2000, 1)
    event("uA", 2, "pX", 2, 3000, 2)
    event("uA", 2, "pX", 2, 4000, 3)
    event("uA", 2, "oov-one", 1, 5000, 4)
    event("uA", 2, "oov-two", 1, 6000, 5)
    event("uA", 2, "oov-one", 1, 7000, 6)
    event("uA", 2, "pX", 2, 8000, 7)
    # user B: equal visit counts at the probe anchor (home tie).
    event("uB", 3, "pY", 3, 100, 10)
    event("uB", 3, "pX", 2, 200, 11)
    event("uB", 3, "pX", 2, 300, 12)
    # user C: pre-cutoff events favour pY, post-cutoff events favour pX.
    event("uC", 4, "pY", 3, 1000, 20)
    event("uC", 4, "pY", 3, 2000, 21)
    event("uC", 4, "pX", 2, 5000, 22)
    event("uC", 4, "pX", 2, 6000, 23)
    event("uC", 4, "pX", 2, 7000, 24)
    # user D: no event strictly before the train cutoff.
    event("uD", 5, "pX", 2, 5000, 30)
    event("uD", 5, "pY", 3, 6000, 31)
    # user E: the most-visited POI carries no coordinates.
    event("uE", 6, "p4", 4, 100, 40)
    event("uE", 6, "p4", 4, 200, 41)
    event("uE", 6, "pX", 2, 300, 42)
    event("uE", 6, "pY", 3, 400, 43)

    events_path = tmp_path / "events.parquet"
    pd.DataFrame(rows).to_parquet(events_path, index=False)

    catalog = pd.DataFrame(
        [
            {"poi_id": 2, "raw_poi_id": "pX", "latitude": 10.0, "longitude": 20.0,
             "train_count": 5, "first_train_time_s": 0, "last_train_time_s": 100},
            {"poi_id": 3, "raw_poi_id": "pY", "latitude": -5.0, "longitude": 30.0,
             "train_count": 2, "first_train_time_s": 0, "last_train_time_s": 100},
            {"poi_id": 4, "raw_poi_id": "p4", "latitude": float("nan"),
             "longitude": float("nan"), "train_count": 3, "first_train_time_s": 0,
             "last_train_time_s": 100},
        ]
    )
    catalog_path = tmp_path / "poi_catalog.parquet"
    catalog.to_parquet(catalog_path, index=False)
    return events_path, catalog_path


@pytest.fixture()
def index(tmp_path) -> FullHistoryIndex:
    events_path, catalog_path = _write_fixture(tmp_path)
    return FullHistoryIndex(events_path, catalog_path, train_cutoff_s=CUTOFF_S)


def test_repeated_visits_are_counted_not_deduplicated(index):
    state = index.seen_state(_gap(3, 4))  # user A, anchor at the third pX visit
    assert state.poi_ids.tolist() == [2, 3]
    assert state.counts.tolist() == [3, 1]
    assert state.last_visit_s.tolist() == [4000, 2000]
    assert state.size == 2


def test_seen_state_filters_reserved_ids_but_keeps_exact_counts(index):
    # Anchor after the out-of-vocabulary stretch: UNK (1) never enters the seen set.
    state = index.seen_state(_gap(6, 7))
    assert state.poi_ids.tolist() == [2, 3]
    assert state.counts.tolist() == [3, 1]
    assert state.last_visit_s.tolist() == [4000, 2000]


def test_repeat_label_reads_the_full_past_not_a_window(index):
    # pX was last visited four events before this anchor; the raw ID first
    # appeared at the user's very first event, so the gap is still a repeat
    # even when an encoder window would have truncated it away.
    assert index.repeat_label(_gap(6, 7)) is True


def test_shared_unk_encoding_never_fakes_a_repeat(index):
    # oov-two encodes to UNK like the earlier oov-one, yet it is a new raw ID.
    assert index.repeat_label(_gap(4, 5)) is False
    # The same raw out-of-vocabulary ID seen twice is a genuine repeat.
    assert index.repeat_label(_gap(5, 6)) is True
    # Censored gaps carry no target and therefore no label.
    assert index.repeat_label("ds:000000000003:censored") is None


def test_home_is_most_visited_geocoded_poi_with_count_then_id_tie_rule(index):
    # user A at anchor e3: pX visited 3x vs pY 1x.
    assert index.home(_gap(3, 4), policy="causal") == (10.0, 20.0)
    # user B at anchor src 11: pY and pX both visited once; the tie goes to the
    # smaller POI ID (pX == 2).
    assert index.home(_gap(11, 12), policy="causal") == (10.0, 20.0)
    # user E: the top POI (p4, 2 visits) has no coordinates; home falls to pX.
    assert index.home(_gap(42, 43), policy="causal") == (10.0, 20.0)


def test_frozen_home_uses_only_pre_cutoff_events(index):
    gap = "ds:000000000024:censored"  # user C, anchor at the last pX visit
    assert index.home(gap, policy="causal") == (10.0, 20.0)   # pX dominates causally
    assert index.home(gap, policy="frozen") == (-5.0, 30.0)   # only pY precedes the cutoff


def test_frozen_home_without_pre_cutoff_events_is_none_and_callers_fall_back(index):
    """A user with no pre-cutoff event has no frozen home: the index answers
    None (so the fallback is detectable and countable), and the feature bridge
    performs the causal fallback explicitly."""
    from adrec_still.models.adrec_still.memory import memory_features

    gap = _gap(30, 31)  # user D's events all sit at or after the cutoff
    assert index.home(gap, policy="frozen") is None
    assert index.home(gap, policy="causal") == (10.0, 20.0)
    features = memory_features(index, {"gap_id": [gap]}, home_policy="frozen")
    assert bool(features.home_available[0])
    assert (float(features.home_lat[0]), float(features.home_lon[0])) == (10.0, 20.0)


def test_frozen_home_requires_a_cutoff(tmp_path):
    events_path, catalog_path = _write_fixture(tmp_path)
    bare = FullHistoryIndex(events_path, catalog_path)
    with pytest.raises(ValueError, match="train cutoff"):
        bare.home(_gap(3, 4), policy="frozen")


def test_parse_gap_id_handles_every_tail():
    assert FullHistoryIndex.parse_gap_id("ds:000000000003:000000000004") == (3, 4)
    assert FullHistoryIndex.parse_gap_id("ds:000000000005:censored") == (5, None)
    assert FullHistoryIndex.parse_gap_id("ds:000000000005:train-cutoff-9999") == (5, None)
    with pytest.raises(ValueError, match="not a gap identity"):
        FullHistoryIndex.parse_gap_id("no-colons")


def test_gap_duration_and_target_follow_the_identity(index):
    assert index.gap_duration_s(_gap(3, 4)) == 1000
    assert index.target_poi(_gap(3, 4)) == 1  # the target is the first OOV event
    assert index.gap_duration_s("ds:000000000003:censored") is None
    assert index.target_poi("ds:000000000003:censored") is None
