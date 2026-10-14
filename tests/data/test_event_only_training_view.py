"""The event-only view selects existing canonical gaps before decoding histories."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from adrec_still.data.dataset import LandmarkDataset
from adrec_still.data.training_views import EVENT_ONLY_TRAINING_PROFILE
from adrec_still.models.baselines.trainer import RowGroupLocalShuffleSampler


def row(user_id, *, split="train", query=0, residual=3600, observed=True,
        known=True, target=3):
    return {
        "user_id": user_id, "split": split, "history_poi": [2],
        "history_timestamp": [100], "last_timestamp": 100, "query_s_sec": query,
        "event_observed": observed, "target_in_train_catalog": known,
        "residual_time_sec": residual if observed else None,
        "next_poi": target if observed else None,
        "censor_time_sec": None if observed else 50,
    }


@pytest.fixture
def landmarks(tmp_path):
    rows = [
        row(1, residual=3600),
        row(2, residual=200 * 3600),
        row(2, query=6 * 3600, residual=194 * 3600),
        row(2, query=24 * 3600, residual=176 * 3600),
        row(3, observed=False, known=False),
        row(4, known=False, target=1),
        row(5, residual=0),  # invalid source row must never reach canonical collation
        row(6, split="validation"),
        row(7, split="test"),
        row(8, known=True, target=1),  # inconsistent catalog flag is not exact-known
        row(9, known=True, target=0),
        row(10, residual=-1),
    ]
    path = tmp_path / "landmarks.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=3)
    return path


def test_only_existing_positive_known_train_gap_origins_remain(landmarks):
    default = LandmarkDataset(landmarks, split="train")
    selected = LandmarkDataset(landmarks, split="train", training_view=EVENT_ONLY_TRAINING_PROFILE)
    assert len(default) == 10
    assert selected.source_row_count == len(default)
    assert [sample.user_id for sample in selected] == [1, 2]
    assert [sample.residual_time_sec for sample in selected] == [3600, 720000]
    assert selected.row_group_lengths() == [2, 0, 0, 0]
    order = list(RowGroupLocalShuffleSampler(selected.row_group_lengths(), seed=41))
    assert sorted(order) == [0, 1]
    assert len({selected[index].user_id for index in order}) == 2


def test_eligibility_index_only_reads_scalar_columns(landmarks, monkeypatch):
    calls = []
    original = pq.ParquetFile.read_row_group

    def read(self, group, *args, **kwargs):
        calls.append(kwargs.get("columns"))
        return original(self, group, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", read)
    selected = LandmarkDataset(landmarks, split="train", training_view=EVENT_ONLY_TRAINING_PROFILE)
    assert len(calls) == 4
    assert all(columns and "history_poi" not in columns for columns in calls)
    calls.clear()
    list(selected)
    assert calls == [None], "the lazy cache should decode only the one nonempty row group"


@pytest.mark.parametrize("split", [None, "validation", "test"])
def test_training_view_cannot_filter_an_evaluation_split(landmarks, split):
    with pytest.raises(ValueError, match="split='train'"):
        LandmarkDataset(landmarks, split=split, training_view=EVENT_ONLY_TRAINING_PROFILE)


def test_unknown_training_view_is_refused(landmarks):
    with pytest.raises(ValueError, match="unsupported training_view"):
        LandmarkDataset(landmarks, split="train", training_view="event-only-typo")
