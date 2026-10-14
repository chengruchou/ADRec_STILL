"""Sequence corpora: pkl ID shifting, eval prefixes, and the strict train cutoff."""

from __future__ import annotations

import pickle

import pandas as pd
import pytest
import torch

from adrec_still.data.sequences import (
    SequenceEvalDataset,
    canonical_train_sequences,
    expand_prefixes,
    load_dataset_pkl,
)


def test_load_dataset_pkl_shifts_ids_reserving_pad_and_unk(tmp_path):
    path = tmp_path / "dataset.pkl"
    raw = {
        "train": [[1, 2, 1], [3, 1]],
        "val": [[2], [3]],
        "test": [[1], []],
    }
    path.write_bytes(pickle.dumps(raw))
    corpus = load_dataset_pkl(path)

    assert corpus.train == [[2, 3, 2], [4, 2]]
    assert corpus.validation == [[3], [4]]
    assert corpus.test == [[2], []]
    # PAD (0) and UNK (1) are reserved: every real item sits at raw + 1 >= 2,
    # and the largest raw ID (3) lands on number_of_pois - 1.
    smallest = min(item for split in (corpus.train, corpus.validation, corpus.test)
                   for seq in split for item in seq)
    assert smallest == 2
    assert corpus.number_of_pois == 5
    assert corpus.identity["max_raw_item_id"] == 3
    assert corpus.identity["id_shift"] == 1
    assert corpus.has_timestamps is False


def test_one_event_history_yields_the_answer_only_in_the_final_slot():
    # Upstream regression: with a single-event history the fork's
    # ``seq[-(len(hist)-1):]`` is ``seq[-0:]`` (the whole list), duplicating the
    # history into the label prefix. The fixed prefix is empty padding.
    dataset = SequenceEvalDataset([[7]], [[9]], max_len=4)
    history, labels = dataset[0]
    assert history.tolist() == [0, 0, 0, 7]
    assert labels.tolist() == [0, 0, 0, 9]
    # Two events: the prefix carries the shifted history, the final slot the answer.
    dataset = SequenceEvalDataset([[5, 7]], [[9]], max_len=4)
    history, labels = dataset[0]
    assert history.tolist() == [0, 0, 5, 7]
    assert labels.tolist() == [0, 0, 7, 9]


def test_expand_prefixes_yields_every_prefix_of_the_tail_window():
    assert expand_prefixes([[1, 2, 3, 4, 5]], max_len=3) == [
        [2, 3], [2, 3, 4], [2, 3, 4, 5]
    ]
    # A short sequence expands from its own start.
    assert expand_prefixes([[8, 9]], max_len=3) == [[8, 9]]


def test_canonical_train_sequences_exclude_events_at_the_cutoff_second(tmp_path):
    events = pd.DataFrame(
        [
            # user u: four events; the third sits exactly at the cutoff second.
            ("u", 2, 0, 0),
            ("u", 3, 10, 1),
            ("u", 4, 20, 2),
            ("u", 5, 30, 3),
            # user v: both events disclosed strictly before the cutoff.
            ("v", 2, 0, 10),
            ("v", 3, 15, 11),
            # user w: only one of its events survives the strict filter.
            ("w", 3, 19, 20),
            ("w", 2, 20, 21),
        ],
        columns=["raw_user_id", "poi_id", "event_time_utc_s", "source_row_id"],
    )
    manifest = pd.DataFrame(
        [
            ("train", 10, 0, 1),
            ("train", 20, 1, 2),   # cutoff = max train endpoint = 20
            ("train", 15, 10, 11),
            ("train", 20, 20, 21),
            ("test", 30, 2, 3),
        ],
        columns=["split", "next_event_time_s", "source_row_id", "next_source_row_id"],
    )
    events_path = tmp_path / "events.parquet"
    manifest_path = tmp_path / "split_manifest.parquet"
    events.to_parquet(events_path, index=False)
    manifest.to_parquet(manifest_path, index=False)

    corpus = canonical_train_sequences(events_path, manifest_path, number_of_pois=6)

    # u's event AT the cutoff second (poi 4, t == 20) never enters a training
    # sequence even though a train gap's source union names it; w is dropped
    # entirely because one disclosed event is not a trainable sequence.
    assert corpus.train == [[2, 3], [2, 3]]
    assert all(4 not in seq for seq in corpus.train)
    assert corpus.identity["cutoff_time_sec"] == 20
    assert corpus.identity["users"] == 2
    assert corpus.has_timestamps is True


def test_canonical_train_sequences_feed_the_training_view_shapes(tmp_path):
    # The corpus above collates into the left-padded parallel view used by the
    # staged trainer: [PAD..., history] with per-position next-item targets.
    from adrec_still.data.sequences import SequenceTrainDataset

    dataset = SequenceTrainDataset([[2, 3, 4]], max_len=4, parallel=True)
    history, targets = dataset[0]
    assert history.tolist() == [0, 0, 2, 3]
    assert targets.tolist() == [0, 0, 3, 4]
    assert isinstance(history, torch.Tensor)
    with pytest.raises(ValueError, match="at least two events"):
        SequenceTrainDataset([[2]], max_len=4)
