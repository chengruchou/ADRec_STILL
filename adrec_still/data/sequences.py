"""Per-user item sequences for the upstream-family models.

Two sources produce the same structure:

- **canonical events** (``events.parquet`` + ``split_manifest.parquet``): training
  sequences are each user's disclosed events strictly before the train cutoff
  (the same leakage rule as the survival training view), encoded in the frozen
  training vocabulary. Validation/test stay on the canonical landmark rows and
  the shared evaluator; no sequence-form split is fabricated for them.
- **upstream ``dataset.pkl``** (the ADRec fork's bundled Amazon/Yelp/ML data):
  per-user train/val/test item lists. Raw IDs are shifted by +1 on load so the
  package-wide reserved IDs hold (PAD=0, UNK=1 — unused here, real items >= 2).
  These files carry **no timestamps**: silence, time, and cell semantics are
  unavailable by construction and requesting them is an explicit error.

The evaluation prefix construction fixes the fork's one-event bug
(``seq[-(len(hist)-1):]`` is ``seq[-0:]`` — the whole list — when the history
holds a single event); the shifted prefix here is ``hist[1:]``, which is empty
in that case.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

PKL_ID_SHIFT = 1  # raw pkl item IDs are 1-based with 0 = padding; +1 frees ID 1 as UNK


@dataclass
class SequenceCorpus:
    """Chronological per-user item sequences in a frozen vocabulary."""

    train: list[list[int]]
    validation: list[list[int]] = field(default_factory=list)
    test: list[list[int]] = field(default_factory=list)
    number_of_pois: int = 0
    identity: dict[str, Any] = field(default_factory=dict)
    has_timestamps: bool = False
    kind: str = "canonical"

    def validate(self) -> None:
        if not self.train or all(len(seq) < 2 for seq in self.train):
            raise ValueError(
                "the sequence corpus has no trainable user: every training sequence "
                "needs at least two events"
            )
        if self.number_of_pois <= 2:
            raise ValueError("number_of_pois must include at least one real item")
        limit = self.number_of_pois
        for name, split in (("train", self.train), ("validation", self.validation),
                            ("test", self.test)):
            for seq in split:
                for item in seq:
                    if not 0 < item < limit:
                        raise ValueError(
                            f"{name} split contains item ID {item} outside (0, {limit})"
                        )


def load_dataset_pkl(path: str | Path) -> SequenceCorpus:
    """Load an upstream ``dataset.pkl`` (train/val/test per-user item lists)."""
    path = Path(path)
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    missing = [key for key in ("train", "val", "test") if key not in raw]
    if missing:
        raise ValueError(
            f"{path} is not an upstream dataset.pkl: missing keys {missing}; expected a "
            "dict with per-user 'train'/'val'/'test' item-ID lists"
        )
    train = [[int(i) + PKL_ID_SHIFT for i in seq] for seq in raw["train"]]
    validation = [[int(i) + PKL_ID_SHIFT for i in seq] for seq in raw["val"]]
    test = [[int(i) + PKL_ID_SHIFT for i in seq] for seq in raw["test"]]
    if not (len(train) == len(validation) == len(test)):
        raise ValueError(f"{path}: train/val/test must be aligned per user")
    max_item = max(
        (max(seq) for split in (train, validation, test) for seq in split if seq),
        default=0,
    )
    corpus = SequenceCorpus(
        train=train, validation=validation, test=test,
        number_of_pois=max_item + 1,
        identity={
            "kind": "dataset_pkl",
            "path": str(path.resolve()),
            "users": len(train),
            "max_raw_item_id": max_item - PKL_ID_SHIFT,
            "id_shift": PKL_ID_SHIFT,
            "number_of_pois": max_item + 1,
        },
        has_timestamps=False,
        kind="pkl",
    )
    corpus.validate()
    return corpus


def canonical_train_sequences(
    events_path: str | Path, split_manifest_path: str | Path, number_of_pois: int,
) -> SequenceCorpus:
    """Per-user training sequences from the disclosed pre-cutoff event prefix.

    Mirrors the survival training view's leakage rule: membership comes from the
    train gaps' source rows, then a strict ``event_time < cutoff`` filter; events
    at or after the boundary second never enter a training sequence.
    """
    events = pd.read_parquet(
        events_path,
        columns=["raw_user_id", "poi_id", "event_time_utc_s", "source_row_id"],
    )
    gaps = pd.read_parquet(split_manifest_path)
    train = gaps.loc[gaps["split"].astype(str) == "train"]
    if train.empty:
        raise ValueError("canonical sequences need completed train gaps")
    cutoff = int(train["next_event_time_s"].max())
    source_ids = set(train["source_row_id"].astype(int)) | set(
        train["next_source_row_id"].astype(int)
    )
    disclosed = events.loc[
        events["source_row_id"].isin(source_ids)
        & (events["event_time_utc_s"] < cutoff)
    ]
    disclosed = disclosed.sort_values(
        ["raw_user_id", "event_time_utc_s", "source_row_id"], kind="mergesort"
    )
    sequences = [
        user_events["poi_id"].astype(int).tolist()
        for _, user_events in disclosed.groupby("raw_user_id", sort=False, observed=True)
        if len(user_events) >= 2
    ]
    corpus = SequenceCorpus(
        train=sequences, number_of_pois=int(number_of_pois),
        identity={"kind": "canonical_sequences", "cutoff_time_sec": cutoff,
                  "users": len(sequences),
                  "events": int(sum(len(seq) for seq in sequences))},
        has_timestamps=True, kind="canonical",
    )
    corpus.validate()
    return corpus


def shifted_prefix(history: list[int], answer: list[int]) -> list[int]:
    """Per-position next-item labels aligned with ``history`` (one-event safe)."""
    return list(history[1:]) + list(answer)


class SequenceTrainDataset(Dataset):
    """Upstream ``TrainDataset``: left-padded history plus per-position targets."""

    def __init__(self, sequences: list[list[int]], max_len: int,
                 parallel: bool = True) -> None:
        self.sequences = [seq for seq in sequences if len(seq) >= 2]
        if not self.sequences:
            raise ValueError("no sequence with at least two events to train on")
        self.max_len = int(max_len)
        self.parallel = parallel

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        seq = self.sequences[index]
        hist = seq[:-1][-self.max_len:]
        mask_len = self.max_len - len(hist)
        hist_pad = [0] * mask_len + hist
        if self.parallel:
            target = [0] * mask_len + seq[-len(hist):]
        else:
            target = [0] * (self.max_len - 1) + [seq[-1]]
        return torch.LongTensor(hist_pad), torch.LongTensor(target)


def expand_prefixes(sequences: list[list[int]], max_len: int) -> list[list[int]]:
    """Upstream ``split_onebyone``: every prefix of the tail window, target last."""
    expanded: list[list[int]] = []
    for seq in sequences:
        tail = seq[-max_len - 1:]
        for star in range(len(tail) - 1):
            expanded.append(tail[: star + 2])
    return expanded


class SequenceEvalDataset(Dataset):
    """Upstream Val/Test datasets with the one-event prefix fixed.

    ``input_sequences[i]`` is the model input history; ``answers[i]`` the
    held-out target list (upstream: exactly one item). The label row carries the
    shifted history on the prefix positions (teacher-forcing context for ADRec's
    reverse loop) and the answer in the final slot; only the final slot is a
    label.
    """

    def __init__(self, input_sequences: list[list[int]], answers: list[list[int]],
                 max_len: int) -> None:
        if len(input_sequences) != len(answers):
            raise ValueError("input sequences and answers must be aligned")
        keep = [index for index, answer in enumerate(answers) if answer]
        self.inputs = [input_sequences[i] for i in keep]
        self.answers = [answers[i] for i in keep]
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        seq = self.inputs[index]
        answer = self.answers[index]
        hist = seq[-self.max_len:]
        padding_len = self.max_len - len(hist)
        hist_pad = [0] * padding_len + hist
        answer_pad = [0] * padding_len + shifted_prefix(hist, answer[:1])
        if len(answer_pad) != len(hist_pad):
            raise AssertionError("prefix construction misaligned history and labels")
        return torch.LongTensor(hist_pad), torch.LongTensor(answer_pad)
