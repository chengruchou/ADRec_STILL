"""Lazy canonical landmark dataset."""

from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from adrec_still.data.schema import CanonicalSample
from adrec_still.data.training_views import (
    EVENT_ONLY_TRAINING_PROFILE,
    GAP_CENSOR_TRAINING_PROFILE,
)


class LandmarkDataset(Dataset[CanonicalSample]):
    """Read one Parquet row group at a time, optionally filtered by split."""

    def __init__(
        self, path: str | Path, split: str | None = None, *, training_view: str | None = None,
    ) -> None:
        if split is not None and split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported split: {split}")
        if training_view not in {None, EVENT_ONLY_TRAINING_PROFILE,
                                 GAP_CENSOR_TRAINING_PROFILE}:
            raise ValueError(f"unsupported training_view: {training_view}")
        if training_view is not None and split != "train":
            raise ValueError("a training_view requires split='train'; evaluation is unchanged")
        self.path = Path(path)
        self.parquet = pq.ParquetFile(self.path)
        self.split = split
        self.training_view = training_view
        self.source_row_count = 0
        self._selected_rows: list[np.ndarray | None] = []
        self._cumulative_count: list[int] = []
        total = 0
        for row_group in range(self.parquet.num_row_groups):
            row_count = self.parquet.metadata.row_group(row_group).num_rows
            if split is None:
                selected = None
                selected_count = row_count
                self.source_row_count += row_count
            else:
                # Only scalar eligibility columns are decoded during indexing;
                # histories remain lazy and the row-group-local sampler still works.
                columns = ["split"]
                if training_view is not None:
                    columns += ["query_s_sec", "event_observed", "target_in_train_catalog",
                                "next_poi", "residual_time_sec"]
                eligibility = self.parquet.read_row_group(row_group, columns=columns)
                keep = pc.equal(eligibility["split"], split)
                self.source_row_count += int(pc.sum(keep).as_py() or 0)
                if training_view == GAP_CENSOR_TRAINING_PROFILE:
                    # One s=0 row per gap, observed and censored alike: the base
                    # distribution trains once per actual gap, with censor rows kept.
                    keep = pc.and_kleene(keep, pc.equal(eligibility["query_s_sec"], 0))
                    keep = pc.fill_null(keep, False)
                elif training_view is not None:
                    for condition in (
                        pc.equal(eligibility["query_s_sec"], 0),
                        eligibility["event_observed"],
                        eligibility["target_in_train_catalog"],
                        pc.greater(eligibility["next_poi"], 1),
                        pc.greater(eligibility["residual_time_sec"], 0),
                    ):
                        keep = pc.and_kleene(keep, condition)
                    keep = pc.fill_null(keep, False)
                selected = (
                    pc.indices_nonzero(keep)
                    .to_numpy(zero_copy_only=False)
                    .astype(np.int32)
                )
                selected_count = len(selected)
            self._selected_rows.append(selected)
            total += selected_count
            self._cumulative_count.append(total)
        self._length = total
        self._cached_row_group_index: int | None = None
        self._cached_row_group: pa.Table | None = None

    def __len__(self) -> int:
        return self._length

    def row_group_lengths(self) -> list[int]:
        """Selected-row count per Parquet row group, in dataset index order.

        Read-only structural metadata for access patterns (e.g. shuffling) that
        want to stay compatible with the one-row-group cache above.
        """
        lengths: list[int] = []
        previous = 0
        for total in self._cumulative_count:
            lengths.append(total - previous)
            previous = total
        return lengths

    def __getitem__(self, index: int) -> CanonicalSample:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        row_group_index = bisect.bisect_right(self._cumulative_count, index)
        previous_count = 0 if row_group_index == 0 else self._cumulative_count[row_group_index - 1]
        selected_index = index - previous_count
        selected = self._selected_rows[row_group_index]
        local_row = selected_index if selected is None else int(selected[selected_index])
        if self._cached_row_group_index != row_group_index:
            self._cached_row_group = self.parquet.read_row_group(row_group_index)
            self._cached_row_group_index = row_group_index
        assert self._cached_row_group is not None
        row = {
            name: self._cached_row_group[name][local_row].as_py()
            for name in self._cached_row_group.column_names
        }
        observed = bool(row["event_observed"])
        sample = CanonicalSample(
            user_id=int(row["user_id"]),
            history_poi=[int(value) for value in row["history_poi"]],
            history_timestamp=[int(value) for value in row["history_timestamp"]],
            last_timestamp=int(row["last_timestamp"]),
            query_s_sec=int(row["query_s_sec"]),
            event_observed=observed,
            target_in_train_catalog=bool(row["target_in_train_catalog"]),
            residual_time_sec=int(row["residual_time_sec"]) if observed else None,
            next_poi=int(row["next_poi"]) if observed else None,
            censor_time_sec=int(row["censor_time_sec"]) if not observed else None,
            gap_id=str(row["gap_id"]) if row.get("gap_id") is not None else None,
        )
        sample.validate()
        return sample
