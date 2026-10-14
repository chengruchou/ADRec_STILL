"""Seen-set matching must be exact under heterogeneous batch padding.

Regression for the end-zero-padding defect: ``memory_features`` pads each row's
ascending seen-POI IDs with trailing zeros to the batch-max width, which broke
``torch.searchsorted``'s sortedness precondition — genuinely seen items then
read as unseen for every row shorter than the batch width, so repeat targets
were masked out of their own branch and forecasts changed with batch
composition. The fix searches over sentinel keys; these tests pin membership
exactness on mixed-width batches.
"""

from __future__ import annotations

import torch

from adrec_still.models.adrec_still.memory import MemoryFeatures
from adrec_still.models.adrec_still.model import (
    BRANCH_EXPLORE,
    BRANCH_REPEAT,
    MASKED,
    AdrecStillModel,
)

NUMBER_OF_POIS = 20


def _model() -> AdrecStillModel:
    torch.manual_seed(7)
    return AdrecStillModel(
        "adrec_still", NUMBER_OF_POIS, hidden_size=8, diffusion_steps=4,
        mixture_components=2, time_samples=1,
    )


def _heterogeneous_memory() -> tuple[MemoryFeatures, list[set[int]]]:
    seen_sets = [{2, 5, 9}, {2, 3, 4, 5, 6, 7}, {11}]
    width = max(len(seen) for seen in seen_sets)
    rows = len(seen_sets)
    seen_poi = torch.zeros((rows, width), dtype=torch.long)
    seen_count = torch.zeros((rows, width), dtype=torch.float32)
    seen_last = torch.zeros((rows, width), dtype=torch.long)
    seen_mask = torch.zeros((rows, width), dtype=torch.bool)
    for row, seen in enumerate(seen_sets):
        ids = sorted(seen)
        seen_poi[row, :len(ids)] = torch.tensor(ids)
        seen_count[row, :len(ids)] = torch.arange(1, len(ids) + 1, dtype=torch.float32)
        seen_last[row, :len(ids)] = 1000
        seen_mask[row, :len(ids)] = True
    memory = MemoryFeatures(
        seen_poi=seen_poi, seen_count=seen_count, seen_last_s=seen_last,
        seen_mask=seen_mask, seen_sizes=seen_mask.sum(dim=1),
        home_lat=torch.zeros(rows, dtype=torch.float64),
        home_lon=torch.zeros(rows, dtype=torch.float64),
        home_available=torch.ones(rows, dtype=torch.bool),
    )
    return memory, seen_sets


def test_matched_equals_true_membership_on_mixed_width_batches():
    model = _model()
    memory, seen_sets = _heterogeneous_memory()
    ids = torch.arange(2, NUMBER_OF_POIS).unsqueeze(0).repeat(len(seen_sets), 1)
    reference = torch.full((len(seen_sets),), 2000.0, dtype=torch.float64)
    _bonus, matched = model._memory_bonus(ids, memory, reference)
    for row, seen in enumerate(seen_sets):
        for column, poi in enumerate(range(2, NUMBER_OF_POIS)):
            assert bool(matched[row, column]) == (poi in seen), (row, poi)


def test_row_matching_is_independent_of_batch_composition():
    model = _model()
    memory, seen_sets = _heterogeneous_memory()
    ids = torch.arange(2, NUMBER_OF_POIS)
    reference = torch.full((len(seen_sets),), 2000.0, dtype=torch.float64)
    _bonus, together = model._memory_bonus(
        ids.unsqueeze(0).repeat(len(seen_sets), 1), memory, reference
    )
    for row in range(len(seen_sets)):
        alone = MemoryFeatures(
            seen_poi=memory.seen_poi[row:row + 1, :int(memory.seen_sizes[row]) or 1],
            seen_count=memory.seen_count[row:row + 1, :int(memory.seen_sizes[row]) or 1],
            seen_last_s=memory.seen_last_s[row:row + 1, :int(memory.seen_sizes[row]) or 1],
            seen_mask=memory.seen_mask[row:row + 1, :int(memory.seen_sizes[row]) or 1],
            seen_sizes=memory.seen_sizes[row:row + 1],
            home_lat=memory.home_lat[row:row + 1],
            home_lon=memory.home_lon[row:row + 1],
            home_available=memory.home_available[row:row + 1],
        )
        _b, solo = model._memory_bonus(ids.unsqueeze(0), alone, reference[row:row + 1])
        assert torch.equal(together[row], solo[0])


def test_branch_supports_stay_disjoint_under_mixed_width_padding():
    model = _model()
    memory, seen_sets = _heterogeneous_memory()
    torch.manual_seed(11)
    generated = torch.randn(len(seen_sets), model.hidden_size)
    ids = torch.arange(2, NUMBER_OF_POIS).unsqueeze(0).repeat(len(seen_sets), 1)
    reference = torch.full((len(seen_sets),), 2000.0, dtype=torch.float64)
    repeat_scores = model.branch_candidate_scores(
        generated, ids, memory, BRANCH_REPEAT, reference
    )
    explore_scores = model.branch_candidate_scores(
        generated, ids, memory, BRANCH_EXPLORE, reference
    )
    for row, seen in enumerate(seen_sets):
        for column, poi in enumerate(range(2, NUMBER_OF_POIS)):
            if poi in seen:
                assert repeat_scores[row, column] > MASKED / 2
                assert explore_scores[row, column] == MASKED
            else:
                assert repeat_scores[row, column] == MASKED
                assert explore_scores[row, column] > MASKED / 2
