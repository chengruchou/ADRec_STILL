"""Step-mode horizon support for the silence floors: same estimator, any s."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from adrec_still.data.landmarks import LANDMARK_SCHEMA
from adrec_still.models.baselines.popularity import GlobalPop
from adrec_still.models.baselines.silence import SilenceAwareFloor

HORIZONS = (0, 3600, 7200)


def _landmarks(path, rows):
    pq.write_table(pa.Table.from_pylist(rows, schema=LANDMARK_SCHEMA), path)
    return path


def _row(gap, duration, target, query=0):
    return {
        "sample_id": f"g{gap}:q{query:09d}", "gap_id": f"toy:{gap:012d}:{gap + 1:012d}",
        "user_id": 2, "split": "train", "history_poi": [2], "history_timestamp": [100],
        "last_timestamp": 100, "query_s_sec": query, "event_observed": True,
        "target_in_train_catalog": True, "residual_time_sec": duration,
        "next_poi": target, "censor_time_sec": None,
    }


@pytest.fixture()
def fitted_floors(tmp_path):
    train_rows = [
        _row(0, 1800, 2), _row(1, 5400, 2), _row(2, 9000, 3), _row(3, 600, 3),
    ]
    landmark_path = _landmarks(tmp_path / "landmark_samples.parquet", train_rows)
    base = GlobalPop(number_of_pois=4).fit({2: 2, 3: 2})
    grid = SilenceAwareFloor(base, landmark_seconds=HORIZONS, horizon_support="grid")
    grid.fit(landmark_path)
    step = SilenceAwareFloor(base, landmark_seconds=HORIZONS, horizon_support="step")
    step.fit(landmark_path)
    return grid, step


def _query(query_s):
    return {
        "user_id": torch.tensor([2]),
        "history_poi": torch.tensor([[2]]),
        "history_timestamp": torch.tensor([[100]]),
        "history_mask": torch.tensor([[True]]),
        "last_timestamp": torch.tensor([100]),
        "query_s_sec": torch.tensor([query_s]),
        "next_poi": torch.tensor([2]),
        "observed_mask": torch.tensor([True]),
        "censored_mask": torch.tensor([False]),
        "target_in_train_catalog": torch.tensor([True]),
    }


def test_step_mode_matches_the_fitted_grid_exactly(fitted_floors):
    grid, step = fitted_floors
    for horizon in HORIZONS:
        torch.testing.assert_close(
            step.predict(_query(horizon)).poi_logits,
            grid.predict(_query(horizon)).poi_logits,
        )


def test_grid_mode_rejects_off_grid_instead_of_rounding(fitted_floors):
    grid, _ = fitted_floors
    with pytest.raises(ValueError, match="fitted horizons"):
        grid.predict(_query(4000))


def test_step_mode_answers_off_grid_with_the_step_survivor(fitted_floors):
    _, step = fitted_floors
    # Durations: POI 2 -> {1800, 5400}, POI 3 -> {9000, 600}. Between 1800 and
    # 5400 the step survivor is flat, so any s in (1800, 5400) gives one answer;
    # crossing 5400 changes it.
    inside = step.predict(_query(4000)).poi_logits
    same_plateau = step.predict(_query(5000)).poi_logits
    beyond = step.predict(_query(6000)).poi_logits
    torch.testing.assert_close(inside, same_plateau)
    assert not torch.allclose(inside, beyond)
    # Hand check at s=4000: survivors are 5400 (POI 2) and 9000 (POI 3), pooled 2/4.
    # fraction_j = (count_j + alpha * pooled) / (count0_j + alpha), alpha = 1.
    base = step.base_model.predict(_query(4000)).poi_logits
    expected_fraction = torch.tensor([1.0, 1.0, (1 + 0.5) / 3.0, (1 + 0.5) / 3.0])
    adjusted = base + expected_fraction.log()
    adjusted[:, :2] = -1.0e9
    expected = torch.log_softmax(adjusted, dim=1)
    torch.testing.assert_close(inside, expected)


def test_step_mode_without_survivors_keeps_the_original_scores(fitted_floors):
    _, step = fitted_floors
    beyond_all = step.predict(_query(10_000)).poi_logits
    original = step.base_model.predict(_query(10_000)).poi_logits
    torch.testing.assert_close(beyond_all, original)
    stats = step.prediction_stats()
    assert stats["unsupported_horizon_fallback_row_count"] >= 1
