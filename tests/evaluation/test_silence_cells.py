"""Four silence cells: labels, count-weighted metrics, distances, zero-early readout.

One hand-built user walks through all four cells once (home at the equator
origin, a local POI ~5.56 km east, a displaced POI ~111.2 km east; gaps of one
hour and one hundred hours), plus one row per exclusion reason. The metric
accumulator is fed two rows whose NLL/Brier/accuracy are computed by hand.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
import torch

from adrec_still.data.history import FullHistoryIndex
from adrec_still.evaluation.cells import CellLabels, CellMetricAccumulator, cell_labels
from adrec_still.models.adrec_still.forecast import (
    CELL_NAMES,
    EARTH_RADIUS_KM,
    haversine_km,
)
from adrec_still.models.adrec_still.model import AdrecStillModel

HOUR = 3600
THRESHOLD_HOURS = 24.0
THRESHOLD_S = THRESHOLD_HOURS * 3600.0
RADIUS_KM = 10.0

HOME, LOCAL, DISPLACED, UNGEOCODED = 2, 3, 4, 5


def _gap(anchor: int, target: int) -> str:
    return f"toy:{anchor:012d}:{target:012d}"


def _write_fixture(tmp_path):
    rows = []

    def event(user, raw, poi, time_s, source):
        rows.append(
            {
                "dataset": "toy",
                "raw_user_id": user,
                "user_id": 2,
                "raw_poi_id": raw,
                "poi_id": poi,
                "event_time_utc_s": time_s,
                "latitude": 0.0,
                "longitude": 0.0,
                "event_index": len(rows),
                "source_row_id": source,
            }
        )

    # user u1: home (three visits), then one gap per cell, then a long-total
    # gap and an out-of-vocabulary target.
    moment = 0
    for source in (0, 1, 2):
        event("u1", "home", HOME, moment, source)
        moment += 100
    moment = 200
    moment += 1 * HOUR
    event("u1", "local", LOCAL, moment, 3)          # gap 2->3: T = 1h, local
    moment += 1 * HOUR
    event("u1", "displaced", DISPLACED, moment, 4)  # gap 3->4: T = 1h, displaced
    moment += 100 * HOUR
    event("u1", "local", LOCAL, moment, 5)          # gap 4->5: T = 100h, local
    moment += 100 * HOUR
    event("u1", "displaced", DISPLACED, moment, 6)  # gap 5->6: T = 100h, displaced
    moment += 31 * HOUR
    event("u1", "local", LOCAL, moment, 7)          # gap 6->7: T = 31h > 24h, local
    moment += 100
    event("u1", "oov-raw", 1, moment, 8)            # gap 7->8: OOV target
    # user u3: only a known-but-ungeocoded POI before the probe target.
    event("u3", "nowhere", UNGEOCODED, 0, 20)
    event("u3", "local", LOCAL, HOUR, 21)

    events_path = tmp_path / "events.parquet"
    pd.DataFrame(rows).to_parquet(events_path, index=False)
    catalog = pd.DataFrame(
        [
            {"poi_id": HOME, "raw_poi_id": "home", "latitude": 0.0, "longitude": 0.0,
             "train_count": 3, "first_train_time_s": 0, "last_train_time_s": 100},
            {"poi_id": LOCAL, "raw_poi_id": "local", "latitude": 0.0, "longitude": 0.05,
             "train_count": 2, "first_train_time_s": 0, "last_train_time_s": 100},
            {"poi_id": DISPLACED, "raw_poi_id": "displaced", "latitude": 0.0,
             "longitude": 1.0, "train_count": 2, "first_train_time_s": 0,
             "last_train_time_s": 100},
            {"poi_id": UNGEOCODED, "raw_poi_id": "nowhere", "latitude": float("nan"),
             "longitude": float("nan"), "train_count": 1, "first_train_time_s": 0,
             "last_train_time_s": 100},
        ]
    )
    catalog_path = tmp_path / "poi_catalog.parquet"
    catalog.to_parquet(catalog_path, index=False)
    return events_path, catalog_path


@pytest.fixture()
def index(tmp_path) -> FullHistoryIndex:
    events_path, catalog_path = _write_fixture(tmp_path)
    return FullHistoryIndex(events_path, catalog_path, train_cutoff_s=10**9)


def test_cell_labels_cover_all_four_cells_and_count_every_exclusion(index):
    batch = {
        "gap_id": [
            _gap(2, 3),               # local_early
            _gap(3, 4),               # displaced_early
            _gap(4, 5),               # local_late
            _gap(5, 6),               # displaced_late
            _gap(6, 7),               # s >= threshold, but TOTAL T = 31h decides: late
            _gap(7, 8),               # OOV target
            _gap(20, 21),             # home unresolvable (ungeocoded prefix)
            "toy:000000000008:censored",
        ],
        "observed_mask": torch.tensor([True] * 7 + [False]),
        "target_in_train_catalog": torch.tensor(
            [True, True, True, True, True, False, True, False]
        ),
    }
    labels = cell_labels(
        index, batch, cell_radius_km=RADIUS_KM,
        cell_time_threshold_hours=THRESHOLD_HOURS, home_policy="causal",
    )
    assert CELL_NAMES == ("local_early", "displaced_early", "local_late",
                          "displaced_late")
    assert labels.available.tolist() == [True] * 5 + [False] * 3
    assert labels.cell[:5].tolist() == [
        CELL_NAMES.index("local_early"),
        CELL_NAMES.index("displaced_early"),
        CELL_NAMES.index("local_late"),
        CELL_NAMES.index("displaced_late"),
        CELL_NAMES.index("local_late"),  # total T, never the residual
    ]
    assert labels.early[:5].tolist() == [True, True, False, False, False]
    assert labels.reasons == {
        "censored": 1, "no_gap_id": 0, "oov_target": 1, "no_home": 1,
        "target_not_geocoded": 0, "labeled": 5,
    }


def test_cell_metric_accumulator_matches_hand_computed_sums():
    accumulator = CellMetricAccumulator()
    probabilities = torch.tensor(
        [[0.7, 0.1, 0.1, 0.1], [0.0, 0.0, 0.25, 0.75]], dtype=torch.float64
    )
    labels = CellLabels(
        cell=torch.tensor([0, 3]),
        available=torch.tensor([True, True]),
        early=torch.tensor([True, False]),
        reasons={"labeled": 2, "censored": 4},
    )
    query_s = torch.tensor([0, int(2 * THRESHOLD_S)])
    accumulator.add(
        probabilities, torch.tensor([True, True]), labels, query_s, THRESHOLD_S,
        unresolved_mass=torch.tensor([0.25, 0.05], dtype=torch.float64),
    )
    result = accumulator.result()

    assert result["cells"] == list(CELL_NAMES)
    assert result["label_reasons"] == {"censored": 4, "labeled": 2}
    assert result["mean_unresolved_spatial_mass"] == pytest.approx(0.15)

    nontrivial = result["nontrivial"]
    assert nontrivial["count"] == 1
    assert nontrivial["nll"] == pytest.approx(-math.log(0.7))
    assert nontrivial["brier"] == pytest.approx(0.3**2 + 3 * 0.1**2)
    assert nontrivial["accuracy"] == 1.0
    assert nontrivial["confusion_matrix"][0][0] == 1

    late = result["late_determined"]
    assert late["count"] == 1
    assert late["nll"] == pytest.approx(-math.log(0.75))
    assert late["brier"] == pytest.approx(0.25**2 + 0.25**2)
    assert late["accuracy"] == 1.0
    assert late["confusion_matrix"][3][3] == 1


def test_haversine_km_reproduces_known_distances():
    # One degree of longitude on the equator is exactly R * pi / 180.
    equator_degree = haversine_km(
        torch.tensor([0.0]), torch.tensor([0.0]),
        torch.tensor([0.0]), torch.tensor([1.0]),
    )
    assert float(equator_degree[0]) == pytest.approx(
        EARTH_RADIUS_KM * math.pi / 180.0, rel=1e-9
    )
    # Paris <-> London great-circle distance is about 343.5 km.
    paris_london = haversine_km(
        torch.tensor([48.8566]), torch.tensor([2.3522]),
        torch.tensor([51.5074]), torch.tensor([-0.1278]),
    )
    assert float(paris_london[0]) == pytest.approx(343.5, rel=0.01)


def test_cell_readout_has_zero_early_mass_once_the_threshold_has_passed(index):
    torch.manual_seed(17)
    model = AdrecStillModel(
        "adrec_still", number_of_pois=6, hidden_size=8, mixture_components=2,
        diffusion_steps=2, time_samples=1, diffusion_samples=1,
    )
    model.attach_memory(index)
    model.is_fitted = True
    model.eval()
    history = torch.tensor([[2, 2, 2, 3, 4]], dtype=torch.long)
    batch = {
        "history_poi": history,
        "history_timestamp": torch.zeros_like(history),
        "history_mask": history.gt(0),
        "last_timestamp": torch.tensor([727400]),
        "query_s_sec": torch.tensor([int(2 * THRESHOLD_S)]),
        "gap_id": ["toy:000000000006:censored"],
    }
    forecast = model.prepare_forecast(batch, chunk_size=6)
    readout = forecast.cell_readout(cell_radius_km=RADIUS_KM)
    assert bool(readout["available"][0])
    probabilities = readout["probabilities"][0]
    early = (CELL_NAMES.index("local_early"), CELL_NAMES.index("displaced_early"))
    assert float(probabilities[list(early)].abs().sum()) == pytest.approx(0.0, abs=1e-12)
    assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-9)
