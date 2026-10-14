import json

import pandas as pd

from adrec_still.data.dataset import LandmarkDataset
from adrec_still.data.survival import SURVIVAL_PROFILE_VERSION, build_survival_training_profile


def _artifacts(tmp_path):
    events = pd.DataFrame(
        [
            ("toy", "u1", 2, 2, 10, 0),
            ("toy", "u1", 2, 3, 20, 1),
            ("toy", "u1", 2, 4, 40, 2),
            ("toy", "u2", 3, 2, 12, 3),
            ("toy", "u2", 3, 3, 25, 4),
            # Held-out endpoint tied at the cutoff. Its membership, mark and identity
            # must remain outside the training projection.
            ("toy", "u2", 3, 4, 25, 5),
        ],
        columns=[
            "dataset",
            "raw_user_id",
            "user_id",
            "poi_id",
            "event_time_utc_s",
            "source_row_id",
        ],
    )
    gaps = pd.DataFrame(
        [
            ("toy:000000000000:000000000001", "u1", 0, 1, 10, 20, 10, "train"),
            ("toy:000000000003:000000000004", "u2", 3, 4, 12, 25, 13, "train"),
            ("toy:000000000001:000000000002", "u1", 1, 2, 20, 40, 20, "validation"),
            ("toy:000000000004:000000000005", "u2", 4, 5, 25, 25, 0, "validation"),
        ],
        columns=[
            "gap_id",
            "raw_user_id",
            "source_row_id",
            "next_source_row_id",
            "event_time_utc_s",
            "next_event_time_s",
            "delta_t_sec",
            "split",
        ],
    )
    events_path, splits_path = tmp_path / "events.parquet", tmp_path / "splits.parquet"
    events.to_parquet(events_path, index=False)
    gaps.to_parquet(splits_path, index=False)
    return events_path, splits_path


def _build(tmp_path, suffix="", schedule=(0, 5, 10)):
    events, splits = _artifacts(tmp_path)
    output = tmp_path / f"survival{suffix}.parquet"
    manifest_path = tmp_path / f"survival{suffix}.json"
    manifest = build_survival_training_profile(
        events, splits, output, manifest_path, landmark_seconds=schedule
    )
    return events, splits, output, manifest_path, manifest


def test_profile_contains_train_completed_and_positive_censor_rows(tmp_path):
    _, _, output, manifest_path, manifest = _build(tmp_path)
    frame = pd.read_parquet(output)
    assert manifest["version"] == SURVIVAL_PROFILE_VERSION
    assert manifest["cutoff_time_sec"] == 25
    assert manifest["row_count"] == len(frame)
    assert json.loads(manifest_path.read_text()) == manifest
    censored = frame.loc[~frame["event_observed"]]
    assert manifest["version"] == "still-train-admin-censor-v1"
    assert manifest["boundary_event_count"] == 1
    assert manifest["completed_gap_count"] == 1
    assert manifest["observed_row_count"] == 2
    assert len(censored) == manifest["censored_row_count"] == 4
    assert (censored["censor_time_sec"] > 0).all()
    assert censored["next_poi"].isna().all()
    assert censored["residual_time_sec"].isna().all()
    # Every event at t=cutoff is excluded, whether train source row 4 or held-out row 5.
    assert not frame["gap_id"].str.contains("000000000004").any()
    assert not frame["gap_id"].str.contains("000000000005").any()
    u2 = frame.loc[frame["user_id"].eq(3)]
    assert (~u2["event_observed"]).all()
    assert set(u2["history_timestamp"].map(tuple)) == {(12,)}
    dataset = LandmarkDataset(output, split="train")
    assert len(dataset) == len(frame)
    assert any(not dataset[i].event_observed for i in range(len(dataset)))


def test_heldout_endpoint_mutation_cannot_change_profile_or_identity(tmp_path):
    events_path, splits_path, first_output, _, first = _build(tmp_path, "-a")
    events = pd.read_parquet(events_path)
    heldout = events["source_row_id"].eq(2) | events["source_row_id"].eq(5)
    events.loc[heldout, "poi_id"] = 999
    events.loc[heldout, "event_time_utc_s"] += 100000
    events.to_parquet(events_path, index=False)
    second_output = tmp_path / "survival-b.parquet"
    second = build_survival_training_profile(
        events_path,
        splits_path,
        second_output,
        tmp_path / "survival-b.json",
        landmark_seconds=(0, 5, 10),
    )
    assert first == second
    assert first_output.read_bytes() == second_output.read_bytes()


def test_boundary_event_mark_cannot_change_strict_prefix_or_identity(tmp_path):
    events_path, splits_path, first_output, _, first = _build(tmp_path, "-boundary-a")
    events = pd.read_parquet(events_path)
    events.loc[events["source_row_id"].eq(4), "poi_id"] = 999
    events.to_parquet(events_path, index=False)
    second_output = tmp_path / "survival-boundary-b.parquet"
    second = build_survival_training_profile(
        events_path,
        splits_path,
        second_output,
        tmp_path / "survival-boundary-b.json",
        landmark_seconds=(0, 5, 10),
    )
    assert first == second
    assert first_output.read_bytes() == second_output.read_bytes()


def test_exact_cutoff_and_zero_followup_boundaries(tmp_path):
    _, _, output, _, manifest = _build(tmp_path, schedule=(0, 5, 13, 15))
    censored = pd.read_parquet(output).query("not event_observed")
    assert (censored["last_timestamp"] + censored["query_s_sec"] < 25).all()
    assert (censored["censor_time_sec"] > 0).all()
    assert manifest["zero_followup_gap_count"] == 0
    assert manifest["zero_followup_landmark_count"] == 2
    assert manifest["exclusion_reasons"]["zero_followup_landmarks"]
