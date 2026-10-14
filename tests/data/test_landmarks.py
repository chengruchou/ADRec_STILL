import json
from pathlib import Path

import pandas as pd
import torch

from adrec_still.data.adapters.gowalla import parse_and_clean_events
from adrec_still.data.collate import collate_canonical_samples
from adrec_still.data.dataset import LandmarkDataset
from adrec_still.data.landmarks import LANDMARK_SECONDS, iter_landmark_records
from adrec_still.data.pipeline import encode_events, fit_training_vocabularies, run_preprocessing
from adrec_still.data.split import assign_chronological_splits, build_base_gaps

FIXTURE = Path(__file__).parents[1] / "fixtures" / "gowalla_small.txt"
CONFIG = Path(__file__).parents[1] / "fixtures" / "configs" / "dataset" / "gowalla.yaml"


def _records():
    events, _ = parse_and_clean_events(FIXTURE)
    split = assign_chronological_splits(build_base_gaps(events))
    user_vocab, poi_vocab, _, _ = fit_training_vocabularies(events, split)
    encoded = encode_events(events, user_vocab, poi_vocab)
    return split, list(iter_landmark_records(encoded, split))


def test_landmark_eligibility_and_censoring():
    split, records = _records()
    observed = [record for record in records if record["event_observed"]]
    censored = [record for record in records if not record["event_observed"]]
    assert observed and censored
    assert all(record["residual_time_sec"] > 0 for record in observed)
    assert all(
        record["target_in_train_catalog"] == (record["next_poi"] != 1) for record in observed
    )
    assert all(record["next_poi"] is None for record in censored)
    assert all(record["residual_time_sec"] is None for record in censored)
    assert all(record["censor_time_sec"] >= 0 for record in censored)
    assert all(not record["target_in_train_catalog"] for record in censored)
    test_period_start_s = int(
        split.loc[split["split"].astype(str) == "test", "next_event_time_s"].min()
    )
    assert all(
        record["last_timestamp"] + record["query_s_sec"] >= test_period_start_s
        for record in censored
    )
    gap_splits: dict[str, set[str]] = {}
    for record in records:
        gap_splits.setdefault(record["gap_id"], set()).add(record["split"])
    assert all(len(values) == 1 for values in gap_splits.values())
    queries = {
        record["query_s_sec"] for record in observed if record["gap_id"] == split.iloc[-1]["gap_id"]
    }
    assert queries.issubset(set(LANDMARK_SECONDS))


def test_pipeline_dataset_and_collation(tmp_path: Path):
    paths = run_preprocessing(
        config_path=CONFIG,
        input_path=FIXTURE,
        output_dir=tmp_path / "processed",
        manifest_path=tmp_path / "manifest.json",
        audit_path=tmp_path / "audit.json",
    )
    assert all(path.exists() for path in paths.values())
    dataset = LandmarkDataset(paths["landmarks"])
    samples = [dataset[index] for index in range(min(4, len(dataset)))]
    batch = collate_canonical_samples(samples, max_sequence_length=2)
    assert batch["history_poi"].shape[1] <= 2
    assert torch.equal(batch["observed_mask"], ~batch["censored_mask"])
    assert batch["target_in_train_catalog"].dtype == torch.bool

    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert set(manifest["raw_files"]) == {"checkins"}
    assert "raw_file" not in manifest
    assert "raw_file_size_bytes" not in manifest
    assert "raw_file_sha256" not in manifest
    assert manifest["processing"]["workers"] == 1
    split_manifest = pd.read_parquet(paths["split_manifest"])
    test_period_start_s = int(
        split_manifest.loc[split_manifest["split"].astype(str) == "test", "next_event_time_s"].min()
    )
    query_range = audit["censored_query_time_range_s"]
    assert query_range["test_period_start"] == test_period_start_s
    assert query_range["minimum"] >= test_period_start_s
    landmarks = pd.read_parquet(
        paths["landmarks"], columns=["event_observed", "last_timestamp", "query_s_sec"]
    )
    censored = landmarks.loc[~landmarks["event_observed"]]
    assert len(censored) == sum(audit["censored_samples_by_landmark_hours"].values())
    assert (censored["last_timestamp"] + censored["query_s_sec"] >= test_period_start_s).all()
