import json
from pathlib import Path

import pandas as pd

from adrec_still.data.adapters.gowalla import parse_and_clean_events
from adrec_still.data.pipeline import run_preprocessing

FIXTURE = Path(__file__).parents[1] / "fixtures" / "gowalla_small.txt"
CONFIG = Path(__file__).parents[1] / "fixtures" / "configs" / "dataset" / "gowalla.yaml"


def test_parser_cleaning_is_deterministic():
    first, first_audit = parse_and_clean_events(FIXTURE)
    second, second_audit = parse_and_clean_events(FIXTURE)
    pd.testing.assert_frame_equal(first, second)
    assert first_audit == second_audit
    assert first_audit.raw_rows == 8
    assert first_audit.retained_rows == 5
    assert first_audit.duplicate_rows == 1
    assert first_audit.invalid_timestamp_rows == 1
    assert first_audit.invalid_coordinate_rows == 1
    assert first.iloc[0]["event_time_utc_s"] == 1_577_836_800


def test_same_timestamp_different_pois_are_preserved(tmp_path: Path):
    raw = tmp_path / "same-time.txt"
    raw.write_text(
        "u1\t2020-01-01T00:00:00Z\t25\t121\tp1\nu1\t2020-01-01T00:00:00Z\t25.1\t121.1\tp2\n",
        encoding="utf-8",
    )
    events, audit = parse_and_clean_events(raw)
    assert len(events) == 2
    assert audit.duplicate_rows == 0
    assert events["event_index"].tolist() == [0, 1]


def test_parallel_pipeline_matches_sequential_outputs(tmp_path: Path):
    common = {
        "config_path": CONFIG,
        "input_path": FIXTURE,
    }
    serial = run_preprocessing(
        **common,
        output_dir=tmp_path / "serial" / "processed",
        manifest_path=tmp_path / "serial" / "manifest.json",
        audit_path=tmp_path / "serial" / "audit.json",
        workers=1,
    )
    parallel = run_preprocessing(
        **common,
        output_dir=tmp_path / "parallel" / "processed",
        manifest_path=tmp_path / "parallel" / "manifest.json",
        audit_path=tmp_path / "parallel" / "audit.json",
        workers=2,
    )

    for artifact in ("events", "poi_catalog", "split_manifest", "landmarks"):
        pd.testing.assert_frame_equal(
            pd.read_parquet(serial[artifact]),
            pd.read_parquet(parallel[artifact]),
        )
    serial_audit = json.loads(serial["audit"].read_text(encoding="utf-8"))
    parallel_audit = json.loads(parallel["audit"].read_text(encoding="utf-8"))
    assert serial_audit == parallel_audit
    serial_manifest = json.loads(serial["manifest"].read_text(encoding="utf-8"))
    parallel_manifest = json.loads(parallel["manifest"].read_text(encoding="utf-8"))
    assert serial_manifest["generated_file_sha256"] == parallel_manifest[
        "generated_file_sha256"
    ]
    assert parallel_manifest["processing"]["workers"] == 2
    assert not list((tmp_path / "parallel" / "processed").glob("still-*-*"))
