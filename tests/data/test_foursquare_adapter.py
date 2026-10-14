import json
from pathlib import Path

import pandas as pd
import pytest

from adrec_still.data.adapters.foursquare import (
    parse_and_clean_global_events,
    parse_and_clean_subset_events,
)
from adrec_still.data.pipeline import run_preprocessing

FIXTURES = Path(__file__).parents[1] / "fixtures"
GLOBAL_CHECKINS = FIXTURES / "foursquare_global_checkins_small.txt"
GLOBAL_POIS = FIXTURES / "foursquare_global_pois_small.txt"
SUBSET = FIXTURES / "foursquare_subset_small.txt"
CONFIG_ROOT = Path(__file__).parents[1] / "fixtures" / "configs" / "dataset"


def test_global_parser_joins_pois_and_audits_missingness_deterministically():
    first, first_audit = parse_and_clean_global_events(GLOBAL_CHECKINS, GLOBAL_POIS)
    second, second_audit = parse_and_clean_global_events(GLOBAL_CHECKINS, GLOBAL_POIS)

    pd.testing.assert_frame_equal(first, second)
    assert first_audit == second_audit
    assert first_audit.raw_rows == 11
    assert first_audit.retained_rows == 6
    assert first_audit.duplicate_rows == 1
    assert first_audit.invalid_timestamp_rows == 1
    assert first_audit.invalid_coordinate_rows == 1
    assert first_audit.invalid_identifier_rows == 1
    assert first_audit.invalid_timezone_rows == 1
    assert first_audit.missing_poi_metadata_rows == 1
    assert first_audit.missing_category_name_rows == 1
    assert first_audit.missing_country_code_rows == 1
    assert first_audit.duplicate_poi_metadata_rows == 1
    assert first_audit.poi_coordinate_conflict_count == 1
    p4 = first.loc[first["raw_poi_id"] == "p4"].iloc[0]
    assert p4["latitude"] == pytest.approx(35.0)
    assert p4["longitude"] == pytest.approx(139.0)


def test_subset_parser_preserves_optional_category_missingness_and_exact_timestamps():
    events, audit = parse_and_clean_subset_events(SUBSET)

    assert audit.raw_rows == 11
    assert audit.retained_rows == 7
    assert audit.duplicate_rows == 1
    assert audit.invalid_timestamp_rows == 1
    assert audit.invalid_coordinate_rows == 1
    assert audit.invalid_timezone_rows == 1
    assert audit.invalid_identifier_rows == 1
    assert audit.missing_category_id_rows == 1
    assert audit.missing_category_name_rows == 1
    assert audit.missing_country_code_rows == 0
    u1_times = events.loc[events["raw_user_id"] == "u1", "event_time_utc_s"].tolist()
    assert u1_times[1] - u1_times[0] == 6 * 3600
    assert events.groupby("raw_user_id")["event_index"].apply(list).to_dict() == {
        "u1": [0, 1, 2],
        "u2": [0, 1, 2],
        "u3": [0],
    }


def test_subset_parser_preserves_different_pois_at_the_same_timestamp(tmp_path: Path):
    raw = tmp_path / "same-time.txt"
    raw.write_text(
        "u1\tv1\tc1\tPark\t25\t121\t480\tTue Apr 03 18:00:00 +0000 2012\n"
        "u1\tv2\tc2\tCafe\t25.1\t121.1\t480\tTue Apr 03 18:00:00 +0000 2012\n",
        encoding="latin-1",
    )

    events, audit = parse_and_clean_subset_events(raw)

    assert len(events) == 2
    assert audit.duplicate_rows == 0
    assert events["event_index"].tolist() == [0, 1]


@pytest.mark.parametrize(
    ("dataset_name", "config_name", "input_path", "poi_path"),
    [
        (
            "foursquare_global",
            "foursquare_global.yaml",
            GLOBAL_CHECKINS,
            GLOBAL_POIS,
        ),
        ("foursquare_nyc", "foursquare_nyc.yaml", SUBSET, None),
        ("foursquare_tky", "foursquare_tky.yaml", SUBSET, None),
    ],
)
def test_foursquare_configs_run_the_shared_pipeline(
    tmp_path: Path,
    dataset_name: str,
    config_name: str,
    input_path: Path,
    poi_path: Path | None,
):
    paths = run_preprocessing(
        config_path=CONFIG_ROOT / config_name,
        input_path=input_path,
        poi_path=poi_path,
        output_dir=tmp_path / dataset_name / "processed",
        manifest_path=tmp_path / dataset_name / "manifest.json",
        audit_path=tmp_path / dataset_name / "audit.json",
        workers=1,
    )

    events = pd.read_parquet(paths["events"])
    split_manifest = pd.read_parquet(paths["split_manifest"])
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert set(events["dataset"]) == {dataset_name}
    assert split_manifest["gap_id"].str.startswith(f"{dataset_name}:").all()
    assert audit["dataset"] == dataset_name
    assert "eligible_unique_users_by_split_and_landmark_hours" in audit
    assert audit["derived_cell_support_by_protocol"] == {}
    assert audit["coverage"]["limitations"]
    assert manifest["adapter"].startswith("foursquare")
    assert set(manifest["raw_files"]) == ({"checkins", "pois"} if poi_path else {"checkins"})
    assert "raw_file" not in manifest
    assert "raw_file_size_bytes" not in manifest
    assert "raw_file_sha256" not in manifest
    assert manifest["processing"] == {
        "parallel_stages": ["base_gaps", "landmarks"],
        "partitioning": "contiguous_user_ranges",
        "workers": 1,
    }


def test_parallel_pipeline_matches_sequential_outputs(tmp_path: Path):
    common = {
        "config_path": CONFIG_ROOT / "foursquare_nyc.yaml",
        "input_path": SUBSET,
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
    assert serial_manifest["generated_file_sha256"] == parallel_manifest["generated_file_sha256"]
    assert parallel_manifest["processing"]["workers"] == 2
    assert not list((tmp_path / "parallel" / "processed").glob("still-*-*"))
