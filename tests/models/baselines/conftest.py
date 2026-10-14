"""Shared fixtures for the baseline test suite.

The synthetic Gowalla-shaped benchmark is built once per session through the
committed preprocessing pipeline, so every baseline is tested against the same
canonical artifacts the real runs consume.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from adrec_still.data.pipeline import run_preprocessing

REPOSITORY_ROOT = Path(__file__).parents[3]
DATASET_CONFIG = REPOSITORY_ROOT / "tests" / "fixtures" / "configs" / "dataset" / "gowalla.yaml"
MODEL_CONFIG = REPOSITORY_ROOT / "tests" / "fixtures" / "configs" / "model" / "global_pop.yaml"
EXPERIMENT_CONFIG = REPOSITORY_ROOT / "tests" / "fixtures" / "configs" / "experiment" / "gowalla_global_pop.yaml"

GAP_HOURS = (1, 7, 13, 25, 49, 73, 169, 2, 5, 30, 100, 200, 3)
POI_CYCLE = ("p000", "p000", "p000", "p001", "p000", "p001", "p002", "p000", "p001", "p003")
USER_COUNT = 8
EVENTS_PER_USER = 14
LATE_EVENT_INDEX = 11


#: The canonical evaluation profile the committed Gowalla protocol registers under.
PROFILE_ID = "gowalla-v0.1"


@pytest.fixture(autouse=True)
def pinned_git_metadata(monkeypatch):
    """Git state is recorded, never gated; pin it so tests do not read the developer's
    real worktree."""
    monkeypatch.setattr("adrec_still.models.baselines.harness.git_worktree_dirty", lambda: False)
    monkeypatch.setattr("adrec_still.models.baselines.harness.git_commit_sha", lambda: "1" * 40)


def write_synthetic_raw(path: Path) -> Path:
    """Deterministic Gowalla-shaped fixture with skewed POIs and late cold-start POIs."""
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    lines = []
    for user in range(USER_COUNT):
        moment = start + timedelta(hours=user)
        for index in range(EVENTS_PER_USER):
            # Odd users introduce brand-new POIs late in time, so the held-out
            # splits contain both cold-start and train-catalog targets.
            cold_start = user % 2 == 1 and index >= LATE_EVENT_INDEX
            poi = f"p_late_{user}_{index}" if cold_start else POI_CYCLE[index % len(POI_CYCLE)]
            lines.append(
                "\t".join(
                    [
                        f"u{user}",
                        moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        f"{25.0 + user * 0.1:.4f}",
                        f"{121.0 + index * 0.1:.4f}",
                        poi,
                    ]
                )
            )
            moment += timedelta(hours=GAP_HOURS[index % len(GAP_HOURS)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_synthetic_foursquare_raw(path: Path) -> Path:
    """The same check-in schedule as :func:`write_synthetic_raw`, in the TSMC2014 layout.

    Tab-separated user, venue, category ID, category name, latitude, longitude,
    timezone offset in minutes and UTC time, as the ``foursquare_subset`` adapter reads
    the NYC and TKY files. Sharing the schedule keeps the two fixtures' populations
    comparable while exercising the Foursquare parser.
    """
    gowalla = write_synthetic_raw(path.with_name(f"{path.stem}.gowalla-layout.txt"))
    lines = []
    for line in gowalla.read_text(encoding="utf-8").splitlines():
        user, moment, latitude, longitude, poi = line.split("\t")
        when = datetime.strptime(moment, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        lines.append(
            "\t".join(
                [
                    user,
                    poi,
                    "c000",
                    "Fixture Venue",
                    latitude,
                    longitude,
                    "0",
                    when.strftime("%a %b %d %H:%M:%S +0000 %Y"),
                ]
            )
        )
    gowalla.unlink()
    path.write_text("\n".join(lines) + "\n", encoding="latin-1")
    return path


@pytest.fixture(scope="session")
def synthetic_benchmark(tmp_path_factory) -> dict[str, Path]:
    """Build a tiny end-to-end canonical benchmark from the committed dataset protocol."""
    root = tmp_path_factory.mktemp("baselines")
    raw = write_synthetic_raw(root / "loc-gowalla_totalCheckins.txt")
    processed = root / "processed"
    paths = run_preprocessing(
        config_path=DATASET_CONFIG,
        input_path=raw,
        output_dir=processed,
        manifest_path=root / "gowalla_v0.1.json",
        audit_path=root / "gowalla_audit.json",
    )
    dataset_config = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))
    dataset_config["processed_root"] = str(processed)
    dataset_config_path = root / "dataset.yaml"
    dataset_config_path.write_text(yaml.safe_dump(dataset_config), encoding="utf-8")
    experiment = yaml.safe_load(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    experiment["dataset_config"] = str(dataset_config_path)
    experiment["model_config"] = str(MODEL_CONFIG)
    experiment["manifest"] = str(paths["manifest"])
    experiment["output_dir"] = str(root / "results")
    experiment["evaluation"]["batch_size"] = 16
    experiment_path = root / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    return {
        "root": root,
        "processed": processed,
        "landmarks": paths["landmarks"],
        "catalog": paths["poi_catalog"],
        "experiment": experiment_path,
        "dataset_config": dataset_config_path,
        "manifest": Path(paths["manifest"]),
    }
