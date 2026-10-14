"""Shared fixtures for the external-baseline tests (#51).

The canonical benchmark builder is rebuilt here rather than imported from
``tests/models/still/``: a ``conftest.py`` fixture is visible only within its own
directory, and importing across test packages fails under the import mode CI
collects with (``ModuleNotFoundError: No module named 'tests'``). Same convention
as ``tests/models/still/conftest.py`` and ``tests/campaign/conftest.py``.

Nothing here is a Gowalla result: eight synthetic users through the committed
preprocessing pipeline, which yields real ``events.parquet`` /
``split_manifest.parquet`` artifacts (the #38 cohort index needs both), a #39
training view that can be built from them, and a non-empty natural 24h validation
risk set for the Table 2 selection population.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from adrec_still.data.pipeline import run_preprocessing

REPOSITORY_ROOT = Path(__file__).parents[3]
DATASET_CONFIG = REPOSITORY_ROOT / "tests" / "fixtures" / "configs" / "dataset" / "gowalla.yaml"

GAP_HOURS = (1, 7, 13, 25, 49, 73, 169, 2, 5, 30, 100, 200, 3)
POI_CYCLE = ("p000", "p000", "p000", "p001", "p000", "p001", "p002", "p000", "p001", "p003")
USER_COUNT = 8
EVENTS_PER_USER = 14
LATE_EVENT_INDEX = 11


def write_synthetic_raw(path: Path) -> Path:
    """Deterministic Gowalla-shaped fixture with skewed POIs and late cold-start POIs."""
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    lines = []
    for user in range(USER_COUNT):
        moment = start + timedelta(hours=user)
        for index in range(EVENTS_PER_USER):
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


@pytest.fixture(scope="session")
def canonical_benchmark(tmp_path_factory) -> dict[str, Path]:
    """A tiny end-to-end canonical benchmark from the committed dataset protocol."""
    root = tmp_path_factory.mktemp("external_canonical")
    raw = write_synthetic_raw(root / "loc-gowalla_totalCheckins.txt")
    processed = root / "processed"
    paths = run_preprocessing(
        config_path=DATASET_CONFIG,
        input_path=raw,
        output_dir=processed,
        manifest_path=root / "gowalla_v0.1.json",
        audit_path=root / "gowalla_audit.json",
    )
    return {
        "root": root,
        "processed": processed,
        "events": Path(paths["events"]),
        "split_manifest": Path(paths["split_manifest"]),
        "landmarks": Path(paths["landmarks"]),
        "catalog": Path(paths["poi_catalog"]),
        "manifest": Path(paths["manifest"]),
    }
