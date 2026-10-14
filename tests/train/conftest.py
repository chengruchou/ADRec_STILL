"""Fixtures for the dataset-first training tests: a tiny raw file and its canonical form."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[2]
_BASELINE_CONFTEST = REPOSITORY_ROOT / "tests" / "models" / "baselines" / "conftest.py"


@pytest.fixture(scope="session")
def raw_gowalla_file(tmp_path_factory) -> Path:
    """The Gowalla-shaped synthetic check-in file shared with the baseline tests."""
    writer = runpy.run_path(str(_BASELINE_CONFTEST))["write_synthetic_raw"]
    root = tmp_path_factory.mktemp("raw")
    return writer(root / "loc-gowalla_totalCheckins.txt")


@pytest.fixture(scope="session")
def canonical_data(raw_gowalla_file, tmp_path_factory) -> Path:
    """The raw file preprocessed once through the public input path."""
    from adrec_still.training.inputs import resolve_data

    cache = tmp_path_factory.mktemp("cache")
    source = resolve_data(raw_gowalla_file, cache_dir=cache)
    return source.processed_root


#: Small enough to train in seconds on CPU; the recipe is otherwise the shared one.
SMALL_NEURAL = {
    "embedding_dim": 8,
    "hidden_dim": 8,
    "training.negative_samples": 4,
    "training.batch_size": 16,
}


def small_spec(data: Path, output: Path, model: str, **overrides):
    from adrec_still.training.service import RunSpec

    fields = {
        "data": data, "output": output, "model": model, "device": "cpu", "progress": "off",
        "overrides": dict(SMALL_NEURAL), "seed": 41,
    }
    fields.update(overrides)
    return RunSpec(**fields)
