"""Deterministic synthetic canonical-format benchmark for bounded #41 smoke runs.

FIXTURE ONLY: this writes a tiny canonical-schema ``landmark_samples.parquet`` and
``poi_catalog.parquet`` so the CLI -> config -> shared trainer -> validation selection ->
checkpoint -> shared evaluation path can run end to end without any real dataset. It is
not a preprocessing pipeline, produces no Gowalla identity, and results computed on it
must always be labeled fixture/smoke evidence, never empirical model quality.

The generated task is learnable on purpose: each synthetic user mostly revisits one
preferred POI, so a bounded training run can show a decreasing loss and real-inference
movement. Rows include observed known-target training rows plus a few censored and
observed-unknown rows so exclusion counters and censor-aware evaluation paths are
exercised; ``query_s_sec`` values come from the frozen landmark schedule, and every
eligible landmark of a gap (``s < total gap``) is emitted so the per-gap weights sum
to one. Every third user has a long (> 24h) gap, so the fixture carries a non-empty
natural 24h risk set in each split and the ``still-gowalla-core-v1-20260909``
24h checkpoint selection can be exercised end to end on fixture data.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from adrec_still.data.landmarks import LANDMARK_SECONDS

FIXTURE_VERSION = "still-synthetic-fixture-v0.2"
_BASE_TIMESTAMP = 1_600_000_000  # 2020-09-13 UTC; arbitrary fixed anchor
#: Added to every third user's residual so the gap exceeds the 24h landmark.
_LONG_GAP_EXTRA_SECONDS = 30 * 3600


def write_synthetic_benchmark(
    root: str | Path,
    number_of_pois: int = 30,
    users_per_split: tuple[int, int, int] = (24, 8, 8),
    seed: int = 41,
) -> dict[str, object]:
    """Write the fixture tables under ``root`` and return their identity metadata."""
    if number_of_pois < 6:
        raise ValueError("the fixture needs a few non-reserved POIs")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    catalog_ids = list(range(2, number_of_pois))

    rows: list[dict[str, object]] = []
    user_id = 1
    for split, user_count in zip(("train", "validation", "test"), users_per_split):
        for user_index in range(user_count):
            preferred = catalog_ids[(user_id * 3) % len(catalog_ids)]
            filler = catalog_ids[(user_id * 5 + 1) % len(catalog_ids)]
            history_length = 2 + (user_index % 4)
            history = [preferred if i % 2 == 0 else filler for i in range(history_length)]
            timestamps = [
                _BASE_TIMESTAMP + user_id * 1000 + i * 7200 for i in range(history_length)
            ]
            last_timestamp = timestamps[-1]
            residual = int(3600 * (2 + (user_id % 4)) + int(rng.integers(0, 1800)))
            if user_index % 3 == 2:
                residual += _LONG_GAP_EXTRA_SECONDS
            for query_s in LANDMARK_SECONDS:
                if residual <= query_s:
                    continue
                rows.append(
                    {
                        "split": split,
                        "user_id": user_id,
                        "history_poi": history,
                        "history_timestamp": timestamps,
                        "last_timestamp": last_timestamp,
                        "query_s_sec": int(query_s),
                        "event_observed": True,
                        "target_in_train_catalog": True,
                        "residual_time_sec": residual - int(query_s),
                        "next_poi": preferred,
                        "censor_time_sec": 0,
                    }
                )
            if user_index % 5 == 0:
                # One censored row per fifth user: excluded from POI training under the
                # observed-known profile, consumed by censor-aware evaluation.
                rows.append(
                    {
                        "split": split,
                        "user_id": user_id,
                        "history_poi": history,
                        "history_timestamp": timestamps,
                        "last_timestamp": last_timestamp,
                        "query_s_sec": int(LANDMARK_SECONDS[1]),
                        "event_observed": False,
                        "target_in_train_catalog": False,
                        "residual_time_sec": 0,
                        "next_poi": 0,
                        "censor_time_sec": 3600 * 5,
                    }
                )
            if user_index % 7 == 0:
                # One observed row with an unknown (cold-start) exact target.
                rows.append(
                    {
                        "split": split,
                        "user_id": user_id,
                        "history_poi": history,
                        "history_timestamp": timestamps,
                        "last_timestamp": last_timestamp,
                        "query_s_sec": 0,
                        "event_observed": True,
                        "target_in_train_catalog": False,
                        "residual_time_sec": residual,
                        "next_poi": 1,
                        "censor_time_sec": 0,
                    }
                )
            user_id += 1

    table = pa.Table.from_pylist(rows)
    landmark_path = root / "landmark_samples.parquet"
    pq.write_table(table, landmark_path)

    catalog = pa.Table.from_pylist(
        [{"poi_id": poi_id, "train_count": 1} for poi_id in range(number_of_pois)]
    )
    catalog_path = root / "poi_catalog.parquet"
    pq.write_table(catalog, catalog_path)

    digest = hashlib.sha256(landmark_path.read_bytes()).hexdigest()
    return {
        "kind": "synthetic-fixture",
        "fixture_version": FIXTURE_VERSION,
        "landmark_path": str(landmark_path),
        "catalog_path": str(catalog_path),
        "number_of_pois": number_of_pois,
        "row_count": len(rows),
        "seed": seed,
        "landmark_table_sha256": digest,
        "claim_boundary": (
            "fixture-only smoke data: results on it are engineering evidence, never "
            "empirical model quality"
        ),
    }
