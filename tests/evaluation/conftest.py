import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from adrec_still.data.landmarks import iter_landmark_records, write_landmarks
from adrec_still.data.split import build_base_gaps, split_manifest_frame


@pytest.fixture
def cohort_artifacts(tmp_path):
    records = []
    for raw, encoded, sources, times in (
        ("cold-a", 1, [10, 11, 12], [0, 20, 40]),
        ("cold-b", 1, [20, 21], [0, 10]),
        ("known", 2, [30, 31], [0, 50]),
        ("early-final", 1, [40], [5]),
    ):
        for ordinal, (source, timestamp) in enumerate(zip(sources, times, strict=True)):
            records.append(
                {
                    "dataset": "toy",
                    "raw_user_id": raw,
                    "user_id": encoded,
                    "source_row_id": source,
                    "event_time_utc_s": timestamp,
                    "event_index": ordinal,
                    "poi_id": 2 + ordinal % 3,
                    "raw_poi_id": str(2 + ordinal % 3),
                }
            )
    events = pd.DataFrame(records)
    gaps = build_base_gaps(events, dataset_name="toy")
    gaps["split"] = "test"
    paths = {
        key: tmp_path / f"{key}.parquet" for key in ("events", "splits", "landmarks", "catalog")
    }
    events.to_parquet(paths["events"], index=False)
    split_manifest_frame(gaps).to_parquet(paths["splits"], index=False)
    write_landmarks(
        iter_landmark_records(events, gaps, landmark_seconds=(0, 10, 20)),
        paths["landmarks"],
        batch_size=3,
    )
    pq.write_table(
        pa.Table.from_pylist([{"poi_id": i, "train_count": 1} for i in (2, 3, 4)]), paths["catalog"]
    )
    return paths
