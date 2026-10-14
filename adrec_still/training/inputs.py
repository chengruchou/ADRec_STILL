"""Resolve ``--data`` into canonical training inputs.

``--data`` is either a canonical directory (at least ``landmark_samples.parquet``
and ``poi_catalog.parquet``; ``events.parquet`` and ``split_manifest.parquet`` when
present enable the survival training profile) or a raw check-in file. A raw file is
run through the existing adapter and preprocessing pipeline once, into a
user-writable cache, and reused on later runs when the same file is passed again.

Validity checks here are about the supplied data being usable (files present,
catalog non-empty, splits readable). Nothing compares the data to a committed
manifest or hash.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from adrec_still.training import checkpoint as ckpt
from adrec_still.training.progress import Reporter

ADAPTERS = ("gowalla", "foursquare_subset", "foursquare_global")
DEFAULT_HORIZON_HOURS = (0, 6, 12, 24, 48, 72, 168)
DEFAULT_MAX_HISTORY_LENGTH = 100
LANDMARKS = "landmark_samples.parquet"
CATALOG = "poi_catalog.parquet"
EVENTS = "events.parquet"
SPLIT_MANIFEST = "split_manifest.parquet"


@dataclass
class DataSource:
    processed_root: Path
    landmark_path: Path
    catalog_path: Path
    events_path: Path | None
    split_manifest_path: Path | None
    number_of_pois: int
    identity: dict[str, Any]
    splits: list[str]
    horizons_seconds: list[int]
    max_history_length: int | None = DEFAULT_MAX_HISTORY_LENGTH
    origin: dict[str, Any] = field(default_factory=dict)

    def has_split(self, split: str) -> bool:
        return split in self.splits

    def record(self) -> dict[str, Any]:
        return {
            "processed_root": str(self.processed_root),
            "number_of_pois": self.number_of_pois,
            "splits": self.splits,
            "horizons_seconds": self.horizons_seconds,
            "max_history_length": self.max_history_length,
            "identity": self.identity,
            "inputs": self.input_stamp(),
            "origin": self.origin,
        }

    def input_stamp(self) -> dict[str, Any]:
        """What the user supplied, as (path, size, mtime) plus preprocessing parameters:
        which observations, not just which item-ID table. For a raw file that is the
        raw file itself, so re-preprocessing the same file yields the same stamp; for
        a canonical directory it is its Parquet files. Cheap; not a content hash and
        not an admission check, only a way to tell two runs' inputs apart."""
        if self.origin.get("source"):
            return {"kind": "raw", **self.origin["source"]}
        stamp: dict[str, Any] = {"kind": "canonical"}
        for name, path in (("landmarks", self.landmark_path), ("catalog", self.catalog_path),
                           ("events", self.events_path),
                           ("split_manifest", self.split_manifest_path)):
            if path is not None and path.exists():
                info = path.stat()
                stamp[name] = {"path": str(path.resolve()), "size": info.st_size,
                               "mtime_ns": info.st_mtime_ns}
        return stamp


def infer_adapter(path: Path) -> str | None:
    name = path.name.lower()
    if "gowalla" in name:
        return "gowalla"
    if "tsmc2014" in name:
        return "foursquare_subset"
    if "tist2015" in name or "global" in name:
        return "foursquare_global"
    return None


def resolve_data(
    data: str | Path,
    *,
    adapter: str | None = None,
    poi_file: str | Path | None = None,
    cache_dir: str | Path | None = None,
    workers: int | None = None,
    row_limit: int | None = None,
    horizons_hours: tuple[int, ...] = DEFAULT_HORIZON_HOURS,
    max_history_length: int = DEFAULT_MAX_HISTORY_LENGTH,
    reporter: Reporter | None = None,
) -> DataSource:
    reporter = reporter or Reporter()
    path = Path(data)
    if path.is_dir():
        return canonical_source(path, origin={"kind": "canonical", "path": str(path)})
    if not path.is_file():
        raise FileNotFoundError(f"--data {path} is neither a canonical directory nor a raw file")
    adapter = adapter or infer_adapter(path)
    if adapter not in ADAPTERS:
        raise ValueError(
            f"cannot tell which adapter reads {path.name}; pass --adapter "
            f"{{{', '.join(ADAPTERS)}}}"
        )
    if cache_dir is None:
        raise ValueError("a raw file needs --cache-dir (or --output) to hold the processed data")
    return preprocess_raw(
        path, adapter=adapter, poi_file=Path(poi_file) if poi_file else None,
        cache_dir=Path(cache_dir), workers=workers, row_limit=row_limit,
        horizons_hours=horizons_hours, max_history_length=max_history_length,
        reporter=reporter,
    )


def canonical_source(root: Path, *, origin: dict[str, Any] | None = None,
                     max_history_length: int | None = None) -> DataSource:
    landmark_path = root / LANDMARKS
    catalog_path = root / CATALOG
    missing = [str(p) for p in (landmark_path, catalog_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"canonical directory {root} is missing {', '.join(missing)}; a training input "
            f"needs {LANDMARKS} and {CATALOG}"
        )
    events_path = root / EVENTS if (root / EVENTS).is_file() else None
    split_manifest_path = root / SPLIT_MANIFEST if (root / SPLIT_MANIFEST).is_file() else None
    table = pq.read_table(landmark_path, columns=["split", "query_s_sec"])
    if not table.num_rows:
        raise ValueError(f"{landmark_path} has no rows")
    splits = sorted(str(value) for value in pc.unique(table["split"]).to_pylist())
    if "train" not in splits:
        raise ValueError(f"{landmark_path} has no train rows; nothing to fit on")
    horizons = sorted(int(value) for value in pc.unique(table["query_s_sec"]).to_pylist())
    identity = ckpt.data_identity(catalog_path)
    if identity["number_of_pois"] <= 2:
        raise ValueError(f"{catalog_path} holds no train-catalog POIs")
    origin = dict(origin or {})
    if max_history_length is None:
        max_history_length = _recorded_history_length(root)
    return DataSource(
        processed_root=root, landmark_path=landmark_path, catalog_path=catalog_path,
        events_path=events_path, split_manifest_path=split_manifest_path,
        number_of_pois=identity["number_of_pois"], identity=identity, splits=splits,
        horizons_seconds=horizons, max_history_length=max_history_length, origin=origin,
    )


def _recorded_history_length(root: Path) -> int:
    """The history length the landmarks were written with, when the cache recorded it."""
    record = root / "source.json"
    if record.is_file():
        try:
            return int(json.loads(record.read_text(encoding="utf-8"))["max_history_length"])
        except (KeyError, ValueError, TypeError):
            pass
    return DEFAULT_MAX_HISTORY_LENGTH


def _source_stamp(path: Path, poi_file: Path | None, adapter: str, row_limit: int | None,
                  horizons_hours: tuple[int, ...], max_history_length: int) -> dict[str, Any]:
    def stat(p: Path) -> dict[str, Any]:
        info = os.stat(p)
        return {"path": str(p.resolve()), "size": info.st_size, "mtime_ns": info.st_mtime_ns}

    return {
        "adapter": adapter,
        "raw": stat(path),
        "poi_file": stat(poi_file) if poi_file else None,
        "row_limit": row_limit,
        "horizons_hours": list(horizons_hours),
        "max_history_length": max_history_length,
    }


def preprocess_raw(
    path: Path, *, adapter: str, poi_file: Path | None, cache_dir: Path,
    workers: int | None, row_limit: int | None, horizons_hours: tuple[int, ...],
    max_history_length: int, reporter: Reporter,
) -> DataSource:
    from adrec_still.data.pipeline import run_preprocessing

    processed = cache_dir / "processed"
    stamp_path = processed / "source.json"
    stamp = _source_stamp(path, poi_file, adapter, row_limit, horizons_hours, max_history_length)
    if stamp_path.is_file() and (processed / LANDMARKS).is_file():
        try:
            recorded = json.loads(stamp_path.read_text(encoding="utf-8"))
        except ValueError:
            recorded = None
        if recorded == stamp:
            reporter.message(f"reusing preprocessed data in {processed}")
            return canonical_source(
                processed, origin={"kind": "raw", "adapter": adapter, "raw": str(path),
                                   "cached": True, "source": stamp},
                max_history_length=max_history_length,
            )
    name = {"gowalla": "gowalla", "foursquare_subset": "foursquare_subset",
            "foursquare_global": "foursquare_global"}[adapter]
    config: dict[str, Any] = {
        "name": name,
        "adapter": adapter,
        "protocol_version": "local",
        "raw_root": str(path.parent),
        "processed_root": str(processed),
        "source_page": "user-supplied",
        "download_url": "user-supplied",
        "split": {"type": "chronological", "train": 0.7, "validation": 0.1, "test": 0.2},
        "candidate_horizons_hours": list(horizons_hours),
        "max_history_length": int(max_history_length),
        "preserve_long_gaps": True,
        "processing": {"workers": int(workers or 1)},
        "cleaning": {
            "invalid_rows": "count_and_drop",
            "duplicates": "exact_rows_only",
            "preserve_same_timestamp_different_pois": True,
        },
    }
    if adapter == "foursquare_global":
        if poi_file is None:
            raise ValueError("the foursquare_global adapter needs --poi-file")
        config["raw_checkins_filename"] = path.name
        config["raw_pois_filename"] = poi_file.name
    else:
        config["raw_filename"] = path.name
    if adapter.startswith("foursquare"):
        config["coverage"] = {"limitations": ["user-supplied file; coverage not audited"]}
    cache_dir.mkdir(parents=True, exist_ok=True)
    config_path = cache_dir / "dataset.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with reporter.phase("preprocess", adapter=adapter, source=path.name):
        run_preprocessing(
            config_path=config_path, input_path=path, poi_path=poi_file,
            output_dir=processed, manifest_path=cache_dir / "preprocessing_manifest.json",
            audit_path=cache_dir / "preprocessing_audit.json", row_limit=row_limit,
            workers=workers,
        )
    stamp_path.write_text(json.dumps(stamp, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return canonical_source(
        processed, origin={"kind": "raw", "adapter": adapter, "raw": str(path), "cached": False,
                           "source": stamp},
        max_history_length=max_history_length,
    )
