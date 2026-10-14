"""Config-driven canonical preprocessing orchestration."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import yaml

from adrec_still.data.adapters.base import CleaningAudit
from adrec_still.data.adapters.foursquare import (
    parse_and_clean_global_events,
    parse_and_clean_subset_events,
)
from adrec_still.data.adapters.gowalla import parse_and_clean_events
from adrec_still.data.audit import build_audit_report, sha256_file
from adrec_still.data.parallel import build_base_gaps_parallel, write_landmarks_parallel
from adrec_still.data.schema import UNKNOWN_POI_ID
from adrec_still.data.split import (
    assign_chronological_splits,
    split_manifest_frame,
)

logger = logging.getLogger(__name__)


@contextmanager
def _pipeline_stage(number: int, total: int, label: str) -> Iterator[None]:
    """Log the boundary and duration of one coarse preprocessing stage."""
    started_at = perf_counter()
    logger.info("[%d/%d] %s: started", number, total, label)
    try:
        yield
    except Exception:
        logger.error(
            "[%d/%d] %s: failed after %.1f s",
            number,
            total,
            label,
            perf_counter() - started_at,
        )
        raise
    logger.info(
        "[%d/%d] %s: completed in %.1f s",
        number,
        total,
        label,
        perf_counter() - started_at,
    )


def fit_training_vocabularies(
    events: pd.DataFrame, split_gaps: pd.DataFrame
) -> tuple[dict[str, int], dict[str, int], pd.DataFrame, dict[str, int]]:
    """Fit user/POI IDs and the canonical POI catalog from training events only."""
    train_gaps = split_gaps.loc[split_gaps["split"] == "train"]
    training_source_rows = pd.Index(
        np.unique(
            np.concatenate(
                [
                    train_gaps["source_row_id"].to_numpy(dtype=np.int64),
                    train_gaps["next_source_row_id"].to_numpy(dtype=np.int64),
                ]
            )
        )
    )
    training_events = events.loc[events["source_row_id"].isin(training_source_rows)].copy()
    user_vocab = {
        raw_id: index + 2
        for index, raw_id in enumerate(sorted(training_events["raw_user_id"].unique().tolist()))
    }
    poi_vocab = {
        raw_id: index + 2
        for index, raw_id in enumerate(sorted(training_events["raw_poi_id"].unique().tolist()))
    }
    support = training_events.groupby("raw_poi_id", observed=True).agg(
        train_count=("source_row_id", "size"),
        first_train_time_s=("event_time_utc_s", "min"),
        last_train_time_s=("event_time_utc_s", "max"),
    )
    coordinate_counts = (
        training_events.groupby(["raw_poi_id", "latitude", "longitude"], observed=True)
        .size()
        .rename("coordinate_count")
        .reset_index()
    )
    coordinate_variants = coordinate_counts.groupby("raw_poi_id", observed=True).size()
    coordinate_counts = coordinate_counts.sort_values(
        ["raw_poi_id", "coordinate_count", "latitude", "longitude"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    canonical_coordinates = coordinate_counts.drop_duplicates("raw_poi_id", keep="first")
    catalog = canonical_coordinates.merge(support, on="raw_poi_id", how="inner")
    catalog.insert(0, "poi_id", catalog["raw_poi_id"].map(poi_vocab).astype("int64"))
    catalog = catalog[
        [
            "poi_id",
            "raw_poi_id",
            "latitude",
            "longitude",
            "train_count",
            "first_train_time_s",
            "last_train_time_s",
        ]
    ].sort_values("poi_id")
    metadata = {
        "training_event_count": len(training_events),
        "training_user_vocabulary_size": len(user_vocab),
        "training_poi_vocabulary_size": len(poi_vocab),
        "poi_coordinate_conflict_count": int((coordinate_variants > 1).sum()),
    }
    return user_vocab, poi_vocab, catalog.reset_index(drop=True), metadata


def encode_events(
    events: pd.DataFrame,
    user_vocab: dict[str, int],
    poi_vocab: dict[str, int],
    dataset_name: str = "gowalla",
) -> pd.DataFrame:
    """Apply frozen training vocabularies; unseen held-out IDs map to one."""
    encoded = events.copy()
    encoded.insert(
        1,
        "user_id",
        encoded["raw_user_id"].map(user_vocab).fillna(UNKNOWN_POI_ID).astype("int64"),
    )
    encoded.insert(
        3,
        "poi_id",
        encoded["raw_poi_id"].map(poi_vocab).fillna(UNKNOWN_POI_ID).astype("int64"),
    )
    encoded.insert(0, "dataset", dataset_name)
    return encoded[
        [
            "dataset",
            "raw_user_id",
            "user_id",
            "raw_poi_id",
            "poi_id",
            "event_time_utc_s",
            "latitude",
            "longitude",
            "event_index",
            "source_row_id",
        ]
    ]


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a dataset config against the frozen shared protocol."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not config.get("name"):
        raise ValueError("dataset config must define a non-empty name")
    adapter = config.get("adapter", config["name"])
    supported_adapters = {"gowalla", "foursquare_global", "foursquare_subset"}
    if adapter not in supported_adapters:
        raise ValueError(f"unsupported dataset adapter: {adapter}")
    config["adapter"] = adapter
    if config.get("preserve_long_gaps") is not True:
        raise ValueError("the primary protocol must preserve long gaps")
    split = config["split"]
    if split.get("type") != "chronological":
        raise ValueError("the primary protocol requires a chronological split")
    ratio_sum = float(split["train"]) + float(split["validation"]) + float(split["test"])
    if not math.isclose(ratio_sum, 1.0):
        raise ValueError("train, validation, and test ratios must sum to one")
    cleaning = config.get("cleaning", {})
    expected_cleaning = {
        "invalid_rows": "count_and_drop",
        "duplicates": "exact_rows_only",
        "preserve_same_timestamp_different_pois": True,
    }
    if cleaning != expected_cleaning:
        raise ValueError(f"cleaning must match the frozen protocol: {expected_cleaning}")
    if adapter == "foursquare_global":
        required_files = {"raw_checkins_filename", "raw_pois_filename"}
    else:
        required_files = {"raw_filename"}
    missing_files = sorted(key for key in required_files if not config.get(key))
    if missing_files:
        raise ValueError(f"dataset config is missing raw file settings: {missing_files}")
    if adapter.startswith("foursquare"):
        coverage = config.get("coverage")
        if not isinstance(coverage, dict) or not coverage.get("limitations"):
            raise ValueError("Foursquare configs must document coverage limitations")
    configured_workers = config.get("processing", {}).get("workers", 1)
    if isinstance(configured_workers, bool) or not isinstance(configured_workers, int):
        raise TypeError("processing.workers must be an integer")
    if configured_workers <= 0:
        raise ValueError("processing.workers must be positive")
    return config


def load_gowalla_config(path: str | Path) -> dict[str, Any]:
    """Load the Gowalla config while preserving the original public helper."""
    config = load_dataset_config(path)
    if config["name"] != "gowalla" or config["adapter"] != "gowalla":
        raise ValueError("the Gowalla pipeline requires name and adapter: gowalla")
    return config


def _resolve_raw_paths(
    config: dict[str, Any],
    input_path: str | Path | None,
    poi_path: str | Path | None,
) -> dict[str, Path]:
    raw_root = Path(config["raw_root"])
    if config["adapter"] == "foursquare_global":
        return {
            "checkins": (
                Path(input_path) if input_path else raw_root / config["raw_checkins_filename"]
            ),
            "pois": Path(poi_path) if poi_path else raw_root / config["raw_pois_filename"],
        }
    if poi_path is not None:
        raise ValueError("--poi-input is only valid for the Foursquare global adapter")
    return {"checkins": Path(input_path) if input_path else raw_root / config["raw_filename"]}


def _parse_configured_events(
    config: dict[str, Any],
    raw_paths: dict[str, Path],
    row_limit: int | None,
) -> tuple[pd.DataFrame, CleaningAudit]:
    adapter = config["adapter"]
    if adapter == "gowalla":
        return parse_and_clean_events(raw_paths["checkins"], row_limit=row_limit)
    if adapter == "foursquare_global":
        return parse_and_clean_global_events(
            raw_paths["checkins"], raw_paths["pois"], row_limit=row_limit
        )
    return parse_and_clean_subset_events(raw_paths["checkins"], row_limit=row_limit)


def run_preprocessing(
    config_path: str | Path = "configs/dataset/gowalla.yaml",
    input_path: str | Path | None = None,
    poi_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    audit_path: str | Path | None = None,
    row_limit: int | None = None,
    workers: int | None = None,
) -> dict[str, Path]:
    """Run the complete canonical pipeline and return generated artifact paths."""
    config_path = Path(config_path)
    config = load_dataset_config(config_path)
    dataset_name = str(config["name"])
    if row_limit is not None and row_limit <= 0:
        raise ValueError("row_limit must be positive when provided")
    if workers is not None and (isinstance(workers, bool) or not isinstance(workers, int)):
        raise TypeError("workers must be an integer")
    worker_count = (
        workers if workers is not None else config.get("processing", {}).get("workers", 1)
    )
    if worker_count <= 0:
        raise ValueError("workers must be positive")
    raw_paths = _resolve_raw_paths(config, input_path, poi_path)
    output_dir = Path(output_dir) if output_dir else Path(config["processed_root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        Path(manifest_path)
        if manifest_path
        else Path("data/manifests") / f"{dataset_name}_{config['protocol_version']}.json"
    )
    audit_path = (
        Path(audit_path) if audit_path else Path("results/data") / f"{dataset_name}_audit.json"
    )
    landmark_hours = tuple(int(value) for value in config["candidate_horizons_hours"])
    if any(hours < 0 for hours in landmark_hours) or len(set(landmark_hours)) != len(
        landmark_hours
    ):
        raise ValueError("candidate horizons must be unique and non-negative")

    total_stages = 7
    logger.info(
        "Preprocessing dataset=%s with workers=%d; output=%s",
        dataset_name,
        worker_count,
        output_dir,
    )

    with _pipeline_stage(1, total_stages, "parse and clean raw data"):
        events, cleaning = _parse_configured_events(config, raw_paths, row_limit)
        if len(events) < 2:
            raise ValueError("at least two retained events are required")
        logger.info("Retained %s cleaned events", f"{len(events):,}")

    with _pipeline_stage(2, total_stages, "build consecutive-event gaps"):
        base_gaps = build_base_gaps_parallel(
            events,
            dataset_name=dataset_name,
            workers=worker_count,
            work_dir=output_dir,
        )
        if base_gaps.empty:
            raise ValueError("at least one user must have two retained events")
        logger.info("Built %s gaps", f"{len(base_gaps):,}")

    with _pipeline_stage(3, total_stages, "split data and fit vocabularies"):
        split_config = config["split"]
        split_gaps = assign_chronological_splits(
            base_gaps,
            train_ratio=float(split_config["train"]),
            validation_ratio=float(split_config["validation"]),
        )
        user_vocab, poi_vocab, catalog, vocabulary_metadata = fit_training_vocabularies(
            events, split_gaps
        )
        encoded_events = encode_events(events, user_vocab, poi_vocab, dataset_name=dataset_name)
    paths = {
        "events": output_dir / "events.parquet",
        "poi_catalog": output_dir / "poi_catalog.parquet",
        "split_manifest": output_dir / "split_manifest.parquet",
        "landmarks": output_dir / "landmark_samples.parquet",
        "manifest": manifest_path,
        "audit": audit_path,
    }
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["audit"].parent.mkdir(parents=True, exist_ok=True)
    with _pipeline_stage(4, total_stages, "write core Parquet artifacts"):
        encoded_events.to_parquet(paths["events"], index=False, compression="zstd")
        catalog.to_parquet(paths["poi_catalog"], index=False, compression="zstd")
        split_manifest = split_manifest_frame(split_gaps)
        split_manifest.to_parquet(paths["split_manifest"], index=False, compression="zstd")

    with _pipeline_stage(5, total_stages, "generate landmark samples"):
        landmark_count = write_landmarks_parallel(
            encoded_events,
            split_gaps,
            paths["landmarks"],
            max_history_length=int(config["max_history_length"]),
            landmark_seconds=tuple(hours * 3600 for hours in landmark_hours),
            dataset_name=dataset_name,
            workers=worker_count,
        )
        logger.info("Generated %s landmark samples", f"{landmark_count:,}")

    with _pipeline_stage(6, total_stages, "build audit report"):
        split_manifest_hash = sha256_file(paths["split_manifest"])
        audit = build_audit_report(
            events,
            split_gaps,
            cleaning,
            vocabulary_metadata,
            split_manifest_hash,
            protocol_version=str(config["protocol_version"]),
            landmark_hours=landmark_hours,
            dataset_name=dataset_name,
            coverage_notes=config.get("coverage"),
        )
        audit["overall"]["landmark_samples"] = landmark_count
        paths["audit"].write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    with _pipeline_stage(7, total_stages, "hash artifacts and write manifest"):
        raw_files = {
            role: {
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for role, path in raw_paths.items()
        }
        manifest = {
            "dataset": dataset_name,
            "adapter": config["adapter"],
            "protocol_version": str(config["protocol_version"]),
            "source_page": config["source_page"],
            "download_url": config["download_url"],
            "raw_files": raw_files,
            "config_file": config_path.as_posix(),
            "config_sha256": sha256_file(config_path),
            "split_manifest_sha256": split_manifest_hash,
            "split": split_config,
            "landmark_hours": list(landmark_hours),
            "max_history_length": int(config["max_history_length"]),
            "row_limit": row_limit,
            "processing": {
                "workers": worker_count,
                "partitioning": "contiguous_user_ranges",
                "parallel_stages": ["base_gaps", "landmarks"],
            },
            "generated_files": {
                name: path.as_posix() for name, path in paths.items() if name != "manifest"
            },
            "generated_file_sha256": {
                name: sha256_file(path) for name, path in paths.items() if name != "manifest"
            },
            "cleaning": audit["cleaning"],
            "coverage": audit["coverage"],
            "notes": audit["notes"],
        }
        paths["manifest"].write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    logger.info("Preprocessing completed successfully for dataset=%s", dataset_name)
    return paths
