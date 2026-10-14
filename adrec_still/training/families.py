"""Model families: how each backend is built, fed and reloaded.

Three families share the training loop and differ only in construction and in
what they need prepared from the data: the encoder baselines read the canonical
landmark table directly; the STILL arms and the external LNM train on the
administrative-censor survival view (built once per run from ``events.parquet``
and ``split_manifest.parquet``, because that view *is* the definition of their
loss), and the external model additionally fits its log-time statistics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from adrec_still.data.dataset import LandmarkDataset
from adrec_still.data.training_views import EVENT_ONLY_TRAINING_PROFILE
from adrec_still.models.baselines.harness import build_fitted_model
from adrec_still.models.baselines.trainer import build_baseline_model
from adrec_still.models.external.registry import EXTERNAL_MODEL_FAMILY, build_external_model
from adrec_still.models.still.losses import SURVIVAL_TRAINING_PROFILE
from adrec_still.models.still.registry import STILL_MODEL_FAMILY, build_still_model
from adrec_still.training import checkpoint as ckpt
from adrec_still.training.defaults import backend
from adrec_still.training.inputs import DataSource
from adrec_still.training.progress import Reporter

FAMILY_BASELINE = "baseline"
SURVIVAL_VIEW_NAME = "survival_training.parquet"
SURVIVAL_MANIFEST_NAME = "survival_training_manifest.json"


def build_model(model_config: dict[str, Any], number_of_pois: int):
    family = model_config.get("family") or backend(model_config["name"])["family"]
    config = {**model_config, "family": family}
    if family == FAMILY_BASELINE:
        return build_baseline_model(config, number_of_pois)
    if family == STILL_MODEL_FAMILY:
        return build_still_model(config, number_of_pois)
    if family == EXTERNAL_MODEL_FAMILY:
        return build_external_model(config, number_of_pois)
    raise ValueError(f"unknown model family: {family}")


def fit_floor(model_config: dict[str, Any], data: DataSource, reporter: Reporter | None = None):
    return build_fitted_model(model_config, data.processed_root, reporter=reporter)


def load_model(path: str | Path, *, device: torch.device | None = None):
    """Rebuild the model a ``best.pt`` holds, on ``device`` (CPU by default)."""
    payload = ckpt.load_payload(path, ckpt.BEST_FORMAT)
    family = payload["family"]
    resolved = dict(payload["model"])
    model = build_model({**resolved, "family": family}, int(resolved["number_of_pois"]))
    model.load_state_dict(payload["state_dict"])
    if family == EXTERNAL_MODEL_FAMILY:
        restore_external_statistics(model, resolved.get("log_time_statistics"))
    model.is_fitted = True
    if device is not None:
        model.to(device)
    return model, payload


def restore_external_statistics(model, statistics: dict[str, Any] | None) -> None:
    """Put a checkpoint's recorded log-time statistics back on an external model."""
    if not statistics:
        return
    from adrec_still.models.external.statistics import LogTimeStatistics, legacy_population

    restored: dict[str, Any] = {
        "mean": float(statistics["log_time_mean"]),
        "std": float(statistics["log_time_std"]),
        "source": str(statistics["source"]),
        "unique_gap_count": int(statistics["unique_gap_count"]),
        "status": str(statistics["status"]),
    }
    if statistics.get("population") is not None:
        restored["population"] = str(statistics["population"])
    else:
        restored["population"] = legacy_population(restored["source"], restored["status"])
    for key in ("view_profile_id", "view_landmarks_sha256"):
        if statistics.get(key) is not None:
            restored[key] = str(statistics[key])
    model.set_log_time_statistics(LogTimeStatistics(**restored))


# ------------------------------------------------------------- training data


def prepare_training_source(
    settings: dict[str, Any],
    data: DataSource,
    cache_dir: Path,
    *,
    reporter: Reporter | None = None,
) -> tuple[LandmarkDataset, dict[str, Any]]:
    """The train-split dataset for a training profile, plus a record of what was built."""
    reporter = reporter or Reporter()
    profile = settings.get("training_profile")
    if profile == SURVIVAL_TRAINING_PROFILE:
        view, manifest = survival_view(data, cache_dir, reporter=reporter)
        return LandmarkDataset(view, split="train"), {
            "training_view": SURVIVAL_TRAINING_PROFILE,
            "survival_view": str(view),
            "survival_profile_id": manifest.get("profile_id"),
        }
    if profile == EVENT_ONLY_TRAINING_PROFILE:
        return LandmarkDataset(
            data.landmark_path, split="train", training_view=EVENT_ONLY_TRAINING_PROFILE
        ), {"training_view": EVENT_ONLY_TRAINING_PROFILE}
    return LandmarkDataset(data.landmark_path, split="train"), {
        "training_view": "canonical-landmarks",
    }


def survival_view(
    data: DataSource, cache_dir: Path, *, reporter: Reporter | None = None
) -> tuple[Path, dict[str, Any]]:
    """Build (or reuse) the administrative-censor training view for this data."""
    from adrec_still.data.survival import build_survival_training_profile

    reporter = reporter or Reporter()
    if data.events_path is None or data.split_manifest_path is None:
        raise FileNotFoundError(
            "the survival training profile needs events.parquet and split_manifest.parquet "
            f"beside {data.landmark_path}; run preprocessing on the raw file (pass the raw "
            "file to --data) or point --data at a complete canonical directory"
        )
    view = cache_dir / SURVIVAL_VIEW_NAME
    manifest_path = cache_dir / SURVIVAL_MANIFEST_NAME
    stamp = _survival_view_stamp(data)
    if view.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # A derived cache is reused only for the inputs and parameters it was built
        # from; changed events, splits, horizons or history length rebuild it.
        if manifest.get("landmarks_sha256") and manifest.get("built_from") == stamp:
            reporter.message(f"reusing survival training view {view}")
            return view, manifest
        reporter.message("survival training view is stale for this data; rebuilding")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with reporter.phase("survival-view"):
        manifest = build_survival_training_profile(
            data.events_path,
            data.split_manifest_path,
            view,
            manifest_path,
            landmark_seconds=tuple(data.horizons_seconds),
            max_history_length=data.max_history_length,
        )
    manifest = {**manifest, "source_events": str(data.events_path), "built_from": stamp}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    return view, manifest


def _survival_view_stamp(data: DataSource) -> dict[str, Any]:
    """What the view depends on: the two input files (path, size, mtime) and the build
    parameters. Cheap to compute; a content hash of large Parquet is not needed for
    cache invalidation."""

    def stat(path: Path) -> dict[str, Any]:
        info = path.stat()
        return {"path": str(path.resolve()), "size": info.st_size, "mtime_ns": info.st_mtime_ns}

    return {
        "events": stat(data.events_path),
        "split_manifest": stat(data.split_manifest_path),
        "horizons_seconds": list(data.horizons_seconds),
        "max_history_length": data.max_history_length,
    }


def fit_external_statistics(
    model,
    model_config: dict[str, Any],
    data: DataSource,
    train_record: dict[str, Any],
    cache_dir: Path,
    *,
    reporter: Reporter | None = None,
) -> dict[str, Any]:
    """Apply the external model's log-time statistics policy and return its record."""
    from adrec_still.models.external.statistics import (
        IDENTITY_STATISTICS,
        fit_log_time_statistics,
        fit_visible_view_log_time_statistics,
    )

    reporter = reporter or Reporter()
    mode = model_config.get("log_time_statistics", "identity")
    with reporter.phase("log-time-statistics", mode=mode):
        if mode == "identity":
            model.set_log_time_statistics(IDENTITY_STATISTICS)
        elif mode == "fit_train_unique_gaps":
            model.set_log_time_statistics(fit_log_time_statistics(data.landmark_path))
        elif mode == "fit_visible_training_view":
            view = train_record.get("survival_view")
            if view is None:
                raise ValueError(
                    "log_time_statistics: fit_visible_training_view needs the survival "
                    "training profile; the model config's training_profile does not build one"
                )
            manifest = json.loads(
                (cache_dir / SURVIVAL_MANIFEST_NAME).read_text(encoding="utf-8")
            )
            model.set_log_time_statistics(
                fit_visible_view_log_time_statistics(view, manifest)
            )
        else:
            raise ValueError(f"unknown log_time_statistics mode: {mode!r}")
    return dict(model.log_time_statistics_record)
