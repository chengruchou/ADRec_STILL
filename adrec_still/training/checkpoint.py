"""Two-file checkpoint retention: ``last.pt`` for resuming, ``best.pt`` for loading.

``last.pt`` is replaced atomically after every completed epoch and carries what a
resume needs: weights, optimizer state, counters, RNG state, the curve and the
scalar best-so-far. ``best.pt`` is replaced only when the monitored metric improves
and carries the selected weights with the metadata needed to rebuild the model.
Neither file embeds the other's weights, and no per-epoch generation is kept.

Compatibility on load is about what actually has to match for the state to be
usable: the model configuration, the parameter shapes, the item-ID mapping the
embedding table was trained against, and the recipe-defining training settings.
Host, interpreter, boot, process namespace and library versions are recorded as
information and never compared.
"""

from __future__ import annotations

import getpass
import hashlib
import io
import json
import os
import random
import socket
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from adrec_still.common.atomic_files import replace_file

LAST_FORMAT = "still-train-last-v1"
BEST_FORMAT = "still-train-best-v1"
LAST_NAME = "last.pt"
BEST_NAME = "best.pt"
LOCK_NAME = "run.lock"

#: The one training setting a resume cannot change: the optimizer state that is
#: restored belongs to this optimizer type. Every other setting (learning rate,
#: batch size, negatives, weighting, profile, monitor, horizon, budget, patience,
#: device, workers) is an experiment choice; changes are applied and recorded.
RESUME_RECIPE_KEYS = ("optimizer",)

#: Settings whose change between segments is worth recording in the summary.
RESUME_RECORDED_KEYS = (
    "learning_rate", "weight_decay", "betas", "eps", "amsgrad", "batch_size",
    "negative_samples", "landmark_weighting", "training_profile",
    "early_stopping_metric", "early_stopping_mode", "selection_horizon_sec",
    "max_epochs", "early_stopping_patience", "validation_max_batches",
)

#: Model-config keys that determine parameter shapes or the item-ID table. Only
#: these (plus the state-dict shapes themselves) decide whether a checkpoint can be
#: loaded; fit provenance such as an external model's normalization statistics,
#: dropout, or parameter counts never do.
STRUCTURAL_MODEL_KEYS = (
    "name", "number_of_pois", "catalog_size", "embedding_dim", "hidden_dim",
    "num_layers", "num_heads", "max_sequence_length", "time_hidden_dim",
    "mixture_components", "mark_embedding_dim", "context_dim",
)


class CheckpointError(ValueError):
    """A checkpoint cannot be used for the requested purpose; the message says why."""


class RunLockedError(RuntimeError):
    """Another live process owns the output directory."""


# --------------------------------------------------------------------------- files


def _sync_directory(directory: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_bytes(path: str | Path, content: bytes) -> None:
    """Write to a unique sibling, fsync, then replace ``path`` in one step."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        replace_file(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    _sync_directory(path.parent)


def atomic_json(path: str | Path, record: dict[str, Any]) -> None:
    atomic_bytes(path, json.dumps(record, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def torch_bytes(payload: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def write_torch(path: str | Path, payload: dict[str, Any]) -> str:
    """Atomically persist a torch payload; return its SHA-256."""
    content = torch_bytes(payload)
    atomic_bytes(path, content)
    return hashlib.sha256(content).hexdigest()


def load_payload(path: str | Path, expected_format: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise CheckpointError(f"cannot read checkpoint {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("format") != expected_format:
        found = payload.get("format") if isinstance(payload, dict) else type(payload).__name__
        raise CheckpointError(
            f"{path} is not a {expected_format} checkpoint (found {found!r}); pass last.pt "
            "to --resume and best.pt to --weights"
        )
    return payload


# ------------------------------------------------------------------ data identity


def catalog_fingerprint(catalog_path: str | Path) -> str:
    """Digest of the item-ID mapping (``poi_id`` <-> ``raw_poi_id``), in ID order.

    Two catalogs with the same fingerprint assign the same row of the embedding
    table to the same real-world place, which is what makes trained weights
    meaningful on the data they are loaded against.
    """
    table = pq.read_table(Path(catalog_path), columns=["poi_id", "raw_poi_id"])
    table = table.sort_by("poi_id")
    digest = hashlib.sha256()
    for poi_id, raw in zip(
        table.column("poi_id").to_pylist(), table.column("raw_poi_id").to_pylist(), strict=True
    ):
        digest.update(f"{poi_id}\t{raw}\n".encode())
    return digest.hexdigest()


def number_of_pois(catalog_path: str | Path) -> int:
    table = pq.read_table(Path(catalog_path), columns=["poi_id"])
    if not table.num_rows:
        raise CheckpointError("the POI catalog is empty")
    return int(pc.max(table.column("poi_id")).as_py()) + 1


def data_identity(catalog_path: str | Path) -> dict[str, Any]:
    return {
        "catalog_fingerprint": catalog_fingerprint(catalog_path),
        "number_of_pois": number_of_pois(catalog_path),
    }


# ------------------------------------------------------------------------- RNG


def capture_rng(negative_generator: torch.Generator, sampler: Any, device: torch.device) -> dict:
    generator = getattr(sampler, "_generator", getattr(sampler, "generator", None))
    record: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        "negative": negative_generator.get_state(),
    }
    if isinstance(generator, torch.Generator):
        record["sampler"] = generator.get_state()
    return record


def restore_rng(
    record: dict, negative_generator: torch.Generator, sampler: Any, device: torch.device
) -> dict[str, Any]:
    """Restore what can be restored; report what could not.

    A CUDA stream captured on one machine cannot be replayed on a CPU-only host or
    on a host with a different device count. The run continues from the reseeded
    stream and the report says so; bitwise trajectory replay across hardware is
    not promised.
    """
    report: dict[str, Any] = {"restored": [], "skipped": []}
    generator = getattr(sampler, "_generator", getattr(sampler, "generator", None))
    random.setstate(record["python"])
    np.random.set_state(record["numpy"])
    torch.set_rng_state(record["torch_cpu"])
    negative_generator.set_state(record["negative"])
    report["restored"] += ["python", "numpy", "torch_cpu", "negative"]
    if isinstance(generator, torch.Generator) and record.get("sampler") is not None:
        generator.set_state(record["sampler"])
        report["restored"].append("sampler")
    else:
        report["skipped"].append("sampler")
    cuda_states = record.get("torch_cuda") or []
    if device.type == "cuda" and cuda_states and len(cuda_states) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(cuda_states)
        report["restored"].append("torch_cuda")
    elif device.type == "cuda" or cuda_states:
        report["skipped"].append("torch_cuda")
    return report


# ------------------------------------------------------------------------ lock


class RunLock:
    """Exclusive ownership of an output directory for the duration of a run.

    Uses an advisory OS lock on ``<output>/run.lock`` so a second writer is
    refused while the first is alive, and a stale file left by a crash does not
    block the next run.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.path = Path(output_dir) / LOCK_NAME
        self._handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+")
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.seek(0)
            owner = handle.read().strip()
            handle.close()
            raise RunLockedError(
                f"{self.path.parent} is in use by another live run"
                + (f" ({owner})" if owner else "")
                + "; use a different --output"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "user": _current_user(),
                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
        )
        handle.flush()
        self._handle = handle
        return self

    def __exit__(self, *exc: object) -> None:
        if self._handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None
            try:
                self.path.unlink()
            except OSError:
                pass


def _current_user() -> str | None:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - purely informational
        return None


# ---------------------------------------------------------------- compatibility


def state_dict_differences(
    expected: dict[str, torch.Tensor], found: dict[str, torch.Tensor]
) -> list[str]:
    problems = []
    for key in sorted(set(expected) - set(found)):
        problems.append(f"missing parameter {key}")
    for key in sorted(set(found) - set(expected)):
        problems.append(f"unexpected parameter {key}")
    for key in sorted(set(expected) & set(found)):
        left, right = expected[key], found[key]
        if tuple(left.shape) != tuple(right.shape):
            problems.append(f"{key}: shape {tuple(right.shape)} != {tuple(left.shape)}")
        elif left.dtype != right.dtype:
            problems.append(f"{key}: dtype {right.dtype} != {left.dtype}")
    return problems


def _flat_differences(prefix: str, expected: Any, found: Any, out: list[str]) -> None:
    if isinstance(expected, dict) and isinstance(found, dict):
        for key in sorted(set(expected) | set(found)):
            _flat_differences(f"{prefix}.{key}" if prefix else str(key),
                              expected.get(key), found.get(key), out)
        return
    if isinstance(expected, (list, tuple)):
        expected = list(expected)
    if isinstance(found, (list, tuple)):
        found = list(found)
    if expected != found:
        out.append(f"{prefix}: checkpoint has {found!r}, this run has {expected!r}")


def model_differences(expected: dict[str, Any], found: dict[str, Any]) -> list[str]:
    """Structural differences only (see :data:`STRUCTURAL_MODEL_KEYS`)."""
    out: list[str] = []
    for key in STRUCTURAL_MODEL_KEYS:
        if key in expected or key in found:
            _flat_differences(f"model.{key}", expected.get(key), found.get(key), out)
    return out


def recorded_differences(expected: dict[str, Any], found: dict[str, Any]) -> list[str]:
    """Training-setting changes between the checkpoint and this run, for the record."""
    out: list[str] = []
    for key in RESUME_RECORDED_KEYS:
        _flat_differences(f"training.{key}", expected.get(key), found.get(key), out)
    return out


def data_differences(expected: dict[str, Any], found: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if expected.get("number_of_pois") != found.get("number_of_pois"):
        out.append(
            f"catalog width {found.get('number_of_pois')} != {expected.get('number_of_pois')}"
        )
    if expected.get("catalog_fingerprint") != found.get("catalog_fingerprint"):
        out.append(
            "item-ID mapping differs (poi_id <-> raw_poi_id); the embedding rows would "
            "refer to different places"
        )
    return out


def recipe_differences(expected: dict[str, Any], found: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key in RESUME_RECIPE_KEYS:
        _flat_differences(f"training.{key}", expected.get(key), found.get(key), out)
    return out


def require_compatible(purpose: str, path: str | Path, problems: list[str]) -> None:
    if problems:
        listed = "\n  - ".join(problems)
        raise CheckpointError(f"cannot {purpose} from {path}:\n  - {listed}")


# ---------------------------------------------------------------- payloads


def last_payload(
    *,
    family: str,
    model_config: dict[str, Any],
    data: dict[str, Any],
    training: dict[str, Any],
    model_state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    rng: dict[str, Any],
    completed_epoch: int,
    optimizer_updates: int,
    curve: list[dict[str, Any]],
    best: dict[str, Any] | None,
    early_stopped: bool,
    training_complete: bool,
    sampler_identity: dict[str, Any],
    resumed_from: dict[str, Any] | None,
    environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": LAST_FORMAT,
        "family": family,
        "model": model_config,
        "data": data,
        "training": training,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "rng": rng,
        "completed_epoch": int(completed_epoch),
        "optimizer_updates": int(optimizer_updates),
        "curve": curve,
        "best": best,
        "early_stopped": bool(early_stopped),
        "training_complete": bool(training_complete),
        "sampler_identity": sampler_identity,
        "resumed_from": resumed_from,
        "environment": environment,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def best_payload(
    *,
    family: str,
    model_config: dict[str, Any],
    data: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    selection: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": BEST_FORMAT,
        "family": family,
        "model": model_config,
        "data": data,
        "state_dict": state_dict,
        "selection": selection,
        "environment": environment,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def environment_record(device: torch.device) -> dict[str, Any]:
    """Informational only: recorded in every checkpoint, compared by nothing."""
    import platform

    return {
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "host": socket.gethostname(),
    }


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def save_weights(
    path: str | Path, model: torch.nn.Module, *, family: str = "baseline",
    selection: dict[str, Any] | None = None, data: dict[str, Any] | None = None,
) -> str:
    """Write a model's weights in the ``best.pt`` layout; return the SHA-256."""
    payload = best_payload(
        family=family, model_config=model.resolved_config(), data=dict(data or {}),
        state_dict=cpu_state_dict(model), selection=dict(selection or {}),
        environment=environment_record(torch.device("cpu")),
    )
    return write_torch(path, payload)
