"""Registry and checkpoint reload for the external baselines (#51).

Mirrors the still-family registry: the shared trainer's checkpoint payload format
is reused, only name-to-class resolution lives here. External research baselines
are neither ``family: baseline`` nor ``family: still`` and must not be disguised
as either to pass those harnesses' checks. Everything needed to rebuild a model in
a fresh Python process is imported directly here — no registration side effects
from test imports are required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from adrec_still.models.baselines.base import BaselineModel
from adrec_still.models.external.joint_lnm import ExternalJointLNM
from adrec_still.models.external.statistics import LogTimeStatistics, legacy_population

EXTERNAL_MODEL_FAMILY = "external"

EXTERNAL_MODELS: dict[str, type[BaselineModel]] = {
    ExternalJointLNM.name: ExternalJointLNM,
}

#: Constructor keys copied from a model config / recorded checkpoint config when
#: present. Every settable width must be listed: a width the checkpoint could not
#: restore would rebuild at the default and then fail to load its state dict.
EXTERNAL_CONSTRUCTOR_KEYS = (
    "mark_embedding_dim",
    "context_dim",
    "num_layers",
    "dropout",
    "mixture_components",
)


def build_external_model(
    model_config: dict[str, Any], number_of_pois: int
) -> BaselineModel:
    name = model_config["name"]
    model_class = EXTERNAL_MODELS.get(name)
    if model_class is None:
        raise ValueError(f"unknown external model: {name}")
    if model_config.get("family") != EXTERNAL_MODEL_FAMILY:
        raise ValueError("external model configs must declare family: external")
    constructor = {
        key: model_config[key] for key in EXTERNAL_CONSTRUCTOR_KEYS if key in model_config
    }
    return model_class(number_of_pois=number_of_pois, **constructor)


def load_external_checkpoint(path: str | Path) -> BaselineModel:
    """Rebuild the fitted external model that produced a shared-trainer checkpoint.

    Restores the architecture from the recorded resolved config, the parameters
    and the affine log-time buffers from the state dict, and the statistics
    provenance record from the recorded config, so a fresh process reproduces the
    exact scoring model without any prior imports beyond this module.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    resolved = payload["model"]
    model_class = EXTERNAL_MODELS.get(resolved["name"])
    if model_class is None:
        raise ValueError(f"checkpoint names an unknown external model: {resolved['name']}")
    constructor = {
        key: resolved[key] for key in EXTERNAL_CONSTRUCTOR_KEYS if key in resolved
    }
    model = model_class(number_of_pois=resolved["number_of_pois"], **constructor)
    model.load_state_dict(payload["state_dict"])
    statistics = resolved.get("log_time_statistics")
    if statistics:
        restored = {
            "mean": float(statistics["log_time_mean"]),
            "std": float(statistics["log_time_std"]),
            "source": str(statistics["source"]),
            "unique_gap_count": int(statistics["unique_gap_count"]),
            "status": str(statistics["status"]),
        }
        if statistics.get("population") is not None:
            restored["population"] = str(statistics["population"])
        else:
            # A checkpoint written before the population field existed is not
            # blanket-relabeled as the canonical fit: the record's own source
            # decides (identity declarations keep the fixture population, the
            # executable fit's source maps to the canonical one, anything else is
            # an explicit unknown). Historical provenance is never rewritten.
            restored["population"] = legacy_population(
                restored["source"], restored["status"]
            )
        # View identity travels with the fit when the record has it; a checkpoint
        # written before the field existed keeps None rather than a guessed view.
        for key in ("view_profile_id", "view_landmarks_sha256"):
            value = statistics.get(key)
            if value is not None:
                restored[key] = str(value)
        model.set_log_time_statistics(LogTimeStatistics(**restored))
    model.is_fitted = True
    return model
