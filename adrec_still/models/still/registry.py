"""Model registry and checkpoint reload for the #41 research arms.

Reuses the shared trainer's checkpoint payload format (``trainer.save_checkpoint``);
only the name-to-class resolution lives here, because these research models are not
baselines and must not be disguised under ``family: baseline`` to pass the baseline
harness checks. ``variant_override`` lets the registered same-checkpoint ablation be
re-scored from the trained coherent checkpoint without retraining; the override is
recorded by the caller under its own ``variant_id``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from adrec_still.models.baselines.base import BaselineModel
from adrec_still.models.still.coherent import StillCoherentJoint
from adrec_still.models.still.independent import StillIndependent

STILL_MODEL_FAMILY = "still"

STILL_MODELS: dict[str, type[BaselineModel]] = {
    StillIndependent.name: StillIndependent,
    StillCoherentJoint.name: StillCoherentJoint,
}

#: Constructor keys copied from a model config / resolved checkpoint config when present.
STILL_CONSTRUCTOR_KEYS = (
    "embedding_dim",
    "hidden_dim",
    "num_layers",
    "dropout",
    "mixture_components",
    "variant",
)


def build_still_model(model_config: dict[str, Any], number_of_pois: int) -> BaselineModel:
    name = model_config["name"]
    model_class = STILL_MODELS.get(name)
    if model_class is None:
        raise ValueError(f"unknown still model: {name}")
    if model_config.get("family") != STILL_MODEL_FAMILY:
        raise ValueError("still model configs must declare family: still")
    constructor = {
        key: model_config[key] for key in STILL_CONSTRUCTOR_KEYS if key in model_config
    }
    return model_class(number_of_pois=number_of_pois, **constructor)


def load_still_checkpoint(
    path: str | Path, variant_override: str | None = None
) -> BaselineModel:
    """Rebuild the fitted research model that produced a shared-trainer checkpoint."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    resolved = payload["model"]
    model_class = STILL_MODELS.get(resolved["name"])
    if model_class is None:
        raise ValueError(f"checkpoint names an unknown still model: {resolved['name']}")
    constructor = {
        key: resolved[key] for key in STILL_CONSTRUCTOR_KEYS if key in resolved
    }
    if variant_override is not None:
        if "variant" not in constructor:
            raise ValueError("variant_override applies only to models with variants")
        constructor["variant"] = variant_override
    model = model_class(number_of_pois=resolved["number_of_pois"], **constructor)
    model.load_state_dict(payload["state_dict"])
    model.is_fitted = True
    return model
