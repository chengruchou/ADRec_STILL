"""Construction and capabilities of the ADRec-STILL family."""

from __future__ import annotations

from typing import Any

from adrec_still.models.adrec_still.model import (
    ADREC_STILL_FAMILY,
    VARIANTS,
    AdrecStillModel,
)

__all__ = ["ADREC_STILL_FAMILY", "VARIANTS", "build_adrec_still_model"]


def build_adrec_still_model(config: dict[str, Any], number_of_pois: int) -> AdrecStillModel:
    name = config["name"]
    if name not in VARIANTS:
        raise ValueError(
            f"unknown adrec_still variant {name!r}; implemented: {sorted(VARIANTS)}"
        )
    excluded = {
        "name", "family", "training", "number_of_pois", "interface_version",
        "parameter_count", "context_dim", "embedding_dim", "hidden_dim",
        "branches", "time_model", "elapsed_conditioning", "item_head",
    }
    keyword_arguments = {
        key: value for key, value in config.items() if key not in excluded
    }
    if "hidden_size" not in keyword_arguments and config.get("embedding_dim"):
        keyword_arguments["hidden_size"] = config["embedding_dim"]
    return AdrecStillModel(name, number_of_pois, **keyword_arguments)
