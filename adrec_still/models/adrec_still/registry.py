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
    keyword_arguments = {
        key: value for key, value in config.items()
        if key not in ("name", "family", "training")
    }
    return AdrecStillModel(name, number_of_pois, **keyword_arguments)
