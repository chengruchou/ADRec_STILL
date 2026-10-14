"""Capacity estimation for the literal per-mark LNM head (#51).

The literal head costs ``3 * C * V * (d + 1)`` parameters (locations, raw
log-scales and weight logits per legal mark, from a ``d``-wide context with
bias). This module turns an *actual* catalog size into theoretical
parameter/memory figures **before any model is allocated** — it never guesses a
catalog, and its theoretical totals are exact for :class:`ExternalJointLNM`
(cross-checked against ``parameter_count()`` in the tests), so the pre-allocation
feasibility gate can run without constructing anything.

Component definitions (each parameter counted exactly once):

- ``encoder_parameters`` = GRU + mark embedding + ``context_init`` — everything
  outside the two heads, matching ``ExternalJointLNM.encoder_parameter_count()``.
- ``mark_head_parameters`` / ``time_head_parameters`` = the two linear heads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: float32 training footprint per parameter: weights + gradients + two Adam states.
TRAIN_BYTES_PER_PARAMETER = 4 + 4 + 8
CHECKPOINT_BYTES_PER_PARAMETER = 4
FLOAT64_BYTES = 8

#: [batch, V, C] float64 tensors alive at once in the scoring path (locations,
#: scales, log-weights, per-component densities) — a working-set bound, not a
#: profiler measurement.
SCORING_WORKING_SET_TENSORS = 4


def theoretical_encoder_parameter_count(
    number_of_pois: int,
    mark_embedding_dim: int,
    context_dim: int,
    num_layers: int,
) -> int:
    """Exact parameter count of the paper-shaped encoder without building it.

    Covers the GRU (per layer: ``3d`` gates over ``input + d`` inputs with two
    bias vectors), the mark embedding table and ``context_init`` — the same
    scope as ``ExternalJointLNM.encoder_parameter_count()``.
    """
    gru = 0
    for layer in range(num_layers):
        input_size = (1 + mark_embedding_dim) if layer == 0 else context_dim
        gru += 3 * context_dim * (input_size + context_dim) + 2 * (3 * context_dim)
    embedding = number_of_pois * mark_embedding_dim
    return gru + embedding + context_dim


def literal_head_capacity_estimate(
    catalog_size: int,
    mixture_components: int,
    context_dim: int,
    number_of_pois: int,
    mark_embedding_dim: int,
    batch_size: int,
    num_layers: int = 1,
    encoder_parameters: int | None = None,
) -> dict[str, Any]:
    """Theoretical parameter/memory estimate for one configuration.

    ``encoder_parameters`` may be a measured
    ``ExternalJointLNM.encoder_parameter_count()`` (GRU + embedding +
    ``context_init``); when omitted it is computed exactly from the same formula,
    so the total equals the real model's ``parameter_count()`` in both cases and
    nothing is counted twice.
    """
    time_head = 3 * mixture_components * catalog_size * (context_dim + 1)
    mark_head = number_of_pois * (context_dim + 1)
    if encoder_parameters is None:
        encoder_parameters = theoretical_encoder_parameter_count(
            number_of_pois, mark_embedding_dim, context_dim, num_layers
        )
    total = time_head + mark_head + encoder_parameters
    scoring_tensor = batch_size * catalog_size * mixture_components * FLOAT64_BYTES
    return {
        "estimate_kind": "theoretical",
        "inputs": {
            "catalog_size": catalog_size,
            "mixture_components": mixture_components,
            "context_dim": context_dim,
            "number_of_pois": number_of_pois,
            "mark_embedding_dim": mark_embedding_dim,
            "num_layers": num_layers,
            "batch_size": batch_size,
        },
        "time_head_parameters": time_head,
        "mark_head_parameters": mark_head,
        "encoder_parameters": encoder_parameters,
        "total_parameters": total,
        "training_bytes": total * TRAIN_BYTES_PER_PARAMETER,
        "checkpoint_bytes": total * CHECKPOINT_BYTES_PER_PARAMETER,
        "scoring_tensor_bytes_each": scoring_tensor,
        "scoring_working_set_bytes": scoring_tensor * SCORING_WORKING_SET_TENSORS,
        "notes": (
            "totals are exact for ExternalJointLNM; byte figures are lower bounds "
            "(activations, optimizer overheads beyond Adam moments, allocator "
            "slack and dataloader buffers are excluded). scoring_working_set_bytes "
            "bounds the SCORING path only: the survival training step evaluates the "
            "same [batch, V, C] float64 per-mark mixture inside the autograd graph, "
            "so those intermediates are additionally retained for backward and are "
            "not covered by this figure"
        ),
        "training_working_set_covered": False,
    }


def catalog_capacity_report(
    catalog_path: str | Path,
    mixture_components: int,
    context_dim: int,
    mark_embedding_dim: int,
    batch_size: int,
    num_layers: int = 1,
) -> dict[str, Any]:
    """Pre-allocation estimate for a real catalog, or an explicit unknown.

    Reads only the actual frozen catalog table (no tensors allocated); an absent
    table reports ``unknown_catalog_not_local`` instead of a guessed size.
    """
    catalog_path = Path(catalog_path)
    if not catalog_path.exists():
        return {
            "status": "unknown_catalog_not_local",
            "catalog_path": str(catalog_path),
            "formula": "time_head_parameters = 3 * C * V * (context_dim + 1)",
            "estimate": None,
        }
    from adrec_still.models.baselines.popularity import train_catalog_size

    number_of_pois = train_catalog_size(catalog_path)
    from adrec_still.models.external.joint_lnm import FIRST_LEGAL_MARK_ID

    estimate = literal_head_capacity_estimate(
        catalog_size=number_of_pois - FIRST_LEGAL_MARK_ID,
        mixture_components=mixture_components,
        context_dim=context_dim,
        number_of_pois=number_of_pois,
        mark_embedding_dim=mark_embedding_dim,
        batch_size=batch_size,
        num_layers=num_layers,
    )
    return {
        "status": "estimated_from_local_catalog",
        "catalog_path": str(catalog_path),
        "estimate": estimate,
    }
