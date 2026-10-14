"""Shared full-catalog next-POI ranking metrics."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from adrec_still.data.schema import PADDING_POI_ID, UNKNOWN_POI_ID


def full_catalog_candidate_ids(
    number_of_pois: int,
    padding_id: int = PADDING_POI_ID,
    unknown_id: int = UNKNOWN_POI_ID,
) -> torch.Tensor:
    """Return exact training-catalog POI IDs, excluding reserved IDs."""
    if padding_id == unknown_id:
        raise ValueError("padding_id and unknown_id must be distinct")
    candidates = torch.arange(number_of_pois, dtype=torch.long)
    candidates = candidates[candidates.ne(padding_id) & candidates.ne(unknown_id)]
    if not len(candidates):
        raise ValueError("number_of_pois must include at least one train-catalog POI")
    return candidates


def analytical_random_floors(
    candidate_count: int,
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
    """Return exact uniform-random ranking floors for one relevant item."""
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("ks must contain at least one positive cutoff")

    floors: dict[str, float] = {}
    for k in ks:
        effective_k = min(k, candidate_count)
        floors[f"recall@{k}"] = effective_k / candidate_count
        floors[f"ndcg@{k}"] = (
            sum(1.0 / math.log2(rank + 1.0) for rank in range(1, effective_k + 1)) / candidate_count
        )
        floors[f"mrr@{k}"] = sum(1.0 / rank for rank in range(1, effective_k + 1)) / candidate_count
    return floors


def ranking_contributions(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_in_train_catalog: torch.Tensor,
    observed_mask: torch.Tensor | None = None,
    ks: tuple[int, ...] = (5, 10, 20),
    padding_id: int = PADDING_POI_ID,
    unknown_id: int = UNKNOWN_POI_ID,
) -> dict[str, Any]:
    """Return shared v0.2 per-row values; excluded rows have zero contributions.

    A [POIs] score vector denotes one distribution shared by every row. This
    computes full-catalog top-K and normalization once, without materializing a
    [batch, POIs] matrix. A [batch, POIs] tensor supports general model outputs.
    Both paths use the same candidates, top-K tie handling, masks, and formulas.
    """
    if targets.ndim != 1 or logits.ndim not in (1, 2):
        raise ValueError("logits must be [POIs] or [batch, POIs]; targets must be [batch]")
    shared_scores = logits.ndim == 1
    if not shared_scores and logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must have matching batch sizes")
    if target_in_train_catalog.shape != targets.shape:
        raise ValueError("target_in_train_catalog must have shape [batch]")
    observed = targets.ne(padding_id) if observed_mask is None else observed_mask.bool()
    if observed.shape != targets.shape:
        raise ValueError("observed_mask must have shape [batch]")
    known = target_in_train_catalog.bool()
    if (known & ~observed).any():
        raise ValueError("censored samples cannot be marked as train-catalog targets")
    if (observed & targets.eq(padding_id)).any():
        raise ValueError("observed samples cannot target the padding POI")
    if (known & targets.eq(unknown_id)).any():
        raise ValueError("the shared unknown POI cannot be a known ranking target")
    cold_start = observed & ~known
    if (cold_start & targets.ne(unknown_id)).any():
        raise ValueError("cold-start targets must use the frozen unknown POI ID")

    mask = observed & known
    observed_target_count = int(observed.sum())
    ranking_sample_count = int(mask.sum())
    logits = logits.unsqueeze(0) if shared_scores else logits[mask]
    targets = targets[mask]
    candidate_ids = full_catalog_candidate_ids(
        logits.shape[1], padding_id=padding_id, unknown_id=unknown_id
    ).to(logits.device)
    candidate_count = len(candidate_ids)
    random_floor = analytical_random_floors(candidate_count, ks)
    result: dict[str, Any] = {
        "sample_count": ranking_sample_count,
        "observed_target_count": observed_target_count,
        "cold_start_target_count": int(cold_start.sum()),
        "ranking_coverage": (
            ranking_sample_count / observed_target_count if observed_target_count else None
        ),
        "candidate_count": candidate_count,
        "random_floor": random_floor,
    }
    values = {
        key: torch.zeros(len(mask), dtype=torch.float64, device=logits.device)
        for key in (*random_floor, "poi_nll")
    }
    result["values"] = values
    result["eligible_mask"] = mask
    if not len(targets):
        return result
    if targets.min() < 0 or targets.max() >= logits.shape[1]:
        raise ValueError("target index is outside the POI vocabulary")
    logits = logits.clone()
    for reserved_id in (padding_id, unknown_id):
        if 0 <= reserved_id < logits.shape[1]:
            logits[:, reserved_id] = -torch.inf
    candidate_logits = logits.index_select(1, candidate_ids)
    ranking_positions = torch.topk(
        candidate_logits, k=min(max(ks), len(candidate_ids)), dim=1
    ).indices
    ranking = candidate_ids[ranking_positions]
    for k in ks:
        matches = ranking[:, : min(k, len(candidate_ids))].eq(targets[:, None])
        hit = matches.any(dim=1)
        positions = matches.float().argmax(dim=1)
        discount = torch.where(
            hit,
            1.0 / torch.log2(positions.to(torch.float64) + 2.0),
            torch.zeros_like(positions, dtype=torch.float64),
        )
        reciprocal_rank = torch.where(
            hit,
            1.0 / (positions.to(torch.float64) + 1.0),
            torch.zeros_like(positions, dtype=torch.float64),
        )
        values[f"recall@{k}"][mask] = hit.to(torch.float64)
        values[f"ndcg@{k}"][mask] = discount
        values[f"mrr@{k}"][mask] = reciprocal_rank
    if shared_scores:
        nll = -F.log_softmax(logits[0], dim=0)[targets]
    else:
        nll = F.cross_entropy(logits, targets, reduction="none")
    values["poi_nll"][mask] = nll.to(torch.float64)
    return result


def ranking_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_in_train_catalog: torch.Tensor,
    observed_mask: torch.Tensor | None = None,
    ks: tuple[int, ...] = (5, 10, 20),
    padding_id: int = PADDING_POI_ID,
    unknown_id: int = UNKNOWN_POI_ID,
) -> dict[str, Any]:
    """Evaluate known observed targets and report cold-start coverage explicitly."""
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("logits must be [batch, POIs] and targets must be [batch]")
    result = ranking_contributions(
        logits, targets, target_in_train_catalog, observed_mask, ks, padding_id, unknown_id
    )
    values = result.pop("values")
    result.pop("eligible_mask")
    count = result["sample_count"]
    result.update(
        {key: float(value.sum()) / count if count else None for key, value in values.items()}
    )
    return result
