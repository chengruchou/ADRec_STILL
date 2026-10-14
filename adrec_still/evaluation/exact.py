"""Opt-in exact catalog scoring, with an explicit proposed tie contract (#38).

The v0.2 evaluator stays unchanged. This path orders equal scores by ascending
POI ID, independently of chunk size/order. Callbacks score only the supplied IDs;
they must represent the same fixed forecast on every call (eval mode, no dropout).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

from adrec_still.evaluation.ranking import analytical_random_floors, full_catalog_candidate_ids

EXACT_VERSION = "still-exact-v0.1-proposed"
TIE_POLICY = "score-descending-poi-id-ascending"
SCORE_KINDS = {"logits", "log_probabilities", "probabilities", "scores"}


def _topk(scores: torch.Tensor, ids: torch.Tensor, k: int):
    # Sort IDs first, then stably sort scores: lexicographic order without jitter.
    by_id = ids.argsort(dim=1, stable=True)
    ids, scores = ids.gather(1, by_id), scores.gather(1, by_id)
    order = scores.argsort(dim=1, descending=True, stable=True)[:, :k]
    return scores.gather(1, order), ids.gather(1, order)


@torch.no_grad()
def exact_ranking_contributions(
    score_candidates: Callable[[torch.Tensor], torch.Tensor],
    targets: torch.Tensor,
    known: torch.Tensor,
    observed: torch.Tensor,
    number_of_pois: int,
    *,
    score_kind: str,
    chunk_size: int = 4096,
    ks: tuple[int, ...] = (5, 10, 20),
    candidate_order: torch.Tensor | None = None,
    normalization_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """One bounded pass: exact top-K and float64 log normalization over all IDs.

    Callback returns CPU or device [B, chunk] (or shared [chunk]) scores. Only
    IDs >= 2 are requested. ``logits`` declares a normalized softmax model;
    ``scores`` explicitly has no probability interpretation. Probability/log-
    probability inputs must sum to one over the frozen catalog; no smoothing or
    renormalization is silently applied to these declared distributions.

    Excluded rows have zero contributions, plus an explicit mask. Infinite NLL
    remains infinity in memory; the export layer encodes status/count + JSON null.
    """
    started = time.perf_counter()
    if score_kind not in SCORE_KINDS or chunk_size <= 0:
        raise ValueError("declare a supported score_kind and positive chunk_size")
    if normalization_tolerance <= 0 or normalization_tolerance > 1e-3:
        raise ValueError("normalization_tolerance must be in (0, 1e-3]")
    targets = targets.detach().cpu()
    known, observed = known.detach().cpu().bool(), observed.detach().cpu().bool()
    if targets.ndim != 1 or targets.dtype != torch.long:
        raise ValueError("targets must be int64 [batch]")
    if known.shape != targets.shape or observed.shape != targets.shape:
        raise ValueError("masks must match targets")
    if (known & ~observed).any() or (observed & (targets == 0)).any():
        raise ValueError("invalid observed/known target masks")
    if (observed & ~known & (targets != 1)).any():
        raise ValueError("cold targets must use unknown ID 1")
    if (known & ((targets < 2) | (targets >= number_of_pois))).any():
        raise ValueError("known target outside frozen catalog")
    if (~observed & (targets != 0)).any():
        raise ValueError("censored targets must use padding ID 0")
    candidates = full_catalog_candidate_ids(number_of_pois)
    if candidate_order is not None:
        candidate_order = candidate_order.detach().cpu()
        if candidate_order.dtype != torch.long or not torch.equal(
            candidate_order.sort().values, candidates
        ):
            raise ValueError("candidate_order must be a permutation of the exact catalog")
        candidates = candidate_order
    floors = analytical_random_floors(len(candidates), ks)
    eligible = observed & known
    batch = len(targets)
    values = {key: torch.zeros(batch, dtype=torch.float64) for key in (*floors, "poi_nll")}
    top_scores = torch.empty((batch, 0), dtype=torch.float64)
    top_ids = torch.empty((batch, 0), dtype=torch.long)
    normalizer = torch.full((batch,), -torch.inf, dtype=torch.float64)
    target_scores = torch.full_like(normalizer, -torch.inf)
    k = min(max(ks), len(candidates))
    calls = peak_elements = 0
    # No callback calls are necessary when there are no ranking-eligible rows.
    if eligible.any():
        for offset in range(0, len(candidates), chunk_size):
            ids = candidates[offset : offset + chunk_size]
            raw = score_candidates(ids)
            if raw.ndim == 1:
                raw = raw.unsqueeze(0).expand(batch, -1)
            if raw.shape != (batch, len(ids)) or not raw.is_floating_point():
                raise ValueError("scorer must return floating [batch, chunk] or [chunk]")
            scores = raw.detach().to(device="cpu", dtype=torch.float64)
            if torch.isnan(scores).any() or torch.isposinf(scores).any():
                raise ValueError("NaN/+infinite scores cannot define an exact ordering")
            if score_kind == "probabilities":
                if (scores < 0).any() or (scores > 1).any():
                    raise ValueError("probabilities must lie in [0, 1]")
                scores = scores.log()
            if score_kind == "log_probabilities" and (scores > 0).any():
                raise ValueError("log probabilities cannot be positive")
            calls += 1
            peak_elements = max(peak_elements, scores.numel())
            normalizer = torch.logaddexp(normalizer, torch.logsumexp(scores, dim=1))
            matches = targets[:, None].eq(ids)
            target_scores = torch.where(
                matches.any(dim=1),
                scores.gather(1, matches.long().argmax(1)[:, None])[:, 0],
                target_scores,
            )
            top_scores, top_ids = _topk(
                torch.cat((top_scores, scores), dim=1),
                torch.cat((top_ids, ids.expand(batch, -1)), dim=1),
                k,
            )
        if score_kind != "scores" and not torch.isfinite(normalizer[eligible]).all():
            raise ValueError("eligible rows must have positive finite catalog mass")
        if score_kind in {"probabilities", "log_probabilities"} and not torch.allclose(
            normalizer[eligible],
            torch.zeros_like(normalizer[eligible]),
            atol=normalization_tolerance,
            rtol=0,
        ):
            raise ValueError("declared probabilities are not normalized over the frozen catalog")
        for cutoff in ks:
            matches = top_ids[:, :cutoff].eq(targets[:, None])
            hit = matches.any(dim=1) & eligible
            rank = matches.long().argmax(dim=1).to(torch.float64) + 1
            values[f"recall@{cutoff}"] = hit.to(torch.float64)
            values[f"ndcg@{cutoff}"] = torch.where(hit, 1 / torch.log2(rank + 1), 0)
            values[f"mrr@{cutoff}"] = torch.where(hit, 1 / rank, 0)
        if score_kind != "scores":
            nll = -target_scores
            if score_kind == "logits":
                nll = normalizer - target_scores
            values["poi_nll"][eligible] = nll[eligible]
    return {
        "evaluator_version": EXACT_VERSION,
        "tie_policy": TIE_POLICY,
        "normalization": score_kind,
        "normalization_tolerance": normalization_tolerance,
        "candidate_count": len(candidates),
        "random_floor": floors,
        "eligible_mask": eligible,
        "values": values,
        "top_ids": top_ids,
        "poi_nll_supported": score_kind != "scores",
        "poi_nll_reason": "ranking-only scores have no declared probability model"
        if score_kind == "scores"
        else None,
        "resource_cost": {
            "callback_calls": calls,
            "peak_score_elements": peak_elements,
            "chunk_size": chunk_size,
            "batch_size": batch,
            "seconds": time.perf_counter() - started,
            "memory_convention": "score-block elements only, not total process/model RSS",
        },
    }


def dense_exact_contributions(logits: torch.Tensor, targets, known, observed, **kwargs):
    """Dense adapter for the same proposed tie/score contract; v0.2 is unchanged."""
    return exact_ranking_contributions(
        lambda ids: logits.index_select(-1, ids.to(logits.device)),
        targets,
        known,
        observed,
        logits.shape[-1],
        chunk_size=logits.shape[-1],
        **kwargs,
    )
