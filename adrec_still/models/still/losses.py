"""Model-specific loss assembly for the #41 arms.

Row policy: valid rows are selected **before** any log/gather so masked fill values
(0-padded residual/censor columns) can never leak NaN or infinities into gradients
through an unselected ``torch.where`` branch.

Training population: the shared trainer currently delivers only observed rows with
exact known-catalog targets (``select_training_rows``). That is the explicitly labeled
**observed-known-v0-proposed** profile: it is not an approved survival-training view.
The censored-exposure and observed-unknown time contributions below are implemented and
unit-tested so the accepted #39 profile can consume them, but no real-data training may
call them until that contract exists; affected Table 2 cells stay blocked.

Censoring convention: ``censor_time_sec = c`` is the remaining exposure measured from
the query. For a survival-conditioned model its contribution is
``-(log S(s + c) - log S(s))`` — obtained by evaluating the *conditional* residual
distribution's ``log_survival(c)`` — never ``-(log S(c) - log S(s))``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from adrec_still.data.survival import SURVIVAL_PROFILE_VERSION
from adrec_still.time.protocol import ResidualTimeDistribution

#: Explicit label for the only population the shared trainer currently supplies.
TRAINING_PROFILE = "observed-known-v0-proposed"
SURVIVAL_TRAINING_PROFILE = SURVIVAL_PROFILE_VERSION


def survival_training_masks(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Validate and partition every #39 row into exactly one loss contribution."""
    observed = batch["observed_mask"].bool()
    known = batch["target_in_train_catalog"].bool()
    residual = batch["residual_time_sec"].to(torch.float64)
    censor = batch["censor_time_sec"].to(torch.float64)
    if any(value.shape != observed.shape for value in (known, residual, censor)):
        raise ValueError("survival training masks require aligned one-dimensional rows")
    if (residual[observed] <= 0).any() or (censor[observed] != 0).any():
        raise ValueError("observed rows require positive residual and no censor exposure")
    if known[~observed].any():
        raise ValueError("censored rows cannot have a known POI target")
    if (censor[~observed] <= 0).any() or (residual[~observed] != 0).any():
        raise ValueError("training censors require positive follow-up and no residual target")
    return {
        "observed_known": observed & known,
        "observed_unknown": observed & ~known,
        "censored_positive_followup": ~observed,
    }


def observed_known_joint_losses(
    model,
    kept: dict[str, torch.Tensor],
    fused_context: torch.Tensor,
    time_log_density: torch.Tensor,
    negative_samples: int,
    generator: torch.Generator,
    mark_logits_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Per-row joint loss on the observed-known population the shared trainer delivers.

    ``time_log_density`` is the exact per-row log density at the observed residual time
    (already conditional for the coherent arm). The mark term consumes the shared
    sampled-candidate builder, so the sampler/correction policy is identical across
    arms and is an explicitly labeled approximation of the exact catalog softmax.
    """
    from adrec_still.models.baselines.trainer import build_sampled_candidates

    del fused_context  # context is consumed by mark_logits_fn; kept for signature clarity
    targets = kept["next_poi"]
    residual = kept["residual_time_sec"]
    observed = kept["observed_mask"].bool() & kept["target_in_train_catalog"].bool()
    if not bool(observed.all()):
        raise ValueError(
            "observed_known_joint_losses expects the shared trainer's observed-known "
            f"population ({TRAINING_PROFILE}); censored/unknown rows need the #39 profile"
        )
    if (residual.to(torch.float64) <= 0).any():
        raise ValueError("observed rows require positive residual_time_sec")
    if not torch.isfinite(time_log_density).all():
        raise ValueError("time log density must be finite on observed rows")

    candidates, accidental_hit_penalty = build_sampled_candidates(
        targets, negative_samples, model.number_of_pois, generator
    )
    logits = mark_logits_fn(candidates) - accidental_hit_penalty
    log_mark = torch.log_softmax(logits, dim=-1)[..., 0]
    if log_mark.ndim != 1:
        raise ValueError("mark_logits_fn must reduce to one positive log-probability per row")
    return -(time_log_density.to(log_mark.dtype) + log_mark)


def observed_time_nll(
    distribution: ResidualTimeDistribution, residual_time_sec: torch.Tensor
) -> torch.Tensor:
    """Exact per-row time NLL for observed rows (rows already selected)."""
    residual = residual_time_sec.to(torch.float64)
    if (residual <= 0).any():
        raise ValueError("observed residual times must be positive")
    return -distribution.log_density(residual_time_sec)

def censored_exposure_nll(
    distribution: ResidualTimeDistribution, censor_time_sec: torch.Tensor
) -> torch.Tensor:
    """Per-row censored contribution ``-log S_res(c)`` (rows already selected).

    ``distribution`` must be the residual distribution conditional at the query, so for
    the coherent arm this equals ``-(log S(s + c) - log S(s))`` by construction. Zero
    remaining exposure contributes exactly 0. Unit-tested now; wired into real training
    only under an accepted #39 survival profile.
    """
    censor = censor_time_sec.to(torch.float64)
    if (censor < 0).any():
        raise ValueError("censor times must be non-negative")
    return -distribution.log_survival(censor_time_sec)
