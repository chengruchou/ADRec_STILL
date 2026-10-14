"""Censor-aware return-time metrics (ported from STILL's evaluation.time, CRPS removed).

Two adapter paths coexist:

- the single-``LogNormalPrediction`` path (:func:`lognormal_time_metrics`);
- a generic residual-time distribution protocol (:class:`ResidualTimeDistribution` /
  :func:`residual_time_distribution_metrics`) needed by mixture and survival-conditioned
  models. A mixture must not be moment-matched and misrepresented as a single LogNormal,
  so models expose their own exact ``log_density`` / ``log_survival`` in **native
  seconds**.

CRPS is intentionally not part of this port: the donor's quadrature/certification
machinery stays in the STILL repository. Time quality is reported as observed density
NLL and censored survival NLL with separate masks and counts.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

#: Version tag for the residual-time adapter as ported into adrec_still (no CRPS).
RESIDUAL_TIME_ADAPTER_VERSION = "adrec-still-residual-time-adapter-v1"


def lognormal_component_tail_mass_log(
    mu: torch.Tensor, sigma: torch.Tensor, from_sec: torch.Tensor
) -> torch.Tensor:
    """``log`` of an upper bound on ``E[(T_k - a)^+]`` per LogNormal component.

    ``E[(T - a)^+] = exp(mu + sigma^2/2) Phi((mu + sigma^2 - log a)/sigma) - a Phi((mu -
    log a)/sigma)``. Dropping the (non-negative) subtracted term leaves an upper bound
    that is a single ``log_ndtr`` in log space, so it stays finite and stable arbitrarily
    far into the tail where the exact difference would cancel catastrophically.

    ``mu``/``sigma`` are ``[batch, K]`` native-second parameters and ``from_sec`` is
    ``[batch]``; the result is ``[batch, K]``. ``a == 0`` gives the component mean.
    """
    time64 = from_sec.to(torch.float64).unsqueeze(1)
    positive = time64 > 0
    safe = torch.where(positive, time64, torch.ones_like(time64))
    variance = sigma.square()
    argument = (mu + variance - safe.log()) / sigma
    argument = torch.where(positive, argument, torch.full_like(argument, math.inf))
    return mu + 0.5 * variance + torch.special.log_ndtr(argument)


def score_time_joint_rows(model, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Shared native-second row adapter for time and joint scoring (no CRPS)."""
    prediction = model.predict(batch)
    distribution = getattr(prediction, "time_distribution", None)
    if distribution is None:
        raise TypeError("the joint scorer requires a ResidualTimeDistribution")
    observed = batch["observed_mask"].bool()
    residual = batch["residual_time_sec"].to(torch.float64)
    censor = batch["censor_time_sec"].to(torch.float64)
    if (residual < 0).any() or (censor < 0).any():
        raise ValueError("time scorer inputs must be non-negative")
    time_eligible = (observed & residual.gt(0)) | (~observed & censor.gt(0))
    safe_residual = torch.where(observed & time_eligible, residual, torch.ones_like(residual))
    safe_censor = torch.where(~observed & time_eligible, censor, torch.zeros_like(censor))
    time_nll = torch.where(
        observed & time_eligible,
        -distribution.log_density(safe_residual),
        torch.where(
            ~observed & time_eligible,
            -distribution.log_survival(safe_censor),
            torch.zeros_like(residual),
        ),
    )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            joint = model.score_joint(batch)
    finally:
        model.train(was_training)
    eligible = joint["eligible"].detach().cpu().bool()
    log_joint = joint["log_joint"].detach().cpu().to(torch.float64)
    log_time = joint["log_time"].detach().cpu().to(torch.float64)
    log_mark = joint["log_mark_given_time"].detach().cpu().to(torch.float64)
    finite = (
        eligible
        & torch.isfinite(log_joint)
        & torch.isfinite(log_time)
        & torch.isfinite(log_mark)
    )
    if finite.any() and not torch.allclose(
        log_joint[finite], log_time[finite] + log_mark[finite], atol=1e-7, rtol=1e-7
    ):
        raise ValueError("joint score decomposition failed on the common eligible subset")
    return {
        "time_nll": time_nll.detach().cpu(),
        "time_eligible": time_eligible.detach().cpu(),
        "joint_nll": -log_joint,
        "joint_eligible": eligible,
        "joint_time_nll": -log_time,
        "conditional_mark_nll": -log_mark,
    }


@dataclass(frozen=True)
class LogNormalPrediction:
    """Batch of log-normal residual-time parameters."""

    mu: torch.Tensor
    sigma: torch.Tensor

    def validate(self) -> None:
        if self.mu.shape != self.sigma.shape or self.mu.ndim != 1:
            raise ValueError("mu and sigma must have equal [batch] shapes")
        if not torch.isfinite(self.mu).all() or not torch.isfinite(self.sigma).all():
            raise ValueError("time-distribution parameters must be finite")
        if (self.sigma <= 0).any():
            raise ValueError("sigma must be positive")


def _split_nll_summary(
    individual_nll: torch.Tensor, observed: torch.Tensor
) -> dict[str, float | int | None]:
    """Pooled and per-population NLL summary; masks and counts stay separate."""
    observed_count = int(observed.sum())
    censored_count = int((~observed).sum())
    observed_sum = float(individual_nll[observed].sum()) if observed_count else 0.0
    censored_sum = float(individual_nll[~observed].sum()) if censored_count else 0.0
    return {
        "sample_count": int(individual_nll.shape[0]),
        "observed_count": observed_count,
        "censored_count": censored_count,
        "return_time_nll": float(individual_nll.mean()),
        "observed_time_nll_sum": observed_sum,
        "observed_time_nll": observed_sum / observed_count if observed_count else None,
        "censored_time_nll_sum": censored_sum,
        "censored_time_nll": censored_sum / censored_count if censored_count else None,
    }


def lognormal_time_metrics(
    prediction: LogNormalPrediction,
    residual_time_sec: torch.Tensor,
    censor_time_sec: torch.Tensor,
    observed_mask: torch.Tensor,
) -> dict[str, float | int | None]:
    """Compute censor-aware NLL for a single-LogNormal residual prediction."""
    prediction.validate()
    mu = prediction.mu.to(torch.float64)
    sigma = prediction.sigma.to(torch.float64)
    residual = residual_time_sec.to(torch.float64)
    censor = censor_time_sec.to(torch.float64)
    observed = observed_mask.bool()
    if not (mu.shape == residual.shape == censor.shape == observed.shape):
        raise ValueError("all time tensors must have equal [batch] shapes")

    individual_nll = torch.empty_like(mu)
    if observed.any():
        observed_time = residual[observed]
        if (observed_time <= 0).any():
            raise ValueError("observed residual times must be positive")
        z = (torch.log(observed_time) - mu[observed]) / sigma[observed]
        log_pdf = (
            -torch.log(observed_time)
            - torch.log(sigma[observed])
            - 0.5 * math.log(2.0 * math.pi)
            - 0.5 * z.square()
        )
        individual_nll[observed] = -log_pdf
    if (~observed).any():
        censored_time = censor[~observed]
        if (censored_time < 0).any():
            raise ValueError("censor times must be non-negative")
        positive = censored_time > 0
        log_survival = torch.zeros_like(censored_time)
        if positive.any():
            z = (torch.log(censored_time[positive]) - mu[~observed][positive]) / sigma[~observed][
                positive
            ]
            log_survival[positive] = torch.special.log_ndtr(-z)
        individual_nll[~observed] = -log_survival

    return _split_nll_summary(individual_nll, observed)


class ResidualTimeDistribution(ABC):
    """Per-row residual-time distribution measured from the query, in native seconds.

    One instance describes a batch: every method takes and returns ``[batch]`` tensors.
    For a survival-conditioned model the instance must already be the conditional
    residual distribution at its query (so ``log_survival(c)`` is the conditional
    remaining-exposure survival ``log S(s + c) - log S(s)``, not the base ``log S(c)``).
    """

    @abstractmethod
    def batch_size(self) -> int:
        """Number of rows this distribution instance describes."""

    @abstractmethod
    def log_density(self, residual_time_sec: torch.Tensor) -> torch.Tensor:
        """Exact log density at positive residual times (native seconds).

        Callers may pass a full batch tensor whose entries at rows they will discard
        (e.g. censored rows padded with 0) are non-positive: implementations must return
        ``-inf`` there without raising and without contaminating other rows with NaN.
        """

    @abstractmethod
    def log_survival(self, time_sec: torch.Tensor) -> torch.Tensor:
        """Exact log survival at non-negative times; ``log_survival(0) == 0``."""

    def residual_tail_mass_bound(self, from_sec: torch.Tensor) -> torch.Tensor | None:
        """Finite upper bound on ``E[(R - from)^+]``, or None when unavailable."""
        del from_sec
        return None


def residual_time_distribution_metrics(
    distribution: ResidualTimeDistribution,
    residual_time_sec: torch.Tensor,
    censor_time_sec: torch.Tensor,
    observed_mask: torch.Tensor,
) -> dict[str, object]:
    """Censor-aware NLL (native seconds) for a generic residual-time distribution.

    Observed rows contribute ``-log_density(r)``; censored rows contribute
    ``-log_survival(c)`` where ``c`` is the remaining exposure measured from the query.
    Observed and censored contributions are reported with separate sums and counts.
    """
    residual = residual_time_sec.to(torch.float64)
    censor = censor_time_sec.to(torch.float64)
    observed = observed_mask.bool()
    if not (residual.shape == censor.shape == observed.shape) or residual.ndim != 1:
        raise ValueError("all time tensors must have equal [batch] shapes")
    if distribution.batch_size() != int(residual.shape[0]):
        raise ValueError("distribution batch size must match the metric tensors")

    individual_nll = torch.empty_like(residual)
    if observed.any():
        observed_time = residual[observed]
        if (observed_time <= 0).any():
            raise ValueError("observed residual times must be positive")
        individual_nll[observed] = -distribution.log_density(residual)[observed]
    if (~observed).any():
        censored_time = censor[~observed]
        if (censored_time < 0).any():
            raise ValueError("censor times must be non-negative")
        individual_nll[~observed] = -distribution.log_survival(censor)[~observed]

    return {
        "time_adapter_version": RESIDUAL_TIME_ADAPTER_VERSION,
        **_split_nll_summary(individual_nll, observed),
    }
