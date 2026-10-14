"""Log-normal mixture total-gap distributions and their survival conditioning.

The coherent joint model (Issue #41, `docs/JOINT_MODEL_PLAN.md`) needs one fixed base
distribution over the total gap ``T`` (native seconds) and exact conditional objects for a
query at elapsed time ``s``:

```text
f(T) = sum_k pi_k * g_k(T)            g_k log-normal(mu_k, sigma_k)
S(s) = sum_k pi_k * S_k(s)
q_s(r) = f(s + r) / S(s)              residual density, r > 0
omega_k(s) = pi_k * S_k(s) / S(s)     survivor-reweighted component weights
S_res(c | s) = S(s + c) / S(s)        conditional remaining-exposure survival
```

Numerical policy: all math runs in float64; mixing weights are renormalized in float64
before use, so a float32 ``log_softmax`` head's ~1e-7 of normalization slack never lands
in a survival; component log survival uses ``torch.special.log_ndtr`` on
``-(ln t - mu) / sigma`` (never ``log(1 - CDF)`` with an epsilon clamp); mixtures combine
through ``logsumexp``; ``log S(0) == 0`` exactly and negative times are rejected.
Non-positive density query points return ``-inf`` without raising or contaminating other
rows, so callers may pass padded full-batch tensors and select rows afterwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from adrec_still.time.protocol import (
    ResidualTimeDistribution,
    lognormal_component_tail_mass_log,
)

_LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)


def _validate_times(time_sec: torch.Tensor, batch: int, name: str) -> torch.Tensor:
    if time_sec.ndim != 1 or int(time_sec.shape[0]) != batch:
        raise ValueError(f"{name} must have shape [batch]")
    time64 = time_sec.to(torch.float64)
    if not torch.isfinite(time64).all():
        raise ValueError(f"{name} must be finite")
    return time64


@dataclass(frozen=True)
class LognormalMixtureParameters:
    """Batched mixture parameters: ``log_weights``/``mu``/``sigma`` of shape [batch, K].

    ``log_weights`` must be normalized per row (``logsumexp == 0``); ``sigma`` must be
    strictly positive. Parameters are treated as float64 for all downstream math.
    """

    log_weights: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor

    def validate(self) -> None:
        shape = self.log_weights.shape
        if len(shape) != 2 or self.mu.shape != shape or self.sigma.shape != shape:
            raise ValueError("log_weights, mu, and sigma must share one [batch, K] shape")
        for name, tensor in (("log_weights", self.log_weights), ("mu", self.mu)):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite")
        if not torch.isfinite(self.sigma).all() or (self.sigma <= 0).any():
            raise ValueError("sigma must be finite and strictly positive")
        normalization = torch.logsumexp(self.log_weights.to(torch.float64), dim=1)
        if not torch.allclose(
            normalization, torch.zeros_like(normalization), atol=1.0e-6
        ):
            raise ValueError("log_weights must be normalized per row (logsumexp == 0)")

    @property
    def batch(self) -> int:
        return int(self.log_weights.shape[0])

    @property
    def components(self) -> int:
        return int(self.log_weights.shape[1])

    def normalized_log_weights(self) -> torch.Tensor:
        """``log pi_k`` in float64, renormalized -- the only correct way to read them.

        :meth:`validate` admits a row whose ``logsumexp`` is within ``1e-6`` of zero,
        which is what a float32 ``log_softmax`` head actually delivers: its float64
        ``logsumexp`` sits a few ``1e-7`` away from zero. That slack is not harmless.
        It passes straight through every mixture ``logsumexp``, so wherever all
        components still survive, ``log S(t)`` comes out a few ``1e-7`` **above** zero,
        a survival greater than one.

        Renormalizing costs one ``logsumexp`` and makes ``sum_k pi_k == 1`` hold to
        float64 precision, so the certificate sees only genuine float64 roundoff. Rows
        already normalized in float64 are unchanged.

        Read the weights through this accessor, never through ``self.log_weights``
        directly: a consumer that mixes a raw numerator with a normalized denominator
        (any of them computed here) keeps the slack instead of cancelling it.
        """
        log_weights = self.log_weights.to(torch.float64)
        return log_weights - torch.logsumexp(log_weights, dim=1, keepdim=True)

    def _float64(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Float64 parameters, the weights through :meth:`normalized_log_weights`."""
        return (
            self.normalized_log_weights(),
            self.mu.to(torch.float64),
            self.sigma.to(torch.float64),
        )

    def _component_log_density_terms(
        self, time_sec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Finite per-component log-density terms plus the positive-time row mask.

        Non-positive rows are evaluated at a safe stand-in time so every term stays
        finite; callers must mask those rows to ``-inf`` **after** any mixture
        reduction. Reducing over per-component ``-inf`` first (an all-``-inf``
        ``logsumexp`` row) makes the backward pass emit NaN even for rows the loss
        never selects, and that NaN pollutes shared parameters.
        """
        _, mu, sigma = self._float64()
        time64 = _validate_times(time_sec, self.batch, "time_sec")
        positive = time64 > 0
        safe = torch.where(positive, time64, torch.ones_like(time64)).unsqueeze(1)
        z = (safe.log() - mu) / sigma
        terms = -safe.log() - sigma.log() - _LOG_SQRT_TWO_PI - 0.5 * z.square()
        return terms, positive

    def component_log_density(self, time_sec: torch.Tensor) -> torch.Tensor:
        """Per-component log density [batch, K]; ``-inf`` at non-positive times.

        Do not feed these masked values into a mixture ``logsumexp`` under autograd;
        use :meth:`_component_log_density_terms` and mask after reducing instead.
        """
        terms, positive = self._component_log_density_terms(time_sec)
        return torch.where(
            positive.unsqueeze(1), terms, torch.full_like(terms, -math.inf)
        )

    def component_log_survival(self, time_sec: torch.Tensor) -> torch.Tensor:
        """Per-component log survival [batch, K]; exactly 0 at ``t == 0``."""
        _, mu, sigma = self._float64()
        time64 = _validate_times(time_sec, self.batch, "time_sec")
        if (time64 < 0).any():
            raise ValueError("survival times must be non-negative")
        positive = time64 > 0
        safe = time64.clamp_min(torch.finfo(torch.float64).tiny).unsqueeze(1)
        z = (safe.log() - mu) / sigma
        log_survival = torch.special.log_ndtr(-z)
        return torch.where(positive.unsqueeze(1), log_survival, torch.zeros_like(log_survival))

    def log_density(self, time_sec: torch.Tensor) -> torch.Tensor:
        """Mixture log density over the total gap; ``-inf`` at non-positive times.

        The reduction runs on finite safe terms and the non-positive rows are masked
        afterwards: masked rows must contribute exactly zero gradient to shared
        parameters, never an all-``-inf`` ``logsumexp`` NaN.
        """
        log_weights, _, _ = self._float64()
        terms, positive = self._component_log_density_terms(time_sec)
        value = torch.logsumexp(log_weights + terms, dim=1)
        return torch.where(positive, value, torch.full_like(value, -math.inf))

    def log_survival(self, time_sec: torch.Tensor) -> torch.Tensor:
        """Mixture log survival; exactly 0 at ``t == 0``."""
        log_weights, _, _ = self._float64()
        time64 = _validate_times(time_sec, self.batch, "time_sec")
        value = torch.logsumexp(log_weights + self.component_log_survival(time_sec), dim=1)
        # logsumexp(log_weights) is only ~1e-16 close to 0; the t == 0 boundary is exact
        # by contract, so pin it rather than leaking float rounding into ratios.
        return torch.where(time64 > 0, value, torch.zeros_like(value))

    def survivor_log_weights(self, elapsed_sec: torch.Tensor) -> torch.Tensor:
        """``log omega_k(s)``: normalized survivor-reweighted component weights."""
        log_weights, _, _ = self._float64()
        joint = log_weights + self.component_log_survival(elapsed_sec)
        return joint - torch.logsumexp(joint, dim=1, keepdim=True)

    def mean(self) -> torch.Tensor:
        """Finite mixture mean in native seconds."""
        log_weights, mu, sigma = self._float64()
        return torch.sum(log_weights.exp() * torch.exp(mu + 0.5 * sigma.square()), dim=1)

    def component_tail_mass_log(self, from_sec: torch.Tensor) -> torch.Tensor:
        """``log`` upper bound on ``E[(T_k - a)^+]`` per component [batch, K]."""
        _, mu, sigma = self._float64()
        return lognormal_component_tail_mass_log(mu, sigma, from_sec)


class MixtureResidualDistribution(ResidualTimeDistribution):
    """Direct residual-time mixture (the independent arm's ``g_s(r | H)``)."""

    def __init__(self, parameters: LognormalMixtureParameters) -> None:
        parameters.validate()
        self.parameters = parameters

    def batch_size(self) -> int:
        return self.parameters.batch

    def log_density(self, residual_time_sec: torch.Tensor) -> torch.Tensor:
        return self.parameters.log_density(residual_time_sec)

    def log_survival(self, time_sec: torch.Tensor) -> torch.Tensor:
        return self.parameters.log_survival(time_sec)

    def residual_tail_mass_bound(self, from_sec: torch.Tensor) -> torch.Tensor:
        """``E[(R - a)^+]`` bound: the residual is the mixture itself, so no
        conditioning divides it."""
        log_weights, _, _ = self.parameters._float64()
        return torch.logsumexp(
            log_weights + self.parameters.component_tail_mass_log(from_sec), dim=1
        ).exp()


class SurvivalConditionedResidualDistribution(ResidualTimeDistribution):
    """The coherent arm's exact residual conditional ``q_s(r) = f(s + r) / S(s)``.

    The base mixture is fixed per row; every waiting effect enters only through this
    conditioning. ``log_survival(c)`` is the conditional remaining-exposure survival
    ``log S(s + c) - log S(s)`` — never the base ``log S(c)``.
    """

    def __init__(
        self, base: LognormalMixtureParameters, elapsed_sec: torch.Tensor
    ) -> None:
        base.validate()
        elapsed = _validate_times(elapsed_sec, base.batch, "elapsed_sec")
        if (elapsed < 0).any():
            raise ValueError("elapsed_sec must be non-negative")
        self.base = base
        self.elapsed_sec = elapsed
        self._log_survival_at_query = base.log_survival(elapsed)

    def batch_size(self) -> int:
        return self.base.batch

    def log_density(self, residual_time_sec: torch.Tensor) -> torch.Tensor:
        residual = _validate_times(residual_time_sec, self.base.batch, "residual_time_sec")
        total = self.elapsed_sec + residual.clamp_min(0.0)
        conditional = self.base.log_density(total) - self._log_survival_at_query
        return torch.where(
            residual > 0, conditional, torch.full_like(conditional, -math.inf)
        )

    def log_survival(self, time_sec: torch.Tensor) -> torch.Tensor:
        remaining = _validate_times(time_sec, self.base.batch, "time_sec")
        if (remaining < 0).any():
            raise ValueError("survival times must be non-negative")
        return self.base.log_survival(self.elapsed_sec + remaining) - self._log_survival_at_query

    def survivor_log_weights(self) -> torch.Tensor:
        """``log omega_k(s)`` for the query this conditional was built at."""
        return self.base.survivor_log_weights(self.elapsed_sec)

    def residual_tail_mass_bound(self, from_sec: torch.Tensor) -> torch.Tensor:
        """``E[(R - a)^+] = E[(T - (s + a))^+] / S(s)``, taken in log space so the deep
        tail neither overflows nor cancels."""
        log_weights, _, _ = self.base._float64()
        total = self.elapsed_sec + from_sec.to(torch.float64)
        return (
            torch.logsumexp(
                log_weights + self.base.component_tail_mass_log(total), dim=1
            )
            - self._log_survival_at_query
        ).exp()


class FixedPiTruncatedResidualDistribution(ResidualTimeDistribution):
    """Registered ablation ``q_abl,s``: per-component truncation with the original ``pi``.

    ``q_abl,s(r) = sum_k pi_k * g_k(s + r) / S_k(s)``. Each component keeps its correct
    truncation, but the cross-component survivor reweighting is disabled. This variant is
    normalized per query yet does not satisfy the full cross-query conditioning identity;
    it must be scored under its own ``variant_id`` and never under the coherent model's
    likelihood label.
    """

    def __init__(
        self, base: LognormalMixtureParameters, elapsed_sec: torch.Tensor
    ) -> None:
        base.validate()
        elapsed = _validate_times(elapsed_sec, base.batch, "elapsed_sec")
        if (elapsed < 0).any():
            raise ValueError("elapsed_sec must be non-negative")
        self.base = base
        self.elapsed_sec = elapsed
        self._component_log_survival_at_query = base.component_log_survival(elapsed)

    def batch_size(self) -> int:
        return self.base.batch

    def _log_mix(self, per_component: torch.Tensor) -> torch.Tensor:
        log_weights, _, _ = self.base._float64()
        return torch.logsumexp(
            log_weights + per_component - self._component_log_survival_at_query, dim=1
        )

    def log_density(self, residual_time_sec: torch.Tensor) -> torch.Tensor:
        residual = _validate_times(residual_time_sec, self.base.batch, "residual_time_sec")
        total = self.elapsed_sec + residual.clamp_min(0.0)
        # Reduce over finite safe terms and mask afterwards; at s == 0 a masked row has
        # total == 0 and per-component -inf terms would NaN the mixture backward.
        terms, _ = self.base._component_log_density_terms(total)
        mixed = self._log_mix(terms)
        return torch.where(residual > 0, mixed, torch.full_like(mixed, -math.inf))

    def log_survival(self, time_sec: torch.Tensor) -> torch.Tensor:
        remaining = _validate_times(time_sec, self.base.batch, "time_sec")
        if (remaining < 0).any():
            raise ValueError("survival times must be non-negative")
        value = self._log_mix(self.base.component_log_survival(self.elapsed_sec + remaining))
        return torch.where(remaining > 0, value, torch.zeros_like(value))

    def residual_tail_mass_bound(self, from_sec: torch.Tensor) -> torch.Tensor:
        """Per-component truncation keeps its own divisor, matching this variant's own
        density: ``sum_k pi_k E[(T_k - (s + a))^+] / S_k(s)``."""
        log_weights, _, _ = self.base._float64()
        total = self.elapsed_sec + from_sec.to(torch.float64)
        return torch.logsumexp(
            log_weights
            + self.base.component_tail_mass_log(total)
            - self._component_log_survival_at_query,
            dim=1,
        ).exp()

