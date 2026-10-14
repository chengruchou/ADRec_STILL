"""Per-mark log-normal-mixture joint distribution for the external LNM baseline (#51).

Math layer for the conditionally *dependent* factorization of Waghmare et al.
(CIKM 2022, arXiv:2210.15294), official code
https://github.com/waghmaregovind/joint_tpp @ a32bca006ba87beb6f71453b9660f81a845eadda
(MIT, Copyright (c) 2021 Oleksandr Shchur). For one fixed history ``H`` over the
legal catalog of ``V`` marks with ``C`` mixture components per mark:

```text
p_y            = p(y | H)                       mark probabilities, sum_y p_y = 1
g_y(T)         = sum_c w_yc LogNormal(T; mu_yc, sigma_yc)   native seconds
f(T, y)        = p_y * g_y(T)
S(t)           = sum_y p_y * S_y(t)             log S(0) == 0 exactly
log p_s(y)     = log p_y + log S_y(s) - log S(s)            waiting-marginal ranking
log q_s(r)     = log sum_y[p_y g_y(s + r)] - log S(s)       marginal residual density
log q_s(r, y)  = log p_y + log g_y(s + r) - log S(s)        joint score
log p(y|T=t,H) = log p_y + log g_y(t) - log sum_j[p_j g_j(t)]
censor NLL     = -(log S(s + c) - log S(s))
```

Every waiting effect is derived by conditioning this fixed base; the query elapsed
time ``s`` never parameterizes the base. Sums over marks are exact over the whole
declared legal catalog (never sampled candidates). Per-component math reuses the
tested NaN-safe :class:`adrec_still.time.distributions.LognormalMixtureParameters`
(stable ``log_ndtr`` survival, ``logsumexp`` mixing, float64, ``-inf`` pinned after
reductions so masked rows cannot poison shared-parameter gradients).

Source tags: the dependent factorization and formulas above are the paper's
(Eq. 5/7/11 lineage); the parameter constraints (unrestricted locations,
straight-through-clamped log-scales, per-mark component ``log_softmax``) follow the
official implementation; the waiting/survival conditioning is the STILL adaptation
recorded in ``UPSTREAM.md`` (the official end-of-sequence survival selects the last
observed mark's component and is deliberately **not** reproduced here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from adrec_still.time.distributions import LognormalMixtureParameters
from adrec_still.time.protocol import (
    ResidualTimeDistribution,
    lognormal_component_tail_mass_log,
)

#: Official straight-through clamp bounds for normalized-space log-scales
#: (joint_tpp code/dpp/models/log_norm_mix.py: clamp_preserve_gradients(x, -5.0, 3.0)).
LOG_SCALE_CLAMP_MIN = -5.0
LOG_SCALE_CLAMP_MAX = 3.0


def clamp_preserve_gradients(
    value: torch.Tensor, minimum: float, maximum: float
) -> torch.Tensor:
    """Straight-through clamp: clamped forward value, identity gradient.

    Official-implementation semantics (Shchur's ``clamp_preserve_gradients``): the
    forward pass is ``value.clamp(minimum, maximum)`` but the backward pass treats
    the operation as the identity, so parameters outside the bounds still receive
    gradient signal instead of going dead. A plain ``torch.clamp`` is not the same
    operation and must not be silently substituted.
    """
    return value + (value.clamp(minimum, maximum) - value).detach()


@dataclass(frozen=True)
class PerMarkJointDistribution:
    """One fixed base joint ``f(T, y) = p_y g_y(T)`` per row, in native seconds.

    ``log_mark_probabilities`` is ``[batch, V]`` normalized over the legal catalog
    (PAD/UNK excluded before any normalization). ``locs``/``log_scales_raw``/
    ``weight_logits`` are ``[batch, V, C]`` in the official *normalized log-time*
    space; ``log_time_mean``/``log_time_std`` are the affine statistics ``a``/``b``
    with ``log T_seconds = a + b * Z``, so the native-seconds component parameters
    are ``a + b * mu`` and ``b * sigma`` (the log-normal density in native seconds
    then carries the correct change-of-variables Jacobian by construction).
    """

    log_mark_probabilities: torch.Tensor
    locs: torch.Tensor
    log_scales_raw: torch.Tensor
    weight_logits: torch.Tensor
    log_time_mean: torch.Tensor
    log_time_std: torch.Tensor

    def __post_init__(self) -> None:
        if self.log_mark_probabilities.ndim != 2:
            raise ValueError("log_mark_probabilities must be [batch, V]")
        if self.locs.shape != self.log_scales_raw.shape or self.locs.shape != (
            self.weight_logits.shape
        ):
            raise ValueError("locs, log_scales_raw and weight_logits must share one shape")
        if self.locs.ndim != 3 or self.locs.shape[:2] != self.log_mark_probabilities.shape:
            raise ValueError("mixture parameters must be [batch, V, C]")
        if float(self.log_time_std) <= 0.0:
            raise ValueError("log_time_std must be strictly positive")
        normalization = torch.logsumexp(
            self.log_mark_probabilities.to(torch.float64), dim=1
        )
        if not torch.allclose(normalization, torch.zeros_like(normalization), atol=1e-6):
            raise ValueError("log_mark_probabilities must be normalized over the catalog")

    @property
    def batch(self) -> int:
        return int(self.log_mark_probabilities.shape[0])

    @property
    def catalog_size(self) -> int:
        return int(self.log_mark_probabilities.shape[1])

    @property
    def components(self) -> int:
        return int(self.locs.shape[2])

    def _native_flat_mixture(self) -> LognormalMixtureParameters:
        """All ``batch * V`` per-mark mixtures as one flat tested mixture object."""
        batch, catalog, components = self.locs.shape
        log_scales = clamp_preserve_gradients(
            self.log_scales_raw.to(torch.float64), LOG_SCALE_CLAMP_MIN, LOG_SCALE_CLAMP_MAX
        )
        mean = self.log_time_mean.to(torch.float64)
        std = self.log_time_std.to(torch.float64)
        native_mu = mean + std * self.locs.to(torch.float64)
        native_sigma = std * log_scales.exp()
        log_weights = torch.log_softmax(self.weight_logits.to(torch.float64), dim=2)
        return LognormalMixtureParameters(
            log_weights=log_weights.reshape(batch * catalog, components),
            mu=native_mu.reshape(batch * catalog, components),
            sigma=native_sigma.reshape(batch * catalog, components),
        )

    def _expand_time(self, time_sec: torch.Tensor) -> torch.Tensor:
        return (
            time_sec.to(torch.float64)
            .unsqueeze(1)
            .expand(self.batch, self.catalog_size)
            .reshape(self.batch * self.catalog_size)
        )

    def per_mark_log_density(self, time_sec: torch.Tensor) -> torch.Tensor:
        """``log g_y(t)`` [batch, V]; ``-inf`` at non-positive rows."""
        flat = self._native_flat_mixture().log_density(self._expand_time(time_sec))
        return flat.reshape(self.batch, self.catalog_size)

    def per_mark_log_survival(self, time_sec: torch.Tensor) -> torch.Tensor:
        """``log S_y(t)`` [batch, V]; exactly 0 at ``t == 0``."""
        flat = self._native_flat_mixture().log_survival(self._expand_time(time_sec))
        return flat.reshape(self.batch, self.catalog_size)

    def log_survival(self, time_sec: torch.Tensor) -> torch.Tensor:
        """``log S(t) = log sum_y p_y S_y(t)`` [batch]; pinned to 0 at ``t == 0``."""
        time64 = time_sec.to(torch.float64)
        value = torch.logsumexp(
            self.log_mark_probabilities.to(torch.float64)
            + self.per_mark_log_survival(time_sec),
            dim=1,
        )
        return torch.where(time64 > 0, value, torch.zeros_like(value))

    def marginal_and_per_mark_log_density(
        self, time_sec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(log sum_y p_y g_y(t) [batch], log g_y(t) [batch, V])`` in one pass.

        The per-mark tensor is the largest object in the training step
        (``[batch, V, C]`` float64 inside the mixture), so a caller that needs both
        the marginal and the per-mark densities at the same times takes them from
        here instead of evaluating the mixture twice.

        The mark reduction runs on finite safe values and the mask is applied after
        the ``logsumexp`` (same policy as the tested mixture primitives), so a
        non-positive row contributes zero gradient instead of NaN. The returned
        per-mark tensor is the one evaluated at those safe values.
        """
        time64 = time_sec.to(torch.float64)
        positive = time64 > 0
        safe = torch.where(positive, time64, torch.ones_like(time64))
        per_mark = self.per_mark_log_density(safe)
        value = torch.logsumexp(
            self.log_mark_probabilities.to(torch.float64) + per_mark, dim=1
        )
        return torch.where(positive, value, torch.full_like(value, -math.inf)), per_mark

    def log_marginal_density(self, time_sec: torch.Tensor) -> torch.Tensor:
        """``log sum_y p_y g_y(t)`` [batch]; ``-inf`` at non-positive rows."""
        return self.marginal_and_per_mark_log_density(time_sec)[0]

    def log_marginal_mean(self) -> torch.Tensor:
        """``log E[T] = log sum_y p_y sum_k w_yk exp(mu_yk + sigma_yk^2 / 2)`` [batch].

        Computed in log space on purpose. The naive product ``p_y * E[T_y]`` overflows
        as soon as one of the ``V * C`` components sits near the upstream log-scale
        clamp with a fitted log-time spread of ~2 -- and ``p_y * inf`` is ``inf`` for
        any positive ``p_y`` and ``NaN`` when a log-probability underflows to zero, so
        a single mark carrying no probability mass would poison the whole batch. In
        log space that mark contributes ``log p_y + log E[T_y]``, which is small
        exactly when its probability is negligible.
        """
        flat = self._native_flat_mixture()
        per_mark_log_mean = torch.logsumexp(
            flat.log_weights + flat.mu + 0.5 * flat.sigma.square(), dim=1
        ).reshape(self.batch, self.catalog_size)
        return torch.logsumexp(
            self.log_mark_probabilities.to(torch.float64) + per_mark_log_mean, dim=1
        )

    def waiting_marginal_log_mark_probabilities(
        self, elapsed_sec: torch.Tensor
    ) -> torch.Tensor:
        """Ranking ``log p_s(y) = log p_y + log S_y(s) - log S(s)`` [batch, V].

        The unknown future residual is integrated out; at ``s == 0`` this equals
        ``log p_y`` exactly (both survival pins are exact zeros).
        """
        elapsed = elapsed_sec.to(torch.float64)
        if (elapsed < 0).any():
            raise ValueError("elapsed_sec must be non-negative")
        log_joint_mass = self.log_mark_probabilities.to(
            torch.float64
        ) + self.per_mark_log_survival(elapsed)
        return log_joint_mass - torch.logsumexp(log_joint_mass, dim=1, keepdim=True)

    def conditional_log_mark_probabilities(self, time_sec: torch.Tensor) -> torch.Tensor:
        """``log p(y | T = t, H)`` [batch, V]: conditions on an exact event time.

        Generally different from the waiting-marginal ranking above; only the
        scoring path may feed an observed target time in here.
        """
        time64 = time_sec.to(torch.float64)
        if (time64 <= 0).any():
            raise ValueError("conditional mark probabilities require positive times")
        weighted = self.log_mark_probabilities.to(
            torch.float64
        ) + self.per_mark_log_density(time_sec)
        return weighted - torch.logsumexp(weighted, dim=1, keepdim=True)


class MarginalResidualDistribution(ResidualTimeDistribution):
    """Residual-time distribution after waiting ``s``, marginal over the unknown mark.

    ``log_density(r) = log sum_y[p_y g_y(s + r)] - log S(s)`` and ``log_survival(c)
    = log S(s + c) - log S(s)`` (the conditional remaining-exposure survival — never
    the base ``log S(c)``, and never the official last-observed-mark selection).
    Native seconds throughout.
    """

    def __init__(self, base: PerMarkJointDistribution, elapsed_sec: torch.Tensor) -> None:
        elapsed = elapsed_sec.to(torch.float64)
        if elapsed.ndim != 1 or int(elapsed.shape[0]) != base.batch:
            raise ValueError("elapsed_sec must have shape [batch]")
        if (elapsed < 0).any():
            raise ValueError("elapsed_sec must be non-negative")
        self.base = base
        self.elapsed_sec = elapsed
        self._log_survival_at_query = base.log_survival(elapsed)

    def batch_size(self) -> int:
        return self.base.batch

    def log_density(self, residual_time_sec: torch.Tensor) -> torch.Tensor:
        residual = residual_time_sec.to(torch.float64)
        if residual.ndim != 1 or int(residual.shape[0]) != self.base.batch:
            raise ValueError("residual_time_sec must have shape [batch]")
        total = self.elapsed_sec + residual.clamp_min(0.0)
        conditional = self.base.log_marginal_density(total) - self._log_survival_at_query
        return torch.where(
            residual > 0, conditional, torch.full_like(conditional, -math.inf)
        )

    def log_survival(self, time_sec: torch.Tensor) -> torch.Tensor:
        remaining = time_sec.to(torch.float64)
        if remaining.ndim != 1 or int(remaining.shape[0]) != self.base.batch:
            raise ValueError("time_sec must have shape [batch]")
        if (remaining < 0).any():
            raise ValueError("survival times must be non-negative")
        value = (
            self.base.log_survival(self.elapsed_sec + remaining)
            - self._log_survival_at_query
        )
        return torch.where(remaining > 0, value, torch.zeros_like(value))

    def residual_tail_mass_bound(self, from_sec: torch.Tensor) -> torch.Tensor:
        """``E[(R - a)^+] = E[(T - (s + a))^+] / S(s)`` over marks and components.

        Every mark's every component contributes its own bounded tail moment, so a mark
        carrying almost no probability still counts for exactly what it holds: the
        shared walk can then keep going across the plateau such a mark leaves in the
        survival instead of stopping at it. Taken in log space throughout.
        """
        base = self.base
        flat = base._native_flat_mixture()
        total = self.elapsed_sec + from_sec.to(torch.float64)
        per_component = lognormal_component_tail_mass_log(
            flat.mu,
            flat.sigma,
            total.unsqueeze(1)
            .expand(base.batch, base.catalog_size)
            .reshape(base.batch * base.catalog_size),
        )
        per_mark = torch.logsumexp(flat.log_weights + per_component, dim=1).reshape(
            base.batch, base.catalog_size
        )
        return (
            torch.logsumexp(
                base.log_mark_probabilities.to(torch.float64) + per_mark, dim=1
            )
            - self._log_survival_at_query
        ).exp()

