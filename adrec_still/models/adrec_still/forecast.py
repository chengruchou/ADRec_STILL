"""Silence-conditioned forecasting for the ADRec-STILL family.

A forecast is built once per batch from history and elapsed silence ``s`` only:

1. branch weights — ``pi_raw`` from H; the catalog projection ``pi_cat`` masks
   empty-support branches; with a time model the survivor posterior
   ``kappa_b(s) = pi_cat_b S_b(s) / S_cat(s)`` reweights them;
2. stratified truncated time samples — with ``a`` the cell time threshold,
   per-branch strata ``[s, a)`` (early, when ``s < a``) and ``[max(s, a), inf)``
   (late) carry analytic masses ``w[b,early] = pi_cat_b (S_b(s) - S_b(a)) /
   S_cat(s)`` and ``w[b,late] = pi_cat_b S_b(max(s, a)) / S_cat(s)`` (stable
   log-space differences). Truncated samples first reweight mixture components
   by prior x in-stratum mass, then invert each component's survival — no
   rejection loops, no hard caps;
3. item generation — one reverse-diffusion (or direct-head) pass per
   (branch, stratum, time sample, latent); the same noise and samples serve
   every catalog chunk and the cell readout;
4. scoring — per-sample branch-restricted scores are normalized over the frozen
   catalog once (chunked, float64) and item probabilities are averaged in
   probability space across samples with the stratum weights.

The cell readout aggregates the same samples: early/late mass is analytic (the
stratum weights), the spatial split sums normalized item probabilities over the
local/displaced groups induced by the causal home and ``d0``. Geo-unresolvable
item mass is reported as unresolved, never silently renormalized per stratum.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from adrec_still.evaluation.ranking import full_catalog_candidate_ids
from adrec_still.models.adrec_still.memory import memory_features
from adrec_still.models.baselines.base import BaselinePrediction
from adrec_still.models.still.conditioning import SECONDS_PER_HOUR  # noqa: F401
from adrec_still.time.distributions import (
    LognormalMixtureParameters,
    SurvivalConditionedResidualDistribution,
)

MASKED = -1.0e9
EARTH_RADIUS_KM = 6371.0088
CELL_NAMES = ("local_early", "displaced_early", "local_late", "displaced_late")


def log1mexp(log_value: torch.Tensor) -> torch.Tensor:
    bound = math.log(0.5)
    return torch.where(
        log_value > bound,
        torch.log(-torch.expm1(log_value.clamp(max=-1e-12))),
        torch.log1p(-torch.exp(log_value)),
    )


def haversine_km(lat1, lon1, lat2, lon2) -> torch.Tensor:
    lat1, lon1, lat2, lon2 = (torch.deg2rad(value.to(torch.float64))
                              for value in (lat1, lon1, lat2, lon2))
    half_dlat = (lat2 - lat1) / 2
    half_dlon = (lon2 - lon1) / 2
    a = (torch.sin(half_dlat) ** 2
         + torch.cos(lat1) * torch.cos(lat2) * torch.sin(half_dlon) ** 2)
    return 2 * EARTH_RADIUS_KM * torch.asin(torch.sqrt(a.clamp(0, 1)))


def _truncated_lognormal_sample(
    parameters: LognormalMixtureParameters,
    lower: torch.Tensor,
    upper: torch.Tensor | None,
    generator: torch.Generator,
) -> torch.Tensor:
    """One truncated sample per row from the mixture restricted to (lower, upper].

    Component weights are reweighted by prior x in-interval survival mass before
    a component is drawn (never the untruncated weights); the time inverts that
    component's survival between its interval endpoints.
    """
    log_weights, mu, sigma = parameters._float64()
    device = mu.device
    lower64 = lower.to(torch.float64)
    log_survival_lower = parameters.component_log_survival(lower64)
    if upper is None:
        log_mass = log_survival_lower
        log_survival_upper = torch.full_like(log_survival_lower, -math.inf)
    else:
        log_survival_upper = parameters.component_log_survival(upper.to(torch.float64))
        delta = (log_survival_upper - log_survival_lower).clamp(max=0.0)
        log_mass = log_survival_lower + log1mexp(delta.clamp(min=-1e30))
    joint = log_weights + log_mass
    # Rows whose interval holds no mass are placeholders (their stratum weight is
    # zero); give them a uniform draw so sampling stays finite everywhere.
    degenerate = torch.isinf(joint).all(dim=1)
    safe_joint = torch.where(degenerate.unsqueeze(1), torch.zeros_like(joint), joint)
    probabilities = torch.softmax(safe_joint, dim=1)
    component = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
    pick = component.unsqueeze(1)
    survival_lower = log_survival_lower.gather(1, pick).squeeze(1).exp()
    survival_upper = log_survival_upper.gather(1, pick).squeeze(1).exp()
    u = torch.rand(lower64.shape, generator=generator, device=device,
                   dtype=torch.float64)
    target_survival = (survival_upper + u * (survival_lower - survival_upper)).clamp(
        min=torch.finfo(torch.float64).tiny, max=1.0 - 1e-16
    )
    # S(T) = Phi(-z) => z = -ndtri(S); T = exp(mu + sigma z)
    z = -torch.special.ndtri(target_survival)
    mu_pick = mu.gather(1, pick).squeeze(1)
    sigma_pick = sigma.gather(1, pick).squeeze(1)
    sample = torch.exp(mu_pick + sigma_pick * z)
    floor = lower64 * (1.0 + 1e-12) + 1e-9
    return torch.maximum(sample, floor)


class SilenceForecast:
    """One batch's forecast: fixed samples reused across chunks and readouts."""

    def __init__(self, model, batch: dict[str, Any], *, chunk_size: int = 4096) -> None:
        self.model = model
        self.batch = batch
        self.chunk_size = int(chunk_size)
        self.device = model.device
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                self._build()
        finally:
            model.train(was_training)

    # ------------------------------------------------------------------- build

    def _build(self) -> None:
        model = self.model
        batch = self.batch
        self.rows = int(batch["history_poi"].shape[0])
        self.memory = (
            memory_features(model.require_memory(), batch,
                            home_policy=model.home_policy_eval)
            if model.use_branches else None
        )
        self.encoded, _mask = model.encode_history(batch)
        context = model.base_context(batch, encoded=self.encoded)
        generator = model._forecast_generator(batch)
        elapsed = batch["query_s_sec"].to(self.device, torch.float64)
        anchor = batch["last_timestamp"].to(self.device, torch.float64)

        self.extras: dict[str, Any] = {
            "estimator": "stratified_truncated_mc",
            "time_samples": model.time_samples,
            "diffusion_samples": model.diffusion_samples,
            "inference_steps": model.inference_steps,
        }
        samples: list[dict[str, Any]] = []
        self.time_distribution = None

        if model.use_branches:
            log_prior_raw = model.branch_log_prior(context).to(torch.float64)
            support_mask = model.catalog_support_log_mask(self.memory)
            log_prior_cat = log_prior_raw + support_mask
            log_prior_cat = log_prior_cat - torch.logsumexp(log_prior_cat, dim=1,
                                                            keepdim=True)
        else:
            log_prior_raw = log_prior_cat = None

        if model.time_model == "lognormal":
            parameters = model.branch_time_parameters(context)
            threshold = torch.full_like(elapsed, model.cell_time_threshold_s)
            log_surv_s = torch.stack(
                [p.log_survival(elapsed) for p in parameters], dim=1
            )
            log_surv_a = torch.stack(
                [p.log_survival(torch.maximum(elapsed, threshold)) for p in parameters],
                dim=1,
            )
            # Catalog-view survivor posterior and stratum weights.
            joint_cat = log_prior_cat + log_surv_s
            log_s_cat = torch.logsumexp(joint_cat, dim=1)
            self.extras["repeat_probability_catalog"] = (
                (joint_cat[:, 0] - log_s_cat).exp().cpu()
            )
            joint_raw = log_prior_raw + log_surv_s
            log_s_raw = torch.logsumexp(joint_raw, dim=1)
            self.extras["repeat_probability_all"] = (
                (joint_raw[:, 0] - log_s_raw).exp().cpu()
            )
            early_possible = elapsed < threshold
            delta = (log_surv_a - log_surv_s).clamp(max=0.0)
            log_early = log_prior_cat + log_surv_s + log1mexp(delta.clamp(min=-1e30))
            log_early = torch.where(
                early_possible.unsqueeze(1), log_early,
                torch.full_like(log_early, -math.inf),
            )
            log_late = log_prior_cat + log_surv_a
            weights = torch.stack(
                [log_early, log_late], dim=2
            )  # [B, branch, stratum]
            weights = (weights - log_s_cat.view(-1, 1, 1)).exp()
            weights = torch.where(torch.isfinite(weights), weights,
                                  torch.zeros_like(weights))
            self.extras["stratum_weights"] = weights.cpu()
            # All-event residual distribution (pi_raw view): merge both branches'
            # components into one mixture, then condition on survival at s.
            merged = LognormalMixtureParameters(
                log_weights=torch.cat(
                    [log_prior_raw[:, b:b + 1] + parameters[b].normalized_log_weights()
                     for b in range(2)], dim=1,
                ),
                mu=torch.cat([p.mu.to(torch.float64) for p in parameters], dim=1),
                sigma=torch.cat([p.sigma.to(torch.float64) for p in parameters], dim=1),
            )
            self.time_distribution = SurvivalConditionedResidualDistribution(
                LognormalMixtureParameters(
                    log_weights=merged.log_weights.detach().cpu(),
                    mu=merged.mu.detach().cpu(),
                    sigma=merged.sigma.detach().cpu(),
                ),
                elapsed.detach().cpu(),
            )
            for branch in range(2):
                for stratum, (lower, upper) in enumerate((
                    (elapsed, torch.maximum(elapsed, threshold)),
                    (torch.maximum(elapsed, threshold), None),
                )):
                    stratum_weight = weights[:, branch, stratum]
                    for _ in range(model.time_samples):
                        total = _truncated_lognormal_sample(
                            parameters[branch], lower, upper, generator
                        )
                        samples.append({
                            "branch": branch,
                            "stratum": "early" if stratum == 0 else "late",
                            "weight": stratum_weight / model.time_samples,
                            "total_sec": total,
                        })
        elif model.use_branches:
            kappa = log_prior_cat.exp()
            self.extras["repeat_probability_catalog"] = kappa[:, 0].cpu()
            self.extras["repeat_probability_all"] = log_prior_raw.exp()[:, 0].cpu()
            for branch in range(2):
                samples.append({
                    "branch": branch, "stratum": "all",
                    "weight": kappa[:, branch], "total_sec": None,
                })
        else:
            samples.append({
                "branch": None, "stratum": "all",
                "weight": torch.ones(self.rows, dtype=torch.float64,
                                     device=self.device),
                "total_sec": None,
            })

        # Latent replication: each (branch, stratum, T) sample spawns
        # diffusion_samples independent reverse chains; probabilities are
        # averaged, never logits.
        replicated: list[dict[str, Any]] = []
        for sample in samples:
            share = sample["weight"] / model.diffusion_samples
            for _ in range(model.diffusion_samples):
                replicated.append({**sample, "weight": share})
        self.samples = replicated
        self.extras["mc_sample_count"] = len(self.samples)
        anchor64 = anchor
        for sample in self.samples:
            branch = sample["branch"]
            branch_tensor = (
                torch.full((self.rows,), branch, dtype=torch.long, device=self.device)
                if branch is not None else None
            )
            sample["generated"] = model._generate_inference(
                self.encoded, batch, branch_tensor, sample["total_sec"], generator
            )
            sample["reference_time"] = (
                anchor64 + sample["total_sec"] if sample["total_sec"] is not None
                else anchor64
            )
        self._compute_normalizers()

    def _compute_normalizers(self) -> None:
        candidates = full_catalog_candidate_ids(self.model.number_of_pois).to(self.device)
        for sample in self.samples:
            normalizer = torch.full((self.rows,), -math.inf, dtype=torch.float64,
                                    device=self.device)
            for start in range(0, len(candidates), self.chunk_size):
                ids = candidates[start:start + self.chunk_size].expand(self.rows, -1)
                scores = self._sample_scores(sample, ids).to(torch.float64)
                normalizer = torch.logaddexp(normalizer,
                                             torch.logsumexp(scores, dim=1))
            sample["normalizer"] = normalizer

    def _sample_scores(self, sample: dict[str, Any], ids: torch.Tensor) -> torch.Tensor:
        return self.model.branch_candidate_scores(
            sample["generated"], ids, self.memory, sample["branch"],
            sample["reference_time"],
        )

    # ----------------------------------------------------------------- scoring

    def score(self, ids: torch.Tensor) -> torch.Tensor:
        """Exact normalized log marginal probabilities for explicit IDs, [B, C]."""
        with torch.no_grad():
            expanded = ids.to(self.device)
            if expanded.ndim == 1:
                expanded = expanded.expand(self.rows, -1)
            probability = torch.zeros((self.rows, expanded.shape[1]),
                                      dtype=torch.float64, device=self.device)
            for sample in self.samples:
                log_p = (self._sample_scores(sample, expanded).to(torch.float64)
                         - sample["normalizer"].unsqueeze(1))
                probability = probability + sample["weight"].unsqueeze(1) * log_p.exp()
            return probability.clamp_min(torch.finfo(torch.float64).tiny).log().cpu()

    def prediction(self) -> BaselinePrediction:
        candidates = full_catalog_candidate_ids(self.model.number_of_pois)
        logits = torch.full((self.rows, self.model.number_of_pois), MASKED,
                            dtype=torch.float32)
        for start in range(0, len(candidates), self.chunk_size):
            ids = candidates[start:start + self.chunk_size]
            logits[:, ids] = self.score(ids).to(torch.float32)
        context = self.encoded[:, -1, :].detach().cpu()
        return BaselinePrediction(
            poi_logits=logits, context=context,
            time_distribution=self.time_distribution,
            extras={key: value for key, value in self.extras.items()},
        )

    # ------------------------------------------------------------- cell readout

    def cell_readout(self, *, cell_radius_km: float) -> dict[str, torch.Tensor]:
        """P(cell | H, T > s) over the four silence cells, plus coverage masses.

        Early/late mass is analytic (stratum weights); within each stratum the
        spatial split sums the normalized item probabilities over the
        local/displaced groups induced by the row's home. Known items without
        coordinates contribute to ``unresolved_mass``; the four-cell table over
        geocoded support is normalized once across all strata, never per
        stratum. Rows without a home or without a time model report
        ``available=False``.
        """
        model = self.model
        if model.time_model != "lognormal" or not model.use_branches:
            return {
                "available": torch.zeros(self.rows, dtype=torch.bool),
                "probabilities": torch.zeros((self.rows, 4), dtype=torch.float64),
                "unresolved_mass": torch.ones(self.rows, dtype=torch.float64),
            }
        memory = self.memory
        home_available = memory.home_available.clone()
        home_lat = memory.home_lat.to(self.device)
        home_lon = memory.home_lon.to(self.device)
        index = model.require_memory()
        candidates = full_catalog_candidate_ids(model.number_of_pois).to(self.device)
        with torch.no_grad():
            mass = torch.zeros((self.rows, 2, 2), dtype=torch.float64,
                               device=self.device)  # [row, early/late, local/displaced]
            unresolved = torch.zeros(self.rows, dtype=torch.float64, device=self.device)
            for start in range(0, len(candidates), self.chunk_size):
                ids = candidates[start:start + self.chunk_size]
                coordinates = [index.poi_coordinates(int(poi)) for poi in ids.cpu()]
                geocoded = torch.tensor(
                    [coordinate is not None for coordinate in coordinates],
                    device=self.device,
                )
                lat = torch.tensor(
                    [c[0] if c is not None else 0.0 for c in coordinates],
                    dtype=torch.float64, device=self.device,
                )
                lon = torch.tensor(
                    [c[1] if c is not None else 0.0 for c in coordinates],
                    dtype=torch.float64, device=self.device,
                )
                distance = haversine_km(
                    home_lat.unsqueeze(1), home_lon.unsqueeze(1),
                    lat.unsqueeze(0), lon.unsqueeze(0),
                )
                local = distance <= cell_radius_km
                expanded = ids.expand(self.rows, -1)
                for sample in self.samples:
                    if sample["stratum"] == "all":
                        continue
                    stratum_index = 0 if sample["stratum"] == "early" else 1
                    log_p = (self._sample_scores(sample, expanded).to(torch.float64)
                             - sample["normalizer"].unsqueeze(1))
                    weighted = sample["weight"].unsqueeze(1) * log_p.exp()
                    resolved = weighted * geocoded.unsqueeze(0)
                    mass[:, stratum_index, 0] += (resolved * local).sum(dim=1)
                    mass[:, stratum_index, 1] += (resolved * ~local).sum(dim=1)
                    unresolved += (weighted * ~geocoded.unsqueeze(0)).sum(dim=1)
        probabilities = mass.reshape(self.rows, 4)[
            :, [0, 1, 2, 3]
        ]  # local_early, displaced_early, local_late, displaced_late
        resolved_total = probabilities.sum(dim=1)
        available = home_available.to(self.device) & (resolved_total > 0)
        normalized = torch.where(
            available.unsqueeze(1),
            probabilities / resolved_total.clamp_min(1e-300).unsqueeze(1),
            torch.zeros_like(probabilities),
        )
        return {
            "available": available.cpu(),
            "probabilities": normalized.cpu(),
            "raw_cell_mass": probabilities.cpu(),
            "unresolved_mass": unresolved.cpu(),
            "geocoded_mass": resolved_total.cpu(),
        }
