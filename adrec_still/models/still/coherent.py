"""Coherent finite-return joint model (Issue #41).

One fixed base joint distribution per row, conditioned only on history/anchor metadata:

```text
f(T, y | H) = sum_k pi_k(H) * g_k(T | H) * p_k(y | H)
S(s | H)    = sum_k pi_k(H) * S_k(s | H)
q_s(r, y)   = f(s + r, y) / S(s)                       r > 0
omega_k(s)  = pi_k * S_k(s) / S(s)
p_s(y)      = sum_k omega_k(s) * p_k(y)                ranking marginal
S_res(c|s)  = S(s + c) / S(s)
```

The base parameters ``pi, mu, sigma, p_k`` never see ``query_s_sec``: every waiting
effect comes from conditioning the fixed base, so re-querying the same gap re-derives
the same base (prediction runs in eval mode; no fresh dropout draw may masquerade as
consistency). Ranking integrates out the unknown future residual time; the observed
target time or POI enters only the scoring path, never inference.

The registered same-checkpoint ablation ``fixed_pi_truncated`` keeps each component's
correct truncation but freezes the original ``pi`` (no cross-component survivor
reweighting). It is normalized per query, no longer satisfies the full cross-query
conditioning identity, must carry its own ``variant_id``, equals the coherent model at
``s == 0`` (and when ``K == 1`` or all component survivals coincide), and leaves the POI
marginal unchanged when all component mark heads coincide.

``K`` (``mixture_components``, pilot default 2) is a bounded starting point, not a
frozen recipe; component-collapse diagnostics are exported rather than hidden, and no
latent component is given a semantic-cell label.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from adrec_still.models.baselines.base import (
    BaselineModel,
    BaselinePrediction,
    masked_log_probabilities,
    validate_batch,
)
from adrec_still.models.baselines.gru4rec import GRU4RecH
from adrec_still.models.still.conditioning import ANCHOR_FEATURE_DIM, anchor_features
from adrec_still.time.distributions import (
    FixedPiTruncatedResidualDistribution,
    LognormalMixtureParameters,
    SurvivalConditionedResidualDistribution,
)
from adrec_still.models.still.heads import FeatureFusion, LognormalMixtureHead
from adrec_still.models.still.losses import TRAINING_PROFILE, survival_training_masks

VARIANT_SURVIVOR_REWEIGHTED = "survivor_reweighted"
VARIANT_FIXED_PI_TRUNCATED = "fixed_pi_truncated"
_VARIANTS = (VARIANT_SURVIVOR_REWEIGHTED, VARIANT_FIXED_PI_TRUNCATED)


class StillCoherentJoint(BaselineModel, nn.Module):
    """Fixed log-normal-mixture base joint with exact survival conditioning."""

    name = "still_joint"
    family = "still"

    def __init__(
        self,
        number_of_pois: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        mixture_components: int = 2,
        variant: str = VARIANT_SURVIVOR_REWEIGHTED,
    ) -> None:
        nn.Module.__init__(self)
        BaselineModel.__init__(self, number_of_pois)
        if mixture_components <= 0:
            raise ValueError("mixture_components must be positive")
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {_VARIANTS}")
        self.mixture_components = int(mixture_components)
        self.variant = variant
        self.history_encoder = GRU4RecH(
            number_of_pois,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        context_dim = self.history_encoder.context_dim
        self.anchor_fusion = FeatureFusion(ANCHOR_FEATURE_DIM, context_dim)
        self.time_head = LognormalMixtureHead(context_dim, self.mixture_components)
        # One shared candidate/item table (inside the composed encoder); components only
        # add a per-component context projection, never a per-component item table.
        self.component_projection = nn.Linear(
            context_dim, self.mixture_components * context_dim
        )

    @property
    def context_dim(self) -> int:
        return self.history_encoder.context_dim

    @property
    def device(self) -> torch.device:
        return self.history_encoder.item_embedding.weight.device

    # ------------------------------------------------------------------ base pieces

    def base_context(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """History context fused with anchor-clock features; ``query_s_sec`` never enters."""
        context = self.history_encoder.encode(batch)
        features = anchor_features(batch).to(context.device)
        return self.anchor_fusion(context, features)

    def base_parameters(self, base_context: torch.Tensor) -> LognormalMixtureParameters:
        return self.time_head(base_context)

    def component_contexts(self, base_context: torch.Tensor) -> torch.Tensor:
        batch = base_context.shape[0]
        return self.component_projection(base_context).reshape(
            batch, self.mixture_components, self.context_dim
        )

    def component_mark_log_probabilities(self, base_context: torch.Tensor) -> torch.Tensor:
        """Exact normalized ``log p_k(y)`` over the frozen catalog, [batch, K, catalog]."""
        contexts = self.component_contexts(base_context)
        batch, components, width = contexts.shape
        flat_scores = self.history_encoder.score_full_catalog(
            contexts.reshape(batch * components, width)
        )
        flat_log_probabilities = masked_log_probabilities(flat_scores)
        return flat_log_probabilities.reshape(batch, components, -1)

    def component_candidate_scores(
        self, base_context: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        """Unnormalized per-component scores for explicit candidates, [batch, K, C]."""
        contexts = self.component_contexts(base_context)
        batch, components, width = contexts.shape
        repeated_candidates = candidates.repeat_interleave(components, dim=0)
        flat = self.history_encoder.score_candidates(
            contexts.reshape(batch * components, width), repeated_candidates
        )
        return flat.reshape(batch, components, -1)

    # ------------------------------------------------------------------- inference

    def _log_component_weights_at_query(
        self, base: LognormalMixtureParameters, elapsed_sec: torch.Tensor
    ) -> torch.Tensor:
        if self.variant == VARIANT_SURVIVOR_REWEIGHTED:
            return base.survivor_log_weights(elapsed_sec)
        return base.normalized_log_weights()

    def marginal_poi_log_probabilities(
        self,
        base: LognormalMixtureParameters,
        component_log_probabilities: torch.Tensor,
        elapsed_sec: torch.Tensor,
    ) -> torch.Tensor:
        """Ranking marginal ``log p_s(y)``: the unknown future residual is integrated out."""
        weights = self._log_component_weights_at_query(base, elapsed_sec)
        return torch.logsumexp(
            weights.unsqueeze(2) + component_log_probabilities.to(torch.float64), dim=1
        )

    def predict(self, batch: dict[str, torch.Tensor]) -> BaselinePrediction:
        self.require_fitted()
        validate_batch(batch)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                context = self.base_context(batch)
                base = self.base_parameters(context)
                component_log_probabilities = self.component_mark_log_probabilities(context)
                # The shared harness delivers CPU collate batches; scalar query fields
                # must join the model's parameter device before mixing with base math.
                elapsed = batch["query_s_sec"].to(device=context.device, dtype=torch.float64)
                marginal = self.marginal_poi_log_probabilities(
                    base, component_log_probabilities, elapsed
                )
                base_cpu = LognormalMixtureParameters(
                    log_weights=base.log_weights.detach().cpu(),
                    mu=base.mu.detach().cpu(),
                    sigma=base.sigma.detach().cpu(),
                )
                elapsed_cpu = elapsed.cpu()
        finally:
            self.train(was_training)
        if self.variant == VARIANT_SURVIVOR_REWEIGHTED:
            time_distribution = SurvivalConditionedResidualDistribution(base_cpu, elapsed_cpu)
        else:
            time_distribution = FixedPiTruncatedResidualDistribution(base_cpu, elapsed_cpu)
        return BaselinePrediction(
            poi_logits=marginal.to(torch.float32).cpu(),
            context=context.cpu(),
            time_distribution=time_distribution,
        )

    #: Score declaration for the merged #38 exact chunked scorer. A mixture of
    #: per-component softmaxes has no single logit vector whose softmax is the
    #: ranking marginal, so the chunk scores from ``exact_candidate_factory`` are
    #: exact normalized log-probabilities (each component normalized once over the
    #: frozen catalog); the chunker verifies the full-catalog mass within its
    #: declared tolerance.
    exact_score_kind = "log_probabilities"

    # --------------------------------------------------------------------- scoring

    def score_joint(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Per-row joint log score and its exact decomposition on observed-known rows.

        Scoring only: the observed residual time and target enter here as evaluation
        targets. Returns full-batch tensors plus ``eligible`` (observed exact-known
        rows); values outside the eligible mask are meaningless and must be discarded
        by the caller. The decomposition satisfies
        ``log_joint == log_time + log_mark_given_time`` per eligible row, where the
        mark term conditions on the observed residual time (generally different from
        the ranking marginal).
        """
        context = self.base_context(batch)
        device = context.device
        # Batch fields cross the model boundary here: the caller's (typically CPU
        # collate) batch is never mutated, and every tensor entering base math or a
        # gather lives on the model's parameter device.
        eligible = (
            batch["observed_mask"].to(device).bool()
            & batch["target_in_train_catalog"].to(device).bool()
        )
        base = self.base_parameters(context)
        component_log_probabilities = self.component_mark_log_probabilities(context)
        elapsed = batch["query_s_sec"].to(device=device, dtype=torch.float64)
        residual = batch["residual_time_sec"].to(device=device, dtype=torch.float64)
        # Select-first policy: padded residuals (0 on censored rows) never reach log().
        safe_residual = torch.where(eligible, residual, torch.ones_like(residual))
        total = elapsed + safe_residual

        component_log_density = base.component_log_density(total)
        log_weights = base.normalized_log_weights()
        if self.variant == VARIANT_SURVIVOR_REWEIGHTED:
            components = log_weights + component_log_density
            normalizer = base.log_survival(elapsed)
        else:
            components = (
                log_weights + component_log_density - base.component_log_survival(elapsed)
            )
            normalizer = torch.zeros_like(elapsed)

        target = batch["next_poi"].to(device).clamp_min(0)
        target_log_probability = component_log_probabilities.to(torch.float64).gather(
            2, target.view(-1, 1, 1).expand(-1, self.mixture_components, 1)
        )[:, :, 0]

        log_time = torch.logsumexp(components, dim=1) - normalizer
        log_joint = torch.logsumexp(components + target_log_probability, dim=1) - normalizer
        # The conditional mark term is computed on its own path (gamma-weighted mark
        # mixture), not derived as log_joint - log_time: the exported decomposition
        # error is then a genuine cross-check of two computations of the same
        # identity instead of a tautology.
        log_gamma = components - torch.logsumexp(components, dim=1, keepdim=True)
        log_mark_given_time = torch.logsumexp(log_gamma + target_log_probability, dim=1)
        return {
            "eligible": eligible,
            "log_joint": log_joint,
            "log_time": log_time,
            "log_mark_given_time": log_mark_given_time,
        }

    def exact_candidate_factory(self, chunk_size: int = 4096):
        """Two-pass bounded exact marginal callback with per-component normalizers."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.require_fitted()

        def prepare(features):
            was_training = self.training
            self.eval()
            try:
                with torch.no_grad():
                    context = self.base_context(features)
                    base = self.base_parameters(context)
                    elapsed = features["query_s_sec"].to(context.device, torch.float64)
                    weights = self._log_component_weights_at_query(base, elapsed)
                    normalizer = torch.full(
                        (len(context), self.mixture_components),
                        -torch.inf,
                        dtype=torch.float64,
                        device=context.device,
                    )
                    for start in range(2, self.number_of_pois, chunk_size):
                        ids = torch.arange(
                            start,
                            min(start + chunk_size, self.number_of_pois),
                            device=context.device,
                        ).expand(len(context), -1)
                        scores = self.component_candidate_scores(context, ids).to(torch.float64)
                        normalizer = torch.logaddexp(
                            normalizer, torch.logsumexp(scores, dim=2)
                        )
            finally:
                self.train(was_training)

            def score(ids):
                was_training = self.training
                self.eval()
                try:
                    with torch.no_grad():
                        candidates = ids.to(context.device).expand(len(context), -1)
                        values = self.component_candidate_scores(context, candidates).to(
                            torch.float64
                        )
                        return torch.logsumexp(
                            weights[:, :, None] + values - normalizer[:, :, None], dim=1
                        )
                finally:
                    self.train(was_training)

            return score

        return prepare

    # -------------------------------------------------------------------- training

    def per_row_sampled_training_losses(
        self,
        kept: dict[str, torch.Tensor],
        negative_samples: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Observed-known joint loss ``-log q_s(r) - log p(y | r)`` (mark part sampled).

        The time term is the exact survival-conditioned density; the conditional mark
        term uses the shared sampled-candidate policy per component (an explicitly
        labeled approximation of the exact catalog softmax). Only the
        ``survivor_reweighted`` variant trains; the ablation re-scores the same
        checkpoint and is never a separate training run.
        """
        if self.variant != VARIANT_SURVIVOR_REWEIGHTED:
            raise RuntimeError(
                "the fixed-pi ablation re-scores the trained coherent checkpoint; "
                "training it as a separate model is not a registered configuration"
            )
        from adrec_still.models.baselines.trainer import build_sampled_candidates

        observed = kept["observed_mask"].bool() & kept["target_in_train_catalog"].bool()
        if not bool(observed.all()):
            raise ValueError(
                "the shared trainer must deliver the observed-known population "
                f"({TRAINING_PROFILE}); censored/unknown training needs the #39 profile"
            )
        residual = kept["residual_time_sec"].to(torch.float64)
        if (residual <= 0).any():
            raise ValueError("observed rows require positive residual_time_sec")

        context = self.base_context(kept)
        base = self.base_parameters(context)
        elapsed = kept["query_s_sec"].to(torch.float64)
        total = elapsed + residual

        log_weights = base.normalized_log_weights()
        components = log_weights + base.component_log_density(total)
        log_time = torch.logsumexp(components, dim=1) - base.log_survival(elapsed)
        # gamma_k(r) = pi_k g_k(s+r) / sum_j pi_j g_j(s+r): S(s) cancels.
        log_gamma = components - torch.logsumexp(components, dim=1, keepdim=True)

        candidates, accidental_hit_penalty = build_sampled_candidates(
            kept["next_poi"], negative_samples, self.number_of_pois, generator
        )
        candidate_scores = self.component_candidate_scores(context, candidates)
        candidate_scores = candidate_scores - accidental_hit_penalty.unsqueeze(1)
        sampled_log_probabilities = torch.log_softmax(candidate_scores, dim=2)
        log_mark_given_time = torch.logsumexp(
            log_gamma.to(sampled_log_probabilities.dtype)
            + sampled_log_probabilities[:, :, 0],
            dim=1,
        )
        return -(log_time.to(log_mark_given_time.dtype) + log_mark_given_time)

    def per_row_survival_training_losses(
        self,
        batch: dict[str, torch.Tensor],
        negative_samples: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Three-way #39 likelihood under one survival-conditioned base joint."""
        if self.variant != VARIANT_SURVIVOR_REWEIGHTED:
            raise RuntimeError("the fixed-pi ablation cannot be a separate training run")
        masks = survival_training_masks(batch)
        context = self.base_context(batch)
        device = context.device
        base = self.base_parameters(context)
        elapsed = batch["query_s_sec"].to(device=device, dtype=torch.float64)
        residual = batch["residual_time_sec"].to(device=device, dtype=torch.float64)
        censor = batch["censor_time_sec"].to(device=device, dtype=torch.float64)
        observed = batch["observed_mask"].to(device).bool()
        safe_residual = torch.where(observed, residual, torch.ones_like(residual))
        safe_censor = torch.where(~observed, censor, torch.ones_like(censor))

        log_weights = base.normalized_log_weights()
        components = log_weights + base.component_log_density(elapsed + safe_residual)
        log_time = torch.logsumexp(components, dim=1) - base.log_survival(elapsed)
        censor_nll = -(
            base.log_survival(elapsed + safe_censor) - base.log_survival(elapsed)
        )
        losses = torch.where(observed, -log_time, censor_nll)

        known = masks["observed_known"].to(device)
        if known.any():
            from adrec_still.models.baselines.trainer import build_sampled_candidates

            targets = batch["next_poi"].to(device)[known]
            candidates, penalty = build_sampled_candidates(
                targets, negative_samples, self.number_of_pois, generator
            )
            candidate_scores = self.component_candidate_scores(context[known], candidates)
            candidate_scores = candidate_scores - penalty.unsqueeze(1)
            sampled_log_probabilities = torch.log_softmax(candidate_scores, dim=2)
            known_components = components[known]
            log_gamma = known_components - torch.logsumexp(
                known_components, dim=1, keepdim=True
            )
            log_mark = torch.logsumexp(
                log_gamma.to(sampled_log_probabilities.dtype)
                + sampled_log_probabilities[:, :, 0],
                dim=1,
            )
            losses = losses.clone()
            losses[known] = losses[known] - log_mark.to(losses.dtype)
        return losses

    # ------------------------------------------------------------------ diagnostics

    def component_diagnostics(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Component-collapse diagnostics at the batch's queries (reported, not hidden).

        Runs in eval mode (restoring the caller's mode) so a dropout draw can never
        masquerade as component behavior, and moves the query field to the model's
        device like every other inference entry point.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                context = self.base_context(batch)
                base = self.base_parameters(context)
                elapsed = batch["query_s_sec"].to(device=context.device, dtype=torch.float64)
                log_omega = self._log_component_weights_at_query(base, elapsed)
                omega = log_omega.exp()
                entropy = -(omega * log_omega.clamp_min(-1.0e9)).sum(dim=1)
        finally:
            self.train(was_training)
        return {
            "mixture_components": float(self.mixture_components),
            "mean_max_component_weight": float(omega.max(dim=1).values.mean()),
            "mean_effective_components": float(entropy.exp().mean()),
            "min_component_weight": float(omega.min()),
        }

    # ------------------------------------------------------------------- interface

    def fit(self, training_source: Any) -> StillCoherentJoint:
        from adrec_still.models.baselines.trainer import TrainingPlan, train_model

        if not isinstance(training_source, TrainingPlan):
            raise TypeError("StillCoherentJoint.fit expects a shared-trainer TrainingPlan")
        train_model(self, training_source)
        return self

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def head_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for module in (self.anchor_fusion, self.time_head, self.component_projection)
            for parameter in module.parameters()
        )

    def resolved_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "interface_version": self.interface_version,
            "number_of_pois": self.number_of_pois,
            "embedding_dim": self.history_encoder.embedding_dim,
            "hidden_dim": self.history_encoder.hidden_dim,
            "num_layers": self.history_encoder.num_layers,
            "dropout": self.history_encoder.dropout,
            "mixture_components": self.mixture_components,
            "variant": self.variant,
            "conditioning_features": "anchor_clock(base); waiting via survival conditioning",
            "encoder": "gru4rec_h(composed)",
            "encoder_parameter_count": self.history_encoder.parameter_count(),
            "head_parameter_count": self.head_parameter_count(),
            "parameter_count": self.parameter_count(),
            "context_dim": self.context_dim,
        }
