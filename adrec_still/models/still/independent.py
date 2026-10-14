"""Independent time/POI control arm (Issue #41).

For every query the model outputs a valid normalized joint
``q_ind,s(r, y | H) = g_s(r | H, features) * p_s(y | H, features)`` where the features
are the legitimately available elapsed time and query clock. Both heads read the same
fused representation, so this arm is not information-starved relative to the coherent
arm. Each query's joint is valid on its own; consistency between queries of one gap is
not enforced — which is a property of this particular decoder, not a theorem about every
factorized model (a fixed factorized base can also be survival-conditioned correctly).

The residual-time head is a configurable log-normal mixture (``mixture_components``)
so a K-component coherent arm has a matching-K independent control; ``K == 1`` remains
available and then also exposes the accepted single-``LogNormalPrediction`` path (with
its exact analytic CRPS). This arm trains separately and selects its own checkpoint:
attaching an untrained time head to a POI-only checkpoint is not this configuration.
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
from adrec_still.models.still.conditioning import QUERY_FEATURE_DIM, query_features
from adrec_still.models.still.heads import FeatureFusion, LognormalMixtureHead
from adrec_still.models.still.losses import observed_known_joint_losses, survival_training_masks
from adrec_still.time.distributions import (
    LognormalMixtureParameters,
    MixtureResidualDistribution,
)
from adrec_still.time.protocol import LogNormalPrediction


class StillIndependent(BaselineModel, nn.Module):
    """Directly conditioned residual-time mixture and POI head on shared information."""

    name = "still_independent"
    family = "still"

    def __init__(
        self,
        number_of_pois: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        mixture_components: int = 2,
    ) -> None:
        nn.Module.__init__(self)
        BaselineModel.__init__(self, number_of_pois)
        if mixture_components <= 0:
            raise ValueError("mixture_components must be positive")
        self.mixture_components = int(mixture_components)
        # Composition of the accepted shared encoder (tied item table included); this
        # instance is private to the model so separately trained runs never share
        # updating parameters.
        self.history_encoder = GRU4RecH(
            number_of_pois,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.feature_fusion = FeatureFusion(QUERY_FEATURE_DIM, self.history_encoder.context_dim)
        self.time_head = LognormalMixtureHead(
            self.history_encoder.context_dim, self.mixture_components
        )

    @property
    def context_dim(self) -> int:
        return self.history_encoder.context_dim

    @property
    def device(self) -> torch.device:
        return self.history_encoder.item_embedding.weight.device

    def fused_context(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """History context fused with the elapsed/query-clock features."""
        context = self.history_encoder.encode(batch)
        features = query_features(batch).to(context.device)
        return self.feature_fusion(context, features)

    def residual_time_distribution(self, fused: torch.Tensor) -> MixtureResidualDistribution:
        return MixtureResidualDistribution(self.time_head(fused))

    def per_row_sampled_training_losses(
        self,
        kept: dict[str, torch.Tensor],
        negative_samples: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Observed-known joint loss: exact time NLL plus the shared sampled mark loss.

        The mark term is the shared sampled-softmax approximation (explicitly not an
        exact full-catalog likelihood); the sampler/correction policy comes from the
        shared trainer builder so both #41 arms train identically.
        """
        fused = self.fused_context(kept)
        parameters = self.time_head(fused)
        return observed_known_joint_losses(
            model=self,
            kept=kept,
            fused_context=fused,
            time_log_density=parameters.log_density(kept["residual_time_sec"]),
            negative_samples=negative_samples,
            generator=generator,
            mark_logits_fn=lambda candidates: self.history_encoder.score_candidates(
                fused, candidates
            ),
        )

    def per_row_survival_training_losses(
        self,
        batch: dict[str, torch.Tensor],
        negative_samples: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Three-way #39 loss: known joint, unknown time, or censor survival."""
        masks = survival_training_masks(batch)
        fused = self.fused_context(batch)
        parameters = self.time_head(fused)
        residual = batch["residual_time_sec"].to(fused.device, torch.float64)
        censor = batch["censor_time_sec"].to(fused.device, torch.float64)
        observed = batch["observed_mask"].to(fused.device).bool()
        safe_residual = torch.where(observed, residual, torch.ones_like(residual))
        safe_censor = torch.where(~observed, censor, torch.ones_like(censor))
        losses = -parameters.log_density(safe_residual)
        losses = torch.where(~observed, -parameters.log_survival(safe_censor), losses)

        known = masks["observed_known"].to(fused.device)
        if known.any():
            from adrec_still.models.baselines.trainer import build_sampled_candidates

            targets = batch["next_poi"].to(fused.device)[known]
            candidates, penalty = build_sampled_candidates(
                targets, negative_samples, self.number_of_pois, generator
            )
            logits = self.history_encoder.score_candidates(fused[known], candidates) - penalty
            mark_nll = -torch.log_softmax(logits, dim=1)[:, 0]
            losses = losses.clone()
            losses[known] = losses[known] + mark_nll.to(losses.dtype)
        return losses

    def predict(self, batch: dict[str, torch.Tensor]) -> BaselinePrediction:
        """Exact normalized POI log-probabilities plus the residual-time distribution."""
        self.require_fitted()
        validate_batch(batch)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                fused = self.fused_context(batch)
                log_probabilities = masked_log_probabilities(
                    self.history_encoder.score_full_catalog(fused)
                )
                parameters = self.time_head(fused)
        finally:
            self.train(was_training)
        time_prediction = None
        time_distribution = None
        if self.mixture_components == 1:
            # A one-component mixture is exactly one log-normal: expose the accepted
            # single-LogNormal adapter so its analytic CRPS stays available.
            time_prediction = LogNormalPrediction(
                mu=parameters.mu[:, 0].detach().cpu(),
                sigma=parameters.sigma[:, 0].detach().cpu(),
            )
        else:
            time_distribution = MixtureResidualDistribution(
                LognormalMixtureParameters(
                    log_weights=parameters.log_weights.detach().cpu(),
                    mu=parameters.mu.detach().cpu(),
                    sigma=parameters.sigma.detach().cpu(),
                )
            )
        return BaselinePrediction(
            poi_logits=log_probabilities.cpu(),
            context=fused.cpu(),
            time_prediction=time_prediction,
            time_distribution=time_distribution,
        )

    def score_joint(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Per-row joint log score and decomposition on observed-known rows (scoring only).

        For this factorized arm the mark term conditional on the observed residual time
        equals the marginal (``p_s(y | r) == p_s(y)`` by construction); the shared
        decomposition contract still applies:
        ``log_joint == log_time + log_mark_given_time`` on eligible rows.
        """
        fused = self.fused_context(batch)
        device = fused.device
        # Batch fields cross the model boundary here: the caller's (typically CPU
        # collate) batch is never mutated, and every tensor entering density math or a
        # gather lives on the model's parameter device.
        eligible = (
            batch["observed_mask"].to(device).bool()
            & batch["target_in_train_catalog"].to(device).bool()
        )
        parameters = self.time_head(fused)
        residual = batch["residual_time_sec"].to(device=device, dtype=torch.float64)
        safe_residual = torch.where(eligible, residual, torch.ones_like(residual))
        log_time = parameters.log_density(safe_residual)
        log_probabilities = masked_log_probabilities(
            self.history_encoder.score_full_catalog(fused)
        ).to(torch.float64)
        target = batch["next_poi"].to(device).clamp_min(0)
        log_mark = log_probabilities.gather(1, target.view(-1, 1))[:, 0]
        return {
            "eligible": eligible,
            "log_joint": log_time + log_mark,
            "log_time": log_time,
            "log_mark_given_time": log_mark,
        }

    #: Score declaration for the merged #38 exact chunked scorer. The factory
    #: below normalizes its chunk scores over the frozen catalog, so what the
    #: evaluator (and checkpoint selection) receives are exact log-probabilities.
    exact_score_kind = "log_probabilities"

    def exact_candidate_factory(self, chunk_size: int = 4096):
        """Two-pass bounded normalized marginal callback for the #38 evaluator."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.require_fitted()

        def prepare(features):
            was_training = self.training
            self.eval()
            try:
                with torch.no_grad():
                    fused = self.fused_context(features)
                    normalizer = torch.full(
                        (len(fused),), -torch.inf, dtype=torch.float64, device=fused.device
                    )
                    for start in range(2, self.number_of_pois, chunk_size):
                        ids = torch.arange(
                            start,
                            min(start + chunk_size, self.number_of_pois),
                            device=fused.device,
                        ).expand(len(fused), -1)
                        scores = self.history_encoder.score_candidates(fused, ids).to(torch.float64)
                        normalizer = torch.logaddexp(normalizer, torch.logsumexp(scores, dim=1))
            finally:
                self.train(was_training)

            def score(ids):
                was_training = self.training
                self.eval()
                try:
                    with torch.no_grad():
                        candidates = ids.to(fused.device).expand(len(fused), -1)
                        values = self.history_encoder.score_candidates(fused, candidates)
                        return values.to(torch.float64) - normalizer[:, None]
                finally:
                    self.train(was_training)

            return score

        return prepare

    def fit(self, training_source: Any) -> StillIndependent:
        from adrec_still.models.baselines.trainer import TrainingPlan, train_model

        if not isinstance(training_source, TrainingPlan):
            raise TypeError("StillIndependent.fit expects a shared-trainer TrainingPlan")
        train_model(self, training_source)
        return self

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def head_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for module in (self.feature_fusion, self.time_head)
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
            "conditioning_features": "elapsed+query_clock",
            "encoder": "gru4rec_h(composed)",
            "encoder_parameter_count": self.history_encoder.parameter_count(),
            "head_parameter_count": self.head_parameter_count(),
            "parameter_count": self.parameter_count(),
            "context_dim": self.context_dim,
        }
