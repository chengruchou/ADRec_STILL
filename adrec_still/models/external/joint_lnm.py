"""External dependent time–mark LNM baseline (Issue #51).

Port of the conditionally *dependent* intensity-free model of Waghmare, Debnath,
Asthana and Malhotra, "Modeling Inter-Dependence Between Time and Mark in
Multivariate Temporal Point Processes" (CIKM 2022, arXiv:2210.15294), official code
https://github.com/waghmaregovind/joint_tpp @ a32bca006ba87beb6f71453b9660f81a845eadda
(MIT, Copyright (c) 2021 Oleksandr Shchur). Source-by-source fidelity notes and every
deviation live in ``UPSTREAM.md`` next to this module; the highlights:

- **Official implementation** reproduced: paper-shaped GRU over per-event features
  ``[(log(dt + 1e-8) - mean) / std, mark embedding]``; a learnable ``context_init``
  as the context when no history event precedes the prediction; a literal per-mark
  head (one linear producing ``V * 3C`` values reshaped to locations / raw
  log-scales / component-weight logits per legal mark); unrestricted log-locations;
  straight-through-clamped log-scales in ``[-5, 3]``; per-mark component
  ``log_softmax``; the log-time affine transform parameterized by the training
  statistics.
- **STILL adaptations** (named, never called faithful reproduction): the first
  valid history event's inter-event time is defined as 0 seconds
  (``first-interval-zero-v1``, history-only — the canonical history gives no
  predecessor and no future value may be read); prediction after waiting ``s`` uses
  the survival-conditioned mark posterior ``p_y S_y(s) / S(s)`` because the
  official mark inference conditions on the *observed* next inter-event time,
  which STILL's protocol forbids; censoring uses the marginal remaining-exposure
  survival ``log S(s+c) - log S(s)`` — the official end-of-sequence term selects
  the last observed mark's component and is recorded as a source difference only.

The base joint ``f(T, y | H) = p_y g_y(T)`` reads only legal history; the query
elapsed ``s`` never enters the parameter heads. Ranking/predict never read
``residual_time_sec``, the future target or the censor label; only the loss and
``score_joint`` consume evaluation targets.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from adrec_still.data.schema import PADDING_POI_ID, UNKNOWN_POI_ID
from adrec_still.models.baselines.base import (
    MASKED_LOGIT,
    BaselineModel,
    BaselinePrediction,
    validate_batch,
)
from adrec_still.models.external.lnm_distributions import (
    MarginalResidualDistribution,
    PerMarkJointDistribution,
)
from adrec_still.models.external.statistics import IDENTITY_STATISTICS, LogTimeStatistics

#: The shared schema's reserved (non-catalog) mark ids, excluded from every
#: prediction/normalization denominator; UNK may still appear in history input.
RESERVED_MARK_IDS = (PADDING_POI_ID, UNKNOWN_POI_ID)
FIRST_LEGAL_MARK_ID = len(RESERVED_MARK_IDS)

#: STILL adaptation id for how canonical histories become inter-event features.
HISTORY_TIME_POLICY = "first-interval-zero-v1"

#: Official input-feature epsilon (recurrent_tpp.py: log(inter_times + 1e-8)).
INTER_TIME_EPSILON = 1.0e-8

class ExternalJointLNM(BaselineModel, nn.Module):
    """Dependent time–mark LNM: ``p(y|H)`` times a per-mark log-normal mixture."""

    name = "external_joint_lnm"
    family = "external"

    def __init__(
        self,
        number_of_pois: int,
        mark_embedding_dim: int = 16,
        context_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        mixture_components: int = 2,
    ) -> None:
        nn.Module.__init__(self)
        BaselineModel.__init__(self, number_of_pois)
        if number_of_pois <= FIRST_LEGAL_MARK_ID:
            raise ValueError("the catalog needs at least one non-reserved mark")
        if mark_embedding_dim <= 0 or context_dim <= 0 or num_layers <= 0:
            raise ValueError("mark_embedding_dim, context_dim, num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if mixture_components <= 0:
            raise ValueError("mixture_components must be positive")
        self.mark_embedding_dim = int(mark_embedding_dim)
        self.context_dim_value = int(context_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.mixture_components = int(mixture_components)

        self.mark_embedding = nn.Embedding(
            self.number_of_pois, self.mark_embedding_dim, padding_idx=PADDING_POI_ID
        )
        self.encoder = nn.GRU(
            1 + self.mark_embedding_dim,
            self.context_dim_value,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        # Official semantics: the context in force before any event has occurred.
        self.context_init = nn.Parameter(torch.zeros(self.context_dim_value))
        self.mark_head = nn.Linear(self.context_dim_value, self.number_of_pois)
        # Literal per-mark head: one linear producing V * 3C raw values
        # (locations, raw log-scales, component-weight logits per legal mark).
        self.time_head = nn.Linear(
            self.context_dim_value, self.catalog_size * 3 * self.mixture_components
        )
        # Affine statistics a/b with log T_seconds = a + b * Z; identity until the
        # runner fits/declares them. Buffers so every checkpoint restores them.
        self.register_buffer(
            "log_time_mean", torch.tensor(0.0, dtype=torch.float64), persistent=True
        )
        self.register_buffer(
            "log_time_std", torch.tensor(1.0, dtype=torch.float64), persistent=True
        )
        self.log_time_statistics_record: dict[str, Any] = IDENTITY_STATISTICS.as_record()

    # ------------------------------------------------------------------ properties

    @property
    def catalog_size(self) -> int:
        """``V``: legal (non-reserved) marks in the frozen training vocabulary."""
        return self.number_of_pois - FIRST_LEGAL_MARK_ID

    @property
    def context_dim(self) -> int:
        return self.context_dim_value

    @property
    def device(self) -> torch.device:
        return self.mark_embedding.weight.device

    # ------------------------------------------------------------------ statistics

    def set_log_time_statistics(self, statistics: LogTimeStatistics) -> None:
        """Install the affine statistics (training-only fit or declared identity)."""
        if statistics.std <= 0:
            raise ValueError("log_time_std must be strictly positive")
        with torch.no_grad():
            self.log_time_mean.fill_(float(statistics.mean))
            self.log_time_std.fill_(float(statistics.std))
        self.log_time_statistics_record = statistics.as_record()

    # --------------------------------------------------------------------- encoder

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Paper-shaped history encoding from mark ids and inter-event times only.

        Consumes ``history_poi``/``history_mask``/``history_timestamp``; neither
        ``query_s_sec`` nor any future field is read, so the context is identical
        for every landmark of one gap by construction. Left-padded rows are
        re-packed right-aligned exactly like the shared GRU4Rec-H encoder; a row
        with no valid event gets the learnable ``context_init``.
        """
        history = batch["history_poi"].to(self.device)
        mask = batch["history_mask"].to(self.device)
        timestamps = batch["history_timestamp"].to(self.device)
        batch_size, sequence_length = history.shape
        if sequence_length == 0:
            return self.context_init.unsqueeze(0).expand(batch_size, -1)
        lengths = mask.sum(dim=1)
        offsets = (sequence_length - lengths).unsqueeze(1)
        positions = torch.arange(sequence_length, device=self.device).unsqueeze(0)
        gather_index = (offsets + positions).clamp(max=max(sequence_length - 1, 0))
        right_marks = history.gather(1, gather_index)
        right_times = timestamps.gather(1, gather_index).to(torch.float64)
        # Inter-event times: diff along the right-aligned valid prefix. The first
        # valid event has no predecessor in the canonical history, so its
        # inter-event time is defined as 0 (HISTORY_TIME_POLICY; history-only).
        # Canonical histories are time-ordered; clamp guards exact ties.
        deltas = torch.diff(right_times, dim=1, prepend=right_times[:, :1]).clamp_min(0.0)
        feature = (
            (deltas + INTER_TIME_EPSILON).log() - self.log_time_mean
        ) / self.log_time_std
        inputs = torch.cat(
            [feature.to(torch.float32).unsqueeze(2), self.mark_embedding(right_marks)],
            dim=2,
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            inputs,
            lengths.clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, final_state = self.encoder(packed)
        context = final_state[-1]
        empty = lengths == 0
        return torch.where(
            empty.unsqueeze(1),
            self.context_init.unsqueeze(0).expand(batch_size, -1),
            context,
        )

    # ------------------------------------------------------------------------ base

    def base_distribution(self, context: torch.Tensor) -> PerMarkJointDistribution:
        """The fixed base joint for each row; the query ``s`` never enters here."""
        batch_size = context.shape[0]
        mark_logits = self.mark_head(context)
        log_mark = torch.log_softmax(
            mark_logits[:, FIRST_LEGAL_MARK_ID:].to(torch.float64), dim=1
        )
        raw = self.time_head(context).reshape(
            batch_size, self.catalog_size, 3 * self.mixture_components
        )
        components = self.mixture_components
        return PerMarkJointDistribution(
            log_mark_probabilities=log_mark,
            locs=raw[..., :components],
            log_scales_raw=raw[..., components : 2 * components],
            weight_logits=raw[..., 2 * components :],
            log_time_mean=self.log_time_mean,
            log_time_std=self.log_time_std,
        )

    def _full_width_log_probabilities(self, legal: torch.Tensor) -> torch.Tensor:
        """Embed normalized legal-catalog log-probs into the shared [B, num_pois]
        layout with reserved ids at the standard finite floor."""
        full = torch.full(
            (legal.shape[0], self.number_of_pois),
            MASKED_LOGIT,
            dtype=torch.float64,
            device=legal.device,
        )
        full[:, FIRST_LEGAL_MARK_ID:] = legal
        return full

    # -------------------------------------------------------------------- training

    def per_row_sampled_training_losses(
        self,
        kept: dict[str, torch.Tensor],
        negative_samples: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Exact marginal time NLL plus the sampled conditional-mark NLL.

        The time term ``-(log sum_y p_y g_y(s+r) - log S(s))`` sums the whole legal
        catalog exactly. The mark term is the shared sampled-softmax approximation
        of ``log p(y | T = s+r, H)``: candidate logits are ``log p_y + log g_y(s+r)``
        (exact-time conditional evidence, **not** the waiting-marginal ranking
        logits), through the shared candidate builder and accidental-hit policy.
        The sum approximates the joint NLL; it is never reported as the exact one.
        """
        from adrec_still.models.baselines.trainer import build_sampled_candidates

        observed = kept["observed_mask"].bool() & kept["target_in_train_catalog"].bool()
        if not bool(observed.all()):
            raise ValueError(
                "the shared trainer must deliver the observed-known population "
                "(observed-known-v0-proposed); censored/unknown training needs the "
                "#39 profile"
            )
        residual = kept["residual_time_sec"].to(torch.float64)
        if (residual <= 0).any():
            raise ValueError("observed rows require positive residual_time_sec")
        context = self.encode(kept)
        base = self.base_distribution(context)
        elapsed = kept["query_s_sec"].to(device=context.device, dtype=torch.float64)
        total = elapsed + residual.to(context.device)
        log_survival_at_query = base.log_survival(elapsed)
        time_nll = -(base.log_marginal_density(total) - log_survival_at_query)

        candidates, accidental_hit_penalty = build_sampled_candidates(
            kept["next_poi"], negative_samples, self.number_of_pois, generator
        )
        weighted = base.log_mark_probabilities + base.per_mark_log_density(total)
        candidate_logits = weighted.gather(
            1, candidates - FIRST_LEGAL_MARK_ID
        ) - accidental_hit_penalty.to(torch.float64)
        mark_nll = -torch.log_softmax(candidate_logits, dim=1)[:, 0]
        return time_nll + mark_nll

    def per_row_survival_training_losses(
        self,
        kept: dict[str, torch.Tensor],
        negative_samples: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """The #39 three-way survival loss for this port; one term per row.

        Population and partition come from the shared
        :func:`adrec_still.models.still.losses.survival_training_masks`; the shared trainer
        applies ``still-survival-gap-weighting-v1`` to the returned per-row values, so
        nothing is weighted here. All three terms condition on having waited
        ``query_s_sec`` (native seconds), exactly as the observed-known path does:

        1. **observed known** -- exact marginal residual-time NLL
           ``-(log sum_y p_y g_y(s+r) - log S(s))`` plus the shared *sampled*
           conditional-mark NLL over ``log p_y + log g_y(s+r)``. The mark term is a
           training surrogate for ``log p(y | T = s+r, H)``, never reported as an
           exact full-catalog likelihood (the exact one is ``score_joint``).
        2. **observed unknown mark** -- the marginal time term only. UNKNOWN is a
           reserved id, not a learnable mark: it never enters a candidate set or a
           mark target.
        3. **positive-followup censor** -- ``-[log S(s + c) - log S(s)]`` with ``S``
           the marginal survival over the *unknown next* mark. The upstream
           end-of-sequence term conditions on the last observed mark; that is not
           this population's likelihood and is not used here.

        Select-first throughout: a row's inactive branch is evaluated at a safe
        placeholder time and discarded by the mask, so no ``-inf``/NaN reaches a
        shared reduction. Nothing is clamped away with ``nan_to_num``.
        """
        from adrec_still.models.baselines.trainer import build_sampled_candidates
        from adrec_still.models.still.losses import survival_training_masks

        masks = survival_training_masks(kept)
        context = self.encode(kept)
        device = context.device
        base = self.base_distribution(context)
        elapsed = kept["query_s_sec"].to(device=device, dtype=torch.float64)
        observed = kept["observed_mask"].to(device).bool()
        residual = kept["residual_time_sec"].to(device=device, dtype=torch.float64)
        censor = kept["censor_time_sec"].to(device=device, dtype=torch.float64)
        log_survival_at_query = base.log_survival(elapsed)

        # Placeholders keep the inactive branch finite; the masks below select.
        safe_total = elapsed + torch.where(observed, residual, torch.ones_like(residual))
        safe_exposure = elapsed + torch.where(~observed, censor, torch.ones_like(censor))
        # One evaluation of the per-mark density at these times, reused by the mark
        # term below: it is the largest tensor in the step at catalog scale.
        marginal_total, per_mark_total = base.marginal_and_per_mark_log_density(
            safe_total
        )
        time_nll = -(marginal_total - log_survival_at_query)
        censor_nll = -(base.log_survival(safe_exposure) - log_survival_at_query)
        losses = torch.where(observed, time_nll, censor_nll)

        known = masks["observed_known"].to(device)
        if bool(known.any()):
            targets = kept["next_poi"].to(device)[known]
            candidates, accidental_hit_penalty = build_sampled_candidates(
                targets, negative_samples, self.number_of_pois, generator
            )
            weighted = base.log_mark_probabilities[known] + per_mark_total[known]
            candidate_logits = weighted.gather(
                1, candidates.to(device) - FIRST_LEGAL_MARK_ID
            ) - accidental_hit_penalty.to(device=device, dtype=torch.float64)
            mark_nll = -torch.log_softmax(candidate_logits, dim=1)[:, 0]
            losses = losses.clone()
            losses[known] = losses[known] + mark_nll.to(losses.dtype)
        return losses

    # ------------------------------------------------------------------- inference

    def predict(self, batch: dict[str, torch.Tensor]) -> BaselinePrediction:
        """Waiting-marginal ranking plus the marginal residual-time distribution.

        Reads history fields and ``query_s_sec`` only: the survivor-conditioned
        ``p_s(y) = p_y S_y(s) / S(s)`` integrates the unknown future residual out.
        At ``s == 0`` it equals ``p_y`` exactly.
        """
        self.require_fitted()
        validate_batch(batch)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                context = self.encode(batch)
                base = self.base_distribution(context)
                elapsed = batch["query_s_sec"].to(
                    device=context.device, dtype=torch.float64
                )
                ranking = base.waiting_marginal_log_mark_probabilities(elapsed)
                full = self._full_width_log_probabilities(ranking)
                base_cpu = PerMarkJointDistribution(
                    log_mark_probabilities=base.log_mark_probabilities.detach().cpu(),
                    locs=base.locs.detach().cpu(),
                    log_scales_raw=base.log_scales_raw.detach().cpu(),
                    weight_logits=base.weight_logits.detach().cpu(),
                    log_time_mean=self.log_time_mean.detach().cpu(),
                    log_time_std=self.log_time_std.detach().cpu(),
                )
                elapsed_cpu = elapsed.detach().cpu()
        finally:
            self.train(was_training)
        return BaselinePrediction(
            poi_logits=full.to(torch.float32).cpu(),
            context=context.detach().cpu(),
            time_distribution=MarginalResidualDistribution(base_cpu, elapsed_cpu),
        )

    def score_joint(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Per-row joint log score and decomposition on observed exact-known rows.

        Scoring only — the observed residual/target enter here as evaluation
        targets. ``log_joint`` and ``log_time`` come from their own reductions and
        ``log_mark_given_time`` from the separately normalized exact-time
        conditional, so the exported decomposition identity is a genuine
        cross-check of independent computation paths, not a subtraction identity.
        """
        context = self.encode(batch)
        device = context.device
        eligible = (
            batch["observed_mask"].to(device).bool()
            & batch["target_in_train_catalog"].to(device).bool()
        )
        base = self.base_distribution(context)
        elapsed = batch["query_s_sec"].to(device=device, dtype=torch.float64)
        residual = batch["residual_time_sec"].to(device=device, dtype=torch.float64)
        # Select-first: padded residuals (0 on censored rows) never reach log().
        safe_residual = torch.where(eligible, residual, torch.ones_like(residual))
        total = elapsed + safe_residual
        target_index = (
            batch["next_poi"].to(device).clamp_min(FIRST_LEGAL_MARK_ID)
            - FIRST_LEGAL_MARK_ID
        ).view(-1, 1)

        log_survival_at_query = base.log_survival(elapsed)
        weighted = base.log_mark_probabilities + base.per_mark_log_density(total)
        log_joint = weighted.gather(1, target_index)[:, 0] - log_survival_at_query
        log_time = base.log_marginal_density(total) - log_survival_at_query
        log_mark_given_time = base.conditional_log_mark_probabilities(total).gather(
            1, target_index
        )[:, 0]
        return {
            "eligible": eligible,
            "log_joint": log_joint,
            "log_time": log_time,
            "log_mark_given_time": log_mark_given_time,
        }

    #: Score declaration for the merged #38 exact chunked scorer: the waiting
    #: marginal ``log p_s(y)`` is normalized over the legal catalog by construction
    #: (its ``S(s)`` denominator sums every mark), so chunk scores are exact
    #: log-probabilities sliced from that one computation.
    exact_score_kind = "log_probabilities"

    def exact_candidate_factory(self, chunk_size: int = 4096):
        """``scorer_factory`` for the #38 evaluator (``exact_model_candidate_factory``).

        Computes the fixed base and the survivor-conditioned ranking marginal once
        per batch from the forecast inputs only (history fields and ``query_s_sec``;
        the official mark inference's observed inter-event time is never read) and
        returns ``score(ids)`` over explicit CPU int64 candidate IDs. The marginal is
        one already-normalized ``[rows, marks]`` block, so ``chunk_size`` only slices
        it -- the two internal arms' two-pass normalization has nothing to add here.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.require_fitted()

        def prepare(features: dict[str, torch.Tensor]):
            was_training = self.training
            self.eval()
            try:
                with torch.no_grad():
                    context = self.encode(features)
                    base = self.base_distribution(context)
                    elapsed = features["query_s_sec"].to(
                        device=context.device, dtype=torch.float64
                    )
                    ranking = base.waiting_marginal_log_mark_probabilities(elapsed)
            finally:
                self.train(was_training)

            def score(ids: torch.Tensor) -> torch.Tensor:
                candidates = ids.to(ranking.device)
                legal = candidates.clamp_min(FIRST_LEGAL_MARK_ID) - FIRST_LEGAL_MARK_ID
                scores = ranking.index_select(1, legal)
                reserved = (candidates < FIRST_LEGAL_MARK_ID).view(1, -1)
                return torch.where(reserved, torch.full_like(scores, MASKED_LOGIT), scores)

            return score

        return prepare

    # ------------------------------------------------------------------- interface

    def fit(self, training_source: Any) -> ExternalJointLNM:
        from adrec_still.models.baselines.trainer import TrainingPlan, train_model

        if not isinstance(training_source, TrainingPlan):
            raise TypeError("ExternalJointLNM.fit expects a shared-trainer TrainingPlan")
        train_model(self, training_source)
        return self

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def time_head_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.time_head.parameters())

    def mark_head_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.mark_head.parameters())

    def encoder_parameter_count(self) -> int:
        return (
            sum(parameter.numel() for parameter in self.encoder.parameters())
            + sum(parameter.numel() for parameter in self.mark_embedding.parameters())
            + self.context_init.numel()
        )

    def resolved_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "interface_version": self.interface_version,
            "number_of_pois": self.number_of_pois,
            "catalog_size": self.catalog_size,
            "mark_embedding_dim": self.mark_embedding_dim,
            "context_dim": self.context_dim_value,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "mixture_components": self.mixture_components,
            "history_time_policy": HISTORY_TIME_POLICY,
            "log_time_statistics": dict(self.log_time_statistics_record),
            "encoder": "paper-shaped-gru(inter-time+mark-embedding)",
            "encoder_parameter_count": self.encoder_parameter_count(),
            "mark_head_parameter_count": self.mark_head_parameter_count(),
            "time_head_parameter_count": self.time_head_parameter_count(),
            "parameter_count": self.parameter_count(),
        }
