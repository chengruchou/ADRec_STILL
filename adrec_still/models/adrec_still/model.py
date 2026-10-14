"""ADRec-STILL: observable repeat/explore branches with silence conditioning (v1).

Base joint law per row (``b`` ranges over the two observable branches)::

    f(T, Y | H) = sum_b pi_b(H) f_b(T | H) q_b(Y | H, T)
    S(s | H)    = sum_b pi_b(H) S_b(s | H)
    kappa_b(s)  = pi_b(H) S_b(s | H) / S(s | H)

``pi_raw`` is the event-type prior over branches and never sees catalog support;
the catalog projection ``pi_cat`` masks a branch only when its known-item support
is empty (no seen known item -> no repeat branch in the *catalog* view) and is
what the marked joint/ranking math uses. ``q_repeat`` is supported on the known
items the user has seen anywhere in their full past; ``q_explore`` on the unseen
known items ??the supports are disjoint, which is what lets ``kappa_repeat(s)``
read as a repeat probability in the closed-catalog model.

One class carries the registered variants (same modules, flags differ):

============== ========= ============ ======== ==========
name           branches  time model   elapsed  item head
============== ========= ============ ======== ==========
adrec_repeat   yes       none         no       diffusion
adrec_elapsed  no        none         yes      diffusion
adrec_repeat_elapsed yes none         yes      diffusion
adrec_still    yes       lognormal    no       diffusion
still_direct   yes       lognormal    no       direct
============== ========= ============ ======== ==========

The diffusion backbone is the ported upstream ADRec core (clean causal encoder,
token-independent noising, x0 prediction, prefix-teacher-forced last-target
inference); conditioning enters through the core's full-scale context hook,
never through the 0.001 ``lambda_uncertainty`` factor. Training conditions the
item denoiser on the observed ``T`` (teacher forcing); inference integrates ``T``
out by stratified truncated sampling, and item probabilities are averaged in
probability space across samples (a log-probability score surface over the
frozen catalog). CE+MSE is the diffusion training surrogate and is never
reported as an exact reverse-mixture marginal NLL.
"""

from __future__ import annotations

import math
import zlib
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from adrec_still.data.history import FullHistoryIndex
from adrec_still.models.adrec_still.memory import (
    MemoryFeatures,
    memory_features,
    memory_labels,
)
from adrec_still.models.baselines.base import (
    BaselineModel,
    BaselinePrediction,
    validate_batch,
)
from adrec_still.models.baselines.trainer import build_sampled_candidates
from adrec_still.models.still.conditioning import clock_features
from adrec_still.models.still.heads import FeatureFusion, LognormalMixtureHead
from adrec_still.models.upstream.adrec import AdRec, AdRecConfig
from adrec_still.models.upstream.transformer import LayerNorm
from adrec_still.models.upstream.wrappers import chunked_cross_entropy
from adrec_still.time.distributions import (
    LognormalMixtureParameters,
    SurvivalConditionedResidualDistribution,
)

ADREC_STILL_FAMILY = "adrec_still"
BRANCH_REPEAT, BRANCH_EXPLORE = 0, 1
BRANCH_NAMES = ("repeat", "explore")
MASKED = -1.0e9
SECONDS_PER_HOUR = 3600.0

VARIANTS: dict[str, dict[str, Any]] = {
    "adrec_repeat": {"branches": True, "time_model": None, "elapsed": False,
                     "item_head": "diffusion"},
    "adrec_elapsed": {"branches": False, "time_model": None, "elapsed": True,
                      "item_head": "diffusion"},
    "adrec_repeat_elapsed": {"branches": True, "time_model": None, "elapsed": True,
                             "item_head": "diffusion"},
    "adrec_still": {"branches": True, "time_model": "lognormal", "elapsed": False,
                    "item_head": "diffusion"},
    "still_direct": {"branches": True, "time_model": "lognormal", "elapsed": False,
                     "item_head": "direct"},
}


def log1mexp(log_value: torch.Tensor) -> torch.Tensor:
    """Numerically stable ``log(1 - exp(x))`` for ``x <= 0``."""
    bound = math.log(0.5)
    return torch.where(
        log_value > bound,
        torch.log(-torch.expm1(log_value.clamp(max=-1e-12))),
        torch.log1p(-torch.exp(log_value)),
    )


def _time_features(total_sec: torch.Tensor, event_timestamp: torch.Tensor) -> torch.Tensor:
    """[B, 5]: log1p(T in hours) plus the arrival wall clock's day/week phases."""
    duration = torch.log1p(total_sec.to(torch.float64) / SECONDS_PER_HOUR)
    return torch.cat(
        [duration.to(torch.float32).unsqueeze(1), clock_features(event_timestamp)], dim=1
    )


class AdrecStillModel(BaselineModel, nn.Module):
    """One backbone, five registered variants (see the module table)."""

    family = ADREC_STILL_FAMILY

    def __init__(self, name: str, number_of_pois: int, *, hidden_size: int = 128,
                 dropout: float = 0.1, emb_dropout: float = 0.3,
                 mixture_components: int = 2, diffusion_steps: int = 32,
                 noise_schedule: str = "trunc_lin", schedule_sampler_name: str = "uniform",
                 lambda_uncertainty: float = 0.001, rescale_timesteps: bool = True,
                 cfg_scale: float = 1.0, is_causal: bool = True,
                 loss_scale: float = 1.0, ce_weight: float = 1.0,
                 time_unit_s: float = SECONDS_PER_HOUR,
                 cell_time_threshold_hours: float = 24.0,
                 time_samples: int = 2, diffusion_samples: int = 1,
                 inference_steps: int | None = None,
                 eval_noise_seed: int = 20251014, **_: Any) -> None:
        if name not in VARIANTS:
            raise ValueError(f"unknown adrec_still variant: {name!r}")
        nn.Module.__init__(self)
        BaselineModel.__init__(self, number_of_pois)
        flags = VARIANTS[name]
        self.name = name
        self.use_branches: bool = flags["branches"]
        self.time_model: str | None = flags["time_model"]
        self.elapsed_conditioning: bool = flags["elapsed"]
        self.item_head: str = flags["item_head"]
        self.hidden_size = int(hidden_size)
        self.dropout_rate = float(dropout)
        self.emb_dropout_rate = float(emb_dropout)
        self.mixture_components = int(mixture_components)
        self.loss_scale = float(loss_scale)
        self.ce_weight = float(ce_weight)
        self.time_unit_s = float(time_unit_s)
        self.cell_time_threshold_s = float(cell_time_threshold_hours) * 3600.0
        self.time_samples = int(time_samples)
        self.diffusion_samples = int(diffusion_samples)
        self.inference_steps = inference_steps
        self.eval_noise_seed = int(eval_noise_seed)
        #: event-only training capability (shared-trainer profile key)
        self.trains_event_only = name == "adrec_repeat"

        self.item_embedding = nn.Embedding(number_of_pois, self.hidden_size, padding_idx=0)
        self.embed_dropout = nn.Dropout(emb_dropout)
        self.hist_norm = LayerNorm(self.hidden_size, eps=1e-12)
        core_config = AdRecConfig(
            hidden_size=self.hidden_size, dropout=self.dropout_rate,
            diffusion_steps=int(diffusion_steps), noise_schedule=noise_schedule,
            schedule_sampler_name=schedule_sampler_name,
            lambda_uncertainty=float(lambda_uncertainty),
            rescale_timesteps=bool(rescale_timesteps), independent=True,
            cfg_scale=float(cfg_scale), geodesic=False, dif_decoder="att",
            is_causal=bool(is_causal),
        )
        self.core = AdRec(core_config)
        self.anchor_fusion = FeatureFusion(4, self.hidden_size)
        if self.use_branches:
            self.branch_prior_head = nn.Linear(self.hidden_size, 2)
            self.branch_embedding = nn.Embedding(2, self.hidden_size)
            # Learnable full-history memory mix-in for the repeat decoder.
            self.frequency_weight = nn.Parameter(torch.zeros(()))
            self.recency_weight = nn.Parameter(torch.zeros(()))
        if self.time_model == "lognormal":
            self.time_heads = nn.ModuleList(
                [LognormalMixtureHead(self.hidden_size, self.mixture_components)
                 for _ in range(2)]
            )
            self.time_context = nn.Sequential(
                nn.Linear(5, self.hidden_size), nn.Tanh(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
        if self.elapsed_conditioning:
            self.elapsed_context = nn.Sequential(
                nn.Linear(5, self.hidden_size), nn.Tanh(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
        if self.item_head == "direct":
            self.direct_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 2), nn.Tanh(),
                nn.Linear(self.hidden_size * 2, self.hidden_size),
            )
        self._memory: FullHistoryIndex | None = None
        self.home_policy_train = "causal"
        self.home_policy_eval = "frozen"

    # ---------------------------------------------------------------- plumbing

    @property
    def device(self) -> torch.device:
        return self.item_embedding.weight.device

    @property
    def context_dim(self) -> int:
        return self.hidden_size

    def attach_memory(self, index: FullHistoryIndex) -> None:
        self._memory = index

    def require_memory(self) -> FullHistoryIndex:
        if self._memory is None:
            raise RuntimeError(
                f"{self.name} needs the full event memory; call attach_memory() with a "
                "FullHistoryIndex built from this run's events.parquet"
            )
        return self._memory

    def fit(self, training_source: Any):
        from adrec_still.models.baselines.trainer import TrainingPlan, train_model

        if not isinstance(training_source, TrainingPlan):
            raise TypeError(f"{self.name}.fit expects a shared-trainer TrainingPlan")
        train_model(self, training_source)
        return self

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def resolved_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "interface_version": self.interface_version,
            "number_of_pois": self.number_of_pois,
            "embedding_dim": self.hidden_size,
            "hidden_dim": self.hidden_size,
            "dropout": self.dropout_rate,
            "emb_dropout": self.emb_dropout_rate,
            "mixture_components": self.mixture_components,
            "diffusion_steps": self.core.diffusion_steps,
            "branches": self.use_branches,
            "time_model": self.time_model,
            "elapsed_conditioning": self.elapsed_conditioning,
            "item_head": self.item_head,
            "loss_scale": self.loss_scale,
            "ce_weight": self.ce_weight,
            "cell_time_threshold_hours": self.cell_time_threshold_s / 3600.0,
            "time_samples": self.time_samples,
            "diffusion_samples": self.diffusion_samples,
            "inference_steps": self.inference_steps,
            "parameter_count": self.parameter_count(),
            "context_dim": self.hidden_size,
        }

    # -------------------------------------------------------------- base pieces

    def embed_history(self, history: torch.Tensor) -> torch.Tensor:
        return self.hist_norm(self.embed_dropout(self.item_embedding(history)))

    def encode_history(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """(encoded sequence [B, L, D], mask [B, L]) ??the clean causal encoder."""
        history = batch["history_poi"].to(self.device)
        mask = (history > 0).float()
        encoded = self.core.ag_encoder(self.embed_history(history), mask)
        return encoded, mask

    def base_context(self, batch: dict[str, Any],
                     encoded: torch.Tensor | None = None) -> torch.Tensor:
        """History state fused with anchor-clock features; never ``query_s_sec``."""
        if encoded is None:
            encoded, _ = self.encode_history(batch)
        context = encoded[:, -1, :]
        anchor = clock_features(batch["last_timestamp"].to("cpu")).to(self.device)
        return self.anchor_fusion(context, anchor)

    def branch_log_prior(self, context: torch.Tensor) -> torch.Tensor:
        """``log pi_raw`` [B, 2]; computed from H only, never from s."""
        if not self.use_branches:
            raise RuntimeError(f"{self.name} has no branch structure")
        return torch.log_softmax(self.branch_prior_head(context), dim=1)

    def branch_time_parameters(self, context: torch.Tensor) -> list[LognormalMixtureParameters]:
        if self.time_model != "lognormal":
            raise RuntimeError(f"{self.name} has no time model")
        return [head(context) for head in self.time_heads]

    def _item_context(self, batch_rows: dict[str, torch.Tensor],
                      branch: torch.Tensor | None,
                      total_sec: torch.Tensor | None) -> torch.Tensor | None:
        """Per-row denoiser context [B, D]; components are additive and optional."""
        pieces = []
        if self.use_branches and branch is not None:
            pieces.append(self.branch_embedding(branch.to(self.device)))
        if self.time_model == "lognormal" and total_sec is not None:
            arrival = batch_rows["last_timestamp"].to("cpu") + total_sec.to("cpu").long()
            features = _time_features(total_sec.to("cpu"), arrival).to(self.device)
            pieces.append(self.time_context(features))
        if self.elapsed_conditioning:
            elapsed = batch_rows["query_s_sec"].to("cpu")
            query_clock = batch_rows["last_timestamp"].to("cpu") + elapsed
            features = _time_features(elapsed, query_clock).to(self.device)
            pieces.append(self.elapsed_context(features))
        if not pieces:
            return None
        return torch.stack(pieces, dim=0).sum(dim=0)

    # ----------------------------------------------------------- item generation

    def _shifted_targets(self, history: torch.Tensor,
                         final_target: torch.Tensor | None) -> torch.Tensor:
        shifted = torch.cat([history[:, 1:], torch.zeros_like(history[:, :1])], dim=1)
        if final_target is not None:
            shifted = shifted.clone()
            shifted[:, -1] = final_target
        return shifted

    def _context_sequence(self, context_rows: torch.Tensor | None,
                          length: int) -> torch.Tensor | None:
        """Context applies to the generated (final) position only: each training
        prefix keeps its own metadata, so earlier token targets receive none."""
        if context_rows is None:
            return None
        sequence = context_rows.new_zeros((context_rows.shape[0], length,
                                           context_rows.shape[1]))
        sequence[:, -1, :] = context_rows
        return sequence

    def _generate_training(self, batch: dict[str, Any], branch: torch.Tensor | None,
                           total_sec: torch.Tensor | None):
        """Training pass: returns (final x0 prediction [B, D], diffusion MSE scalar)."""
        history = batch["history_poi"].to(self.device)
        targets = self._shifted_targets(history, batch["next_poi"].to(self.device))
        mask_seq = (history > 0).float()
        mask_tag = (targets > 0).float()
        context = self._context_sequence(
            self._item_context(batch, branch, total_sec), history.shape[1]
        )
        if self.item_head == "direct":
            pieces = [self.base_context(batch)]
            row_context = self._item_context(batch, branch, total_sec)
            if row_context is not None:
                pieces.append(row_context)
            generated = self.direct_head(torch.stack(pieces, dim=0).sum(dim=0))
            return generated, torch.zeros((), device=self.device)
        item_embeddings = self.embed_history(history)
        tag_embeddings = self.item_embedding(targets)
        denoised, _per_token, mse = self.core(
            item_embeddings, tag_embeddings, mask_seq, mask_tag, context=context
        )
        return denoised[:, -1, :], mse

    def _generate_inference(self, encoded: torch.Tensor, batch: dict[str, Any],
                            branch: torch.Tensor | None,
                            total_sec: torch.Tensor | None,
                            generator: torch.Generator) -> torch.Tensor:
        """One reverse-diffusion (or direct) generation for every row, [B, D]."""
        history = batch["history_poi"].to(self.device)
        if self.item_head == "direct":
            pieces = [self.base_context(batch, encoded=encoded)]
            row_context = self._item_context(batch, branch, total_sec)
            if row_context is not None:
                pieces.append(row_context)
            return self.direct_head(torch.stack(pieces, dim=0).sum(dim=0))
        targets = self._shifted_targets(history, None)
        mask_seq = (history > 0).float()
        mask_tag = (targets > 0).float()
        context = self._context_sequence(
            self._item_context(batch, branch, total_sec), history.shape[1]
        )
        out = self.core.denoise_sample(
            encoded, self.item_embedding(targets), mask_seq, mask_tag,
            generator=generator, inference_steps=self.inference_steps,
            context=context, encoded=True,
        )
        return out[:, -1, :]

    # --------------------------------------------------------------- decoders

    def _memory_bonus(self, ids: torch.Tensor, memory: MemoryFeatures,
                      reference_time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(bonus [B, C], seen-mask [B, C]) for explicit candidate IDs.

        Frequency counts come from H alone; recency is measured from the given
        reference time (the anchor, or the model event time ``t_n + T`` for the
        coherent variant ??never the query's ``s`` unless the variant conditions
        on it explicitly).
        """
        seen_poi = memory.seen_poi.to(self.device)
        positions = torch.searchsorted(seen_poi, ids.contiguous())
        positions = positions.clamp(max=max(seen_poi.shape[1] - 1, 0))
        matched = seen_poi.gather(1, positions).eq(ids) & memory.seen_mask.to(
            self.device
        ).gather(1, positions)
        counts = memory.seen_count.to(self.device).gather(1, positions)
        last_visit = memory.seen_last_s.to(self.device).gather(1, positions)
        delta_hours = (
            (reference_time.unsqueeze(1).to(self.device) - last_visit)
            .clamp_min(0).to(torch.float32) / SECONDS_PER_HOUR
        )
        bonus = (
            self.frequency_weight * torch.log1p(counts)
            - self.recency_weight * torch.log1p(delta_hours)
        )
        return torch.where(matched, bonus, torch.zeros_like(bonus)), matched

    def branch_candidate_scores(self, generated: torch.Tensor, ids: torch.Tensor,
                                memory: MemoryFeatures | None,
                                branch: int | None,
                                reference_time: torch.Tensor) -> torch.Tensor:
        """Unnormalized branch-restricted scores for explicit IDs, [B, C]."""
        base = torch.bmm(self.item_embedding(ids), generated.unsqueeze(2)).squeeze(2)
        if not self.use_branches or branch is None:
            return base
        assert memory is not None
        bonus, seen = self._memory_bonus(ids, memory, reference_time)
        if branch == BRANCH_REPEAT:
            return torch.where(seen, base + bonus, torch.full_like(base, MASKED))
        return torch.where(seen, torch.full_like(base, MASKED), base)

    def catalog_support_log_mask(self, memory: MemoryFeatures) -> torch.Tensor:
        """``log m_b`` [B, 2]: 0 where the branch's known-item support is non-empty.

        Affects only the catalog projection ``pi_cat``; ``pi_raw`` and the
        all-event time mixture never read it.
        """
        seen_sizes = memory.seen_sizes.to(self.device)
        catalog = self.number_of_pois - 2
        repeat_ok = seen_sizes > 0
        explore_ok = seen_sizes < catalog
        mask = torch.full((seen_sizes.shape[0], 2), MASKED, dtype=torch.float64,
                          device=self.device)
        mask[:, BRANCH_REPEAT] = torch.where(repeat_ok, 0.0, MASKED)
        mask[:, BRANCH_EXPLORE] = torch.where(explore_ok, 0.0, MASKED)
        if (~(repeat_ok | explore_ok)).any():
            raise ValueError(
                "a row has no predictable known item in either branch: the training "
                "catalog is empty for this data"
            )
        return mask

    # ------------------------------------------------------------ training loss

    def _branch_labels(self, batch: dict[str, Any]) -> torch.Tensor:
        """Branch indices per row: BRANCH_REPEAT/BRANCH_EXPLORE, -1 unresolvable."""
        labels = memory_labels(self.require_memory(), batch).repeat_label.to(self.device)
        return torch.where(
            labels < 0,
            torch.full_like(labels, -1),
            torch.where(labels == 1,
                        torch.full_like(labels, BRANCH_REPEAT),
                        torch.full_like(labels, BRANCH_EXPLORE)),
        )

    def _time_losses(self, context, branch_labels, batch) -> tuple[torch.Tensor, dict]:
        """Observed/censored time-and-type NLL in the raw (all-event) view."""
        log_prior = self.branch_log_prior(context).to(torch.float64)
        parameters = self.branch_time_parameters(context)
        observed = batch["observed_mask"].to(self.device).bool()
        residual = batch["residual_time_sec"].to(self.device, torch.float64)
        censor = batch["censor_time_sec"].to(self.device, torch.float64)
        total = torch.where(observed, residual, torch.ones_like(residual))
        exposure = torch.where(~observed, censor, torch.zeros_like(censor))
        density = torch.stack(
            [params.log_density(total) for params in parameters], dim=1
        )
        survival = torch.stack(
            [params.log_survival(exposure) for params in parameters], dim=1
        )
        labeled = observed & (branch_labels >= 0)
        safe_branch = branch_labels.clamp_min(0).long()
        type_term = -(
            log_prior.gather(1, safe_branch.unsqueeze(1)).squeeze(1)
            + density.gather(1, safe_branch.unsqueeze(1)).squeeze(1)
        )
        marginal_term = -torch.logsumexp(log_prior + density, dim=1)
        censor_term = -torch.logsumexp(log_prior + survival, dim=1)
        loss = torch.where(
            observed, torch.where(labeled, type_term, marginal_term), censor_term
        )
        components = {
            "observed_labeled": int(labeled.sum()),
            "observed_marginal": int((observed & ~labeled).sum()),
            "censored": int((~observed).sum()),
        }
        return loss, components

    def _item_losses(self, batch, branch_labels, memory, negative_samples, generator):
        """CE surrogate + diffusion MSE on observed-known rows (zeros elsewhere)."""
        from adrec_still.data.collate import select_batch_rows

        observed_known = (
            batch["observed_mask"].to(self.device).bool()
            & batch["target_in_train_catalog"].to(self.device).bool()
        )
        batch_size = int(batch["next_poi"].shape[0])
        ce = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        mse = torch.zeros((), device=self.device)
        if not bool(observed_known.any()):
            return ce, mse, {"observed_known": 0}
        keep = observed_known.cpu()
        sub = select_batch_rows(batch, keep)
        sub_labels = branch_labels[observed_known]
        sub_memory = _subset(memory, keep) if memory is not None else None
        teacher_time = (
            sub["query_s_sec"].to(self.device, torch.float64)
            + sub["residual_time_sec"].to(self.device, torch.float64)
        ) if self.time_model == "lognormal" else None
        branch = sub_labels.clamp_min(0).long() if self.use_branches else None
        generated, mse = self._generate_training(sub, branch, teacher_time)
        targets = sub["next_poi"].to(self.device)
        anchor = sub["last_timestamp"].to(self.device)
        reference_time = (
            anchor.to(torch.float64) + teacher_time if teacher_time is not None
            else anchor.to(torch.float64)
        )
        sub_ce = torch.zeros(len(targets), dtype=torch.float32, device=self.device)
        if self.use_branches:
            for branch_id in (BRANCH_REPEAT, BRANCH_EXPLORE):
                members = torch.nonzero(sub_labels == branch_id).squeeze(1)
                if not len(members):
                    continue
                if branch_id == BRANCH_REPEAT:
                    ids = sub_memory.seen_poi.to(self.device)[members]
                    scores = self.branch_candidate_scores(
                        generated[members], ids, _subset(sub_memory, members.cpu()),
                        BRANCH_REPEAT, reference_time[members],
                    )
                    target_position = ids.eq(targets[members].unsqueeze(1))
                    if not bool(target_position.any(dim=1).all()):
                        raise ValueError(
                            "a repeat-labeled known target is missing from its own "
                            "full-history seen set; the memory index and the landmark "
                            "rows disagree"
                        )
                    log_probabilities = torch.log_softmax(scores, dim=1)
                    sub_ce[members] = -(
                        log_probabilities[target_position.nonzero(as_tuple=True)]
                    )
                else:
                    candidates, penalty = build_sampled_candidates(
                        targets[members], negative_samples, self.number_of_pois,
                        generator,
                    )
                    scores = self.branch_candidate_scores(
                        generated[members], candidates.to(self.device),
                        _subset(sub_memory, members.cpu()), BRANCH_EXPLORE,
                        reference_time[members],
                    )
                    scores = scores - penalty.to(self.device)
                    sub_ce[members] = F.cross_entropy(
                        scores,
                        torch.zeros(len(members), dtype=torch.long, device=self.device),
                        reduction="none",
                    )
        else:
            candidates, penalty = build_sampled_candidates(
                targets, negative_samples, self.number_of_pois, generator
            )
            scores = self.branch_candidate_scores(
                generated, candidates.to(self.device), None, None, reference_time,
            )
            scores = scores - penalty.to(self.device)
            sub_ce = F.cross_entropy(
                scores, torch.zeros(len(targets), dtype=torch.long, device=self.device),
                reduction="none",
            )
        ce[observed_known] = sub_ce
        return ce, mse, {"observed_known": int(observed_known.sum())}

    def per_row_survival_training_losses(self, batch, negative_samples, generator):
        """Gap-censor profile: time/type + censor NLL, plus CE+MSE on known targets."""
        if self.time_model != "lognormal":
            raise RuntimeError(f"{self.name} does not train under a survival profile")
        memory = memory_features(self.require_memory(), batch,
                                 home_policy=self.home_policy_train)
        branch_labels = self._branch_labels(batch)
        context = self.base_context(batch)
        time_loss, time_counts = self._time_losses(context, branch_labels, batch)
        ce, mse, item_counts = self._item_losses(
            batch, branch_labels, memory, negative_samples, generator
        )
        per_row = (
            time_loss.to(torch.float32)
            + self.ce_weight * ce
            + self.loss_scale * mse  # scalar broadcast: logged per batch
        )
        self._last_loss_components = {
            "time_type_nll": float(time_loss.detach().mean()),
            "item_ce": float(ce.detach().sum() / max(item_counts["observed_known"], 1)),
            "diffusion_mse": float(mse.detach()),
            **time_counts, **item_counts,
        }
        return per_row

    def per_row_sampled_training_losses(self, kept, negative_samples, generator):
        """Observed-known profiles (event-only and landmark-grid variants)."""
        memory = memory_features(self.require_memory(), kept,
                                 home_policy=self.home_policy_train) \
            if self.use_branches else None
        branch_labels = (
            self._branch_labels(kept) if self.use_branches
            else torch.zeros(int(kept["next_poi"].shape[0]), dtype=torch.int8,
                             device=self.device)
        )
        if self.use_branches and bool((branch_labels < 0).any()):
            raise ValueError("an observed-known row has no resolvable repeat label")
        ce, mse, item_counts = self._item_losses(
            kept, branch_labels, memory, negative_samples, generator
        )
        per_row = self.ce_weight * ce + self.loss_scale * mse
        if self.use_branches:
            context = self.base_context(kept)
            log_prior = self.branch_log_prior(context)
            type_nll = F.nll_loss(log_prior, branch_labels.long(), reduction="none")
            per_row = per_row + type_nll
            self._last_loss_components = {
                "branch_type_nll": float(type_nll.detach().mean()),
                "item_ce": float(ce.detach().mean()),
                "diffusion_mse": float(mse.detach()),
                **item_counts,
            }
        else:
            self._last_loss_components = {
                "item_ce": float(ce.detach().mean()), "diffusion_mse": float(mse.detach()), **item_counts,
            }
        return per_row

    def loss_components(self) -> dict[str, Any]:
        return dict(getattr(self, "_last_loss_components", {}))

    # ---------------------------------------------------------------- forecast

    def _forecast_generator(self, batch: dict[str, Any]) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        fingerprint = zlib.crc32(batch["history_poi"].detach().cpu().numpy().tobytes())
        generator.manual_seed(
            (self.eval_noise_seed * 1_000_003 + fingerprint) % (2**63 - 1)
        )
        return generator

    def prepare_forecast(self, batch: dict[str, Any], *, chunk_size: int = 4096):
        from adrec_still.models.adrec_still.forecast import SilenceForecast

        return SilenceForecast(self, batch, chunk_size=chunk_size)

    def exact_candidate_factory(self, chunk_size: int = 4096):
        self.require_fitted()

        def prepare(features):
            forecast = self.prepare_forecast(features, chunk_size=chunk_size)
            return forecast.score

        return prepare

    exact_score_kind = "log_probabilities"

    def predict(self, batch: dict[str, Any]) -> BaselinePrediction:
        self.require_fitted()
        validate_batch(batch)
        forecast = self.prepare_forecast(batch, chunk_size=self.number_of_pois)
        return forecast.prediction()


def _subset(memory: MemoryFeatures, rows: torch.Tensor) -> MemoryFeatures:
    return MemoryFeatures(
        seen_poi=memory.seen_poi[rows], seen_count=memory.seen_count[rows],
        seen_last_s=memory.seen_last_s[rows], seen_mask=memory.seen_mask[rows],
        seen_sizes=memory.seen_sizes[rows], home_lat=memory.home_lat[rows],
        home_lon=memory.home_lon[rows], home_available=memory.home_available[rows],
    )
