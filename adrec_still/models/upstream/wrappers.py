"""Shared-interface wrappers for the upstream fork models.

Each wrapper owns the item-embedding table (the fork's ``Att_Diffuse_model``
level), feeds the ported cores, and exposes the package's encoder surface
(``encode`` / ``score_candidates`` / ``predict``) so the exact chunked evaluator
scores these models like any other backend. They are event-only: no silence,
time, or cell semantics — ``query_s_sec`` and timestamps are never read.

Evaluation noise policy: the diffusion cores sample reverse-process noise. Each
``encode`` call derives its generator seed from the wrapper's ``eval_noise_seed``
and a content hash of the batch's history, so a fixed seed plus an identical
history yields an identical forecast (batch composition elsewhere cannot change
a row's randomness source), and candidate chunks never redraw noise — scoring
happens after the single denoising pass.

Training cross-entropy is the fork's full-softmax CE over valid positions; the
``ce_chunk_tokens`` option computes it in gradient-checkpointed token chunks
(identical loss and gradients, bounded activation memory). ``None`` keeps the
original dense behavior.
"""

from __future__ import annotations

import zlib
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from adrec_still.models.baselines.base import (
    BaselineModel,
    BaselinePrediction,
    masked_log_probabilities,
    validate_batch,
)
from adrec_still.models.upstream.adrec import AdRec, AdRecConfig
from adrec_still.models.upstream.diffurec import DiffuRec
from adrec_still.models.upstream.dreamrec import DreamRec
from adrec_still.models.upstream.sasrec import SASRecUpstream
from adrec_still.models.upstream.transformer import LayerNorm

UPSTREAM_MODEL_FAMILY = "upstream"


def chunked_cross_entropy(
    representations: torch.Tensor,
    labels: torch.Tensor,
    embedding_weight: torch.Tensor,
    chunk_tokens: int | None,
) -> torch.Tensor:
    """Mean CE of ``rep @ E.T`` vs labels over rows, optionally in checkpointed chunks.

    ``representations`` is ``[rows, D]`` (already filtered to label > 0 rows).
    Chunking recomputes each chunk's logits during backward instead of keeping
    every ``[chunk, V]`` activation alive, so the loss and gradients equal the
    dense computation while peak memory stays bounded by one chunk.
    """
    rows = representations.shape[0]
    if rows == 0:
        raise ValueError("cross entropy needs at least one labeled position")
    if chunk_tokens is None or rows <= chunk_tokens:
        logits = representations @ embedding_weight.t()
        return F.cross_entropy(logits, labels)

    def chunk_loss_sum(rep_chunk: torch.Tensor, label_chunk: torch.Tensor) -> torch.Tensor:
        logits = rep_chunk @ embedding_weight.t()
        return F.cross_entropy(logits, label_chunk, reduction="sum")

    total = representations.new_zeros(())
    for start in range(0, rows, chunk_tokens):
        rep_chunk = representations[start:start + chunk_tokens]
        label_chunk = labels[start:start + chunk_tokens]
        total = total + checkpoint(chunk_loss_sum, rep_chunk, label_chunk,
                                   use_reentrant=False)
    return total / rows


def _left_pad_to(history: torch.Tensor, mask: torch.Tensor, length: int):
    """Re-pad a left-padded [B, L] batch to exactly ``length`` columns."""
    batch, current = history.shape
    if current == length:
        return history, mask
    if current > length:
        return history[:, -length:], mask[:, -length:]
    pad = history.new_zeros((batch, length - current))
    pad_mask = mask.new_zeros((batch, length - current))
    return torch.cat([pad, history], dim=1), torch.cat([pad_mask, mask], dim=1)


def _history_fingerprint(history: torch.Tensor) -> int:
    return zlib.crc32(history.detach().cpu().numpy().tobytes())


class UpstreamRecommender(BaselineModel, nn.Module):
    """Base wrapper: embedding table, history path, scoring surface."""

    family = UPSTREAM_MODEL_FAMILY
    #: Which corpus view this model trains on ("parallel" per-position targets,
    #: "prefix_last" prefix-expanded last-target rows).
    training_view = "parallel"
    #: Objectives the staged trainer merges through the PCGrad wrapper.
    uses_diffusion_mse = False
    trains_cross_entropy = True
    #: Whether the staged trainer runs an embedding-pretrain stage first.
    embedding_pretrain = False

    def __init__(self, number_of_pois: int, *, hidden_size: int = 128, max_len: int = 50,
                 dropout: float = 0.1, emb_dropout: float = 0.3,
                 ce_chunk_tokens: int | None = 4096, eval_noise_seed: int = 20251014,
                 **_: Any) -> None:
        nn.Module.__init__(self)
        BaselineModel.__init__(self, number_of_pois)
        self.hidden_size = int(hidden_size)
        self.max_len = int(max_len)
        self.emb_dropout_rate = float(emb_dropout)
        self.dropout_rate = float(dropout)
        self.ce_chunk_tokens = ce_chunk_tokens
        self.eval_noise_seed = int(eval_noise_seed)
        self.item_embedding = nn.Embedding(self.number_of_pois, self.hidden_size,
                                           padding_idx=0)
        self.embed_dropout = nn.Dropout(emb_dropout)
        self.hist_norm = LayerNorm(self.hidden_size, eps=1e-12)

    # ------------------------------------------------------------------ helpers

    @property
    def device(self) -> torch.device:
        return self.item_embedding.weight.device

    def embed_history(self, history: torch.Tensor) -> torch.Tensor:
        """Fork order preserved: dropout first, then the TF-style layer norm."""
        return self.hist_norm(self.embed_dropout(self.item_embedding(history)))

    def _eval_generator(self, history: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(
            (self.eval_noise_seed * 1_000_003 + _history_fingerprint(history))
            % (2**63 - 1)
        )
        return generator

    def set_embedding_trainable(self, trainable: bool) -> None:
        self.item_embedding.weight.requires_grad = bool(trainable)

    def load_embedding(self, weight: torch.Tensor, *, freeze: bool) -> None:
        if weight.shape != self.item_embedding.weight.shape:
            raise ValueError(
                f"pretrained embedding shape {tuple(weight.shape)} does not match this "
                f"model's table {tuple(self.item_embedding.weight.shape)}"
            )
        with torch.no_grad():
            self.item_embedding.weight.copy_(weight.to(self.device))
        self.set_embedding_trainable(not freeze)

    # ------------------------------------------------------- scoring interface

    def encode_sequences(self, history: torch.Tensor) -> torch.Tensor:
        """Last-position representation for raw left-padded [B, L] histories."""
        raise NotImplementedError

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        history = batch["history_poi"].to(self.device)
        mask = batch["history_mask"].to(self.device)
        history, _ = _left_pad_to(history, mask, self.max_len)
        return self.encode_sequences(history)

    def score_candidates(self, context: torch.Tensor,
                         candidate_poi: torch.Tensor) -> torch.Tensor:
        candidate_embeddings = self.item_embedding(candidate_poi.to(self.device))
        return torch.bmm(candidate_embeddings, context.unsqueeze(2)).squeeze(2)

    def score_full_catalog(self, context: torch.Tensor) -> torch.Tensor:
        return context @ self.item_embedding.weight.T

    def predict(self, batch: dict[str, torch.Tensor]) -> BaselinePrediction:
        self.require_fitted()
        validate_batch(batch)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                context = self.encode(batch)
                log_probabilities = masked_log_probabilities(self.score_full_catalog(context))
        finally:
            self.train(was_training)
        return BaselinePrediction(
            poi_logits=log_probabilities.cpu(), context=context.cpu(),
        )

    # ------------------------------------------------------------------ training

    def sequence_ce(self, out_seq: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        index = labels > 0
        return chunked_cross_entropy(
            out_seq[index], labels[index], self.item_embedding.weight, self.ce_chunk_tokens
        )

    def training_objectives(self, history: torch.Tensor,
                            targets: torch.Tensor) -> dict[str, Any]:
        """Per-batch objectives; the staged trainer merges and logs them."""
        raise NotImplementedError

    def fit(self, training_source: Any) -> UpstreamRecommender:
        from adrec_still.training.stages import StagedPlan, train_staged_model

        if not isinstance(training_source, StagedPlan):
            raise TypeError(f"{self.name}.fit expects a StagedPlan from the staged trainer")
        train_staged_model(self, training_source)
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
            "max_sequence_length": self.max_len,
            "dropout": self.dropout_rate,
            "emb_dropout": self.emb_dropout_rate,
            "ce_chunk_tokens": self.ce_chunk_tokens,
            "training_view": self.training_view,
            "parameter_count": self.parameter_count(),
            "context_dim": self.hidden_size,
        }

    @property
    def context_dim(self) -> int:
        return self.hidden_size


def _shifted_targets(history: torch.Tensor) -> torch.Tensor:
    """Teacher-forcing prefix for inference: history shifted left, last slot PAD."""
    return torch.cat([history[:, 1:], torch.zeros_like(history[:, :1])], dim=1)


class AdRecRecommender(UpstreamRecommender):
    """ADRec: clean causal encoder + token-independent diffusion + CE + MSE."""

    name = "adrec"
    uses_diffusion_mse = True

    def __init__(self, number_of_pois: int, *, loss_scale: float = 1.0,
                 inference_steps: int | None = None, embedding_pretrain: bool = True,
                 **kwargs: Any) -> None:
        super().__init__(number_of_pois, **kwargs)
        self.loss_scale = float(loss_scale)
        self.inference_steps = inference_steps
        self.embedding_pretrain = bool(embedding_pretrain)
        self.core = AdRec(self._core_config(kwargs))

    def _core_config(self, kwargs: dict[str, Any]) -> AdRecConfig:
        allowed = {name for name in AdRecConfig.__dataclass_fields__}
        picked = {key: value for key, value in kwargs.items() if key in allowed}
        picked.setdefault("hidden_size", self.hidden_size)
        picked.setdefault("dropout", self.dropout_rate)
        return AdRecConfig(**picked)

    def training_objectives(self, history, targets):
        item_embeddings = self.embed_history(history)
        tag_embeddings = self.item_embedding(targets)
        if self.core.geodesic:
            tag_embeddings = F.normalize(tag_embeddings, p=2, dim=-1)
        mask_seq = (history > 0).float()
        mask_tag = (targets > 0).float().view(targets.shape[0], -1)
        out_seq, _per_token, dif_loss = self.core(
            item_embeddings, tag_embeddings, mask_seq, mask_tag
        )
        ce = self.sequence_ce(out_seq, targets)
        return {"ce": ce, "mse": dif_loss,
                "objectives": [ce, self.loss_scale * dif_loss]}

    def encode_sequences(self, history: torch.Tensor) -> torch.Tensor:
        item_embeddings = self.embed_history(history)
        targets = _shifted_targets(history)
        tag_embeddings = self.item_embedding(targets)
        if self.core.geodesic:
            tag_embeddings = F.normalize(tag_embeddings, p=2, dim=-1)
        mask_seq = (history > 0).float()
        mask_tag = (targets > 0).float().view(targets.shape[0], -1)
        out_seq = self.core.denoise_sample(
            item_embeddings, tag_embeddings, mask_seq, mask_tag,
            generator=self._eval_generator(history),
            inference_steps=self.inference_steps,
        )
        return out_seq[:, -1, :]

    def resolved_config(self) -> dict[str, Any]:
        resolved = super().resolved_config()
        config = self.core.config
        resolved.update({
            "diffusion_steps": config.diffusion_steps,
            "noise_schedule": config.noise_schedule,
            "schedule_sampler": config.schedule_sampler_name,
            "lambda_uncertainty": config.lambda_uncertainty,
            "independent": config.independent,
            "cfg_scale": config.cfg_scale,
            "geodesic": config.geodesic,
            "dif_decoder": config.dif_decoder,
            "loss_scale": self.loss_scale,
            "inference_steps": self.inference_steps,
        })
        return resolved


class DiffuRecRecommender(UpstreamRecommender):
    """DiffuRec: prefix-expanded last-target training, CE backpropagated."""

    name = "diffurec"
    training_view = "prefix_last"

    def __init__(self, number_of_pois: int, **kwargs: Any) -> None:
        kwargs.setdefault("is_causal", False)  # the fork forces is_causal=False
        super().__init__(number_of_pois,
                         **{k: v for k, v in kwargs.items()
                            if k in ("hidden_size", "max_len", "dropout", "emb_dropout",
                                     "ce_chunk_tokens", "eval_noise_seed")})
        allowed = {name for name in AdRecConfig.__dataclass_fields__}
        picked = {key: value for key, value in kwargs.items() if key in allowed}
        picked.setdefault("hidden_size", self.hidden_size)
        picked.setdefault("dropout", self.dropout_rate)
        self.core = DiffuRec(AdRecConfig(**picked))

    def training_objectives(self, history, targets):
        item_embeddings = self.embed_history(history)
        tag_embeddings = self.item_embedding(targets)
        mask_seq = (history > 0).float()
        mask_tag = (targets > 0).float().view(targets.shape[0], -1)
        out_seq, dif_loss = self.core(item_embeddings, tag_embeddings, mask_seq, mask_tag)
        ce = self.sequence_ce(out_seq, targets)
        return {"ce": ce, "mse": dif_loss, "objectives": [ce]}

    def encode_sequences(self, history: torch.Tensor) -> torch.Tensor:
        item_embeddings = self.embed_history(history)
        targets = _shifted_targets(history)
        tag_embeddings = self.item_embedding(targets)
        mask_seq = (history > 0).float()
        mask_tag = (targets > 0).float().view(targets.shape[0], -1)
        out = self.core.denoise_sample(item_embeddings, tag_embeddings, mask_seq, mask_tag,
                                       generator=self._eval_generator(history))
        return out[:, -1, :]

    def resolved_config(self) -> dict[str, Any]:
        resolved = super().resolved_config()
        resolved.update({"diffusion_steps": self.core.diffusion_steps})
        return resolved


class DreamRecRecommender(UpstreamRecommender):
    """DreamRec: trained on its diffusion MSE alone (fork recipe)."""

    name = "dreamrec"
    uses_diffusion_mse = True
    trains_cross_entropy = False

    def __init__(self, number_of_pois: int, **kwargs: Any) -> None:
        super().__init__(number_of_pois,
                         **{k: v for k, v in kwargs.items()
                            if k in ("hidden_size", "max_len", "dropout", "emb_dropout",
                                     "ce_chunk_tokens", "eval_noise_seed")})
        allowed = {name for name in AdRecConfig.__dataclass_fields__}
        picked = {key: value for key, value in kwargs.items() if key in allowed}
        picked.setdefault("hidden_size", self.hidden_size)
        picked.setdefault("dropout", self.dropout_rate)
        self.core = DreamRec(AdRecConfig(**picked))

    def training_objectives(self, history, targets):
        item_embeddings = self.embed_history(history)
        tag_embeddings = self.item_embedding(targets)
        mask_seq = (history > 0).float()
        mask_tag = (targets > 0).float().view(targets.shape[0], -1)
        out_seq, dif_loss = self.core(item_embeddings, tag_embeddings, mask_seq, mask_tag)
        with torch.no_grad():
            ce = self.sequence_ce(out_seq, targets)
        return {"ce": ce, "mse": dif_loss, "objectives": [dif_loss]}

    def encode_sequences(self, history: torch.Tensor) -> torch.Tensor:
        item_embeddings = self.embed_history(history)
        targets = _shifted_targets(history)
        tag_embeddings = self.item_embedding(targets)
        mask_seq = (history > 0).float()
        mask_tag = (targets > 0).float().view(targets.shape[0], -1)
        out = self.core.denoise_sample(item_embeddings, tag_embeddings, mask_seq, mask_tag,
                                       generator=self._eval_generator(history))
        return out[:, -1, :]


class SASRecUpstreamRecommender(UpstreamRecommender):
    """The fork's SASRec+ variant behind the shared interface."""

    name = "sasrec_upstream"

    def __init__(self, number_of_pois: int, *, pretrain_mode: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(number_of_pois,
                         **{k: v for k, v in kwargs.items()
                            if k in ("hidden_size", "max_len", "dropout", "emb_dropout",
                                     "ce_chunk_tokens", "eval_noise_seed")})
        is_causal = bool(kwargs.get("is_causal", True))
        self.core = SASRecUpstream(
            item_vocab_size=number_of_pois, hidden_size=self.hidden_size,
            max_len=self.max_len, dropout=self.dropout_rate,
            emb_dropout=self.emb_dropout_rate, is_causal=is_causal,
            pretrain_mode=pretrain_mode,
        )
        # One table: the wrapper's scoring surface and the core share weights.
        self.item_embedding = self.core.item_embedding
        self.pretrain_mode = pretrain_mode

    def embed_history(self, history: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("SASRecUpstream embeds inside its own forward")

    def training_objectives(self, history, targets):
        out_seq, _last = self.core(history)
        ce = self.sequence_ce(out_seq, targets)
        return {"ce": ce, "mse": torch.zeros((), device=ce.device), "objectives": [ce]}

    def encode_sequences(self, history: torch.Tensor) -> torch.Tensor:
        _out, last = self.core(history)
        return last

    def resolved_config(self) -> dict[str, Any]:
        resolved = super().resolved_config()
        resolved["pretrain_mode"] = self.pretrain_mode
        return resolved


UPSTREAM_MODELS = {
    "adrec": AdRecRecommender,
    "diffurec": DiffuRecRecommender,
    "dreamrec": DreamRecRecommender,
    "sasrec_upstream": SASRecUpstreamRecommender,
}


def build_upstream_model(config: dict[str, Any], number_of_pois: int) -> UpstreamRecommender:
    name = config["name"]
    if name not in UPSTREAM_MODELS:
        raise ValueError(f"unknown upstream model: {name!r}")
    keyword_arguments = {
        key: value for key, value in config.items()
        if key not in ("name", "family", "training")
    }
    return UPSTREAM_MODELS[name](number_of_pois, **keyword_arguments)
