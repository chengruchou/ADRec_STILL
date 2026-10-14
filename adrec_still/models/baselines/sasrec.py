"""SASRec-H: the self-attention history-only neural baseline (B2.2).

The model consumes only ``history_poi`` and ``history_mask`` from the shared
collated batch: ``query_s_sec`` and the raw timestamps are never read, so the
returned distribution is identical for every landmark of the same gap by
construction. It reuses GRU4Rec-H's output/candidate policy exactly -- one shared
``[number_of_pois, embedding_dim]`` item table scores the full frozen training
vocabulary and the sampled training candidates -- so the history encoder is the
only intended difference between the two arms trained by
:mod:`adrec_still.models.baselines.trainer`.

Padding and causality
---------------------
:func:`adrec_still.data.collate.collate_canonical_samples` left-pads histories, so the
valid tokens of every row occupy the *last* columns and column ``L - 1`` is the
most recent event. Two consequences are load-bearing here:

* Position embeddings are indexed by **recency distance** (``L - 1 - column``),
  never by absolute column. Absolute indices would shift whenever a longer row
  joins the batch and adds padding columns in front, which would silently break
  padding-mask invariance.
* Attention uses one combined mask: a causal upper triangle plus the per-row key
  padding mask. Under that combination a *padding* query row would have every
  key masked, and a fully masked softmax row is ``NaN``; because ``0 * NaN`` is
  ``NaN``, that value would poison every valid query in the next layer. The mask
  therefore always leaves a query attending to itself, and padding positions are
  re-zeroed after every block so their (finite, meaningless) values never leave
  the row.

Exported context
----------------
:meth:`SASRecH.encode` returns the representation at the last valid position,
``[batch, embedding_dim]``, after the final layer norm. That tensor is the
contract Student C's conditional adapter (Issue #20) and the Issue #40 time
controls consume; see ``docs/SASREC_H_CONTEXT_CONTRACT.md``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from adrec_still.data.schema import PADDING_POI_ID
from adrec_still.models.baselines.base import (
    BaselineModel,
    BaselinePrediction,
    masked_log_probabilities,
    validate_batch,
)

#: Identifier of the exported history-context contract (Issue #20 / #40 consume it).
SASREC_H_CONTEXT_CONTRACT_VERSION = "still-sasrec-h-context-v0.1"

#: Standard deviation of the transformer's normal parameter initialization.
#: Recorded in ``resolved_config`` because it is an architecture-specific choice
#: that GRU4Rec-H (PyTorch defaults) does not share.
INITIALIZER_STD = 0.02


class SASRecH(BaselineModel, nn.Module):
    """History-only self-attention encoder with tied item-embedding output scores."""

    name = "sasrec_h"
    family = "baseline"

    def __init__(
        self,
        number_of_pois: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        max_sequence_length: int = 100,
    ) -> None:
        nn.Module.__init__(self)
        BaselineModel.__init__(self, number_of_pois)
        if embedding_dim <= 0 or hidden_dim <= 0 or num_layers <= 0 or num_heads <= 0:
            raise ValueError("embedding_dim, hidden_dim, num_layers, num_heads must be positive")
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.max_sequence_length = int(max_sequence_length)

        self.item_embedding = nn.Embedding(
            self.number_of_pois, self.embedding_dim, padding_idx=PADDING_POI_ID
        )
        # Indexed by recency distance from the query, not by absolute column.
        self.position_embedding = nn.Embedding(self.max_sequence_length, self.embedding_dim)
        self.input_dropout = nn.Dropout(self.dropout)
        self.attention_norms = nn.ModuleList(
            nn.LayerNorm(self.embedding_dim) for _ in range(self.num_layers)
        )
        self.attentions = nn.ModuleList(
            nn.MultiheadAttention(
                self.embedding_dim, self.num_heads, dropout=self.dropout, batch_first=True
            )
            for _ in range(self.num_layers)
        )
        self.feed_forward_norms = nn.ModuleList(
            nn.LayerNorm(self.embedding_dim) for _ in range(self.num_layers)
        )
        self.feed_forwards = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.embedding_dim),
                nn.Dropout(self.dropout),
            )
            for _ in range(self.num_layers)
        )
        self.final_norm = nn.LayerNorm(self.embedding_dim)
        self._initialize_embeddings()

    def _initialize_embeddings(self) -> None:
        """Small-normal init: PyTorch's default N(0, 1) embeddings would make the
        ``sqrt(embedding_dim)``-scaled inputs and the tied output logits diverge."""
        for embedding in (self.item_embedding, self.position_embedding):
            nn.init.normal_(embedding.weight, mean=0.0, std=INITIALIZER_STD)
        with torch.no_grad():
            self.item_embedding.weight[PADDING_POI_ID].zero_()

    @property
    def context_dim(self) -> int:
        """Width of the exported history representation."""
        return self.embedding_dim

    @property
    def device(self) -> torch.device:
        return self.item_embedding.weight.device

    def _attention_mask(self, valid: torch.Tensor) -> torch.Tensor:
        """Combined causal + key-padding mask, ``True`` where attention is forbidden."""
        sequence_length = int(valid.shape[1])
        future = torch.ones(
            sequence_length, sequence_length, dtype=torch.bool, device=valid.device
        ).triu(1)
        disallow = future.unsqueeze(0) | (~valid).unsqueeze(1)
        # Always leave the diagonal open. A padding query would otherwise have every
        # key masked, and a fully masked softmax row is NaN that ``0 * NaN`` spreads
        # to every valid query in the next layer. Padding rows are re-zeroed after
        # each block and are masked as keys, so their values never leave the row.
        identity = torch.eye(sequence_length, dtype=torch.bool, device=valid.device)
        disallow = disallow & ~identity
        return disallow.repeat_interleave(self.num_heads, dim=0)

    def encode_sequence(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return causal per-position representations ``[batch, sequence, embedding_dim]``.

        Padding positions are exactly zero. Position ``i`` summarizes history events
        up to and including column ``i`` only.
        """
        history = batch["history_poi"].to(self.device)
        valid = batch["history_mask"].to(self.device).bool()
        sequence_length = int(history.shape[1])
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"history of length {sequence_length} exceeds the model's "
                f"max_sequence_length={self.max_sequence_length}"
            )
        if sequence_length == 0:
            # A batch whose every history is empty collates to [batch, 0], and attention
            # cannot run on a zero-length sequence. Return the empty per-position tensor
            # directly so an empty history behaves the same whatever else shares its
            # batch, instead of depending on a longer row to create the columns.
            return self.item_embedding.weight.new_zeros(
                (int(history.shape[0]), 0, self.embedding_dim)
            )
        # Column j is (sequence_length - 1 - j) events before the most recent one, so
        # the same event keeps the same position id however much padding precedes it.
        recency = torch.arange(sequence_length - 1, -1, -1, device=self.device)
        states = self.item_embedding(history) * math.sqrt(self.embedding_dim)
        states = states + self.position_embedding(recency).unsqueeze(0)
        keep = valid.unsqueeze(2)
        states = self.input_dropout(states) * keep
        attention_mask = self._attention_mask(valid)
        for norm, attention, feed_forward_norm, feed_forward in zip(
            self.attention_norms, self.attentions, self.feed_forward_norms, self.feed_forwards
        ):
            queries = norm(states)
            attended, _ = attention(
                queries, states, states, attn_mask=attention_mask, need_weights=False
            )
            states = (states + attended) * keep
            states = (states + feed_forward(feed_forward_norm(states))) * keep
        return self.final_norm(states) * keep

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the exported history context per row: the last valid position.

        Histories are left-padded, so column ``-1`` holds the most recent event of
        every non-empty row. A history that carries no evidence exports an exact zero
        context, matching GRU4Rec-H: a padded row was already zeroed by
        :meth:`encode_sequence`, and a batch with no columns at all is handled below.
        """
        states = self.encode_sequence(batch)
        if states.shape[1] == 0:
            # Every history in the batch is empty, so there is no last position to read.
            # Export the same evidence-free zero context the padded rows of a mixed
            # batch get, rather than indexing into a zero-length dimension.
            return states.new_zeros((int(states.shape[0]), self.context_dim))
        return states[:, -1, :]

    def score_candidates(self, context: torch.Tensor, candidate_poi: torch.Tensor) -> torch.Tensor:
        """Score explicit ``[rows, candidates]`` POI IDs against contexts."""
        candidate_embeddings = self.item_embedding(candidate_poi)
        return torch.bmm(candidate_embeddings, context.unsqueeze(2)).squeeze(2)

    def score_full_catalog(self, context: torch.Tensor) -> torch.Tensor:
        """Score every POI in the frozen training vocabulary."""
        return context @ self.item_embedding.weight.T

    def fit(self, training_source: Any) -> SASRecH:
        """Train through the shared neural trainer (see ``trainer.TrainingPlan``)."""
        from adrec_still.models.baselines.trainer import TrainingPlan, train_model

        if not isinstance(training_source, TrainingPlan):
            raise TypeError("SASRecH.fit expects a shared-trainer TrainingPlan")
        train_model(self, training_source)
        return self

    def predict(self, batch: dict[str, torch.Tensor]) -> BaselinePrediction:
        """Return normalized full-catalog log-probabilities; horizon fields are ignored."""
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
            poi_logits=log_probabilities.cpu(),
            context=context.cpu(),
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def resolved_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "interface_version": self.interface_version,
            "context_contract_version": SASREC_H_CONTEXT_CONTRACT_VERSION,
            "number_of_pois": self.number_of_pois,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "max_sequence_length": self.max_sequence_length,
            "initializer_std": INITIALIZER_STD,
            "tied_item_embeddings": True,
            "context_dim": self.context_dim,
            "parameter_count": self.parameter_count(),
        }
