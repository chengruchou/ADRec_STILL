"""GRU4Rec-H: the recurrent history-only neural baseline (B1).

The model consumes only ``history_poi`` and ``history_mask`` from the shared
collated batch: ``query_s_sec`` and the raw timestamps are never read, so the
returned distribution is identical for every landmark of the same gap by
construction. Output scores are tied to the item embedding table (one shared
``[number_of_pois, embedding_dim]`` matrix scores the full frozen training
vocabulary), which is also the output/candidate policy SASRec-H must reuse.

Histories arrive left-padded from :func:`adrec_still.data.collate.collate_canonical_samples`.
The encoder re-packs each row to a right-aligned sequence and runs the GRU over
the valid positions only, so extra padding columns cannot perturb the recurrent
state.
"""

from __future__ import annotations

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


class GRU4RecH(BaselineModel, nn.Module):
    """History-only GRU encoder with tied item-embedding output scores."""

    name = "gru4rec_h"
    family = "baseline"

    def __init__(
        self,
        number_of_pois: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        nn.Module.__init__(self)
        BaselineModel.__init__(self, number_of_pois)
        if embedding_dim <= 0 or hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("embedding_dim, hidden_dim, and num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.item_embedding = nn.Embedding(
            self.number_of_pois, self.embedding_dim, padding_idx=PADDING_POI_ID
        )
        self.input_dropout = nn.Dropout(self.dropout)
        self.gru = nn.GRU(
            self.embedding_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        # Tied output: scores = project(context) @ item_embedding.T. The projection
        # exists only when the GRU state width differs from the embedding width.
        self.output_projection = (
            nn.Identity()
            if self.hidden_dim == self.embedding_dim
            else nn.Linear(self.hidden_dim, self.embedding_dim, bias=False)
        )

    @property
    def context_dim(self) -> int:
        """Width of the exported final GRU state."""
        return self.hidden_dim

    @property
    def device(self) -> torch.device:
        return self.item_embedding.weight.device

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the final recurrent state per row from history POIs only."""
        history = batch["history_poi"].to(self.device)
        mask = batch["history_mask"].to(self.device)
        sequence_length = int(history.shape[1])
        lengths = mask.sum(dim=1)
        # Left-padded -> right-aligned: valid tokens of a row of length n occupy
        # [sequence_length - n, sequence_length); positions past n are clamped
        # duplicates that pack_padded_sequence never reads.
        offsets = (sequence_length - lengths).unsqueeze(1)
        positions = torch.arange(sequence_length, device=self.device).unsqueeze(0)
        gather_index = (offsets + positions).clamp(max=max(sequence_length - 1, 0))
        right_aligned = history.gather(1, gather_index)
        embedded = self.input_dropout(self.item_embedding(right_aligned))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, final_state = self.gru(packed)
        context = final_state[-1]
        # A fully-padded history carries no evidence: export an exact zero state
        # instead of the GRU's bias response to padding embeddings.
        return context * (lengths > 0).unsqueeze(1)

    def score_candidates(
        self, context: torch.Tensor, candidate_poi: torch.Tensor
    ) -> torch.Tensor:
        """Score explicit ``[rows, candidates]`` POI IDs against contexts."""
        projected = self.output_projection(context)
        candidate_embeddings = self.item_embedding(candidate_poi)
        return torch.bmm(candidate_embeddings, projected.unsqueeze(2)).squeeze(2)

    def score_full_catalog(self, context: torch.Tensor) -> torch.Tensor:
        """Score every POI in the frozen training vocabulary."""
        projected = self.output_projection(context)
        return projected @ self.item_embedding.weight.T

    def fit(self, training_source: Any) -> GRU4RecH:
        """Train through the shared neural trainer (see ``trainer.TrainingPlan``)."""
        from adrec_still.models.baselines.trainer import TrainingPlan, train_model

        if not isinstance(training_source, TrainingPlan):
            raise TypeError("GRU4RecH.fit expects a shared-trainer TrainingPlan")
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
            "number_of_pois": self.number_of_pois,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "tied_item_embeddings": True,
            "context_dim": self.context_dim,
            "parameter_count": self.parameter_count(),
        }
