"""SASRec+ of the fork (ported from ``src/sasrec.py``; Kang & McAuley 2018 lineage).

The fork's actual variant is preserved: learned position embeddings (skipped in
embedding-pretrain mode so the pretrained table is position-agnostic), 2-block
causal transformer, full-softmax CE with ``ignore_index=0``. Named
``sasrec_upstream`` in the registry so it is never confused with the donor
repository's SASRec-H.
"""

from __future__ import annotations

import torch
from torch import nn

from adrec_still.models.upstream.transformer import TransformerEncoder


class SASRecUpstream(nn.Module):
    def __init__(self, item_vocab_size: int, hidden_size: int, max_len: int,
                 dropout: float, emb_dropout: float, is_causal: bool = True,
                 pretrain_mode: bool = False) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.pretrain_mode = pretrain_mode
        self.item_embedding = nn.Embedding(item_vocab_size, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, hidden_size)
        self.trm_encoder = TransformerEncoder(
            hidden_size=hidden_size, dropout=dropout, is_causal=is_causal, num_blocks=2,
            norm_first=False,
        )
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(emb_dropout)
        self.apply(self._init_weights)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)

    @staticmethod
    def _init_weights(module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.1)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def embedding_layer(self, item_seq: torch.Tensor):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        ).unsqueeze(0).expand_as(item_seq)
        return self.item_embedding(item_seq), self.position_embedding(position_ids)

    def forward(self, item_seq: torch.Tensor):
        item_emb, position_emb = self.embedding_layer(item_seq)
        input_emb = item_emb if self.pretrain_mode else item_emb + position_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        mask_seq = (item_seq > 0).float()
        output_seq = self.trm_encoder(input_emb, mask_seq)
        return output_seq, output_seq[:, -1, :]

    def calculate_loss(self, seq_output: torch.Tensor, tgt_seq: torch.Tensor) -> torch.Tensor:
        index = tgt_seq > 0
        seq_output = seq_output[index]
        tgt_seq = tgt_seq[index]
        logits = torch.matmul(seq_output, self.item_embedding.weight.t())
        return self.loss_fct(logits.reshape(-1, logits.shape[-1]), tgt_seq.reshape(-1))

    def calculate_score(self, item: torch.Tensor) -> torch.Tensor:
        return torch.matmul(item.reshape(-1, item.shape[-1]), self.item_embedding.weight.t())
