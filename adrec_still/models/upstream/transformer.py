"""Transformer blocks of the upstream ADRec fork (``src/common.py``), ported.

Only the pieces the live models use are ported: SiLU, the TF-style LayerNorm,
SublayerConnection, PositionwiseFeedForward (tanh-approximated GELU),
MultiHeadedAttention, TransformerEncoderBlock and TransformerEncoder.

One deliberate fix versus the upstream file (MIGRATION.md): the upstream
attention built the padding mask as ``view(B, 1, L, 1)`` — a *query*-side mask —
so padded key positions stayed attendable and left-padding leaked the padding
embedding into real queries. The port masks *keys* (``[B, 1, 1, L]``), keeps the
additive causal mask, and zeroes padded-position outputs after the final block.
The finite ``-1e9`` floor (rather than ``-inf``) keeps a padded query row's
softmax finite, so no all-masked row can emit NaN.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

MASKED_SCORE = -1.0e9


def generate_square_subsequent_mask(size: int, device) -> torch.Tensor:
    """Additive causal mask: 0 on and below the diagonal, ``-inf`` above."""
    return torch.triu(
        torch.full((size, size), float("-inf"), dtype=torch.float32, device=device),
        diagonal=1,
    )


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class LayerNorm(nn.Module):
    """TF-style layer norm (epsilon inside the square root), as upstream."""

    def __init__(self, hidden_size: int, eps: float = 1e-12) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SublayerConnection(nn.Module):
    """Residual connection followed by a layer norm (post-norm by default)."""

    def __init__(self, hidden_size: int, dropout: float, norm_first: bool = False) -> None:
        super().__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x, sublayer):
        if self.norm_first:
            return x + self.dropout(sublayer(self.norm(x)))
        return self.norm(x + self.dropout(sublayer(x)))


class PositionwiseFeedForward(nn.Module):
    """Upstream FFN with the hand-written tanh-approximated GELU."""

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size * 4)
        self.w_2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (
            1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3)))
        )
        return self.w_2(self.dropout(activation))


class MultiHeadedAttention(nn.Module):
    """Upstream multi-head attention with the key-padding mask fix."""

    def __init__(self, heads: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.xavier_normal_(self.w_layer.weight)
        for layer in self.linear_layers:
            nn.init.xavier_normal_(layer.weight)

    def forward(self, q, k, v, padding_mask=None, is_causal=False):
        batch_size = q.shape[0]
        q, k, v = [
            layer(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2)
            for layer, x in zip(self.linear_layers, (q, k, v))
        ]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if padding_mask is not None:
            # Key mask [B, 1, 1, Lk]: padded positions are not attendable by anyone.
            # (The upstream file masked the query axis instead; see module docstring.)
            key_mask = padding_mask.view(batch_size, 1, 1, -1)
            corr = corr.masked_fill(key_mask == 0, MASKED_SCORE)
        if is_causal:
            causal_mask = generate_square_subsequent_mask(corr.shape[-1], device=corr.device)
            corr = corr + causal_mask.unsqueeze(0).unsqueeze(0)
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        return self.w_layer(
            hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head)
        )


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout, is_causal, norm_first=False) -> None:
        super().__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size,
                                              dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size, dropout, norm_first=norm_first)
        self.output_sublayer = SublayerConnection(hidden_size, dropout, norm_first=norm_first)
        self.is_causal = is_causal

    def forward(self, hidden, padding_mask):
        hidden = self.input_sublayer(
            hidden,
            lambda h: self.attention(h, h, h, padding_mask=padding_mask,
                                     is_causal=self.is_causal),
        )
        return self.output_sublayer(hidden, self.feed_forward)


class TransformerEncoder(nn.Module):
    """Upstream encoder stack: 4 heads, causality from the configuration.

    Padded-position outputs are zeroed after the final block so downstream
    consumers never read an embedding computed from padding alone.
    """

    def __init__(self, hidden_size: int, dropout: float, is_causal: bool,
                 num_blocks: int, norm_first: bool = False) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = 4
        self.dropout = dropout
        self.is_causal = is_causal
        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(self.hidden_size, self.heads, self.dropout,
                                     self.is_causal, norm_first=norm_first)
             for _ in range(num_blocks)]
        )

    def forward(self, hidden: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            # Causality is read per call so make_causal applies to every block.
            block.is_causal = self.is_causal
            hidden = block(hidden, padding_mask)
        return hidden * padding_mask.unsqueeze(-1).to(hidden.dtype)

    def make_causal(self, is_causal: bool) -> None:
        self.is_causal = is_causal
