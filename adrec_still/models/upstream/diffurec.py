"""DiffuRec (ported from the fork's ``src/diffurec.py``; WHUIR/DiffuRec lineage).

Preserved: the stochastic ``lambda_uncertainty ~ N(lambda, lambda)`` fusion that
injects only the *last* position's noised target, the 4-block pre-norm
transformer denoiser, sequence-level timesteps, last-position MSE, and the
single-latent reverse loop (no prefix teacher forcing). In the fork's training
recipe DiffuRec backpropagates CE only; the diffusion MSE is logged.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from adrec_still.models.upstream.adrec import AdRecConfig, _extract_into_tensor
from adrec_still.models.upstream.schedules import (
    create_named_schedule_sampler,
    get_named_beta_schedule,
)
from adrec_still.models.upstream.transformer import LayerNorm, SiLU, TransformerEncoder


class DiffuXstart(nn.Module):
    """DiffuRec's x0-prediction network."""

    def __init__(self, config: AdRecConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            SiLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )
        self.transencoder = TransformerEncoder(
            hidden_size=self.hidden_size, dropout=config.dropout,
            is_causal=config.is_causal, num_blocks=4, norm_first=True,
        )
        self.lambda_uncertainty = config.lambda_uncertainty
        self.norm_diffu_rep = LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        assert dim % 2 == 0
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps.unsqueeze(-1).float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, rep_item, x_t, t, mask_seq, mask_tgt, condition=True):
        if condition is not True:
            rep_item = torch.zeros_like(rep_item)
        t = t.reshape(x_t.shape[0], -1)
        time_emb = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        lambda_uncertainty = torch.normal(
            mean=torch.full(rep_item.shape, self.lambda_uncertainty),
            std=torch.full(rep_item.shape, self.lambda_uncertainty),
        ).to(x_t.device)
        assert x_t[:, -1:, :].shape == time_emb.shape
        rep_diffu = rep_item + lambda_uncertainty * (x_t[:, -1:] + time_emb)
        rep_diffu = self.transencoder(rep_diffu, mask_seq)
        return self.norm_diffu_rep(self.dropout(rep_diffu))


class DiffuRec(nn.Module):
    def __init__(self, config: AdRecConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.diffusion_steps = config.diffusion_steps
        betas = np.array(
            get_named_beta_schedule(config.noise_schedule, config.diffusion_steps,
                                    beta_a=config.beta_a, beta_b=config.beta_b),
            dtype=np.float64,
        )
        assert len(betas.shape) == 1 and (betas > 0).all() and (betas <= 1).all()
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.num_timesteps = int(betas.shape[0])
        self.schedule_sampler = create_named_schedule_sampler(
            config.schedule_sampler_name, self.num_timesteps
        )
        self.rescale_timesteps = config.rescale_timesteps
        self.net = DiffuXstart(config)
        self.cfg_scale = config.cfg_scale

    def q_sample(self, x_start, t, noise=None, mask=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        if mask is None:
            return x_t
        mask = torch.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
        return torch.where(mask == 0, x_start, x_t)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        return (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

    def p_mean_variance(self, rep_item, x_t, t, mask_seq, mask_tag):
        out_seq = self.net(rep_item, x_t, self._scale_timesteps(t), mask_seq, mask_tag)
        x_0 = out_seq[:, -1:, :]
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)
        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t, t=t)
        return model_mean, model_log_variance

    def p_sample(self, item_rep, noise_x_t, t, mask_seq, mask_tag, generator=None):
        model_mean, model_log_variance = self.p_mean_variance(
            item_rep, noise_x_t, t, mask_seq, mask_tag
        )
        if generator is None:
            noise = torch.randn_like(noise_x_t)
        else:
            noise = torch.randn(noise_x_t.shape, generator=generator,
                                device=noise_x_t.device, dtype=noise_x_t.dtype)
        nonzero_mask = (t != 0).float().reshape(-1, 1, 1)
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    def denoise_sample(self, seq, tgt, mask_seq, mask_tag, *, generator=None):
        if generator is None:
            noise_x_t = torch.randn_like(tgt[:, -1:])
        else:
            noise_x_t = torch.randn(tgt[:, -1:].shape, generator=generator,
                                    device=tgt.device, dtype=tgt.dtype)
        for i in reversed(range(self.num_timesteps)):
            t = torch.tensor([i] * seq.shape[0], device=seq.device)
            noise_x_t = self.p_sample(seq, noise_x_t, t, mask_seq, mask_tag,
                                      generator=generator)
        return noise_x_t

    def independent_diffuse(self, tgt, mask, is_independent=False):
        if is_independent:
            t, _weights = self.schedule_sampler.sample(tgt.shape[0] * tgt.shape[1], tgt.device)
            t = t * mask.reshape(-1).long()
            x_t = self.q_sample(
                tgt.reshape(-1, tgt.shape[-1]), t, mask=mask.reshape(-1)
            ).reshape(*tgt.shape)
        else:
            t, _weights = self.schedule_sampler.sample(tgt.shape[0], tgt.device)
            x_t = self.q_sample(tgt, t, mask=mask)
        return x_t, t

    def forward(self, item_rep, item_tag, mask_seq, mask_tag):
        x_t, t = self.independent_diffuse(item_tag, mask_tag, False)
        denoised_seq = self.net(item_rep, x_t, self._scale_timesteps(t), mask_seq, mask_tag)
        losses = F.mse_loss(denoised_seq[:, -1], item_tag[:, -1])
        return denoised_seq, losses
