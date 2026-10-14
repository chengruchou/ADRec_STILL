"""ADRec core (KDD'25), ported from the fork's ``src/adrec.py``.

Preserved upstream mechanics: the clean causal history encoder (``ag_encoder``),
token-independent diffusion (per-token timesteps, padding anchored to the clean
embedding), direct x0 prediction, the ``rep_item + lambda_uncertainty * (x_t +
time_emb)`` fusion, DDPM posterior sampling, and inference that regenerates only
the last target while teacher-forcing the clean prefix every reverse step.

Port extensions (both inert unless used):

- ``DenoisedModel.forward`` accepts an optional ``context`` tensor that is added
  to the fusion *without* the ``lambda_uncertainty`` factor. The ADRec-STILL
  extension injects branch/time conditioning through it; plain ADRec never
  passes it.
- ``AdRec.denoise_sample`` accepts an optional ``generator`` (reproducible
  evaluation noise) and an optional ``inference_steps``: a shortened reverse
  schedule built by re-spacing ``alphas_cumprod`` over the retained original
  timesteps (the retained original step index is what the timestep embedding
  sees), never by simply dropping the last steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from adrec_still.models.upstream.schedules import (
    create_named_schedule_sampler,
    get_named_beta_schedule,
    space_timesteps,
)
from adrec_still.models.upstream.transformer import SiLU, TransformerEncoder


@dataclass
class AdRecConfig:
    """Upstream ``config.yaml`` keys the diffusion core reads."""

    hidden_size: int = 128
    dropout: float = 0.1
    diffusion_steps: int = 32
    noise_schedule: str = "trunc_lin"
    schedule_sampler_name: str = "uniform"
    lambda_uncertainty: float = 0.001
    rescale_timesteps: bool = True
    independent: bool = True
    cfg_scale: float = 1.0
    geodesic: bool = False
    dif_decoder: str = "att"
    is_causal: bool = True
    beta_a: float = 0.3
    beta_b: float = 10.0


def _extract_into_tensor(arr: np.ndarray, timesteps: torch.Tensor,
                         broadcast_shape) -> torch.Tensor:
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def exponential_mapping(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Sphere geodesic exp map: ``cos(||v||) x + sin(||v||) v/||v||`` (upstream)."""
    norm_v = v.norm(p=2, dim=-1, keepdim=True)
    v_unit = v / (norm_v + 1e-8)
    return torch.cos(norm_v) * x + torch.sin(norm_v) * v_unit


class DenoisedModel(nn.Module):
    """The x0-prediction network (upstream ``DenoisedModel``)."""

    def __init__(self, config: AdRecConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        if config.dif_decoder == "mlp":
            self.decoder: nn.Module = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 4),
                SiLU(),
                nn.Linear(self.hidden_size * 4, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
            )
        else:
            self.decoder = TransformerEncoder(
                hidden_size=self.hidden_size, dropout=config.dropout,
                is_causal=config.is_causal, num_blocks=2, norm_first=False,
            )
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            SiLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )
        self.lambda_uncertainty = config.lambda_uncertainty

    def timestep_embedding(self, timesteps: torch.Tensor, dim: int,
                           max_period: int = 10000) -> torch.Tensor:
        assert dim % 2 == 0
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps.unsqueeze(-1).float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward_cfg(self, c, x, t, mask_seq, mask_tgt, cfg_scale=1.0, context=None):
        cond_eps = self.forward(c, x, t, mask_seq, mask_tgt, context=context)
        uncond_eps = self.forward(c, x, t, mask_seq, mask_tgt, condition=False, context=context)
        return uncond_eps + cfg_scale * (cond_eps - uncond_eps)

    def forward(self, rep_item, x_t, t, mask_seq, mask_tgt, condition=True, context=None):
        if condition is not True:  # CFG unconditional branch
            rep_item = torch.zeros_like(rep_item)
        t = t.reshape(x_t.shape[0], -1)
        time_emb = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        rep_diffu = rep_item + self.lambda_uncertainty * (x_t + time_emb)
        if context is not None:
            # Extension hook: conditioning enters at full scale, deliberately not
            # multiplied by the small lambda_uncertainty factor.
            rep_diffu = rep_diffu + context
        if isinstance(self.decoder, nn.Sequential):
            rep_diffu = self.decoder(rep_diffu)
        else:
            rep_diffu = self.decoder(rep_diffu, mask_seq)
        return rep_diffu


def _posterior_tables(betas: np.ndarray) -> dict[str, np.ndarray]:
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": np.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": np.sqrt(1.0 - alphas_cumprod),
        "posterior_mean_coef1": betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        "posterior_mean_coef2": (1.0 - alphas_cumprod_prev) * np.sqrt(alphas)
        / (1.0 - alphas_cumprod),
        "posterior_variance": betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
    }


class AdRec(nn.Module):
    """Upstream ``AdRec``: encoder + token-independent diffusion + x0 denoiser."""

    def __init__(self, config: AdRecConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.diffusion_steps = config.diffusion_steps
        betas = np.array(
            get_named_beta_schedule(config.noise_schedule, config.diffusion_steps,
                                    beta_a=config.beta_a, beta_b=config.beta_b),
            dtype=np.float64,
        )
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        tables = _posterior_tables(betas)
        self.betas = tables["betas"]
        self.alphas_cumprod = tables["alphas_cumprod"]
        self.sqrt_alphas_cumprod = tables["sqrt_alphas_cumprod"]
        self.sqrt_one_minus_alphas_cumprod = tables["sqrt_one_minus_alphas_cumprod"]
        self.posterior_mean_coef1 = tables["posterior_mean_coef1"]
        self.posterior_mean_coef2 = tables["posterior_mean_coef2"]
        self.posterior_variance = tables["posterior_variance"]
        self.num_timesteps = int(betas.shape[0])
        self.schedule_sampler = create_named_schedule_sampler(
            config.schedule_sampler_name, self.num_timesteps
        )
        self.rescale_timesteps = config.rescale_timesteps
        self.net = DenoisedModel(config)
        self.independent_diffusion = config.independent
        self.cfg_scale = config.cfg_scale
        self.geodesic = config.geodesic
        self.ag_encoder = TransformerEncoder(
            hidden_size=config.hidden_size, dropout=config.dropout,
            is_causal=config.is_causal, num_blocks=2, norm_first=False,
        )

    # ------------------------------------------------------------- forward noise

    def q_sample(self, x_start, t, noise=None, mask=None, generator=None):
        if noise is None:
            if generator is None:
                noise = torch.randn_like(x_start)
            else:
                noise = torch.randn(x_start.shape, generator=generator,
                                    device=x_start.device, dtype=x_start.dtype)
        assert noise.shape == x_start.shape
        if self.geodesic:
            x_start = F.normalize(x_start, p=2, dim=-1)
        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        if self.geodesic:
            x_t = exponential_mapping(x_start, x_t)
        if mask is None:
            return x_t
        mask = torch.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
        return torch.where(mask == 0, x_start, x_t)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_posterior_mean(self, x_start, x_t, t, tables=None):
        coef1 = (tables or self.__dict__)["posterior_mean_coef1"]
        coef2 = (tables or self.__dict__)["posterior_mean_coef2"]
        return (
            _extract_into_tensor(coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(coef2, t, x_t.shape) * x_t
        )

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

    # ----------------------------------------------------------------- training

    def forward(self, item_rep, item_tag, mask_seq, mask_tag, context=None):
        """Encode clean history, noise the targets token-wise, predict x0.

        Returns ``(denoised_seq, per_token_mse, losses)`` where ``losses`` is the
        upstream scalar (per-token MSE weighted by 1/#real-tokens per row, summed
        over positions, meaned over batch and hidden).
        """
        item_rep = self.ag_encoder(item_rep, mask_seq)
        x_t, t = self.independent_diffuse(item_tag, mask_tag, self.independent_diffusion)
        if self.cfg_scale != 1:
            drop = torch.rand([mask_seq.shape[0], 1, 1], device=item_rep.device) > 0.7
            item_rep = torch.where(drop, torch.zeros_like(item_rep), item_rep)
        denoised_seq = self.net(item_rep, x_t, self._scale_timesteps(t), mask_seq, mask_tag,
                                context=context)
        per_token = F.mse_loss(denoised_seq, item_tag, reduction="none")
        weights = (mask_tag / mask_tag.sum(1, keepdim=True).clamp_min(1.0)).unsqueeze(-1)
        losses = (per_token * weights).sum(1).mean()
        return denoised_seq, per_token, losses

    # ---------------------------------------------------------------- sampling

    def sampling_tables(self, inference_steps: int | None):
        """Reverse-schedule tables: the full chain, or a re-spaced short chain.

        Re-spacing keeps ``inference_steps`` of the original timesteps (upstream
        ``space_timesteps`` striding) and rebuilds betas from the retained
        ``alphas_cumprod`` ratios, so the short chain is a consistent diffusion
        process; ``timestep_map[i]`` is the original step index whose embedding
        the denoiser sees at short-chain position ``i``.
        """
        if inference_steps is None or inference_steps >= self.num_timesteps:
            return list(range(self.num_timesteps)), {
                "posterior_mean_coef1": self.posterior_mean_coef1,
                "posterior_mean_coef2": self.posterior_mean_coef2,
                "posterior_variance": self.posterior_variance,
                "betas": self.betas,
            }
        if inference_steps < 2:
            # The DDPM log-variance table reads index 1
            # (np.append(posterior_variance[1], betas[1:])), so a one-step chain
            # has no valid variance entry.
            raise ValueError("inference_steps must be at least 2")
        kept = sorted(space_timesteps(self.num_timesteps, [inference_steps]))
        last_alpha_cumprod = 1.0
        new_betas = []
        for step in kept:
            alpha_cumprod = self.alphas_cumprod[step]
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
        tables = _posterior_tables(np.array(new_betas, dtype=np.float64))
        return kept, {
            "posterior_mean_coef1": tables["posterior_mean_coef1"],
            "posterior_mean_coef2": tables["posterior_mean_coef2"],
            "posterior_variance": tables["posterior_variance"],
            "betas": tables["betas"],
        }

    def _p_sample_step(self, item_rep, noise_x_t, t_short, t_original, mask_seq, mask_tag,
                       tables, generator=None, context=None):
        """One reverse step at short-chain index ``t_short`` (original ``t_original``)."""
        if self.cfg_scale == 1.0:
            x_0 = self.net(item_rep, noise_x_t, self._scale_timesteps(t_original),
                           mask_seq, mask_tag, context=context)
        else:
            x_0 = self.net.forward_cfg(item_rep, noise_x_t, self._scale_timesteps(t_original),
                                       mask_seq, mask_tag, self.cfg_scale, context=context)
        model_log_variance = np.log(
            np.append(tables["posterior_variance"][1], tables["betas"][1:])
        )
        model_log_variance = _extract_into_tensor(model_log_variance, t_short, noise_x_t.shape)
        model_mean = self.q_posterior_mean(x_0, noise_x_t, t_short, tables)
        if generator is None:
            noise = torch.randn_like(noise_x_t)
        else:
            noise = torch.randn(noise_x_t.shape, generator=generator,
                                device=noise_x_t.device, dtype=noise_x_t.dtype)
        nonzero_mask = (t_short != 0).float().unsqueeze(-1)
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        if self.geodesic:
            sample = F.normalize(sample, p=2, dim=-1)
        return sample

    def denoise_sample(self, seq, tgt, mask_seq, mask_tag, *, generator=None,
                       inference_steps=None, context=None, encoded=False):
        """Regenerate the last target embedding, teacher-forcing the clean prefix.

        ``tgt`` carries clean embeddings on the prefix positions (the shifted
        history); its last position is only a shape/noise anchor. Pass
        ``encoded=True`` when ``seq`` is already the ``ag_encoder`` output (lets
        callers reuse one encoding across many samples).
        """
        if not encoded:
            seq = self.ag_encoder(seq, mask_seq)
        timestep_map, tables = self.sampling_tables(inference_steps)
        if generator is None:
            noise_x_t = torch.randn_like(tgt)
        else:
            noise_x_t = torch.randn(tgt.shape, generator=generator, device=tgt.device,
                                    dtype=tgt.dtype)
        batch, length = seq.shape[0], seq.shape[1]
        for short_index in reversed(range(len(timestep_map))):
            t_short = torch.tensor([0] * (length - 1) + [short_index],
                                   device=seq.device).unsqueeze(0).repeat(batch, 1)
            t_original = torch.tensor([0] * (length - 1) + [timestep_map[short_index]],
                                      device=seq.device).unsqueeze(0).repeat(batch, 1)
            noise_x_t = torch.concat([tgt[:, :-1], noise_x_t[:, -1:]], dim=1)
            noise_x_t = self._p_sample_step(seq, noise_x_t, t_short, t_original,
                                            mask_seq, mask_tag, tables,
                                            generator=generator, context=context)
        return noise_x_t
