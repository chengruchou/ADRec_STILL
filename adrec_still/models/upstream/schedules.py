"""Diffusion beta schedules and timestep samplers (upstream ``src/step_sample.py``).

Ported as-is except: the scipy import is deferred into :func:`beta_noise_schedule`
(only the non-default ``beta`` schedule needs it) and the torch.distributed
all-gather helper is omitted — this package runs single-process.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
import torch


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def beta_noise_schedule(timesteps: int, a: float = 1.0, b: float = 1.0) -> np.ndarray:
    """Beta-distribution schedule; needs scipy, imported only when selected."""
    from scipy.stats import beta as beta_distribution

    s = np.linspace(0, 1, timesteps + 1)
    alphas_cumprod = beta_distribution.ppf(1 - s, a, b)
    alpha_schedule = alphas_cumprod[1:] / alphas_cumprod[:-1]
    return 1 - alpha_schedule


class ScheduleSampler(ABC):
    """A distribution over diffusion timesteps (upstream semantics)."""

    @abstractmethod
    def weights(self) -> np.ndarray:
        """Positive per-step weights; need not be normalized."""

    def sample(self, batch_size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, num_timesteps: int) -> None:
        self.num_timesteps = num_timesteps
        self._weights = np.ones([self.num_timesteps])

    def weights(self) -> np.ndarray:
        return self._weights


class LossSecondMomentResampler(ScheduleSampler):
    """Upstream ``lossaware`` sampler (uniform until its history warms up)."""

    def __init__(self, num_timesteps: int, history_per_term: int = 10,
                 uniform_prob: float = 0.001) -> None:
        self.num_timesteps = num_timesteps
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros([num_timesteps, history_per_term], dtype=np.float64)
        self._loss_counts = np.zeros([num_timesteps], dtype=np.int32)

    def weights(self) -> np.ndarray:
        if not self._warmed_up():
            return np.ones([self.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history**2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses) -> None:
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self) -> bool:
        return bool((self._loss_counts == self.history_per_term).all())


class FixSampler(ScheduleSampler):
    def __init__(self, num_timesteps: int) -> None:
        self.num_timesteps = num_timesteps
        self._weights = np.concatenate(
            [np.ones([num_timesteps // 2]), np.zeros([num_timesteps // 2]) + 0.5]
        )

    def weights(self) -> np.ndarray:
        return self._weights


def create_named_schedule_sampler(name: str, num_timesteps: int) -> ScheduleSampler:
    if name == "uniform":
        return UniformSampler(num_timesteps)
    if name == "lossaware":
        return LossSecondMomentResampler(num_timesteps)
    if name == "fixstep":
        return FixSampler(num_timesteps)
    raise NotImplementedError(f"unknown schedule sampler: {name}")


def get_named_beta_schedule(schedule_name: str, num_diffusion_timesteps: int,
                            beta_a: float = 0.3, beta_b: float = 10.0) -> np.ndarray:
    if schedule_name == "linear":
        return linear_beta_schedule(num_diffusion_timesteps).numpy()
    if schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        )
    if schedule_name == "sqrt":
        return betas_for_alpha_bar(num_diffusion_timesteps, lambda t: 1 - np.sqrt(t + 0.0001))
    if schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(
            num_diffusion_timesteps, lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2
        )
    if schedule_name == "trunc_lin":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        if beta_end > 1:
            beta_end = scale * 0.001 + 0.01
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    if schedule_name == "pw_lin":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001
        beta_end = scale * 0.02
        first_part = np.linspace(beta_start, beta_mid, 10, dtype=np.float64)
        second_part = np.linspace(beta_mid, beta_end, num_diffusion_timesteps - 10,
                                  dtype=np.float64)
        return np.concatenate([first_part, second_part])
    if schedule_name == "beta":
        return beta_noise_schedule(num_diffusion_timesteps, a=beta_a, b=beta_b)
    raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999) -> np.ndarray:
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999) -> np.ndarray:
    betas = [min(1 - alpha_bar(0), max_beta)]
    for i in range(num_diffusion_timesteps - 1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def space_timesteps(num_timesteps: int, section_counts) -> set[int]:
    """Upstream DDIM-style timestep spacing (ported verbatim)."""
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps: list[int] = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
        if section_count <= 1:
            frac_stride = 1.0
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)
