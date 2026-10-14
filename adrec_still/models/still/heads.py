"""Shared head components for the #41 model arms.

Both arms compose the accepted shared encoder (``GRU4RecH.encode`` with its tied item
table) instead of copying it; each training run owns its own encoder instance so two
separately trained arms never share updating parameters. The mixture head keeps positive
scales through softplus and normalized weights through ``log_softmax``; its biases start
in a sane seconds range so a bounded smoke run does not spend its budget escaping an
absurd initialization.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from adrec_still.time.distributions import LognormalMixtureParameters

#: Initial mixture location: about one day, in log-seconds.
_INITIAL_LOG_SECONDS = math.log(86400.0)
#: Raw softplus input that yields an initial sigma close to 1.5.
_INITIAL_SIGMA_RAW = math.log(math.expm1(1.5))
_MIN_SIGMA = 1.0e-3


class FeatureFusion(nn.Module):
    """Two-layer MLP fusing conditioning features into an encoder context (residual add)."""

    def __init__(self, feature_dim: int, context_dim: int) -> None:
        super().__init__()
        if feature_dim <= 0 or context_dim <= 0:
            raise ValueError("feature_dim and context_dim must be positive")
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, context_dim),
            nn.Tanh(),
            nn.Linear(context_dim, context_dim),
        )

    def forward(self, context: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return context + self.projection(features.to(context.dtype))


class LognormalMixtureHead(nn.Module):
    """Predict normalized ``[batch, K]`` mixture parameters from a context vector."""

    def __init__(self, context_dim: int, mixture_components: int) -> None:
        super().__init__()
        if mixture_components <= 0:
            raise ValueError("mixture_components must be positive")
        self.mixture_components = int(mixture_components)
        self.weight_head = nn.Linear(context_dim, self.mixture_components)
        self.mu_head = nn.Linear(context_dim, self.mixture_components)
        self.sigma_head = nn.Linear(context_dim, self.mixture_components)
        with torch.no_grad():
            self.mu_head.bias.fill_(_INITIAL_LOG_SECONDS)
            # Spread the components slightly so K > 1 does not start collapsed.
            spread = torch.linspace(-0.5, 0.5, self.mixture_components)
            self.mu_head.bias.add_(spread)
            self.sigma_head.bias.fill_(_INITIAL_SIGMA_RAW)

    def forward(self, context: torch.Tensor) -> LognormalMixtureParameters:
        log_weights = torch.log_softmax(self.weight_head(context), dim=1)
        mu = self.mu_head(context)
        sigma = torch.nn.functional.softplus(self.sigma_head(context)) + _MIN_SIGMA
        return LognormalMixtureParameters(log_weights=log_weights, mu=mu, sigma=sigma)


def masked_component_log_probabilities(
    component_logits: torch.Tensor, masked_log_probabilities_fn
) -> torch.Tensor:
    """Normalize ``[batch, K, catalog]`` component logits with the shared reserved-ID
    policy, one softmax per component over the exact frozen catalog."""
    batch, components, catalog = component_logits.shape
    flat = component_logits.reshape(batch * components, catalog)
    return masked_log_probabilities_fn(flat).reshape(batch, components, catalog)
