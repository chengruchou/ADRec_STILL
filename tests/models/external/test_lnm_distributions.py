"""Hand-computed math gates for the per-mark LNM distribution layer (#51)."""

import math

import pytest
import torch

from adrec_still.models.external.lnm_distributions import (
    LOG_SCALE_CLAMP_MAX,
    LOG_SCALE_CLAMP_MIN,
    MarginalResidualDistribution,
    PerMarkJointDistribution,
    clamp_preserve_gradients,
)


def lognormal_pdf(t: float, mu: float, sigma: float) -> float:
    z = (math.log(t) - mu) / sigma
    return math.exp(-0.5 * z * z) / (t * sigma * math.sqrt(2.0 * math.pi))


def lognormal_survival(t: float, mu: float, sigma: float) -> float:
    if t <= 0:
        return 1.0
    z = (math.log(t) - mu) / sigma
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def make_base(
    mean: float = 0.0, std: float = 1.0, batch: int = 1
) -> PerMarkJointDistribution:
    """V=2 marks, C=2 components; includes a negative location on purpose."""
    log_p = torch.log(torch.tensor([[0.25, 0.75]], dtype=torch.float64)).repeat(batch, 1)
    locs = torch.tensor(
        [[[-1.0, 8.0], [7.0, 9.5]]], dtype=torch.float64
    ).repeat(batch, 1, 1)
    log_scales_raw = torch.tensor(
        [[[-0.5, 0.2], [0.0, -1.0]]], dtype=torch.float64
    ).repeat(batch, 1, 1)
    weight_logits = torch.log(
        torch.tensor([[[0.6, 0.4], [0.3, 0.7]]], dtype=torch.float64)
    ).repeat(batch, 1, 1)
    return PerMarkJointDistribution(
        log_mark_probabilities=log_p,
        locs=locs,
        log_scales_raw=log_scales_raw,
        weight_logits=weight_logits,
        log_time_mean=torch.tensor(mean, dtype=torch.float64),
        log_time_std=torch.tensor(std, dtype=torch.float64),
    )


def hand_mixture(base_index: tuple, mean: float, std: float, t: float, kind: str) -> float:
    """Manual per-mark mixture value with the affine mapping applied by hand."""
    locs = [[-1.0, 8.0], [7.0, 9.5]][base_index[0]]
    log_scales = [[-0.5, 0.2], [0.0, -1.0]][base_index[0]]
    weights = [[0.6, 0.4], [0.3, 0.7]][base_index[0]]
    total = 0.0
    for weight, loc, log_scale in zip(weights, locs, log_scales, strict=True):
        native_mu = mean + std * loc
        native_sigma = std * math.exp(log_scale)
        if kind == "density":
            total += weight * lognormal_pdf(t, native_mu, native_sigma)
        else:
            total += weight * lognormal_survival(t, native_mu, native_sigma)
    return total


def test_per_mark_density_and_survival_match_hand_computation():
    base = make_base(mean=1.5, std=0.5)
    t = torch.tensor([7200.0], dtype=torch.float64)
    density = base.per_mark_log_density(t).exp()[0]
    survival = base.per_mark_log_survival(t).exp()[0]
    for mark in range(2):
        assert density[mark].item() == pytest.approx(
            hand_mixture((mark,), 1.5, 0.5, 7200.0, "density"), rel=1e-9
        )
        assert survival[mark].item() == pytest.approx(
            hand_mixture((mark,), 1.5, 0.5, 7200.0, "survival"), rel=1e-9
        )
    marginal = base.log_marginal_density(t).exp().item()
    expected = 0.25 * hand_mixture((0,), 1.5, 0.5, 7200.0, "density") + 0.75 * hand_mixture(
        (1,), 1.5, 0.5, 7200.0, "density"
    )
    assert marginal == pytest.approx(expected, rel=1e-9)
    total_survival = base.log_survival(t).exp().item()
    expected_survival = 0.25 * hand_mixture(
        (0,), 1.5, 0.5, 7200.0, "survival"
    ) + 0.75 * hand_mixture((1,), 1.5, 0.5, 7200.0, "survival")
    assert total_survival == pytest.approx(expected_survival, rel=1e-9)


def test_affine_jacobian_two_parameterizations_agree_in_native_seconds():
    """(mean, std) statistics with rescaled normalized-space parameters must give
    the identical native-seconds density/survival — the change-of-variables
    Jacobian is carried by construction, not by an extra factor."""
    mean, std = math.log(3600.0), 0.5
    with_stats = make_base(mean=mean, std=std)
    # Same native distribution expressed with identity statistics.
    identity = PerMarkJointDistribution(
        log_mark_probabilities=with_stats.log_mark_probabilities,
        locs=mean + std * with_stats.locs,
        log_scales_raw=with_stats.log_scales_raw + math.log(std),
        weight_logits=with_stats.weight_logits,
        log_time_mean=torch.tensor(0.0, dtype=torch.float64),
        log_time_std=torch.tensor(1.0, dtype=torch.float64),
    )
    times = torch.tensor([1800.0], dtype=torch.float64)
    assert torch.allclose(
        with_stats.log_marginal_density(times), identity.log_marginal_density(times),
        atol=1e-12,
    )
    assert torch.allclose(
        with_stats.log_survival(times), identity.log_survival(times), atol=1e-12
    )


def test_straight_through_clamp_forward_and_gradient():
    raw = torch.tensor([-7.0, 0.0, 5.0], dtype=torch.float64, requires_grad=True)
    clamped = clamp_preserve_gradients(raw, LOG_SCALE_CLAMP_MIN, LOG_SCALE_CLAMP_MAX)
    assert clamped.detach().tolist() == [-5.0, 0.0, 3.0]
    clamped.sum().backward()
    # Identity gradient even outside the bounds — a plain clamp would zero these.
    assert raw.grad.tolist() == [1.0, 1.0, 1.0]


def test_out_of_bounds_log_scales_stay_clamped_in_the_density():
    """A raw log-scale of +10 must behave exactly like the bound value 3."""
    base = make_base()
    extreme = PerMarkJointDistribution(
        log_mark_probabilities=base.log_mark_probabilities,
        locs=base.locs,
        log_scales_raw=torch.full_like(base.log_scales_raw, 10.0),
        weight_logits=base.weight_logits,
        log_time_mean=base.log_time_mean,
        log_time_std=base.log_time_std,
    )
    at_bound = PerMarkJointDistribution(
        log_mark_probabilities=base.log_mark_probabilities,
        locs=base.locs,
        log_scales_raw=torch.full_like(base.log_scales_raw, LOG_SCALE_CLAMP_MAX),
        weight_logits=base.weight_logits,
        log_time_mean=base.log_time_mean,
        log_time_std=base.log_time_std,
    )
    times = torch.tensor([3600.0], dtype=torch.float64)
    assert torch.equal(
        extreme.log_marginal_density(times), at_bound.log_marginal_density(times)
    )


def test_catalog_normalization_and_boundary_pins():
    base = make_base(mean=2.0, std=0.7, batch=3)
    s = torch.tensor([0.0, 3600.0, 86400.0], dtype=torch.float64)
    ranking = base.waiting_marginal_log_mark_probabilities(s)
    assert torch.allclose(
        ranking.exp().sum(dim=1), torch.ones(3, dtype=torch.float64), atol=1e-12
    )
    # s == 0 restores p_y exactly (survival pins are exact zeros).
    assert torch.allclose(ranking[0], base.log_mark_probabilities[0], atol=1e-12)
    conditional = base.conditional_log_mark_probabilities(
        torch.tensor([3600.0, 3600.0, 3600.0], dtype=torch.float64)
    )
    assert torch.allclose(
        conditional.exp().sum(dim=1), torch.ones(3, dtype=torch.float64), atol=1e-12
    )
    assert base.log_survival(torch.zeros(3, dtype=torch.float64)).tolist() == [0.0, 0.0, 0.0]
    assert bool(
        (
            base.log_marginal_density(torch.zeros(3, dtype=torch.float64)) == -math.inf
        ).all()
    )


def test_residual_distribution_conditioning_and_successive_identity():
    base = make_base(mean=1.0, std=0.8, batch=1)
    s, a, r = 21600.0, 10800.0, 3600.0
    direct = MarginalResidualDistribution(base, torch.tensor([s + a], dtype=torch.float64))
    staged = MarginalResidualDistribution(base, torch.tensor([s], dtype=torch.float64))
    # q_{s+a}(r) == q_s(a + r) / S_res(a | s)
    left = direct.log_density(torch.tensor([r], dtype=torch.float64))
    right = staged.log_density(
        torch.tensor([a + r], dtype=torch.float64)
    ) - staged.log_survival(torch.tensor([a], dtype=torch.float64))
    assert torch.allclose(left, right, atol=1e-10)
    left_survival = direct.log_survival(torch.tensor([r], dtype=torch.float64))
    right_survival = staged.log_survival(
        torch.tensor([a + r], dtype=torch.float64)
    ) - staged.log_survival(torch.tensor([a], dtype=torch.float64))
    assert torch.allclose(left_survival, right_survival, atol=1e-10)
    # Boundary pins and masking on the conditional object itself.
    assert staged.log_survival(torch.zeros(1, dtype=torch.float64)).item() == 0.0
    assert staged.log_density(torch.zeros(1, dtype=torch.float64)).item() == -math.inf
    # CRPS is not part of this port: the distribution exposes no CRPS surface.
    assert not hasattr(staged, "crps")
    assert not hasattr(staged, "crps_unsupported_reason")


def test_marginal_censor_differs_from_last_mark_censor_on_asymmetric_fixture():
    """The STILL censor survival marginalizes the unknown next mark; selecting one
    mark's survival (the official last-observed-mark end term) is a different
    number on any asymmetric fixture and must not be what the adapter computes."""
    base = make_base(mean=2.0, std=1.0)
    s = torch.tensor([3600.0], dtype=torch.float64)
    c = torch.tensor([7200.0], dtype=torch.float64)
    conditional = MarginalResidualDistribution(base, s)
    marginal_value = conditional.log_survival(c).item()
    per_mark = base.per_mark_log_survival
    expected_marginal = (
        torch.logsumexp(
            base.log_mark_probabilities + per_mark(s + c), dim=1
        )
        - torch.logsumexp(base.log_mark_probabilities + per_mark(s), dim=1)
    ).item()
    assert marginal_value == pytest.approx(expected_marginal, abs=1e-12)
    for mark in range(2):
        last_mark_value = (per_mark(s + c)[0, mark] - per_mark(s)[0, mark]).item()
        assert marginal_value != pytest.approx(last_mark_value, abs=1e-6)


def test_waiting_marginal_ranking_differs_from_time_conditional_mark():
    """Dependence fixture: the mark that survives long waits is not the mark that
    explains a specific observed event time — the two conditionals must differ."""
    base = make_base(mean=2.0, std=1.0)
    s = torch.tensor([86400.0], dtype=torch.float64)
    ranking = base.waiting_marginal_log_mark_probabilities(s)[0]
    conditional = base.conditional_log_mark_probabilities(
        torch.tensor([90000.0], dtype=torch.float64)
    )[0]
    assert not torch.allclose(ranking, conditional, atol=1e-3)


def test_masked_nonpositive_rows_do_not_poison_shared_gradients():
    """A zero-residual row through the marginal density must contribute exactly
    zero gradient to shared parameters, never NaN (the #41 F4 policy)."""
    head = torch.nn.Linear(3, 2 * 2 * 3, dtype=torch.float64)
    features = torch.randn(2, 3, dtype=torch.float64)
    raw = head(features).reshape(2, 2, 2, 3)
    base = PerMarkJointDistribution(
        log_mark_probabilities=torch.log_softmax(
            torch.zeros(2, 2, dtype=torch.float64), dim=1
        ),
        locs=raw[..., 0],
        log_scales_raw=raw[..., 1],
        weight_logits=raw[..., 2],
        log_time_mean=torch.tensor(0.0, dtype=torch.float64),
        log_time_std=torch.tensor(1.0, dtype=torch.float64),
    )
    conditional = MarginalResidualDistribution(base, torch.zeros(2, dtype=torch.float64))
    per_row = -conditional.log_density(torch.tensor([0.0, 2.0], dtype=torch.float64))
    assert per_row[0].item() == math.inf
    per_row[1].backward()
    for parameter in head.parameters():
        assert torch.isfinite(parameter.grad).all()


def test_parameter_validation_errors():
    base = make_base()
    with pytest.raises(ValueError):
        PerMarkJointDistribution(
            log_mark_probabilities=torch.zeros(1, 2, dtype=torch.float64),  # unnormalized
            locs=base.locs,
            log_scales_raw=base.log_scales_raw,
            weight_logits=base.weight_logits,
            log_time_mean=base.log_time_mean,
            log_time_std=base.log_time_std,
        )
    with pytest.raises(ValueError):
        PerMarkJointDistribution(
            log_mark_probabilities=base.log_mark_probabilities,
            locs=base.locs,
            log_scales_raw=base.log_scales_raw,
            weight_logits=base.weight_logits,
            log_time_mean=base.log_time_mean,
            log_time_std=torch.tensor(0.0, dtype=torch.float64),
        )
    with pytest.raises(ValueError):
        MarginalResidualDistribution(base, torch.tensor([-1.0], dtype=torch.float64))
