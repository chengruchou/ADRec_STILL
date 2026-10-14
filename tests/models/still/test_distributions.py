import math

import pytest
import torch

from adrec_still.time.distributions import (
    FixedPiTruncatedResidualDistribution,
    LognormalMixtureParameters,
    MixtureResidualDistribution,
    SurvivalConditionedResidualDistribution,
)


def make_mixture(batch: int = 1) -> LognormalMixtureParameters:
    log_weights = torch.log(torch.tensor([[0.3, 0.7]], dtype=torch.float64)).repeat(batch, 1)
    mu = torch.tensor([[math.log(3600.0), math.log(86400.0)]], dtype=torch.float64).repeat(batch, 1)
    sigma = torch.tensor([[0.8, 1.4]], dtype=torch.float64).repeat(batch, 1)
    return LognormalMixtureParameters(log_weights=log_weights, mu=mu, sigma=sigma)


def lognormal_pdf(t: float, mu: float, sigma: float) -> float:
    z = (math.log(t) - mu) / sigma
    return math.exp(-0.5 * z * z) / (t * sigma * math.sqrt(2.0 * math.pi))


def lognormal_survival(t: float, mu: float, sigma: float) -> float:
    if t <= 0:
        return 1.0
    z = (math.log(t) - mu) / sigma
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def test_density_and_survival_match_hand_computation():
    mixture = make_mixture()
    t = torch.tensor([7200.0])
    expected_density = 0.3 * lognormal_pdf(7200.0, math.log(3600.0), 0.8) + 0.7 * lognormal_pdf(
        7200.0, math.log(86400.0), 1.4
    )
    expected_survival = 0.3 * lognormal_survival(
        7200.0, math.log(3600.0), 0.8
    ) + 0.7 * lognormal_survival(7200.0, math.log(86400.0), 1.4)
    assert mixture.log_density(t).exp().item() == pytest.approx(expected_density, rel=1e-9)
    assert mixture.log_survival(t).exp().item() == pytest.approx(expected_survival, rel=1e-9)


def test_survivor_weights_match_hand_computation():
    mixture = make_mixture()
    s = torch.tensor([43200.0])
    s1 = lognormal_survival(43200.0, math.log(3600.0), 0.8)
    s2 = lognormal_survival(43200.0, math.log(86400.0), 1.4)
    expected = torch.tensor([0.3 * s1, 0.7 * s2], dtype=torch.float64) / (
        0.3 * s1 + 0.7 * s2
    )
    omega = mixture.survivor_log_weights(s).exp()[0]
    assert torch.allclose(omega, expected.to(omega.dtype), rtol=1e-9)
    assert omega.sum().item() == pytest.approx(1.0, abs=1e-12)


def test_density_integrates_to_survival_mass():
    """Numerical integration of f over (s, inf) approximates S(s)."""
    s = 21600.0
    grid = torch.logspace(math.log10(s + 1e-3), 8.5, 20001, dtype=torch.float64)
    expanded = make_mixture(batch=len(grid))
    density = expanded.log_density(grid).exp()
    # trapezoid over the log-spaced grid
    integral = torch.trapezoid(density, grid).item()
    survival = make_mixture().log_survival(torch.tensor([s])).exp().item()
    assert integral == pytest.approx(survival, rel=1e-3)


def test_log_survival_boundaries():
    mixture = make_mixture()
    assert mixture.log_survival(torch.tensor([0.0])).item() == 0.0
    with pytest.raises(ValueError):
        mixture.log_survival(torch.tensor([-1.0]))
    # deep tail stays finite in float64 over the registered support
    far = mixture.log_survival(torch.tensor([1.0e12]))
    assert torch.isfinite(far).all()


def test_log_density_padded_rows_return_neg_inf_without_nan():
    mixture = make_mixture(batch=2)
    values = mixture.log_density(torch.tensor([0.0, 7200.0]))
    assert values[0].item() == -math.inf
    assert torch.isfinite(values[1])


def test_conditioned_residual_density_matches_ratio():
    mixture = make_mixture()
    s = torch.tensor([21600.0])
    conditional = SurvivalConditionedResidualDistribution(mixture, s)
    r = torch.tensor([3600.0])
    expected = mixture.log_density(s + r) - mixture.log_survival(s)
    assert torch.allclose(conditional.log_density(r), expected)
    # conditional remaining-exposure survival is S(s+c)/S(s), not S(c)/S(s)
    c = torch.tensor([7200.0])
    expected_survival = mixture.log_survival(s + c) - mixture.log_survival(s)
    assert torch.allclose(conditional.log_survival(c), expected_survival)
    assert conditional.log_survival(torch.tensor([0.0])).item() == pytest.approx(0.0)


def test_successive_conditioning_identity():
    """Conditioning at s+a equals conditioning at s and then surviving a further a."""
    mixture = make_mixture()
    s = torch.tensor([21600.0])
    a = torch.tensor([10800.0])
    direct = SurvivalConditionedResidualDistribution(mixture, s + a)
    staged = SurvivalConditionedResidualDistribution(mixture, s)
    r = torch.tensor([5000.0])
    # residual density: q_{s+a}(r) == q_s(a + r) / S_res(a | s)
    staged_value = staged.log_density(a + r) - staged.log_survival(a)
    assert torch.allclose(direct.log_density(r), staged_value, atol=1e-10)
    # survival: S_res(c | s+a) == S_res(a + c | s) / S_res(a | s)
    c = torch.tensor([4000.0])
    staged_survival = staged.log_survival(a + c) - staged.log_survival(a)
    assert torch.allclose(direct.log_survival(c), staged_survival, atol=1e-10)
    # survivor weights: omega(s + a) equals survivor-reweighting omega(s) by S_k ratios
    direct_weights = mixture.survivor_log_weights(s + a)
    staged_weights = mixture.survivor_log_weights(s) + (
        mixture.component_log_survival(s + a) - mixture.component_log_survival(s)
    )
    staged_weights = staged_weights - torch.logsumexp(staged_weights, dim=1, keepdim=True)
    assert torch.allclose(direct_weights, staged_weights, atol=1e-10)


def test_zero_elapsed_conditioning_is_identity():
    mixture = make_mixture()
    zero = torch.tensor([0.0])
    conditional = SurvivalConditionedResidualDistribution(mixture, zero)
    r = torch.tensor([7200.0])
    assert torch.allclose(conditional.log_density(r), mixture.log_density(r))
    omega = mixture.survivor_log_weights(zero).exp()[0]
    assert torch.allclose(omega, torch.tensor([0.3, 0.7], dtype=omega.dtype), rtol=1e-9)


def test_single_component_mixture_matches_plain_lognormal():
    parameters = LognormalMixtureParameters(
        log_weights=torch.zeros(1, 1, dtype=torch.float64),
        mu=torch.tensor([[math.log(7200.0)]], dtype=torch.float64),
        sigma=torch.tensor([[1.1]], dtype=torch.float64),
    )
    t = torch.tensor([9000.0])
    assert parameters.log_density(t).exp().item() == pytest.approx(
        lognormal_pdf(9000.0, math.log(7200.0), 1.1), rel=1e-9
    )


def test_fixed_pi_ablation_properties():
    mixture = make_mixture()
    r = torch.tensor([5400.0])
    # At s == 0 the ablation equals the coherent conditional.
    zero = torch.tensor([0.0])
    assert torch.allclose(
        FixedPiTruncatedResidualDistribution(mixture, zero).log_density(r),
        SurvivalConditionedResidualDistribution(mixture, zero).log_density(r),
    )
    # With K == 1 the two coincide at every s.
    single = LognormalMixtureParameters(
        log_weights=torch.zeros(1, 1, dtype=torch.float64),
        mu=torch.tensor([[math.log(7200.0)]], dtype=torch.float64),
        sigma=torch.tensor([[1.0]], dtype=torch.float64),
    )
    s = torch.tensor([21600.0])
    assert torch.allclose(
        FixedPiTruncatedResidualDistribution(single, s).log_density(r),
        SurvivalConditionedResidualDistribution(single, s).log_density(r),
    )
    # Identical component survivals also collapse the difference.
    same_scale = LognormalMixtureParameters(
        log_weights=torch.log(torch.tensor([[0.4, 0.6]], dtype=torch.float64)),
        mu=torch.tensor([[math.log(3600.0), math.log(3600.0)]], dtype=torch.float64),
        sigma=torch.tensor([[0.9, 0.9]], dtype=torch.float64),
    )
    assert torch.allclose(
        FixedPiTruncatedResidualDistribution(same_scale, s).log_density(r),
        SurvivalConditionedResidualDistribution(same_scale, s).log_density(r),
        atol=1e-10,
    )
    # In general the two differ (that difference is the ablation's point)...
    ablation = FixedPiTruncatedResidualDistribution(mixture, s)
    coherent = SurvivalConditionedResidualDistribution(mixture, s)
    assert not torch.allclose(ablation.log_density(r), coherent.log_density(r))
    # ...while the ablation remains normalized per query.
    grid = torch.logspace(-2, 8.5, 20001, dtype=torch.float64)
    total = torch.trapezoid(
        torch.exp(_ablation_density_on_grid(ablation, grid)), grid
    ).item()
    assert total == pytest.approx(1.0, rel=1e-3)


def _ablation_density_on_grid(
    distribution: FixedPiTruncatedResidualDistribution, grid: torch.Tensor
) -> torch.Tensor:
    """Evaluate a batch-size-1 distribution's log density over a 1-D grid."""
    outputs = []
    for start in range(0, len(grid), 4096):
        chunk = grid[start : start + 4096]
        base = distribution.base
        expanded = LognormalMixtureParameters(
            log_weights=base.log_weights.expand(len(chunk), -1),
            mu=base.mu.expand(len(chunk), -1),
            sigma=base.sigma.expand(len(chunk), -1),
        )
        expanded_distribution = FixedPiTruncatedResidualDistribution(
            expanded, distribution.elapsed_sec.expand(len(chunk))
        )
        outputs.append(expanded_distribution.log_density(chunk))
    return torch.cat(outputs)


def test_parameter_validation_errors():
    with pytest.raises(ValueError):
        LognormalMixtureParameters(
            log_weights=torch.zeros(1, 2),  # not normalized
            mu=torch.zeros(1, 2),
            sigma=torch.ones(1, 2),
        ).validate()
    with pytest.raises(ValueError):
        LognormalMixtureParameters(
            log_weights=torch.log(torch.tensor([[0.5, 0.5]])),
            mu=torch.zeros(1, 2),
            sigma=torch.tensor([[1.0, -1.0]]),
        ).validate()
    mixture = make_mixture()
    with pytest.raises(ValueError):
        SurvivalConditionedResidualDistribution(mixture, torch.tensor([-1.0]))


def test_mixture_residual_distribution_interface():
    mixture = make_mixture()
    distribution = MixtureResidualDistribution(mixture)
    assert distribution.batch_size() == 1
    density = distribution.log_density(torch.tensor([3600.0]))
    assert density.shape == (1,)
    assert torch.isfinite(density).all()
    survival = distribution.log_survival(torch.tensor([3600.0]))
    assert survival.shape == (1,)
    assert torch.isfinite(survival).all() and (survival <= 0).all()
    # CRPS is not part of the ported distribution protocol.
    assert not hasattr(distribution, "crps")
    assert not hasattr(distribution, "crps_unsupported_reason")


# The donor's CRPS quadrature tests
# (test_single_component_quadrature_crps_matches_analytic_lognormal,
# test_adaptive_mixture_crps_matches_analytic_for_narrow_and_long_tail,
# test_crps_resolves_a_narrow_distribution_and_still_refuses_what_it_cannot_bound,
# test_internal_crps_does_not_accept_an_interval_its_probes_could_not_verify,
# test_survival_conditioned_crps_agrees_across_tail_mappings,
# test_all_internal_adapters_resolve_the_hidden_transition) are not ported: CRPS and
# its certification machinery were deliberately left out of adrec_still.


# --------------------------------------------------------------- gradient safety (F4)


def make_parameters_from_logits(
    logits: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
) -> LognormalMixtureParameters:
    return LognormalMixtureParameters(
        log_weights=torch.log_softmax(logits, dim=1), mu=mu, sigma=sigma
    )


def test_zero_residual_rows_do_not_nan_mixture_weight_gradients():
    """Regression: an all--inf per-component row before logsumexp turned the masked
    row's mixture-weight gradient into NaN even though the loss only used valid rows."""
    logits = torch.zeros(2, 2, dtype=torch.float64, requires_grad=True)
    parameters = make_parameters_from_logits(
        logits,
        torch.zeros(2, 2, dtype=torch.float64),
        torch.ones(2, 2, dtype=torch.float64),
    )
    values = parameters.log_density(torch.tensor([0.0, 2.0], dtype=torch.float64))
    assert values[0].item() == -math.inf  # forward contract unchanged
    loss = -values[1]
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[0], torch.zeros(2, dtype=torch.float64))
    # Valid-row gradients match a reference computed on the valid row alone.
    reference_logits = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
    reference = make_parameters_from_logits(
        reference_logits,
        torch.zeros(1, 2, dtype=torch.float64),
        torch.ones(1, 2, dtype=torch.float64),
    )
    (-reference.log_density(torch.tensor([2.0], dtype=torch.float64))[0]).backward()
    assert torch.allclose(logits.grad[1], reference_logits.grad[0], atol=1e-12)


def _shared_head_gradients(
    distribution_factory, times: torch.Tensor, valid: torch.Tensor
) -> list[torch.Tensor]:
    """Gradients of a shared parameter head when the loss selects only valid rows."""
    torch.manual_seed(0)
    head = torch.nn.Linear(3, 6, dtype=torch.float64)
    features = torch.randn(times.shape[0], 3, dtype=torch.float64)
    raw = head(features)
    parameters = LognormalMixtureParameters(
        log_weights=torch.log_softmax(raw[:, 0:2], dim=1),
        mu=raw[:, 2:4],
        sigma=torch.nn.functional.softplus(raw[:, 4:6]) + 1e-3,
    )
    distribution = distribution_factory(parameters)
    per_row = -distribution.log_density(times)
    loss = per_row[valid].sum()
    loss.backward()
    return [parameter.grad.clone() for parameter in head.parameters()]


@pytest.mark.parametrize(
    "factory",
    [
        MixtureResidualDistribution,
        lambda p: SurvivalConditionedResidualDistribution(
            p, torch.zeros(3, dtype=torch.float64)
        ),
        lambda p: FixedPiTruncatedResidualDistribution(
            p, torch.zeros(3, dtype=torch.float64)
        ),
    ],
    ids=["direct_mixture", "survival_conditioned_s0", "fixed_pi_truncated_s0"],
)
def test_shared_head_gradients_are_finite_and_match_valid_row_reference(factory):
    """Masked rows share head parameters with valid rows: one NaN row would poison
    every gradient. All three distribution families must stay clean at s == 0."""
    times = torch.tensor([0.0, 5400.0, 0.0], dtype=torch.float64)
    valid = torch.tensor([False, True, False])
    grads = _shared_head_gradients(factory, times, valid)
    for grad in grads:
        assert torch.isfinite(grad).all()

    # Reference: same head/features, but the masked rows are evaluated at a harmless
    # positive stand-in time before the same valid-row selection, so every gradient
    # contribution comes from a finite path. The valid row's contribution must match.
    torch.manual_seed(0)
    head = torch.nn.Linear(3, 6, dtype=torch.float64)
    features = torch.randn(3, 3, dtype=torch.float64)
    raw = head(features)
    parameters = LognormalMixtureParameters(
        log_weights=torch.log_softmax(raw[:, 0:2], dim=1),
        mu=raw[:, 2:4],
        sigma=torch.nn.functional.softplus(raw[:, 4:6]) + 1e-3,
    )
    reference = -factory(parameters).log_density(
        torch.tensor([1.0, 5400.0, 1.0], dtype=torch.float64)
    )
    reference[valid].sum().backward()
    for grad, expected in zip(grads, [p.grad for p in head.parameters()], strict=True):
        assert torch.allclose(grad, expected, atol=1e-9)


def test_all_masked_selection_produces_zero_gradients_not_nan():
    """Select-first with an empty selection: the loss contract skips the update, and a
    zero-contribution backward must leave finite (zero) gradients, never NaN."""
    logits = torch.zeros(2, 2, dtype=torch.float64, requires_grad=True)
    parameters = make_parameters_from_logits(
        logits,
        torch.zeros(2, 2, dtype=torch.float64),
        torch.ones(2, 2, dtype=torch.float64),
    )
    values = parameters.log_density(torch.zeros(2, dtype=torch.float64))
    assert bool((values == -math.inf).all())
    empty = -values[torch.zeros(2, dtype=torch.bool)]
    empty.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_gradient_fix_preserves_positive_row_values_exactly():
    """The safe-term restructuring must not change any positive-time value."""
    mixture = make_mixture(batch=3)
    times = torch.tensor([1.0, 7200.0, 3.0e7], dtype=torch.float64)
    direct = mixture.log_density(times)
    expected = torch.logsumexp(
        mixture.log_weights + mixture.component_log_density(times), dim=1
    )
    assert torch.allclose(direct, expected, atol=0.0)
    conditional = SurvivalConditionedResidualDistribution(
        mixture, torch.tensor([0.0, 21600.0, 0.0], dtype=torch.float64)
    )
    fixed_pi = FixedPiTruncatedResidualDistribution(
        mixture, torch.tensor([0.0, 21600.0, 0.0], dtype=torch.float64)
    )
    for distribution in (conditional, fixed_pi):
        values = distribution.log_density(times)
        assert torch.isfinite(values).all()
    # Non-positive rows still report -inf density and the s == 0 survival pin holds.
    zero_rows = mixture.log_density(torch.zeros(3, dtype=torch.float64))
    assert bool((zero_rows == -math.inf).all())
    assert torch.equal(
        mixture.log_survival(torch.zeros(3, dtype=torch.float64)),
        torch.zeros(3, dtype=torch.float64),
    )


@pytest.mark.parametrize(
    "adapter",
    [
        MixtureResidualDistribution,
        SurvivalConditionedResidualDistribution,
        FixedPiTruncatedResidualDistribution,
    ],
    ids=["direct_mixture", "survival_conditioned", "fixed_pi_truncated"],
)
def test_float32_softmax_weights_never_lift_survival_above_one(adapter):
    """A float32 ``log_softmax`` head is normalized only to about ``1e-7``.

    ``validate`` admits such a row on purpose (``atol=1e-6``), and every mixture head
    in the project emits one: ``log_softmax`` of ``[-3, -4, -5]`` in float32 has a
    float64 ``logsumexp`` of roughly ``+3.8e-8``. Carried unchanged into the mixture
    reduction, that slack becomes the answer wherever every component still survives:
    ``log S(t)`` comes out positive, a survival above one.

    Renormalizing the weights in float64 is what keeps this a float64 rounding
    question. Survival must stay at or below one. (The donor also asserted that CRPS
    resolves on this row; CRPS is not part of this port.)
    """
    log_weights = torch.log_softmax(
        torch.tensor([[-3.0, -4.0, -5.0]], dtype=torch.float32), dim=1
    )
    slack = float(torch.logsumexp(log_weights.to(torch.float64), dim=1))
    assert slack > 64.0 * torch.finfo(torch.float64).eps, (
        "this row no longer carries float32 normalization slack, so it no longer "
        "exercises the regression"
    )
    parameters = LognormalMixtureParameters(
        log_weights=log_weights,
        mu=torch.tensor(
            [[math.log(3600.0), math.log(86400.0), math.log(604800.0)]],
            dtype=torch.float32,
        ),
        sigma=torch.tensor([[0.8, 1.2, 1.5]], dtype=torch.float32),
    )
    parameters.validate()
    distribution = (
        adapter(parameters)
        if adapter is MixtureResidualDistribution
        else adapter(parameters, torch.zeros(1, dtype=torch.float64))
    )
    # Every component still survives here, so the mixture reduction is where the
    # slack would otherwise surface undiluted.
    early = torch.tensor([12.0], dtype=torch.float64)
    assert float(parameters.log_survival(early)) <= 0.0
    assert float(distribution.log_survival(early)) <= 0.0


def test_renormalizing_weights_only_removes_normalization_slack():
    """Renormalizing moves a float64 row by roundoff, and mu/sigma not at all.

    The correction any row receives is exactly its own ``logsumexp``, so a mixture
    that is already normalized in float64 keeps its weights to the last ulp and the
    hand-computed expectations elsewhere in this file stay exact.
    """
    mixture = make_mixture()
    log_weights, mu, sigma = mixture._float64()
    slack = torch.logsumexp(mixture.log_weights, dim=1, keepdim=True)
    assert float(slack.abs().max()) < 1.0e-15
    assert torch.equal(log_weights, mixture.log_weights - slack)
    assert torch.equal(mu, mixture.mu) and torch.equal(sigma, mixture.sigma)
    assert float(torch.logsumexp(log_weights, dim=1).abs().max()) <= (
        torch.finfo(torch.float64).eps
    )
