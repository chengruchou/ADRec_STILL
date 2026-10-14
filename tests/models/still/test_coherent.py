import math

import pytest
import torch

from adrec_still.models.still.coherent import (
    VARIANT_FIXED_PI_TRUNCATED,
    VARIANT_SURVIVOR_REWEIGHTED,
    StillCoherentJoint,
)
from adrec_still.time.distributions import LognormalMixtureParameters

NUMBER_OF_POIS = 20


def make_model(variant: str = VARIANT_SURVIVOR_REWEIGHTED, components: int = 2):
    torch.manual_seed(7)
    model = StillCoherentJoint(
        NUMBER_OF_POIS,
        embedding_dim=16,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        mixture_components=components,
        variant=variant,
    )
    model.is_fitted = True
    model.eval()
    return model


def make_batch(query_s: list[int] | None = None) -> dict[str, torch.Tensor]:
    query = torch.tensor(query_s if query_s is not None else [0, 21600, 43200])
    batch_size = len(query)
    history = torch.tensor([[0, 3, 4, 5], [0, 0, 6, 7], [2, 8, 9, 10]])[:batch_size]
    mask = history.ne(0)
    timestamps = torch.where(mask, 1_600_000_000 + torch.arange(4) * 3600, 0)[:batch_size]
    last = timestamps.max(dim=1).values
    return {
        "user_id": torch.arange(1, batch_size + 1),
        "history_poi": history,
        "history_timestamp": timestamps,
        "history_mask": mask,
        "last_timestamp": last,
        "query_s_sec": query,
        "event_observed": torch.ones(batch_size, dtype=torch.bool),
        "observed_mask": torch.ones(batch_size, dtype=torch.bool),
        "censored_mask": torch.zeros(batch_size, dtype=torch.bool),
        "target_in_train_catalog": torch.ones(batch_size, dtype=torch.bool),
        "next_poi": torch.tensor([5, 7, 9])[:batch_size],
        "residual_time_sec": torch.tensor([3600.0, 7200.0, 10800.0])[:batch_size],
        "censor_time_sec": torch.zeros(batch_size, dtype=torch.float64),
    }


def test_predict_shapes_normalization_and_determinism():
    model = make_model()
    batch = make_batch()
    prediction = model.predict(batch)
    prediction.validate(NUMBER_OF_POIS)
    probabilities = prediction.poi_logits.to(torch.float64).exp().sum(dim=1)
    assert torch.allclose(probabilities, torch.ones_like(probabilities), atol=1e-4)
    again = model.predict(batch)
    assert torch.equal(prediction.poi_logits, again.poi_logits)
    assert prediction.time_distribution is not None
    assert prediction.time_prediction is None


def test_base_distribution_ignores_the_query():
    """The fixed base may read history/anchor metadata only; waiting must act purely
    through survival conditioning of that base."""
    model = make_model()
    early = make_batch([0, 0, 0])
    late = make_batch([43200, 43200, 43200])
    with torch.no_grad():
        base_early = model.base_parameters(model.base_context(early))
        base_late = model.base_parameters(model.base_context(late))
    assert torch.equal(base_early.mu, base_late.mu)
    assert torch.equal(base_early.sigma, base_late.sigma)
    assert torch.equal(base_early.log_weights, base_late.log_weights)


def test_prediction_never_reads_hidden_future_targets():
    model = make_model()
    batch = make_batch()
    baseline = model.predict(batch)
    tampered = dict(batch)
    tampered["next_poi"] = torch.tensor([11, 12, 13])
    tampered["residual_time_sec"] = torch.tensor([1.0, 2.0, 3.0])
    changed = model.predict(tampered)
    assert torch.equal(baseline.poi_logits, changed.poi_logits)


def test_marginal_matches_hand_combination():
    model = make_model()
    batch = make_batch()
    with torch.no_grad():
        context = model.base_context(batch)
        base = model.base_parameters(context)
        component_log_probabilities = model.component_mark_log_probabilities(context)
        elapsed = batch["query_s_sec"].to(torch.float64)
        omega = base.survivor_log_weights(elapsed)
        expected = torch.logsumexp(
            omega.unsqueeze(2) + component_log_probabilities.to(torch.float64), dim=1
        )
    prediction = model.predict(batch)
    assert torch.allclose(
        prediction.poi_logits.to(torch.float64), expected, atol=1e-5
    )


def test_model_level_successive_conditioning_identity():
    """Predicting at s+a equals conditioning the s-query state a further a — for the
    mark marginal and the residual distribution — via the real prediction path."""
    model = make_model()
    s, a = 21600, 21600
    at_direct = model.predict(make_batch([s + a] * 3))
    at_staged = model.predict(make_batch([s] * 3))
    staged_distribution = at_staged.time_distribution
    direct_distribution = at_direct.time_distribution
    r = torch.tensor([5000.0, 6000.0, 7000.0], dtype=torch.float64)
    a64 = torch.full((3,), float(a), dtype=torch.float64)
    staged_density = staged_distribution.log_density(a64 + r) - staged_distribution.log_survival(
        a64
    )
    assert torch.allclose(direct_distribution.log_density(r), staged_density, atol=1e-8)
    # Mark marginal at s+a from the s-query state: reweight the s-survivor weights by the
    # additional component survival ratio, then recombine the fixed component marks.
    with torch.no_grad():
        context = model.base_context(make_batch([s] * 3))
        base = model.base_parameters(context)
        marks = model.component_mark_log_probabilities(context).to(torch.float64)
        s64 = torch.full((3,), float(s), dtype=torch.float64)
        staged_weights = base.survivor_log_weights(s64) + (
            base.component_log_survival(s64 + a64) - base.component_log_survival(s64)
        )
        staged_weights = staged_weights - torch.logsumexp(staged_weights, dim=1, keepdim=True)
        expected_marginal = torch.logsumexp(staged_weights.unsqueeze(2) + marks, dim=1)
    assert torch.allclose(
        at_direct.poi_logits.to(torch.float64), expected_marginal, atol=1e-5
    )


def test_negative_fixture_query_dependent_base_fails_the_consistency_check():
    """A decoder that regenerates its base from the query breaks the successive-
    conditioning identity; the check must detect it (checking the survival ratio alone
    would not)."""
    model = make_model()

    original_base_context = model.base_context

    def query_dependent_base_context(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        context = original_base_context(batch)
        # Illegitimate: inject the query into the "fixed" base.
        return context + 0.05 * batch["query_s_sec"].to(context.dtype).unsqueeze(1) / 3600.0

    model.base_context = query_dependent_base_context
    s, a = 21600, 21600
    direct = model.predict(make_batch([s + a] * 3))
    staged_state = model.predict(make_batch([s] * 3))
    r = torch.tensor([5000.0, 6000.0, 7000.0], dtype=torch.float64)
    a64 = torch.full((3,), float(a), dtype=torch.float64)
    staged_density = staged_state.time_distribution.log_density(
        a64 + r
    ) - staged_state.time_distribution.log_survival(a64)
    assert not torch.allclose(
        direct.time_distribution.log_density(r), staged_density, atol=1e-6
    )


def test_score_joint_decomposition_and_hand_value():
    model = make_model()
    batch = make_batch()
    with torch.no_grad():
        scores = model.score_joint(batch)
    eligible = scores["eligible"]
    assert bool(eligible.all())
    recomposed = scores["log_time"] + scores["log_mark_given_time"]
    assert torch.allclose(scores["log_joint"][eligible], recomposed[eligible], atol=1e-9)
    # Hand recomputation for row 0.
    with torch.no_grad():
        context = model.base_context(batch)
        base = model.base_parameters(context)
        marks = model.component_mark_log_probabilities(context).to(torch.float64)
    s = batch["query_s_sec"].to(torch.float64)
    r = batch["residual_time_sec"].to(torch.float64)
    components = base.log_weights.to(torch.float64) + base.component_log_density(s + r)
    target_terms = marks.gather(
        2, batch["next_poi"].view(-1, 1, 1).expand(-1, 2, 1)
    )[:, :, 0]
    expected_joint = torch.logsumexp(components + target_terms, dim=1) - base.log_survival(s)
    assert torch.allclose(scores["log_joint"], expected_joint, atol=1e-9)
    # The mark term conditioned on the observed r generally differs from the ranking
    # marginal (which integrates the unknown future residual out).
    prediction = model.predict(batch)
    marginal_at_target = prediction.poi_logits.to(torch.float64).gather(
        1, batch["next_poi"].view(-1, 1)
    )[:, 0]
    assert not torch.allclose(scores["log_mark_given_time"], marginal_at_target, atol=1e-6)


def test_score_joint_discards_censored_rows_safely():
    model = make_model()
    batch = make_batch()
    batch["observed_mask"] = torch.tensor([True, False, True])
    batch["event_observed"] = batch["observed_mask"]
    batch["censored_mask"] = ~batch["observed_mask"]
    batch["target_in_train_catalog"] = torch.tensor([True, False, True])
    batch["residual_time_sec"] = torch.tensor([3600.0, 0.0, 10800.0])  # masked fill 0
    with torch.no_grad():
        scores = model.score_joint(batch)
    assert scores["eligible"].tolist() == [True, False, True]
    assert torch.isfinite(scores["log_joint"][scores["eligible"]]).all()


def test_ablation_variant_properties():
    trained = make_model(VARIANT_SURVIVOR_REWEIGHTED)
    ablation = make_model(VARIANT_FIXED_PI_TRUNCATED)
    ablation.load_state_dict(trained.state_dict())  # same checkpoint, different variant

    # At s == 0 the ablation and the coherent model coincide.
    zero = make_batch([0, 0, 0])
    assert torch.allclose(
        trained.predict(zero).poi_logits, ablation.predict(zero).poi_logits, atol=1e-6
    )
    # The ablation's POI marginal ignores waiting entirely (fixed pi)...
    late = make_batch([43200, 43200, 43200])
    assert torch.allclose(
        ablation.predict(zero).poi_logits, ablation.predict(late).poi_logits, atol=1e-6
    )
    # ...while the coherent marginal generally moves with s.
    assert not torch.allclose(
        trained.predict(zero).poi_logits, trained.predict(late).poi_logits, atol=1e-6
    )
    # Identical component mark heads make the marginal variant-invariant.
    collapsed = make_model(VARIANT_SURVIVOR_REWEIGHTED)
    with torch.no_grad():
        weight = collapsed.component_projection.weight
        width = collapsed.context_dim
        weight[width:] = weight[:width]
        collapsed.component_projection.bias[width:] = collapsed.component_projection.bias[:width]
    collapsed_ablation = make_model(VARIANT_FIXED_PI_TRUNCATED)
    collapsed_ablation.load_state_dict(collapsed.state_dict())
    assert torch.allclose(
        collapsed.predict(late).poi_logits,
        collapsed_ablation.predict(late).poi_logits,
        atol=1e-6,
    )
    # Training the ablation as its own model is not a registered configuration.
    with pytest.raises(RuntimeError):
        ablation.per_row_sampled_training_losses(
            make_batch(), negative_samples=3, generator=torch.Generator().manual_seed(0)
        )


def test_component_diagnostics_reported():
    model = make_model()
    diagnostics = model.component_diagnostics(make_batch())
    assert diagnostics["mixture_components"] == 2.0
    assert 0.5 <= diagnostics["mean_max_component_weight"] <= 1.0
    assert 1.0 <= diagnostics["mean_effective_components"] <= 2.0


def test_checkpoint_reload_and_variant_override(tmp_path):
    from adrec_still.models.still.registry import load_still_checkpoint
    from adrec_still.training.checkpoint import save_weights

    model = make_model()
    path = tmp_path / "best.pt"
    save_weights(path, model, family="still")
    reloaded = load_still_checkpoint(path)
    reloaded.eval()
    batch = make_batch()
    assert torch.equal(model.predict(batch).poi_logits, reloaded.predict(batch).poi_logits)
    ablation = load_still_checkpoint(path, variant_override=VARIANT_FIXED_PI_TRUNCATED)
    ablation.eval()
    assert ablation.variant == VARIANT_FIXED_PI_TRUNCATED
    zero = make_batch([0, 0, 0])
    assert torch.allclose(
        model.predict(zero).poi_logits, ablation.predict(zero).poi_logits, atol=1e-6
    )


def test_zero_observed_residual_is_rejected_by_scoring_contract():
    """Timestamp resolution contract: observed residual times are strictly positive;
    a zero is a data error, never silently bumped to a positive value."""
    model = make_model()
    kept = make_batch()
    kept["residual_time_sec"] = torch.tensor([0.0, 7200.0, 10800.0])
    with pytest.raises(ValueError):
        model.per_row_sampled_training_losses(
            kept, negative_samples=3, generator=torch.Generator().manual_seed(0)
        )


def test_resolved_config_discloses_capacity():
    model = make_model()
    config = model.resolved_config()
    assert config["variant"] == VARIANT_SURVIVOR_REWEIGHTED
    assert config["mixture_components"] == 2
    assert config["parameter_count"] == (
        config["encoder_parameter_count"] + config["head_parameter_count"]
    )
    assert math.isfinite(config["parameter_count"])


#: A deliberately generous stand-in for the normalization slack
#: ``LognormalMixtureParameters.validate`` admits on purpose: it sits just inside that
#: ``atol=1e-6`` gate, and is about an order of magnitude above the ~4e-8 this project's
#: own float32 ``log_softmax`` head delivers (#91). Chosen to be larger than anything the
#: head can produce, so a term that tracks the slack cannot hide inside float64 roundoff.
ADMISSIBLE_SLACK = 5e-7


def give_the_head_admissible_slack(model, slack: float = ADMISSIBLE_SLACK):
    """Offset every log weight by ``slack``, so the row's logsumexp is ``slack``.

    This is the head every mixture model in the project has: ``validate`` accepts it
    (``atol=1e-6``) and a float32 ``log_softmax`` produces it. Nothing the model
    computes may depend on it.
    """
    emit = model.base_parameters

    def shifted(base_context):
        base = emit(base_context)
        return LognormalMixtureParameters(
            log_weights=base.log_weights.to(torch.float64) + slack,
            mu=base.mu.to(torch.float64),
            sigma=base.sigma.to(torch.float64),
        )

    model.base_parameters = shifted
    return model


@pytest.mark.parametrize("variant", [VARIANT_SURVIVOR_REWEIGHTED, VARIANT_FIXED_PI_TRUNCATED])
def test_admissible_weight_slack_never_reaches_the_ranking_marginal(variant):
    """The marginal mixes the weights with the per-component mark probabilities.

    Reading them raw left the slack in every ``log p_s(y)``, so the reported ranking
    marginal was a few ``1e-7`` away from a normalized log-probability.
    """
    batch = make_batch()
    values = []
    for slack in (None, ADMISSIBLE_SLACK):
        model = make_model(variant)
        if slack is not None:
            give_the_head_admissible_slack(model, slack)
        context = model.base_context(batch)
        base = model.base_parameters(context)
        elapsed = batch["query_s_sec"].to(torch.float64)
        values.append(
            model.marginal_poi_log_probabilities(
                base, model.component_mark_log_probabilities(context), elapsed
            )
        )
    assert torch.allclose(values[0], values[1], rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("variant", [VARIANT_SURVIVOR_REWEIGHTED, VARIANT_FIXED_PI_TRUNCATED])
def test_admissible_weight_slack_never_reaches_the_scored_joint_terms(variant):
    """``score_joint`` took its numerator from the raw weights, so the slack survived.

    Both variants shifted ``log_time`` and ``log_joint`` by the row's slack, for
    different reasons. Under ``VARIANT_SURVIVOR_REWEIGHTED`` the numerator read the raw
    weights while the normalizer came from ``log_survival``, which #92 renormalizes, so
    the slack stopped cancelling between them. Under ``VARIANT_FIXED_PI_TRUNCATED`` there
    is no ``log_survival`` call at all -- the normalizer is zero -- so there was never
    anything for it to cancel against.

    The same shift reaches the observed-row training losses. It does **not** reach the
    decomposition: see the identity assertion below.
    """
    batch = make_batch()
    scored = []
    for slack in (None, ADMISSIBLE_SLACK):
        model = make_model(variant)
        if slack is not None:
            give_the_head_admissible_slack(model, slack)
        scored.append(model.score_joint(batch))
    for key in ("log_time", "log_joint", "log_mark_given_time"):
        assert torch.allclose(scored[0][key], scored[1][key], rtol=0.0, atol=1e-12), key
    # The decomposition is shift-invariant by construction and so was never a symptom
    # here: log_gamma subtracts logsumexp(components), which removes any constant the
    # weights carry, and the same constant enters log_time and log_joint identically.
    # This assertion therefore also holds on the unfixed code -- measured at 1.8e-15
    # there, against the 1e-7 tolerance of the two gates that police the identity
    # (evaluation/time.py and campaign/export_runner.py), so neither gate could ever
    # have fired on this defect. On unfixed code the per-term assertions above are what
    # fail, never this one. It is here to keep the invariant pinned, not as evidence
    # for the fix.
    for outcome in scored:
        eligible = outcome["eligible"]
        residual = (
            outcome["log_joint"] - (outcome["log_time"] + outcome["log_mark_given_time"])
        )[eligible]
        assert torch.allclose(
            residual, torch.zeros_like(residual), rtol=0.0, atol=1e-12
        ), "log_joint == log_time + log_mark_given_time"


@pytest.mark.parametrize(
    "losses", ["per_row_sampled_training_losses", "per_row_survival_training_losses"]
)
def test_admissible_weight_slack_never_reaches_the_training_losses(losses):
    """Same mismatch inside training: the observed-row time term mixed raw weights
    against a renormalized ``log S(s)``, so the reported loss moved with a slack that
    is a property of the head's dtype rather than of the fit."""
    batch = make_batch()
    values = []
    for slack in (None, ADMISSIBLE_SLACK):
        model = make_model(VARIANT_SURVIVOR_REWEIGHTED)
        if slack is not None:
            give_the_head_admissible_slack(model, slack)
        values.append(
            getattr(model, losses)(batch, 4, torch.Generator().manual_seed(11))
        )
    assert torch.allclose(values[0], values[1], rtol=0.0, atol=1e-12)
