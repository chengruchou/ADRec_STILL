import math

import pytest
import torch

from adrec_still.models.still.coherent import StillCoherentJoint
from adrec_still.models.still.independent import StillIndependent
from adrec_still.models.still.losses import (
    SURVIVAL_TRAINING_PROFILE,
    censored_exposure_nll,
    observed_time_nll,
)
from adrec_still.time.distributions import (
    LognormalMixtureParameters,
    SurvivalConditionedResidualDistribution,
)


def make_coherent_batch(query_s: list[int] | None = None) -> dict[str, torch.Tensor]:
    """Local copy of the test_coherent batch fixture; cross-test-module imports break
    under CI's bare ``pytest`` because tests/ is not an importable package."""
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


def make_conditional(s_value: float) -> SurvivalConditionedResidualDistribution:
    base = LognormalMixtureParameters(
        log_weights=torch.log(torch.tensor([[0.4, 0.6]], dtype=torch.float64)),
        mu=torch.tensor([[math.log(3600.0), math.log(86400.0)]], dtype=torch.float64),
        sigma=torch.tensor([[0.9, 1.2]], dtype=torch.float64),
    )
    return SurvivalConditionedResidualDistribution(
        base, torch.tensor([s_value], dtype=torch.float64)
    )


def test_censored_contribution_uses_remaining_exposure_from_the_query():
    """censor_time_sec = c is remaining exposure from the query: the coherent censored
    contribution equals -(log S(s+c) - log S(s)), never -(log S(c) - log S(s))."""
    s, c = 21600.0, 7200.0
    conditional = make_conditional(s)
    base = conditional.base
    value = censored_exposure_nll(conditional, torch.tensor([c], dtype=torch.float64))
    expected = -(
        base.log_survival(torch.tensor([s + c], dtype=torch.float64))
        - base.log_survival(torch.tensor([s], dtype=torch.float64))
    )
    assert torch.allclose(value, expected, atol=1e-12)
    wrong = -(
        base.log_survival(torch.tensor([c], dtype=torch.float64))
        - base.log_survival(torch.tensor([s], dtype=torch.float64))
    )
    assert not torch.allclose(value, wrong)


def test_zero_followup_censor_contributes_exactly_zero():
    conditional = make_conditional(21600.0)
    value = censored_exposure_nll(conditional, torch.zeros(1, dtype=torch.float64))
    assert value.item() == 0.0


def test_observed_time_nll_rejects_nonpositive_residuals():
    conditional = make_conditional(0.0)
    with pytest.raises(ValueError):
        observed_time_nll(conditional, torch.zeros(1, dtype=torch.float64))


def test_training_hooks_reject_censored_rows_until_39_profile():
    torch.manual_seed(3)
    model = StillCoherentJoint(
        20, embedding_dim=16, hidden_dim=16, num_layers=1, dropout=0.0
    )
    kept = make_coherent_batch()
    kept["observed_mask"] = torch.tensor([True, False, True])
    with pytest.raises(ValueError):
        model.per_row_sampled_training_losses(
            kept, negative_samples=3, generator=torch.Generator().manual_seed(0)
        )


@pytest.mark.parametrize("model_class", [StillIndependent, StillCoherentJoint])
def test_training_losses_are_finite_with_finite_gradients(model_class):
    """Masked fill values never reach a log through an unselected branch: the whole
    loss/backward path stays finite for both arms."""
    torch.manual_seed(5)
    model = model_class(20, embedding_dim=16, hidden_dim=16, num_layers=1, dropout=0.0)
    kept = make_coherent_batch()
    per_row = model.per_row_sampled_training_losses(
        kept, negative_samples=4, generator=torch.Generator().manual_seed(1)
    )
    assert per_row.shape == kept["next_poi"].shape
    assert torch.isfinite(per_row).all()
    per_row.sum().backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


def test_shared_weighting_composes_with_the_model_losses():
    from adrec_still.data.weighting import landmark_weights
    from adrec_still.models.baselines.trainer import weighted_mean_loss

    torch.manual_seed(5)
    model = StillIndependent(20, embedding_dim=16, hidden_dim=16, num_layers=1, dropout=0.0)
    kept = make_coherent_batch()
    per_row = model.per_row_sampled_training_losses(
        kept, negative_samples=4, generator=torch.Generator().manual_seed(1)
    )
    weights = landmark_weights(kept, mode="per_gap").to(per_row.dtype)
    loss = weighted_mean_loss(per_row, weights)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("model_class", [StillIndependent, StillCoherentJoint])
def test_survival_profile_three_way_losses_and_backward(model_class):
    from adrec_still.models.baselines.trainer import training_step

    torch.manual_seed(7)
    model = model_class(20, embedding_dim=16, hidden_dim=16, num_layers=1, dropout=0.0)
    batch = make_coherent_batch()
    batch["observed_mask"] = torch.tensor([True, True, False])
    batch["event_observed"] = batch["observed_mask"].clone()
    batch["censored_mask"] = ~batch["observed_mask"]
    batch["target_in_train_catalog"] = torch.tensor([True, False, False])
    batch["next_poi"] = torch.tensor([5, 1, 0])
    batch["residual_time_sec"] = torch.tensor([3600.0, 7200.0, 0.0])
    batch["censor_time_sec"] = torch.tensor([0.0, 0.0, 5400.0])
    step = training_step(
        model,
        batch,
        {
            "training_profile": SURVIVAL_TRAINING_PROFILE,
            "landmark_weighting": "per_gap",
            "negative_samples": 4,
        },
        torch.Generator().manual_seed(1),
        torch.device("cpu"),
    )
    assert step["used_rows"] == 3
    assert step["excluded_rows"] == 0
    assert step["contribution_counts"] == {
        "observed_known": 1,
        "observed_unknown": 1,
        "censored_positive_followup": 1,
    }
    assert torch.isfinite(step["per_row_losses"]).all()
    step["loss"].backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


@pytest.mark.parametrize("model_class", [StillIndependent, StillCoherentJoint])
def test_shared_time_joint_adapter_masks_and_decomposition(model_class):
    from adrec_still.time.protocol import score_time_joint_rows

    torch.manual_seed(9)
    model = model_class(20, embedding_dim=16, hidden_dim=16, num_layers=1, dropout=0.0)
    model.is_fitted = True
    batch = make_coherent_batch()
    batch["observed_mask"] = torch.tensor([True, True, False])
    batch["event_observed"] = batch["observed_mask"].clone()
    batch["censored_mask"] = ~batch["observed_mask"]
    batch["target_in_train_catalog"] = torch.tensor([True, False, False])
    batch["next_poi"] = torch.tensor([5, 1, 0])
    batch["residual_time_sec"] = torch.tensor([3600.0, 7200.0, 0.0])
    batch["censor_time_sec"] = torch.tensor([0.0, 0.0, 0.0])
    scores = score_time_joint_rows(model, batch)
    assert scores["time_eligible"].tolist() == [True, True, False]
    assert scores["joint_eligible"].tolist() == [True, False, False]
    assert torch.isfinite(scores["time_nll"]).all()
    assert scores["time_nll"][2].item() == 0.0
    # CRPS keys ("observed_crps"/"crps_eligible") were removed from the adapter.
    assert not any("crps" in key for key in scores)
    torch.testing.assert_close(
        scores["joint_nll"][scores["joint_eligible"]],
        (
            scores["joint_time_nll"] + scores["conditional_mark_nll"]
        )[scores["joint_eligible"]],
    )


@pytest.mark.parametrize("model_class", [StillIndependent, StillCoherentJoint])
def test_exact_marginal_callback_matches_dense_prediction(model_class):
    torch.manual_seed(11)
    model = model_class(20, embedding_dim=16, hidden_dim=16, num_layers=1, dropout=0.0)
    model.is_fitted = True
    model.eval()
    batch = make_coherent_batch()
    features = {
        key: batch[key]
        for key in (
            "user_id",
            "history_poi",
            "history_timestamp",
            "history_mask",
            "last_timestamp",
            "query_s_sec",
        )
    }
    callback = model.exact_candidate_factory(chunk_size=3)(features)
    ids = torch.arange(2, model.number_of_pois)
    chunked = callback(ids)
    dense = model.predict(batch).poi_logits[:, 2:].to(torch.float64)
    torch.testing.assert_close(chunked, dense, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        torch.logsumexp(chunked, dim=1), torch.zeros(len(batch["user_id"]), dtype=torch.float64)
    )
