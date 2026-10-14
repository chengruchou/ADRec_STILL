import math

import pytest
import torch
import torch.nn.functional as F

from adrec_still.data.collate import collate_canonical_samples
from adrec_still.data.schema import PADDING_POI_ID, UNKNOWN_POI_ID, CanonicalSample
from adrec_still.evaluation.evaluator import evaluate
from adrec_still.models.baselines.popularity import GlobalPop

NUMBER_OF_POIS = 8
TRAINING_COUNTS = {2: 60, 3: 30, 4: 8, 5: 2}


def observed_sample(next_poi: int, history_length: int = 3, query_s_sec: int = 0):
    history = [2 + (index % 4) for index in range(history_length)]
    timestamps = [1_577_836_800 + index * 3600 for index in range(history_length)]
    return CanonicalSample(
        user_id=2,
        history_poi=history,
        history_timestamp=timestamps,
        last_timestamp=timestamps[-1],
        query_s_sec=query_s_sec,
        event_observed=True,
        target_in_train_catalog=next_poi != UNKNOWN_POI_ID,
        residual_time_sec=7200,
        next_poi=next_poi,
    )


def censored_sample(history_length: int = 2, query_s_sec: int = 21600):
    timestamps = [1_577_836_800 + index * 3600 for index in range(history_length)]
    return CanonicalSample(
        user_id=3,
        history_poi=[2 + index for index in range(history_length)],
        history_timestamp=timestamps,
        last_timestamp=timestamps[-1],
        query_s_sec=query_s_sec,
        event_observed=False,
        target_in_train_catalog=False,
        censor_time_sec=3600,
    )


def fitted_model(counts=None, smoothing: float = 1.0) -> GlobalPop:
    return GlobalPop(NUMBER_OF_POIS, smoothing=smoothing).fit(
        TRAINING_COUNTS if counts is None else counts
    )


def test_prediction_shape_is_finite_and_normalized_over_the_train_catalog():
    batch = collate_canonical_samples([observed_sample(2), observed_sample(3)])
    prediction = fitted_model().predict(batch)
    prediction.validate(NUMBER_OF_POIS)
    assert prediction.poi_logits.shape == (2, NUMBER_OF_POIS)
    assert prediction.context.shape == (2, 0)
    assert torch.isfinite(prediction.poi_logits).all()
    probabilities = prediction.poi_logits.exp()
    assert probabilities.sum(dim=1).allclose(torch.ones(2), atol=1e-5)
    assert float(probabilities[0, PADDING_POI_ID]) == 0.0
    assert float(probabilities[0, UNKNOWN_POI_ID]) == 0.0


def test_scores_follow_training_frequency():
    model = fitted_model()
    probabilities = model.log_probabilities.exp()
    assert probabilities.argmax().item() == 2
    ordered = [2, 3, 4, 5]
    values = [float(probabilities[poi_id]) for poi_id in ordered]
    assert values == sorted(values, reverse=True)
    expected = (TRAINING_COUNTS[2] + 1.0) / (sum(TRAINING_COUNTS.values()) + 6.0)
    assert values[0] == pytest.approx(expected, rel=1e-5)


def test_output_is_invariant_to_history_padding():
    model = fitted_model()
    short_batch = collate_canonical_samples([observed_sample(2, history_length=1)])
    padded_batch = collate_canonical_samples(
        [observed_sample(2, history_length=1), observed_sample(3, history_length=9)]
    )
    assert padded_batch["history_mask"][0].sum() == 1
    assert torch.equal(
        model.predict(short_batch).poi_logits[0], model.predict(padded_batch).poi_logits[0]
    )
    truncated_batch = collate_canonical_samples(
        [observed_sample(2, history_length=9)], max_sequence_length=2
    )
    assert truncated_batch["history_poi"].shape[1] == 2
    assert torch.equal(
        model.predict(short_batch).poi_logits[0], model.predict(truncated_batch).poi_logits[0]
    )


def test_fit_is_deterministic_and_seed_independent():
    torch.manual_seed(41)
    first = fitted_model()
    torch.manual_seed(9999)
    second = fitted_model()
    assert torch.equal(first.log_probabilities, second.log_probabilities)
    batch = collate_canonical_samples([observed_sample(2), observed_sample(4)])
    first_loss = F.cross_entropy(first.predict(batch).poi_logits, batch["next_poi"])
    second_loss = F.cross_entropy(second.predict(batch).poi_logits, batch["next_poi"])
    assert float(first_loss) == float(second_loss)


def test_censored_and_cold_start_samples_are_excluded_from_the_poi_target_loss():
    samples = [
        observed_sample(2),
        observed_sample(UNKNOWN_POI_ID),
        censored_sample(),
    ]
    batch = collate_canonical_samples(samples)
    model = fitted_model()
    result = evaluate(model.predict(batch).poi_logits, batch, split="test")
    ranking = result["ranking"]
    assert ranking["sample_count"] == 1
    assert ranking["observed_target_count"] == 2
    assert ranking["cold_start_target_count"] == 1
    assert ranking["ranking_coverage"] == pytest.approx(0.5)
    assert ranking["recall@5"] == pytest.approx(1.0)
    assert result["sample_counts_by_cohort"] == {"observed": 2, "censored": 1}


def test_fitted_prior_scores_better_than_a_uniform_prior():
    held_out = [observed_sample(2) for _ in range(6)] + [observed_sample(3) for _ in range(3)]
    batch = collate_canonical_samples(held_out)
    uniform = fitted_model(counts={})
    fitted = fitted_model()
    uniform_nll = float(F.cross_entropy(uniform.predict(batch).poi_logits, batch["next_poi"]))
    fitted_nll = float(F.cross_entropy(fitted.predict(batch).poi_logits, batch["next_poi"]))
    assert fitted_nll < uniform_nll
    assert uniform_nll == pytest.approx(math.log(NUMBER_OF_POIS - 2), rel=1e-5)


def test_smoothing_keeps_unseen_train_catalog_pois_finite():
    model = fitted_model()
    unseen = float(model.log_probabilities[6])
    assert math.isfinite(unseen)
    assert unseen < float(model.log_probabilities[5])
    assert float(model.log_probabilities.exp()[6]) > 0.0


def test_resolved_config_reports_reproducible_settings():
    model = fitted_model()
    config = model.resolved_config()
    assert config["name"] == "global_pop"
    assert config["family"] == "baseline"
    assert config["number_of_pois"] == NUMBER_OF_POIS
    assert config["count_source"] == "train_targets"
    assert config["parameter_count"] == NUMBER_OF_POIS - 2
    assert config["total_training_count"] == sum(TRAINING_COUNTS.values())
    assert model.parameter_count() == NUMBER_OF_POIS - 2


def test_invalid_configuration_and_unfitted_use_are_rejected():
    with pytest.raises(ValueError):
        GlobalPop(NUMBER_OF_POIS, smoothing=0.0)
    with pytest.raises(ValueError):
        GlobalPop(NUMBER_OF_POIS, count_source="everything")
    with pytest.raises(ValueError):
        GlobalPop(1)
    with pytest.raises(RuntimeError):
        GlobalPop(NUMBER_OF_POIS).predict(collate_canonical_samples([observed_sample(2)]))
    with pytest.raises(ValueError):
        GlobalPop(NUMBER_OF_POIS).fit({UNKNOWN_POI_ID: 5})
    with pytest.raises(ValueError):
        GlobalPop(NUMBER_OF_POIS).fit({NUMBER_OF_POIS: 5})
    with pytest.raises(ValueError):
        GlobalPop(NUMBER_OF_POIS).fit({2: -1})
    with pytest.raises(TypeError):
        GlobalPop(NUMBER_OF_POIS).fit([(2, 5)])
