import pytest
import torch
import torch.nn.functional as F

from adrec_still.data.collate import collate_canonical_samples
from adrec_still.data.schema import PADDING_POI_ID, UNKNOWN_POI_ID, CanonicalSample
from adrec_still.models.baselines.popularity import GlobalPop
from adrec_still.models.baselines.user_history_pop import UserHistoryPop

NUMBER_OF_POIS = 8
TRAINING_COUNTS = {2: 60, 3: 30, 4: 8, 5: 2}


def history_sample(history: list[int], next_poi: int = 2, query_s_sec: int = 0):
    timestamps = [1_577_836_800 + index * 3600 for index in range(len(history))]
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


def fitted_model(**kwargs) -> UserHistoryPop:
    # The tiny 6-POI catalog concentrates GlobalPop mass (max ~0.58), so the
    # frequency-first guard needs a shorter max_history_length than the real
    # Gowalla default of 100: 0.9 / 10 > 0.1 * 0.58 holds, 0.9 / 100 does not.
    kwargs.setdefault("max_history_length", 10)
    return UserHistoryPop(NUMBER_OF_POIS, **kwargs).fit(TRAINING_COUNTS)


def row_probabilities(model, history: list[int]) -> torch.Tensor:
    batch = collate_canonical_samples([history_sample(history)])
    return model.predict(batch).poi_logits[0].exp()


def test_prediction_is_finite_normalized_and_masks_reserved_ids():
    batch = collate_canonical_samples([history_sample([2, 3, 2]), history_sample([4])])
    prediction = fitted_model().predict(batch)
    prediction.validate(NUMBER_OF_POIS)
    probabilities = prediction.poi_logits.exp()
    assert probabilities.sum(dim=1).allclose(torch.ones(2), atol=1e-5)
    assert float(probabilities[0, PADDING_POI_ID]) == 0.0
    assert float(probabilities[0, UNKNOWN_POI_ID]) == 0.0


def test_ranking_follows_in_history_frequency_with_global_pop_tie_breaking():
    probabilities = row_probabilities(fitted_model(), [2, 3, 3, 5])
    # 3 was visited twice; 2 and 5 once each (tie broken by GlobalPop: 60 > 2);
    # 4 was never visited and only carries fallback mass.
    ordered = torch.argsort(probabilities, descending=True).tolist()
    assert ordered[:4] == [3, 2, 5, 4]


def test_any_visited_poi_outranks_any_unvisited_poi():
    # 5 is the rarest training POI but the only visited one, so it must beat
    # the globally most popular POI 2.
    probabilities = row_probabilities(fitted_model(), [5])
    assert int(probabilities.argmax()) == 5


def test_rows_without_known_history_fall_back_to_global_pop():
    model = fitted_model()
    reference = GlobalPop(NUMBER_OF_POIS).fit(TRAINING_COUNTS)
    batch = collate_canonical_samples(
        [history_sample([UNKNOWN_POI_ID, UNKNOWN_POI_ID]), history_sample([2, 3])]
    )
    prediction = model.predict(batch)
    assert torch.allclose(
        prediction.poi_logits[0], reference.predict(batch).poi_logits[0], atol=1e-6
    )
    stats = model.prediction_stats()
    assert stats["predicted_row_count"] == 2
    assert stats["fallback_row_count"] == 1
    assert stats["fallback_rate"] == pytest.approx(0.5)


def test_prediction_stats_reset_between_splits():
    model = fitted_model()
    model.predict(collate_canonical_samples([history_sample([2])]))
    model.reset_prediction_stats()
    stats = model.prediction_stats()
    assert stats["predicted_row_count"] == 0
    assert stats["fallback_rate"] is None


def test_output_is_deterministic_and_invariant_to_batch_padding():
    model = fitted_model()
    alone = collate_canonical_samples([history_sample([2, 3])])
    padded = collate_canonical_samples(
        [history_sample([2, 3]), history_sample([2, 3, 4, 5, 2, 3, 4])]
    )
    assert torch.equal(model.predict(alone).poi_logits[0], model.predict(padded).poi_logits[0])
    torch.manual_seed(9999)
    again = fitted_model()
    assert torch.equal(
        model.predict(alone).poi_logits, again.predict(alone).poi_logits
    )


def test_repeated_visits_lower_the_target_nll_below_global_pop():
    model = fitted_model()
    reference = GlobalPop(NUMBER_OF_POIS).fit(TRAINING_COUNTS)
    batch = collate_canonical_samples([history_sample([3, 3], next_poi=3)])
    model_nll = float(F.cross_entropy(model.predict(batch).poi_logits, batch["next_poi"]))
    reference_nll = float(
        F.cross_entropy(reference.predict(batch).poi_logits, batch["next_poi"])
    )
    assert model_nll < reference_nll


def test_fit_rejects_a_history_weight_that_cannot_dominate_global_pop():
    with pytest.raises(ValueError):
        UserHistoryPop(NUMBER_OF_POIS, history_weight=0.9, max_history_length=100).fit(
            {2: 1_000_000}
        )


def test_histories_longer_than_the_guard_are_rejected():
    model = fitted_model(max_history_length=2)
    with pytest.raises(ValueError):
        model.predict(collate_canonical_samples([history_sample([2, 3, 4])]))


def test_invalid_configuration_and_unfitted_use_are_rejected():
    with pytest.raises(ValueError):
        UserHistoryPop(NUMBER_OF_POIS, history_weight=0.0)
    with pytest.raises(ValueError):
        UserHistoryPop(NUMBER_OF_POIS, history_weight=1.0)
    with pytest.raises(ValueError):
        UserHistoryPop(NUMBER_OF_POIS, max_history_length=0)
    with pytest.raises(RuntimeError):
        UserHistoryPop(NUMBER_OF_POIS).predict(
            collate_canonical_samples([history_sample([2])])
        )


def test_resolved_config_reports_reproducible_settings():
    config = fitted_model().resolved_config()
    assert config["name"] == "user_history_pop"
    assert config["history_weight"] == pytest.approx(0.9)
    assert config["max_history_length"] == 10
    assert config["fallback"]["name"] == "global_pop"
    assert config["parameter_count"] == NUMBER_OF_POIS - 2
