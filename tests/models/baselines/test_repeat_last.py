import pytest
import torch

from adrec_still.data.collate import collate_canonical_samples
from adrec_still.data.schema import PADDING_POI_ID, UNKNOWN_POI_ID, CanonicalSample
from adrec_still.models.baselines.popularity import GlobalPop
from adrec_still.models.baselines.repeat_last import RepeatLast

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


def fitted_model(**kwargs) -> RepeatLast:
    return RepeatLast(NUMBER_OF_POIS, **kwargs).fit(TRAINING_COUNTS)


def row_probabilities(model, history: list[int]) -> torch.Tensor:
    batch = collate_canonical_samples([history_sample(history)])
    return model.predict(batch).poi_logits[0].exp()


def test_prediction_is_finite_normalized_and_masks_reserved_ids():
    batch = collate_canonical_samples([history_sample([2, 3, 4]), history_sample([5])])
    prediction = fitted_model().predict(batch)
    prediction.validate(NUMBER_OF_POIS)
    probabilities = prediction.poi_logits.exp()
    assert probabilities.sum(dim=1).allclose(torch.ones(2), atol=1e-5)
    assert float(probabilities[0, PADDING_POI_ID]) == 0.0
    assert float(probabilities[0, UNKNOWN_POI_ID]) == 0.0


def test_last_known_poi_always_ranks_first():
    model = fitted_model()
    # 5 is the rarest training POI, yet as the last visit it must outrank the
    # globally most popular POI 2.
    assert int(row_probabilities(model, [2, 3, 5]).argmax()) == 5
    # Trailing unknown entries are ignored when picking the repeated POI.
    assert int(row_probabilities(model, [2, 3, UNKNOWN_POI_ID]).argmax()) == 3


def test_remaining_ranks_follow_the_global_pop_order():
    probabilities = row_probabilities(fitted_model(), [2, 3, 5])
    ordered = torch.argsort(probabilities, descending=True).tolist()
    assert ordered[:4] == [5, 2, 3, 4]


def test_rows_without_known_history_fall_back_to_global_pop():
    model = fitted_model()
    reference = GlobalPop(NUMBER_OF_POIS).fit(TRAINING_COUNTS)
    batch = collate_canonical_samples(
        [history_sample([UNKNOWN_POI_ID, UNKNOWN_POI_ID]), history_sample([2, 4])]
    )
    prediction = model.predict(batch)
    assert torch.allclose(
        prediction.poi_logits[0], reference.predict(batch).poi_logits[0], atol=1e-6
    )
    stats = model.prediction_stats()
    assert stats["predicted_row_count"] == 2
    assert stats["fallback_row_count"] == 1
    assert stats["fallback_rate"] == pytest.approx(0.5)
    model.reset_prediction_stats()
    assert model.prediction_stats()["fallback_rate"] is None


def test_output_is_deterministic_and_invariant_to_batch_padding():
    model = fitted_model()
    alone = collate_canonical_samples([history_sample([2, 4])])
    padded = collate_canonical_samples(
        [history_sample([2, 4]), history_sample([2, 3, 4, 5, 2, 3, 4])]
    )
    assert torch.equal(model.predict(alone).poi_logits[0], model.predict(padded).poi_logits[0])
    torch.manual_seed(9999)
    again = fitted_model()
    assert torch.equal(model.predict(alone).poi_logits, again.predict(alone).poi_logits)


def test_invalid_configuration_and_unfitted_use_are_rejected():
    with pytest.raises(ValueError):
        RepeatLast(NUMBER_OF_POIS, repeat_weight=0.4)
    with pytest.raises(ValueError):
        RepeatLast(NUMBER_OF_POIS, repeat_weight=1.0)
    with pytest.raises(RuntimeError):
        RepeatLast(NUMBER_OF_POIS).predict(collate_canonical_samples([history_sample([2])]))


def test_resolved_config_reports_reproducible_settings():
    config = fitted_model().resolved_config()
    assert config["name"] == "repeat_last"
    assert config["repeat_weight"] == pytest.approx(0.5)
    assert config["fallback"]["name"] == "global_pop"
    assert config["parameter_count"] == NUMBER_OF_POIS - 2
