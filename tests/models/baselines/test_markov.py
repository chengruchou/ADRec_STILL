import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from adrec_still.data.collate import collate_canonical_samples
from adrec_still.data.schema import PADDING_POI_ID, UNKNOWN_POI_ID, CanonicalSample
from adrec_still.models.baselines.markov import (
    FirstOrderMarkov,
    MarkovTrainingSource,
    train_transition_counts,
)
from adrec_still.models.baselines.popularity import GlobalPop

NUMBER_OF_POIS = 8
TRAINING_COUNTS = {2: 60, 3: 30, 4: 8, 5: 2}
TRANSITION_COUNTS = {(2, 3): 2, (2, 4): 1, (3, 5): 1}


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


def fitted_model(**kwargs) -> FirstOrderMarkov:
    return FirstOrderMarkov(NUMBER_OF_POIS, **kwargs).fit(
        MarkovTrainingSource(
            transition_counts=TRANSITION_COUNTS, global_counts=TRAINING_COUNTS
        )
    )


def row_probabilities(model, history: list[int]) -> torch.Tensor:
    batch = collate_canonical_samples([history_sample(history)])
    return model.predict(batch).poi_logits[0].exp()


def test_prediction_is_finite_normalized_and_masks_reserved_ids():
    batch = collate_canonical_samples([history_sample([3, 2]), history_sample([5])])
    prediction = fitted_model().predict(batch)
    prediction.validate(NUMBER_OF_POIS)
    probabilities = prediction.poi_logits.exp()
    assert probabilities.sum(dim=1).allclose(torch.ones(2), atol=1e-5)
    assert float(probabilities[0, PADDING_POI_ID]) == 0.0
    assert float(probabilities[0, UNKNOWN_POI_ID]) == 0.0


def test_supported_state_interpolates_its_transition_row_with_global_pop():
    model = fitted_model()
    probabilities = row_probabilities(model, [4, 2])
    # From state 2 the training transitions are 3 (2/3) and 4 (1/3); both must
    # outrank every untransitioned POI, including the globally popular state 2.
    ordered = torch.argsort(probabilities, descending=True).tolist()
    assert ordered[:3] == [3, 4, 2]
    global_pop = GlobalPop(NUMBER_OF_POIS).fit(TRAINING_COUNTS)
    expected_top = 0.9 * (2.0 / 3.0) + 0.1 * float(global_pop.log_probabilities.exp()[3])
    assert float(probabilities[3]) == pytest.approx(expected_top, rel=1e-5)


def test_unseen_state_and_unknown_last_poi_fall_back_to_global_pop():
    model = fitted_model()
    reference = GlobalPop(NUMBER_OF_POIS).fit(TRAINING_COUNTS)
    batch = collate_canonical_samples(
        [
            history_sample([2, 5]),  # state 5 has no fitted transitions
            history_sample([UNKNOWN_POI_ID, UNKNOWN_POI_ID]),  # no known last POI
            history_sample([3, 2]),  # supported state 2
        ]
    )
    prediction = model.predict(batch)
    reference_logits = reference.predict(batch).poi_logits
    assert torch.allclose(prediction.poi_logits[0], reference_logits[0], atol=1e-6)
    assert torch.allclose(prediction.poi_logits[1], reference_logits[1], atol=1e-6)
    assert not torch.allclose(prediction.poi_logits[2], reference_logits[2], atol=1e-4)
    stats = model.prediction_stats()
    assert stats["predicted_row_count"] == 3
    assert stats["supported_row_count"] == 1
    assert stats["fallback_row_count"] == 2
    assert stats["transition_support_rate"] == pytest.approx(1.0 / 3.0)
    assert stats["fallback_rate"] == pytest.approx(2.0 / 3.0)
    model.reset_prediction_stats()
    assert model.prediction_stats()["transition_support_rate"] is None


def test_trailing_unknown_last_poi_falls_back_even_when_older_state_is_supported():
    # Issue #14 contract: unknown last POI -> GlobalPop. State 2 has fitted
    # transitions, but the actual last observation is UNKNOWN, so the row must
    # emit exactly the GlobalPop distribution instead of state 2's row.
    model = fitted_model()
    reference = GlobalPop(NUMBER_OF_POIS).fit(TRAINING_COUNTS)
    batch = collate_canonical_samples([history_sample([2, UNKNOWN_POI_ID])])
    prediction = model.predict(batch)
    reference_logits = reference.predict(batch).poi_logits
    assert torch.allclose(prediction.poi_logits[0], reference_logits[0], atol=1e-6)
    stats = model.prediction_stats()
    assert stats["supported_row_count"] == 0
    assert stats["fallback_row_count"] == 1
    assert stats["fallback_rate"] == pytest.approx(1.0)


def test_transition_fitting_forms_no_transition_across_a_trailing_unknown(tmp_path):
    # The gap whose history ends in UNKNOWN must not fall back to state 2.
    landmark_path = tmp_path / "landmarks.parquet"
    pq.write_table(
        pa.table(
            {
                "split": ["train", "train", "train", "train"],
                "query_s_sec": [0, 0, 0, 0],
                "event_observed": [True, True, True, True],
                "target_in_train_catalog": [True, True, True, True],
                "next_poi": [3, 4, 5, 3],
                "history_poi": [[2], [2, UNKNOWN_POI_ID], [UNKNOWN_POI_ID], []],
            }
        ),
        landmark_path,
    )
    counts, stats = train_transition_counts(landmark_path)
    assert counts == {(2, 3): 1}
    assert stats["training_gap_count"] == 4
    assert stats["training_gaps_with_unknown_last"] == 2
    assert stats["training_gaps_without_observed_last"] == 1


def test_output_is_deterministic_and_invariant_to_batch_padding():
    model = fitted_model()
    alone = collate_canonical_samples([history_sample([3, 2])])
    padded = collate_canonical_samples(
        [history_sample([3, 2]), history_sample([2, 3, 4, 5, 2, 3, 4])]
    )
    assert torch.equal(model.predict(alone).poi_logits[0], model.predict(padded).poi_logits[0])
    torch.manual_seed(9999)
    again = fitted_model()
    assert torch.equal(model.predict(alone).poi_logits, again.predict(alone).poi_logits)


def test_invalid_training_sources_and_configuration_are_rejected():
    with pytest.raises(TypeError):
        FirstOrderMarkov(NUMBER_OF_POIS).fit(TRANSITION_COUNTS)
    with pytest.raises(ValueError):
        FirstOrderMarkov(NUMBER_OF_POIS).fit(
            MarkovTrainingSource(
                transition_counts={(PADDING_POI_ID, 3): 1}, global_counts=TRAINING_COUNTS
            )
        )
    with pytest.raises(ValueError):
        FirstOrderMarkov(NUMBER_OF_POIS).fit(
            MarkovTrainingSource(
                transition_counts={(2, NUMBER_OF_POIS): 1}, global_counts=TRAINING_COUNTS
            )
        )
    with pytest.raises(ValueError):
        FirstOrderMarkov(NUMBER_OF_POIS).fit(
            MarkovTrainingSource(transition_counts={(2, 3): 0}, global_counts=TRAINING_COUNTS)
        )
    with pytest.raises(ValueError):
        FirstOrderMarkov(NUMBER_OF_POIS, transition_weight=0.0)
    with pytest.raises(ValueError):
        FirstOrderMarkov(NUMBER_OF_POIS, transition_weight=1.0)
    with pytest.raises(RuntimeError):
        FirstOrderMarkov(NUMBER_OF_POIS).predict(
            collate_canonical_samples([history_sample([2])])
        )


def test_resolved_config_reports_reproducible_settings():
    config = fitted_model().resolved_config()
    assert config["name"] == "first_order_markov"
    assert config["transition_weight"] == pytest.approx(0.9)
    assert config["fitted_state_count"] == 2
    assert config["fitted_transition_count"] == 3
    assert config["fallback"]["name"] == "global_pop"
    assert config["parameter_count"] == 3 + NUMBER_OF_POIS - 2
