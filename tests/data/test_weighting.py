import pytest
import torch

from adrec_still.data import (
    SURVIVAL_WEIGHTING_CONTRACT_VERSION,
    WEIGHTING_CONTRACT_VERSION,
    landmark_weights,
    survival_landmark_weights,
)
from adrec_still.data.collate import collate_canonical_samples
from adrec_still.data.landmarks import LANDMARK_SECONDS
from adrec_still.data.schema import CanonicalSample


def completed_gap(duration, schedule=LANDMARK_SECONDS):
    return [
        CanonicalSample(
            user_id=2,
            history_poi=[2],
            history_timestamp=[100],
            last_timestamp=100,
            query_s_sec=s,
            event_observed=True,
            target_in_train_catalog=True,
            residual_time_sec=duration - s,
            next_poi=3,
        )
        for s in schedule
        if s < duration
    ]


def test_seven_landmarks_each_receive_one_seventh():
    batch = collate_canonical_samples(completed_gap(169 * 3600))
    weights = landmark_weights(batch)
    assert WEIGHTING_CONTRACT_VERSION == "still-landmark-weighting-v0.1"
    assert weights.dtype == torch.float64
    assert weights.device == batch["query_s_sec"].device
    torch.testing.assert_close(weights, torch.full((7,), 1 / 7, dtype=torch.float64))
    assert weights.sum().item() == pytest.approx(1)


def test_single_landmark_and_exact_horizon_boundary():
    assert landmark_weights(collate_canonical_samples(completed_gap(6 * 3600))).tolist() == [1]
    weights = landmark_weights(collate_canonical_samples(completed_gap(12 * 3600)))
    assert weights.tolist() == [0.5, 0.5]


def test_multiple_gaps_shuffle_partial_batches_and_weighted_loss():
    samples = completed_gap(1) + completed_gap(7 * 3600) + completed_gap(169 * 3600)
    weights = landmark_weights(collate_canonical_samples(samples))
    assert weights.sum().item() == pytest.approx(3)
    permutation = torch.tensor([9, 2, 4, 1, 0, 8, 7, 3, 5, 6])
    shuffled = landmark_weights(collate_canonical_samples([samples[i] for i in permutation]))
    torch.testing.assert_close(shuffled, weights[permutation])
    single_batch_weights = torch.cat(
        [landmark_weights(collate_canonical_samples([sample])) for sample in samples]
    )
    torch.testing.assert_close(single_batch_weights, weights)
    # Each gap's constant loss should contribute equally regardless of its landmark count.
    losses = torch.tensor([3] + [6] * 2 + [12] * 7, dtype=torch.float64, requires_grad=True)
    loss = (weights * losses).sum() / weights.sum()
    assert loss.item() == pytest.approx(7)
    loss.backward()
    torch.testing.assert_close(losses.grad, weights / weights.sum())


def test_per_landmark_sensitivity_and_custom_training_schedule():
    schedule = (0, 6 * 3600, 24 * 3600)
    batch = collate_canonical_samples(completed_gap(169 * 3600, schedule))
    assert landmark_weights(batch, mode="per_landmark").tolist() == [1, 1, 1]
    weights = landmark_weights(batch, landmark_seconds=schedule)
    assert weights.sum().item() == pytest.approx(1)
    torch.testing.assert_close(weights, landmark_weights(batch, landmark_seconds=schedule[::-1]))


@pytest.mark.parametrize("mode", ["per_gap", "per_landmark"])
def test_censoring_rejected_in_both_modes(mode):
    sample = CanonicalSample(
        user_id=1,
        history_poi=[1],
        history_timestamp=[100],
        last_timestamp=100,
        query_s_sec=0,
        event_observed=False,
        target_in_train_catalog=False,
        censor_time_sec=10,
    )
    with pytest.raises(ValueError, match="censoring"):
        landmark_weights(collate_canonical_samples([sample]), mode=mode)


def test_unknown_observed_target_still_receives_completed_gap_weight():
    batch = collate_canonical_samples(completed_gap(169 * 3600))
    batch["next_poi"][:] = 1
    batch["target_in_train_catalog"][:] = False
    assert landmark_weights(batch).sum().item() == pytest.approx(1)


@pytest.mark.parametrize("schedule", [(), (0, 0), (-1, 0), (0, 0.5), (False, 1)])
def test_invalid_schedule_rejected(schedule):
    with pytest.raises(ValueError, match="landmark_seconds"):
        landmark_weights(collate_canonical_samples(completed_gap(1)), landmark_seconds=schedule)


@pytest.mark.parametrize(
    "field,value",
    [
        ("query_s_sec", -1),
        ("query_s_sec", 1),
        ("residual_time_sec", 0),
        ("residual_time_sec", -1),
        ("residual_time_sec", float("nan")),
        ("residual_time_sec", float("inf")),
        ("residual_time_sec", 0.5),
    ],
)
def test_invalid_observed_times_rejected(field, value):
    batch = collate_canonical_samples(completed_gap(1))
    batch[field] = torch.tensor([value], dtype=torch.float64)
    with pytest.raises(ValueError):
        landmark_weights(batch)


def test_unknown_mode_shape_and_empty_batch():
    batch = collate_canonical_samples(completed_gap(1))
    with pytest.raises(ValueError, match="mode"):
        landmark_weights(batch, mode="gap")
    with pytest.raises(ValueError, match="shape"):
        landmark_weights({**batch, "residual_time_sec": torch.ones(2)})
    empty = {key: value[:0] for key, value in batch.items()}
    assert landmark_weights(empty).shape == (0,)


def test_survival_weights_cover_completed_unknown_and_positive_censor_rows():
    schedule = (0, 5, 10)
    samples = completed_gap(11, schedule)
    samples[1] = CanonicalSample(
        **{
            **samples[1].__dict__,
            "target_in_train_catalog": False,
            "next_poi": 1,
        }
    )
    samples += [
        CanonicalSample(
            user_id=3,
            history_poi=[2],
            history_timestamp=[100],
            last_timestamp=100,
            query_s_sec=query,
            event_observed=False,
            target_in_train_catalog=False,
            censor_time_sec=8 - query,
        )
        for query in (0, 5)
    ]
    batch = collate_canonical_samples(samples)
    weights = survival_landmark_weights(batch, landmark_seconds=schedule)
    assert SURVIVAL_WEIGHTING_CONTRACT_VERSION == "still-survival-gap-weighting-v1"
    assert weights.tolist() == pytest.approx([1 / 3] * 3 + [1 / 2] * 2)
    assert weights[:3].sum().item() == pytest.approx(1)
    assert weights[3:].sum().item() == pytest.approx(1)
    permutation = torch.tensor([4, 1, 3, 0, 2])
    shuffled = {key: value[permutation] for key, value in batch.items()}
    torch.testing.assert_close(
        survival_landmark_weights(shuffled, landmark_seconds=schedule), weights[permutation]
    )


def test_survival_weights_reject_zero_followup_and_keep_v01_censor_rejection():
    sample = CanonicalSample(
        user_id=3,
        history_poi=[2],
        history_timestamp=[100],
        last_timestamp=100,
        query_s_sec=0,
        event_observed=False,
        target_in_train_catalog=False,
        censor_time_sec=1,
    )
    batch = collate_canonical_samples([sample])
    batch["censor_time_sec"][:] = 0
    with pytest.raises(ValueError, match="positive follow-up"):
        survival_landmark_weights(batch, landmark_seconds=(0,))
    with pytest.raises(ValueError, match="censoring rejected"):
        landmark_weights(batch, landmark_seconds=(0,))
