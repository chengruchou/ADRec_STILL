import pytest

from adrec_still.data.schema import CanonicalSample


def test_observed_sample_is_valid():
    sample = CanonicalSample(
        user_id=1,
        history_poi=[10, 11],
        history_timestamp=[100, 200],
        last_timestamp=200,
        query_s_sec=10,
        event_observed=True,
        target_in_train_catalog=True,
        residual_time_sec=20,
        next_poi=12,
        cell_label="local_early",
    )
    sample.validate()


def test_censored_sample_rejects_fake_next_poi():
    sample = CanonicalSample(
        user_id=1,
        history_poi=[10],
        history_timestamp=[100],
        last_timestamp=100,
        query_s_sec=10,
        event_observed=False,
        target_in_train_catalog=False,
        next_poi=99,
        censor_time_sec=20,
    )
    with pytest.raises(ValueError):
        sample.validate()


def test_observed_unknown_target_is_explicitly_not_in_train_catalog():
    sample = CanonicalSample(
        user_id=1,
        history_poi=[10],
        history_timestamp=[100],
        last_timestamp=100,
        query_s_sec=10,
        event_observed=True,
        target_in_train_catalog=False,
        residual_time_sec=20,
        next_poi=1,
    )
    sample.validate()
