import torch

from adrec_still.time.protocol import LogNormalPrediction, lognormal_time_metrics


def test_lognormal_metrics_include_observed_and_censored_likelihoods():
    prediction = LogNormalPrediction(
        mu=torch.log(torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)),
        sigma=torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64),
    )
    metrics = lognormal_time_metrics(
        prediction,
        residual_time_sec=torch.tensor([10.0, 0.0, 25.0]),
        censor_time_sec=torch.tensor([0.0, 15.0, 0.0]),
        observed_mask=torch.tensor([True, False, True]),
    )
    assert metrics["observed_count"] == 2
    assert metrics["censored_count"] == 1
    assert metrics["return_time_nll"] > 0
    # CRPS was removed in this port: only NLL summary keys are reported.
    assert metrics["sample_count"] == 3
    assert metrics["observed_time_nll"] is not None
    assert metrics["censored_time_nll"] is not None
    assert not any("crps" in key for key in metrics)
