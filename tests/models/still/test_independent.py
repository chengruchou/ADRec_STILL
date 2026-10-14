import torch

from adrec_still.models.still.independent import StillIndependent

NUMBER_OF_POIS = 20


def make_model(components: int = 2):
    torch.manual_seed(11)
    model = StillIndependent(
        NUMBER_OF_POIS,
        embedding_dim=16,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        mixture_components=components,
    )
    model.is_fitted = True
    model.eval()
    return model


def make_batch(query_s: list[int] | None = None) -> dict[str, torch.Tensor]:
    query = torch.tensor(query_s if query_s is not None else [0, 21600])
    batch_size = len(query)
    history = torch.tensor([[0, 3, 4, 5], [2, 8, 9, 10]])[:batch_size]
    mask = history.ne(0)
    timestamps = torch.where(mask, 1_600_000_000 + torch.arange(4) * 3600, 0)[:batch_size]
    return {
        "user_id": torch.arange(1, batch_size + 1),
        "history_poi": history,
        "history_timestamp": timestamps,
        "history_mask": mask,
        "last_timestamp": timestamps.max(dim=1).values,
        "query_s_sec": query,
        "event_observed": torch.ones(batch_size, dtype=torch.bool),
        "observed_mask": torch.ones(batch_size, dtype=torch.bool),
        "censored_mask": torch.zeros(batch_size, dtype=torch.bool),
        "target_in_train_catalog": torch.ones(batch_size, dtype=torch.bool),
        "next_poi": torch.tensor([5, 9])[:batch_size],
        "residual_time_sec": torch.tensor([3600.0, 7200.0])[:batch_size],
        "censor_time_sec": torch.zeros(batch_size, dtype=torch.float64),
    }


def test_predict_is_a_valid_per_query_joint():
    model = make_model()
    prediction = model.predict(make_batch())
    prediction.validate(NUMBER_OF_POIS)
    probabilities = prediction.poi_logits.to(torch.float64).exp().sum(dim=1)
    assert torch.allclose(probabilities, torch.ones_like(probabilities), atol=1e-5)
    assert prediction.time_distribution is not None
    assert prediction.time_prediction is None
    # residual mixture is a proper distribution: survival at 0 is 1
    assert torch.allclose(
        prediction.time_distribution.log_survival(torch.zeros(2, dtype=torch.float64)),
        torch.zeros(2, dtype=torch.float64),
    )


def test_single_component_exposes_the_accepted_lognormal_path():
    model = make_model(components=1)
    prediction = model.predict(make_batch())
    assert prediction.time_prediction is not None
    assert prediction.time_distribution is None
    prediction.validate(NUMBER_OF_POIS)


def test_direct_arm_uses_elapsed_and_clock_but_never_targets():
    model = make_model()
    base = model.predict(make_batch([0, 21600]))
    waited = model.predict(make_batch([21600, 43200]))
    # The direct arm conditions on elapsed/query clock, so predictions move with s...
    assert not torch.allclose(base.poi_logits, waited.poi_logits, atol=1e-6)
    # ...but hidden future targets never enter the prediction call.
    batch = make_batch()
    tampered = dict(batch)
    tampered["next_poi"] = torch.tensor([12, 13])
    tampered["residual_time_sec"] = torch.tensor([1.0, 5.0])
    assert torch.equal(model.predict(batch).poi_logits, model.predict(tampered).poi_logits)


def test_padding_length_invariance_through_the_composed_encoder():
    model = make_model()
    batch = make_batch()
    padded = dict(batch)
    padded["history_poi"] = torch.cat(
        [torch.zeros(2, 2, dtype=torch.long), batch["history_poi"]], dim=1
    )
    padded["history_timestamp"] = torch.cat(
        [torch.zeros(2, 2, dtype=torch.long), batch["history_timestamp"]], dim=1
    )
    padded["history_mask"] = torch.cat(
        [torch.zeros(2, 2, dtype=torch.bool), batch["history_mask"]], dim=1
    )
    assert torch.allclose(
        model.predict(batch).poi_logits, model.predict(padded).poi_logits, atol=1e-6
    )


def test_checkpoint_reload_reproduces_predictions(tmp_path):
    from adrec_still.models.still.registry import load_still_checkpoint
    from adrec_still.training.checkpoint import save_weights

    model = make_model()
    path = tmp_path / "checkpoint.pt"
    save_weights(path, model, family="still")
    reloaded = load_still_checkpoint(path)
    reloaded.eval()
    batch = make_batch()
    assert torch.equal(model.predict(batch).poi_logits, reloaded.predict(batch).poi_logits)
    original_distribution = model.predict(batch).time_distribution
    reloaded_distribution = reloaded.predict(batch).time_distribution
    r = torch.tensor([1800.0, 3600.0], dtype=torch.float64)
    assert torch.allclose(
        original_distribution.log_density(r), reloaded_distribution.log_density(r)
    )


def test_resolved_config_discloses_capacity():
    model = make_model()
    config = model.resolved_config()
    assert config["mixture_components"] == 2
    assert config["conditioning_features"] == "elapsed+query_clock"
    assert config["parameter_count"] == (
        config["encoder_parameter_count"] + config["head_parameter_count"]
    )
