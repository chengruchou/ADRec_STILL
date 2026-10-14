"""Model-level gates for the external dependent LNM baseline (#51)."""

import math

import pytest
import torch

from adrec_still.models.baselines.trainer import build_sampled_candidates
from adrec_still.models.external.joint_lnm import ExternalJointLNM
from adrec_still.models.external.statistics import LogTimeStatistics

NUMBER_OF_POIS = 12

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def make_model(**overrides) -> ExternalJointLNM:
    torch.manual_seed(23)
    settings = {
        "mark_embedding_dim": 8,
        "context_dim": 16,
        "num_layers": 1,
        "dropout": 0.0,
        "mixture_components": 2,
    }
    settings.update(overrides)
    model = ExternalJointLNM(NUMBER_OF_POIS, **settings)
    model.is_fitted = True
    model.eval()
    return model


def make_batch(query_s: list[int] | None = None) -> dict[str, torch.Tensor]:
    query = torch.tensor(query_s if query_s is not None else [0, 21600, 43200])
    batch_size = len(query)
    history = torch.tensor([[0, 3, 4, 5], [0, 0, 6, 7], [1, 8, 9, 10]])[:batch_size]
    mask = history.ne(0)
    timestamps = torch.where(
        mask, 1_600_000_000 + torch.arange(4) * 3600, torch.zeros(4, dtype=torch.long)
    )[:batch_size]
    return {
        "user_id": torch.arange(1, batch_size + 1),
        "history_poi": history,
        "history_timestamp": timestamps,
        "history_mask": mask,
        "last_timestamp": timestamps.max(dim=1).values,
        "query_s_sec": query,
        "event_observed": torch.tensor([True, True, False])[:batch_size],
        "observed_mask": torch.tensor([True, True, False])[:batch_size],
        "censored_mask": torch.tensor([False, False, True])[:batch_size],
        "target_in_train_catalog": torch.tensor([True, True, False])[:batch_size],
        "next_poi": torch.tensor([5, 7, 0])[:batch_size],
        "residual_time_sec": torch.tensor([3600.0, 7200.0, 0.0])[:batch_size],
        "censor_time_sec": torch.tensor([0.0, 0.0, 5400.0], dtype=torch.float64)[
            :batch_size
        ],
    }


def snapshot(batch):
    return {key: (value, value.dtype, value.clone()) for key, value in batch.items()}


def assert_unchanged(batch, snap):
    for key, (tensor, dtype, values) in snap.items():
        assert batch[key] is tensor and batch[key].dtype == dtype
        assert torch.equal(batch[key], values)


def test_predict_shapes_normalization_and_reserved_floor():
    model = make_model()
    batch = make_batch()
    snap = snapshot(batch)
    prediction = model.predict(batch)
    prediction.validate(NUMBER_OF_POIS)
    assert_unchanged(batch, snap)
    logits = prediction.poi_logits.to(torch.float64)
    assert logits.shape == (3, NUMBER_OF_POIS)
    # Reserved ids carry exactly zero probability; the legal slice is normalized.
    assert (logits[:, :2] <= -1.0e8).all()
    assert torch.allclose(
        logits[:, 2:].exp().sum(dim=1), torch.ones(3, dtype=torch.float64), atol=1e-6
    )
    assert prediction.time_distribution is not None
    assert prediction.time_distribution.batch_size() == 3
    value = prediction.time_distribution.log_survival(
        torch.tensor([0.0, 3600.0, 5400.0], dtype=torch.float64)
    )
    assert value[0].item() == 0.0 and torch.isfinite(value[1:]).all()


def test_ranking_never_reads_future_fields():
    model = make_model()
    batch = make_batch()
    reference = model.predict(batch)
    perturbed = make_batch()
    perturbed["residual_time_sec"] = torch.tensor([9999.0, 1.0, 123.0])
    perturbed["next_poi"] = torch.tensor([9, 3, 4])
    perturbed["censor_time_sec"] = torch.tensor([7.0, 8.0, 9.0], dtype=torch.float64)
    perturbed["observed_mask"] = torch.tensor([False, True, True])
    changed = model.predict(perturbed)
    assert torch.equal(reference.poi_logits, changed.poi_logits)
    grid = torch.tensor([1800.0, 3600.0, 7200.0], dtype=torch.float64)
    assert torch.equal(
        reference.time_distribution.log_density(grid),
        changed.time_distribution.log_density(grid),
    )


def test_query_elapsed_never_enters_the_base_distribution():
    model = make_model()
    early = make_batch([0, 0, 0])
    late = make_batch([43200, 43200, 43200])
    base_early = model.base_distribution(model.encode(early))
    base_late = model.base_distribution(model.encode(late))
    assert torch.equal(base_early.log_mark_probabilities, base_late.log_mark_probabilities)
    assert torch.equal(base_early.locs, base_late.locs)
    assert torch.equal(base_early.log_scales_raw, base_late.log_scales_raw)
    assert torch.equal(base_early.weight_logits, base_late.weight_logits)
    # And s = 0 ranking restores p_y itself.
    ranking_at_zero = model.predict(early).poi_logits.to(torch.float64)[:, 2:]
    assert torch.allclose(
        ranking_at_zero, base_early.log_mark_probabilities.to(torch.float64), atol=1e-6
    )


def test_history_encoding_uses_intervals_not_absolute_time():
    """Shifting every timestamp by a constant leaves all inter-event times — and
    therefore the context — unchanged (first-interval-zero-v1)."""
    model = make_model()
    batch = make_batch()
    shifted = make_batch()
    shifted["history_timestamp"] = torch.where(
        shifted["history_mask"],
        shifted["history_timestamp"] + 500_000,
        shifted["history_timestamp"],
    )
    shifted["last_timestamp"] = shifted["history_timestamp"].max(dim=1).values
    assert torch.equal(model.encode(batch), model.encode(shifted))


def test_padding_columns_do_not_change_the_context():
    model = make_model()
    batch = make_batch()
    padded = make_batch()
    extra = torch.zeros(3, 2, dtype=torch.long)
    padded["history_poi"] = torch.cat([extra, padded["history_poi"]], dim=1)
    padded["history_mask"] = padded["history_poi"].ne(0)
    padded["history_timestamp"] = torch.cat([extra, padded["history_timestamp"]], dim=1)
    assert torch.allclose(model.encode(batch), model.encode(padded), atol=1e-6)


def test_empty_history_rows_and_zero_width_batch_use_context_init():
    model = make_model()
    batch = make_batch()
    batch["history_poi"] = torch.zeros(3, 4, dtype=torch.long)
    batch["history_mask"] = torch.zeros(3, 4, dtype=torch.bool)
    batch["history_timestamp"] = torch.zeros(3, 4, dtype=torch.long)
    context = model.encode(batch)
    assert torch.allclose(context, model.context_init.unsqueeze(0).expand(3, -1))
    zero_width = make_batch()
    zero_width["history_poi"] = torch.zeros(3, 0, dtype=torch.long)
    zero_width["history_mask"] = torch.zeros(3, 0, dtype=torch.bool)
    zero_width["history_timestamp"] = torch.zeros(3, 0, dtype=torch.long)
    context_zero = model.encode(zero_width)
    assert context_zero.shape == (3, model.context_dim)
    assert torch.allclose(context_zero, model.context_init.unsqueeze(0).expand(3, -1))


def test_score_joint_two_path_decomposition_and_eligibility():
    model = make_model()
    batch = make_batch()
    scores = model.score_joint(batch)
    eligible = scores["eligible"]
    assert eligible.tolist() == [True, True, False]
    joint = scores["log_joint"][eligible]
    recomposed = (scores["log_time"] + scores["log_mark_given_time"])[eligible]
    assert torch.allclose(joint, recomposed, atol=1e-9)
    # Independent third path in the test itself: joint from the raw base pieces.
    base = model.base_distribution(model.encode(batch))
    elapsed = batch["query_s_sec"].to(torch.float64)
    total = elapsed + batch["residual_time_sec"].to(torch.float64).clamp_min(1.0)
    target_index = (batch["next_poi"].clamp_min(2) - 2).view(-1, 1)
    manual = (
        base.log_mark_probabilities.gather(1, target_index)[:, 0]
        + base.per_mark_log_density(total).gather(1, target_index)[:, 0]
        - base.log_survival(elapsed)
    )
    assert torch.allclose(joint, manual[eligible], atol=1e-9)


def test_training_hook_is_exact_time_plus_shared_sampled_mark():
    model = make_model()
    model.train()
    batch = make_batch([0, 21600])
    kept = {key: value[:2] for key, value in batch.items()}
    generator = torch.Generator().manual_seed(7)
    losses = model.per_row_sampled_training_losses(kept, negative_samples=4, generator=generator)
    assert losses.shape == (2,) and torch.isfinite(losses).all()
    # Reference: identical candidate draw via the shared builder with the same seed.
    reference_generator = torch.Generator().manual_seed(7)
    candidates, penalty = build_sampled_candidates(
        kept["next_poi"], 4, NUMBER_OF_POIS, reference_generator
    )
    base = model.base_distribution(model.encode(kept))
    elapsed = kept["query_s_sec"].to(torch.float64)
    total = elapsed + kept["residual_time_sec"].to(torch.float64)
    time_nll = -(base.log_marginal_density(total) - base.log_survival(elapsed))
    weighted = base.log_mark_probabilities + base.per_mark_log_density(total)
    logits = weighted.gather(1, candidates - 2) - penalty.to(torch.float64)
    mark_nll = -torch.log_softmax(logits, dim=1)[:, 0]
    assert torch.allclose(losses, time_nll + mark_nll, atol=1e-9)
    # First-update reproducibility: same seed, fresh model, identical parameters.
    losses.sum().backward()
    reference_model = make_model()
    reference_model.train()
    reference_losses = reference_model.per_row_sampled_training_losses(
        kept, negative_samples=4, generator=torch.Generator().manual_seed(7)
    )
    reference_losses.sum().backward()
    for left, right in zip(model.parameters(), reference_model.parameters(), strict=True):
        if left.grad is None:
            assert right.grad is None
        else:
            assert torch.allclose(left.grad, right.grad, atol=0.0)


def test_training_hook_rejects_censored_and_nonpositive_rows():
    model = make_model()
    batch = make_batch()
    with pytest.raises(ValueError):
        model.per_row_sampled_training_losses(
            batch, negative_samples=2, generator=torch.Generator().manual_seed(0)
        )
    kept = {key: value[:2] for key, value in make_batch([0, 21600]).items()}
    kept["residual_time_sec"] = torch.tensor([0.0, 3600.0])
    with pytest.raises(ValueError):
        model.per_row_sampled_training_losses(
            kept, negative_samples=2, generator=torch.Generator().manual_seed(0)
        )


def test_predict_restores_the_callers_training_mode():
    model = make_model(dropout=0.5)
    model.train()
    batch = make_batch()
    first = model.predict(batch)
    second = model.predict(batch)
    assert model.training is True
    assert torch.equal(first.poi_logits, second.poi_logits)


def test_statistics_buffers_change_the_encoder_features_and_distribution():
    model = make_model()
    batch = make_batch()
    before = model.predict(batch)
    model.set_log_time_statistics(
        LogTimeStatistics(
            mean=math.log(3600.0), std=0.5, source="test", unique_gap_count=9,
            status="fitted",
        )
    )
    after = model.predict(batch)
    assert not torch.equal(before.poi_logits, after.poi_logits)
    assert model.log_time_statistics_record["status"] == "fitted"


@requires_cuda
def test_cuda_model_accepts_cpu_collate_batches():
    model = make_model()
    model.to(torch.device("cuda"))
    batch = make_batch()
    snap = snapshot(batch)
    prediction = model.predict(batch)
    assert prediction.poi_logits.device.type == "cpu"
    scores = model.score_joint(batch)
    assert torch.isfinite(scores["log_joint"][scores["eligible"]]).all()
    assert_unchanged(batch, snap)
