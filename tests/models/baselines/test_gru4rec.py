"""GRU4Rec-H unit tests and the Issue #15 tiny-set overfit gates."""

import pytest
import torch

from adrec_still.data.collate import collate_canonical_samples
from adrec_still.data.schema import UNKNOWN_POI_ID, CanonicalSample
from adrec_still.data.weighting import landmark_weights
from adrec_still.models.baselines.gru4rec import GRU4RecH
from adrec_still.models.baselines.harness import set_global_seed
from adrec_still.models.baselines.trainer import (
    load_checkpoint,
    per_row_sampled_losses,
    select_training_rows,
    training_step,
    weighted_mean_loss,
)
from adrec_still.training.checkpoint import save_weights
from adrec_still.training.loop import TrainingPlan, train_model

CPU = torch.device("cpu")


def training_config(**overrides):
    config = {
        "landmark_weighting": "per_gap",
        "optimizer": "adam",
        "learning_rate": 0.05,
        "weight_decay": 0.0,
        "batch_size": 32,
        "max_epochs": 40,
        "early_stopping_patience": 5,
        "early_stopping_metric": "ndcg@10",
        "negative_samples": 8,
    }
    config.update(overrides)
    return config


def observed_sample(history, next_poi, query_s_sec=0, residual_time_sec=7200, user_id=3):
    return CanonicalSample(
        user_id=user_id,
        history_poi=list(history),
        history_timestamp=list(range(1_577_836_800, 1_577_836_800 + 3600 * len(history), 3600)),
        last_timestamp=1_577_836_800 + 3600 * (len(history) - 1),
        query_s_sec=query_s_sec,
        event_observed=True,
        target_in_train_catalog=next_poi != UNKNOWN_POI_ID,
        residual_time_sec=residual_time_sec,
        next_poi=next_poi,
    )


def censored_sample(history, user_id=4):
    return CanonicalSample(
        user_id=user_id,
        history_poi=list(history),
        history_timestamp=list(range(1_577_836_800, 1_577_836_800 + 3600 * len(history), 3600)),
        last_timestamp=1_577_836_800 + 3600 * (len(history) - 1),
        query_s_sec=0,
        event_observed=False,
        target_in_train_catalog=False,
        censor_time_sec=3600,
    )


def fresh_model(number_of_pois=12, **overrides):
    arguments = {"embedding_dim": 16, "hidden_dim": 16, "num_layers": 1, "dropout": 0.0}
    arguments.update(overrides)
    return GRU4RecH(number_of_pois, **arguments)


def collision_free_samples(count, first_target):
    """Unique histories mapped to unique targets; no history repeats with another target."""
    return [
        observed_sample([2 + index, 2 + (index + 3) % count], first_target + index)
        for index in range(count)
    ]


def test_prediction_shape_normalization_and_reserved_masking():
    model = fresh_model()
    model.is_fitted = True
    batch = collate_canonical_samples([observed_sample([2, 3, 4], 5), observed_sample([6], 7)])
    prediction = model.predict(batch)
    prediction.validate(model.number_of_pois)
    assert prediction.poi_logits.shape == (2, 12)
    assert prediction.context.shape == (2, model.context_dim)
    assert torch.isfinite(prediction.poi_logits).all()
    probabilities = prediction.poi_logits.exp()
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-5)
    assert float(probabilities[:, :2].sum()) == pytest.approx(0.0, abs=1e-9)


def test_padding_mask_invariance():
    model = fresh_model()
    model.is_fitted = True
    short = observed_sample([2, 3], 5)
    long = observed_sample([4, 5, 6, 7, 8, 9, 10], 11)
    alone = model.predict(collate_canonical_samples([short]))
    padded = model.predict(collate_canonical_samples([short, long]))
    assert torch.allclose(alone.poi_logits[0], padded.poi_logits[0], atol=1e-5)
    assert torch.allclose(alone.context[0], padded.context[0], atol=1e-5)


def test_duplicated_rows_differing_only_in_query_horizon_predict_identically():
    model = fresh_model()
    model.is_fitted = True
    base = observed_sample([2, 3, 4], 5, query_s_sec=0)
    later = observed_sample([2, 3, 4], 5, query_s_sec=21600)
    prediction = model.predict(collate_canonical_samples([base, later]))
    assert torch.equal(prediction.poi_logits[0], prediction.poi_logits[1])
    assert torch.equal(prediction.context[0], prediction.context[1])


def test_same_seed_reproduces_the_first_step_loss():
    def first_step_loss():
        set_global_seed(123)
        model = fresh_model(dropout=0.2)
        batch = collate_canonical_samples([observed_sample([2, 3], 5), observed_sample([4], 6)])
        generator = torch.Generator().manual_seed(9)
        step = training_step(model, batch, training_config(), generator, CPU)
        return float(step["loss"].detach())

    assert first_step_loss() == first_step_loss()


def test_censored_and_cold_start_targets_are_excluded_from_the_loss():
    batch = collate_canonical_samples(
        [
            observed_sample([2, 3], 5),
            censored_sample([2, 3]),
            observed_sample([4, 5], UNKNOWN_POI_ID),
        ]
    )
    kept, excluded = select_training_rows(batch)
    assert int(kept["next_poi"].shape[0]) == 1
    assert excluded == 2
    step = training_step(fresh_model(), batch, training_config(), torch.Generator(), CPU)
    assert step["used_rows"] == 1
    assert step["excluded_rows"] == 2


def test_per_gap_weights_are_consumed_by_the_contract_loss():
    # One completed gap of duration 30000s appears at landmarks 0 and 21600
    # (eligible for exactly two landmarks -> weight 0.5 each); a second gap of
    # duration 10000s is eligible only at landmark 0 -> weight 1.
    samples = [
        observed_sample([2, 3], 5, query_s_sec=0, residual_time_sec=30000),
        observed_sample([2, 3], 5, query_s_sec=21600, residual_time_sec=8400),
        observed_sample([4], 6, query_s_sec=0, residual_time_sec=10000),
    ]
    batch = collate_canonical_samples(samples)
    kept, _ = select_training_rows(batch)
    weights = landmark_weights(kept, mode="per_gap")
    assert weights.tolist() == pytest.approx([0.5, 0.5, 1.0])
    assert landmark_weights(kept, mode="per_landmark").tolist() == pytest.approx([1.0, 1.0, 1.0])

    model = fresh_model()
    per_row = per_row_sampled_losses(model, kept, 8, torch.Generator().manual_seed(1)).detach()
    weights = weights.to(per_row.dtype)
    combined = weighted_mean_loss(per_row, weights)
    by_hand = float((per_row * weights).sum() / weights.sum())
    assert float(combined) == pytest.approx(by_hand)
    scale_dependent = float((per_row * weights).mean())
    assert float(combined) != pytest.approx(scale_dependent)


def test_checkpoint_roundtrip_reproduces_predictions(tmp_path):
    model = fresh_model(number_of_pois=15, embedding_dim=8, hidden_dim=8)
    model.is_fitted = True
    checkpoint_path = tmp_path / "checkpoint.pt"
    digest = save_weights(checkpoint_path, model)
    assert len(digest) == 64
    reloaded = load_checkpoint(checkpoint_path)
    assert reloaded.resolved_config() == model.resolved_config()
    batch = collate_canonical_samples([observed_sample([2, 3, 4], 5), observed_sample([6], 7)])
    assert torch.equal(model.predict(batch).poi_logits, reloaded.predict(batch).poi_logits)


def test_resolved_config_records_parameter_count_and_context_dim():
    model = fresh_model(number_of_pois=20, embedding_dim=8, hidden_dim=8, num_layers=1)
    resolved = model.resolved_config()
    assert resolved["parameter_count"] == sum(p.numel() for p in model.parameters())
    assert resolved["parameter_count"] > 0
    assert resolved["context_dim"] == model.context_dim == 8
    assert resolved["tied_item_embeddings"] is True


def recall_at_1(model, batch):
    prediction = model.predict(batch)
    top1 = prediction.poi_logits.argmax(dim=1)
    return float((top1 == batch["next_poi"]).float().mean())


def test_ci_smoke_overfit_gate():
    """8-16 collision-free CPU samples: loss decreases and train Recall@1 increases."""
    sample_count = 12
    samples = collision_free_samples(sample_count, first_target=2 + sample_count + 4)
    number_of_pois = 2 + 2 * sample_count + 6
    set_global_seed(41)
    model = fresh_model(number_of_pois, embedding_dim=32, hidden_dim=32)
    batch = collate_canonical_samples(samples)
    model.is_fitted = True
    initial_recall = recall_at_1(model, batch)

    plan = TrainingPlan(
        train_source=samples,
        training=training_config(
            max_epochs=30, batch_size=16, early_stopping_patience=100_000
        ),
        seed=41,
        device=CPU,
    )
    summary = train_model(model, plan)
    assert summary["best"]["epoch"] >= 1
    assert summary["final_train_loss"] < summary["first_train_loss"]
    assert recall_at_1(model, batch) > initial_recall


def test_pr_evidence_overfit_gate():
    """64-128 collision-free samples: train Recall@1 and MRR@1 must reach 0.95."""
    sample_count = 96
    first_target = 2 + sample_count + 8
    samples = collision_free_samples(sample_count, first_target)
    targets = [sample.next_poi for sample in samples]
    assert len({tuple(sample.history_poi) for sample in samples}) == sample_count
    assert len(set(targets)) == sample_count

    number_of_pois = first_target + sample_count + 4
    set_global_seed(41)
    model = fresh_model(number_of_pois, embedding_dim=48, hidden_dim=48)
    plan = TrainingPlan(
        train_source=samples,
        training=training_config(
            max_epochs=200, batch_size=32, negative_samples=64,
            early_stopping_patience=100_000,
        ),
        seed=41,
        device=CPU,
    )
    summary = train_model(model, plan)

    batch = collate_canonical_samples(samples)
    prediction = model.predict(batch)
    top1 = prediction.poi_logits.argmax(dim=1)
    train_recall_at_1 = float((top1 == batch["next_poi"]).float().mean())
    ranks = (
        prediction.poi_logits
        > prediction.poi_logits.gather(1, batch["next_poi"].unsqueeze(1))
    ).sum(dim=1) + 1
    train_mrr_at_1 = float((ranks == 1).float().mean())
    assert train_recall_at_1 >= 0.95
    assert train_mrr_at_1 >= 0.95
    assert summary["final_train_loss"] < 0.5
