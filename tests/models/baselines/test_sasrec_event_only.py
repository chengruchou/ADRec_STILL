"""A matched event-only arm changes training exposure without changing SASRec-H."""

import json
from dataclasses import replace

import pytest
import torch

from adrec_still.data.collate import collate_canonical_samples
from adrec_still.data.dataset import LandmarkDataset
from adrec_still.data.schema import CanonicalSample
from adrec_still.data.training_views import EVENT_ONLY_TRAINING_PROFILE, EVENT_ONLY_WEIGHTING_CONTRACT
from adrec_still.models.baselines import trainer
from adrec_still.models.baselines.sasrec import SASRecH
from adrec_still.models.baselines.sasrec_event_only import SASRecEventOnly
from adrec_still.models.still.losses import TRAINING_PROFILE
from adrec_still.training.families import load_model
from adrec_still.training.loop import TrainingPlan, train_model

CPU = torch.device("cpu")


def sample(target=3, *, query=0, residual=3600, history=(2,)):
    return CanonicalSample(
        user_id=1, history_poi=list(history), history_timestamp=list(range(len(history))),
        last_timestamp=len(history) - 1, query_s_sec=query, event_observed=True,
        target_in_train_catalog=target > 1, next_poi=target, residual_time_sec=residual,
    )


def model(model_class=SASRecEventOnly):
    return model_class(10, embedding_dim=8, hidden_dim=8, num_layers=1,
                       num_heads=2, dropout=0.0, max_sequence_length=20)


def settings(**overrides):
    return {
        "training_profile": EVENT_ONLY_TRAINING_PROFILE, "landmark_weighting": "per_gap",
        "optimizer": "adam", "learning_rate": 0.03, "batch_size": 8,
        "max_epochs": 12, "early_stopping_patience": 20, "early_stopping_metric": "ndcg@10",
        "negative_samples": 16, **overrides,
    }


def test_identical_architecture_and_initialization_with_distinct_model_identity():
    torch.manual_seed(41)
    ordinary = model(SASRecH)
    torch.manual_seed(41)
    event_only = model()
    assert ordinary.parameter_count() == event_only.parameter_count()
    assert ordinary.state_dict().keys() == event_only.state_dict().keys()
    assert all(torch.equal(value, event_only.state_dict()[key])
               for key, value in ordinary.state_dict().items())
    left, right = ordinary.resolved_config(), event_only.resolved_config()
    assert left.pop("name") == "sasrec_h"
    assert right.pop("name") == "sasrec_event_only"
    assert left == right


def test_predictions_and_exact_scores_are_invariant_to_silence_and_clock():
    instance = model()
    instance.is_fitted = True
    instance.eval()
    batch = collate_canonical_samples([sample(history=(2, 4)), sample(history=(2, 4))])
    batch["query_s_sec"][1] = 7 * 86400
    batch["history_timestamp"][1] += 987654
    batch["last_timestamp"][1] += 987654
    batch["residual_time_sec"][1] = 1
    prediction = instance.predict(batch)
    assert torch.equal(prediction.poi_logits[0], prediction.poi_logits[1])
    candidates = torch.arange(2, 10).repeat(2, 1)
    scores = instance.score_candidates(instance.encode(batch), candidates)
    assert torch.equal(scores[0], scores[1])


def test_defensive_selection_and_unit_gap_weights(monkeypatch):
    censored = replace(sample(), event_observed=False, target_in_train_catalog=False,
                       residual_time_sec=None, next_poi=None, censor_time_sec=50)
    batch = collate_canonical_samples([
        sample(3, residual=1), sample(5, residual=200 * 3600),
        sample(query=6 * 3600), sample(1), censored, sample(), sample(),
    ])
    batch["residual_time_sec"][-2] = 0
    batch["next_poi"][-1] = 1  # a false known-catalog flag must not admit UNKNOWN
    monkeypatch.setattr(trainer, "per_row_sampled_losses",
                        lambda _model, kept, _count, _generator: kept["next_poi"].float())
    result = trainer.training_step(model(), batch, settings(), torch.Generator(), CPU)
    assert result["used_rows"] == 2
    assert result["excluded_rows"] == 5
    assert result["weights"].tolist() == [1.0, 1.0]
    assert result["loss"].item() == 4.0


@pytest.mark.parametrize("model_class,profile", [
    (SASRecH, EVENT_ONLY_TRAINING_PROFILE), (SASRecEventOnly, TRAINING_PROFILE),
    (SASRecEventOnly, None),
])
def test_profile_mismatch_is_refused_before_any_optimizer(model_class, profile, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("optimizer allocated before profile refusal")

    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    plan = TrainingPlan([sample()], settings(training_profile=profile), seed=41)
    with pytest.raises(ValueError, match="event-only capability must agree"):
        train_model(model(model_class), plan)


def test_non_gap_weighting_is_not_silently_reinterpreted():
    with pytest.raises(ValueError, match="one unit-weight row per gap"):
        trainer.resolve_training_settings(settings(landmark_weighting="per_landmark"))


def test_tiny_learning_prefilters_before_batching_and_reloads_checkpoint(tmp_path):
    torch.manual_seed(41)
    instance = model()
    samples = [sample() for _ in range(8)] + [sample(query=3600) for _ in range(24)]
    samples += [sample(1), replace(sample(), residual_time_sec=0)]
    plan = TrainingPlan(samples, settings(), seed=41, artifacts_dir=tmp_path / "run")
    summary = train_model(instance, plan)
    assert summary["training_profile"] == EVENT_ONLY_TRAINING_PROFILE
    assert summary["training_view"] == EVENT_ONLY_TRAINING_PROFILE
    assert summary["weighting_contract_version"] == EVENT_ONLY_WEIGHTING_CONTRACT
    assert summary["source_train_row_count"] == 34
    assert summary["train_row_count"] == 8
    assert summary["prefilter_excluded_rows"] == 26
    assert summary["batches_per_epoch"] == 1
    assert summary["optimizer_updates"] == 12
    assert summary["used_rows_total"] == 96
    assert summary["excluded_rows_total"] == 0
    assert summary["final_train_loss"] < summary["first_train_loss"]
    restored, _payload = load_model(summary["checkpoints"]["best"])
    assert isinstance(restored, SASRecEventOnly)
    batch = collate_canonical_samples([sample()])
    assert instance.predict(batch).poi_logits.argmax(dim=1).item() == 3
    assert torch.equal(instance.predict(batch).poi_logits, restored.predict(batch).poi_logits)
    curve = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert len(curve) == 12
    assert all(epoch["used_rows"] == 8 and epoch["excluded_rows"] == 0 for epoch in curve)


def test_training_plan_filters_parquet_without_changing_validation(synthetic_benchmark, tmp_path):
    source = LandmarkDataset(synthetic_benchmark["landmarks"], split="train")
    validation = LandmarkDataset(synthetic_benchmark["landmarks"], split="validation")
    before = [(sample.user_id, sample.query_s_sec) for sample in validation]
    selected = trainer.event_only_training_source(source)
    assert isinstance(selected, LandmarkDataset)
    assert selected.source_row_count == len(source)
    assert len(selected) < len(source)
    assert all(sample.query_s_sec == 0 for sample in selected)
    assert [(sample.user_id, sample.query_s_sec) for sample in validation] == before
    with pytest.raises(ValueError, match="split='train'"):
        trainer.event_only_training_source(validation)


def test_event_only_epoch_resume_matches_uninterrupted_training(tmp_path):
    samples = [sample() for _ in range(8)] + [sample(query=3600) for _ in range(8)]

    def run(destination, *, max_epochs, resume=None):
        torch.manual_seed(41)
        instance = model()
        plan = TrainingPlan(
            samples, settings(max_epochs=max_epochs), seed=41, artifacts_dir=destination,
            resume_from=resume,
        )
        return instance, train_model(instance, plan)

    full, full_summary = run(tmp_path / "full", max_epochs=3)
    # A stop after the second completed epoch leaves last.pt at epoch 2, exactly what
    # an interrupted three-epoch run would leave behind.
    run(tmp_path / "interrupted", max_epochs=2)
    resumed, summary = run(
        tmp_path / "resumed", max_epochs=3, resume=tmp_path / "interrupted/last.pt"
    )
    assert summary["optimizer_updates"] == full_summary["optimizer_updates"] == 3
    assert summary["optimizer_updates_this_segment"] == 1
    assert summary["used_rows_total"] == full_summary["used_rows_total"] == 24
    assert summary["train_row_count"] == 8
    assert all(torch.equal(value, resumed.state_dict()[key])
               for key, value in full.state_dict().items())
