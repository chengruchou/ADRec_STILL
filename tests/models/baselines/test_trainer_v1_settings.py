"""The shared training contract: resolved settings, selection and its scorer surface."""

import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import torch

import adrec_still.training.evaluate as evaluate_module
from adrec_still.data.dataset import LandmarkDataset
from adrec_still.models.baselines.gru4rec import GRU4RecH
from adrec_still.models.baselines.trainer import (
    FIXED_SCHEDULE_SETTINGS,
    resolve_training_settings,
    validate_training_config,
)
from adrec_still.models.still.synthetic_benchmark import write_synthetic_benchmark
from adrec_still.training.evaluate import EvaluationOptions, evaluate_split
from adrec_still.training.loop import SelectionUnavailable, TrainingPlan, train_model

REPOSITORY_ROOT = Path(__file__).parents[3]

V1_TRAINING = {
    "landmark_weighting": "per_gap",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "betas": [0.9, 0.999],
    "eps": 1.0e-8,
    "amsgrad": False,
    "batch_size": 256,
    "max_epochs": 20,
    "early_stopping_patience": 5,
    "early_stopping_metric": "ndcg@10",
    "early_stopping_mode": "max",
    "selection_horizon_sec": 86400,
    "validation_max_batches": None,
    "negative_samples": 128,
}


def make_model(number_of_pois: int) -> GRU4RecH:
    torch.manual_seed(41)
    return GRU4RecH(number_of_pois, embedding_dim=8, hidden_dim=8, num_layers=1, dropout=0.0)


def make_plan(fixture, tmp_path, **overrides) -> TrainingPlan:
    training = dict(V1_TRAINING, **overrides)
    return TrainingPlan(
        train_source=LandmarkDataset(fixture["landmark_path"], split="train"),
        training=training,
        seed=41,
        max_sequence_length=8,
        validation_landmark_path=fixture["landmark_path"],
        validation_batch_size=8,
        artifacts_dir=tmp_path / "artifacts",
    )


def selection_pass(model, landmark_path, horizon=86400, **options):
    return evaluate_split(
        model, landmark_path, split="validation", horizon=horizon,
        options=EvaluationOptions(batch_size=8, max_sequence_length=8, **options),
    )


# --- resolved settings ---------------------------------------------------------------


def test_resolution_fills_the_v1_defaults_and_reports_unconsumed_keys():
    resolved = resolve_training_settings({**V1_TRAINING, "mystery_setting": 3})
    assert resolved["betas"] == (0.9, 0.999)
    assert resolved["eps"] == 1.0e-8 and resolved["amsgrad"] is False
    assert resolved["weight_decay"] == 0.0
    for key, fixed in FIXED_SCHEDULE_SETTINGS.items():
        assert resolved[key] == fixed
    assert resolved["validation_max_batches"] is None
    assert resolved["selection_horizon_sec"] == 86400
    assert resolved["tf32"] is False
    assert resolved["early_stopping_mode"] == "max"
    assert resolved["training_profile"] == "observed-known-v0-proposed"
    assert resolved["unconsumed_training_keys"] == ["mystery_setting"]


def test_a_selection_metric_without_a_source_is_refused_before_any_epoch():
    validate_training_config(dict(V1_TRAINING, early_stopping_metric="joint_nll",
                                  early_stopping_mode="min"))
    validate_training_config(dict(V1_TRAINING, early_stopping_metric="val_loss",
                                  early_stopping_mode=None))
    with pytest.raises(ValueError, match="not a selectable metric"):
        validate_training_config(dict(V1_TRAINING, early_stopping_metric="brier"))


def test_the_selection_direction_follows_the_metric_and_cannot_be_contradicted():
    """NLLs improve downwards, ranking metrics upwards. A config may state the
    direction but never the wrong one; omitting it resolves to the metric's own."""
    for metric in ("poi_nll", "joint_nll", "val_loss", "train_loss"):
        with pytest.raises(ValueError, match="contradicts"):
            validate_training_config(dict(V1_TRAINING, early_stopping_metric=metric,
                                          early_stopping_mode="max"))
        resolved = resolve_training_settings(
            dict(V1_TRAINING, early_stopping_metric=metric, early_stopping_mode=None)
        )
        assert resolved["early_stopping_mode"] == "min"
    for metric in ("ndcg@10", "recall@20", "mrr@5"):
        with pytest.raises(ValueError, match="contradicts"):
            validate_training_config(dict(V1_TRAINING, early_stopping_metric=metric,
                                          early_stopping_mode="min"))
        assert resolve_training_settings(
            dict(V1_TRAINING, early_stopping_metric=metric)
        )["early_stopping_mode"] == "max"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("scheduler", "cosine"),
        ("gradient_clipping", 1.0),
        ("amp", True),
        ("gradient_accumulation", 2),
        ("drop_last", True),
        ("min_delta", 0.01),
    ],
)
def test_unimplemented_schedule_settings_are_refused_not_ignored(key, value):
    with pytest.raises(ValueError, match="not implemented"):
        validate_training_config(dict(V1_TRAINING, **{key: value}))


def test_selection_horizon_and_caps_are_validated():
    """Any non-negative horizon is a legal request (the data decides whether rows
    exist there); negatives, booleans and empty caps are refused up front."""
    validate_training_config(dict(V1_TRAINING, selection_horizon_sec=12345))
    with pytest.raises(ValueError, match="non-negative"):
        validate_training_config(dict(V1_TRAINING, selection_horizon_sec=-1))
    with pytest.raises(ValueError, match="non-negative"):
        validate_training_config(dict(V1_TRAINING, selection_horizon_sec=True))
    with pytest.raises(ValueError, match="validation_max_batches"):
        validate_training_config(dict(V1_TRAINING, validation_max_batches=0))
    with pytest.raises(ValueError, match="patience"):
        validate_training_config(dict(V1_TRAINING, early_stopping_patience=0))


def test_adam_receives_every_resolved_optimizer_setting(tmp_path, monkeypatch):
    fixture = write_synthetic_benchmark(tmp_path / "tables")
    captured = {}
    real_adam = torch.optim.Adam

    def capturing_adam(params, **kwargs):
        captured.update(kwargs)
        return real_adam(params, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", capturing_adam)
    model = make_model(fixture["number_of_pois"])
    plan = make_plan(fixture, tmp_path, max_epochs=1, betas=[0.85, 0.995], eps=2.0e-8)
    plan.max_train_batches = 1
    summary = train_model(model, plan)
    assert captured == {
        "lr": 0.001,
        "weight_decay": 0.0,
        "betas": (0.85, 0.995),
        "eps": 2.0e-8,
        "amsgrad": False,
    }
    recorded = summary["training"]
    assert recorded["optimizer"] == "adam"
    assert recorded["learning_rate"] == 0.001
    assert recorded["betas"] == [0.85, 0.995]
    assert recorded["eps"] == 2.0e-8 and recorded["amsgrad"] is False


# --- selection: population, scorer surface, ties ---------------------------------------


def _fake_exact_contributions(per_row_values):
    """An exact_ranking_contributions stand-in returning one value per pass.

    ``per_row_values[i]`` is the constant contribution every eligible row receives on
    the i-th call; with a validation batch size covering the whole split, one call is
    one selection pass, so the mean the trainer computes equals that constant. The
    stand-in also records the score kind it was handed, so a caller cannot switch
    score sources without the test noticing.
    """
    calls = {"count": 0, "row_counts": [], "score_kinds": []}

    def fake(score_candidates, targets, known, observed, number_of_pois, **kwargs):
        value = per_row_values[calls["count"]]
        calls["count"] += 1
        calls["row_counts"].append(int(targets.shape[0]))
        calls["score_kinds"].append(kwargs.get("score_kind"))
        eligible = observed.bool() & known.bool()
        per_row = torch.zeros(int(targets.shape[0]), dtype=torch.float64)
        per_row[eligible] = value
        return {
            "eligible_mask": eligible,
            "values": {"ndcg@10": per_row},
            "poi_nll_supported": False,
        }

    return fake, calls


def test_horizon_selection_scores_the_horizon_rows_through_the_exact_adapter(
    tmp_path, monkeypatch
):
    """The selection pass feeds only the horizon's rows to the exact adapter; the best
    epoch follows the exact values and the summary names the population."""
    fixture = write_synthetic_benchmark(tmp_path / "tables")
    table = pq.read_table(fixture["landmark_path"])
    horizon_rows = int(pc.sum(pc.and_(
        pc.equal(table.column("query_s_sec"), 86400),
        pc.equal(table.column("split"), "validation"),
    )).as_py())
    assert horizon_rows > 0, "the fixture must carry a natural 24h validation risk set"
    fake, calls = _fake_exact_contributions([0.1, 0.4])
    monkeypatch.setattr(evaluate_module, "exact_ranking_contributions", fake)
    model = make_model(fixture["number_of_pois"])
    plan = make_plan(fixture, tmp_path, max_epochs=2)
    plan.validation_batch_size = 64  # whole split in one batch: one call per pass
    plan.max_train_batches = 1
    summary = train_model(model, plan)
    assert calls["count"] == 2
    assert calls["row_counts"] == [horizon_rows, horizon_rows]
    assert summary["best"]["epoch"] == 2
    assert summary["best"]["value"] == pytest.approx(0.4)
    assert summary["selection"]["horizon"] == 86400
    assert summary["selection"]["population"] == "validation rows at query_s_sec=86400"
    # Encoder baselines are scored through their raw encoder surface, as logits.
    assert calls["score_kinds"] == ["logits", "logits"]


def test_a_validation_stream_without_the_selection_horizon_is_refused(tmp_path):
    fixture = write_synthetic_benchmark(tmp_path / "tables")
    table = pq.read_table(fixture["landmark_path"])
    without_horizon = table.filter(pc.not_equal(table.column("query_s_sec"), 86400))
    filtered_path = tmp_path / "no_24h_landmarks.parquet"
    pq.write_table(without_horizon, filtered_path)
    model = make_model(fixture["number_of_pois"])
    plan = TrainingPlan(
        train_source=LandmarkDataset(filtered_path, split="train"),
        training=dict(V1_TRAINING),
        seed=41,
        max_sequence_length=8,
        validation_landmark_path=filtered_path,
        validation_batch_size=8,
        artifacts_dir=tmp_path / "artifacts",
        max_train_batches=1,
    )
    with pytest.raises(SelectionUnavailable, match="no rows at selection horizon 86400"):
        train_model(model, plan)


def test_a_truncated_prefix_without_horizon_rows_is_refused_not_defaulted(tmp_path):
    """validation_max_batches caps the selection stream; a prefix that never reaches
    the horizon's rows refuses instead of substituting another horizon."""
    fixture = write_synthetic_benchmark(tmp_path / "tables")
    model = make_model(fixture["number_of_pois"])
    plan = make_plan(fixture, tmp_path, max_epochs=1, validation_max_batches=1)
    plan.validation_batch_size = 1
    plan.max_train_batches = 1
    with pytest.raises(SelectionUnavailable, match="no ndcg@10 value"):
        train_model(model, plan)


def test_selection_stop_condition_discards_the_partial_pass(tmp_path):
    fixture = write_synthetic_benchmark(tmp_path / "tables")
    model = make_model(fixture["number_of_pois"])
    model.is_fitted = True
    state = {"polls": 0}

    def stop() -> bool:
        state["polls"] += 1
        return state["polls"] >= 1

    result = evaluate_split(
        model, fixture["landmark_path"], split="validation", horizon=86400,
        options=EvaluationOptions(batch_size=1, max_sequence_length=8),
        stop_condition=stop,
    )
    assert result["stopped"] is True


def test_exact_ties_keep_the_earlier_checkpoint_and_patience_counts_validations(
    tmp_path, monkeypatch
):
    fixture = write_synthetic_benchmark(tmp_path / "tables")
    fake, _ = _fake_exact_contributions([0.3] * 20)
    monkeypatch.setattr(evaluate_module, "exact_ranking_contributions", fake)
    model = make_model(fixture["number_of_pois"])
    plan = make_plan(fixture, tmp_path, max_epochs=20, early_stopping_patience=3)
    plan.validation_batch_size = 64
    plan.max_train_batches = 1
    summary = train_model(model, plan)
    assert summary["best"]["epoch"] == 1, "an exact tie never replaces the earlier checkpoint"
    assert summary["epochs_run"] == 4, "three completed validations without improvement"
    assert summary["early_stopped"] is True
    assert summary["training"]["min_delta"] == 0.0


def test_tied_scores_select_by_the_exact_tie_policy_not_v02_topk(tmp_path):
    """A model whose catalog scores are exactly equal: equal scores order by ascending
    POI ID, so a known observed target ``t`` (IDs start at 2) sits at rank ``t - 1``
    and its NDCG@10 contribution is ``1/log2(t)`` when ``t - 1 <= 10``."""
    fixture = write_synthetic_benchmark(tmp_path / "tables")
    model = make_model(fixture["number_of_pois"])
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.is_fitted = True
    value = selection_pass(model, fixture["landmark_path"])["ndcg@10"]

    rows = pq.read_table(fixture["landmark_path"]).to_pylist()
    contributions = []
    for row in rows:
        if (
            row["split"] == "validation"
            and row["query_s_sec"] == 86400
            and row["event_observed"]
            and row["target_in_train_catalog"]
        ):
            rank = row["next_poi"] - 1  # equal scores: ascending POI ID, IDs start at 2
            contributions.append(1.0 / math.log2(rank + 1) if rank <= 10 else 0.0)
    assert contributions, "the fixture must have eligible tied rows at 24h"
    assert value == pytest.approx(sum(contributions) / len(contributions), abs=1e-12)


class _NearTieEncoder(GRU4RecH):
    """A real encoder whose raw catalog scores separate below the float32 log-softmax
    step: POI 13 leads by 7.1399448e-8, every other catalog POI scores exactly 0."""

    LEADING_POI = 13
    RAW_MARGIN = 7.1399448e-8

    def score_full_catalog(self, context: torch.Tensor) -> torch.Tensor:
        scores = torch.zeros(
            context.shape[0], self.number_of_pois, dtype=torch.float32, device=context.device
        )
        scores[:, self.LEADING_POI] = self.RAW_MARGIN
        return scores

    def score_candidates(
        self, context: torch.Tensor, candidate_poi: torch.Tensor
    ) -> torch.Tensor:
        return self.score_full_catalog(context).gather(1, candidate_poi.to(context.device))


def _target_every_eligible_24h_row(fixture, poi_id: int, destination: Path) -> Path:
    """Rewrite the fixture so every eligible 24h validation row targets ``poi_id``."""
    table = pq.read_table(fixture["landmark_path"])
    eligible = pc.and_(
        pc.and_(
            pc.equal(table.column("split"), "validation"),
            pc.equal(table.column("query_s_sec"), 86400),
        ),
        pc.and_(table.column("event_observed"), table.column("target_in_train_catalog")),
    )
    assert pc.sum(eligible).as_py() > 0, "the fixture must have eligible 24h rows"
    column = table.column("next_poi")
    retargeted = table.set_column(
        table.schema.get_field_index("next_poi"),
        "next_poi",
        pc.if_else(eligible, pa.scalar(poi_id, type=column.type), column),
    )
    pq.write_table(retargeted, destination)
    return destination


def test_selection_scores_the_raw_encoder_surface_not_the_renormalized_predict(tmp_path):
    """A float32 log-softmax turns a real raw ordering into an exact tie. Selection
    scores the encoder's raw surface, where the leading POI ranks first."""
    from adrec_still.data.collate import collate_canonical_samples, select_batch_rows
    from adrec_still.evaluation.exact import dense_exact_contributions

    fixture = write_synthetic_benchmark(tmp_path / "tables")
    landmark_path = _target_every_eligible_24h_row(
        fixture, _NearTieEncoder.LEADING_POI, tmp_path / "near_tie_landmarks.parquet"
    )
    model = _NearTieEncoder(
        fixture["number_of_pois"], embedding_dim=8, hidden_dim=8, num_layers=1, dropout=0.0
    )
    model.is_fitted = True
    dataset = LandmarkDataset(landmark_path, split="validation")
    batch = collate_canonical_samples(
        [dataset[i] for i in range(len(dataset))], max_sequence_length=8
    )
    rows = batch["query_s_sec"].eq(86400)
    sub_batch = select_batch_rows(batch, rows)
    normalized = model.predict(sub_batch).poi_logits
    assert model.RAW_MARGIN > 0.0
    assert torch.equal(normalized[:, _NearTieEncoder.LEADING_POI], normalized[:, 2])
    old_route = dense_exact_contributions(
        normalized, sub_batch["next_poi"], sub_batch["target_in_train_catalog"],
        sub_batch["observed_mask"], score_kind="logits",
    )
    old_eligible = old_route["eligible_mask"]
    old_value = float(old_route["values"]["ndcg@10"][old_eligible].sum()) / int(
        old_eligible.sum()
    )
    assert old_value == pytest.approx(0.0), "tied scores put POI 13 at rank 12, outside K=10"

    result = selection_pass(model, landmark_path)
    assert result["score_source"] == "encoder-candidate-factory"
    assert result["ndcg@10"] == pytest.approx(1.0), "the raw scores rank POI 13 first"


def test_the_selection_scorer_receives_forecast_fields_only(tmp_path):
    """The label-leakage rule, asserted from inside the scorer: the selection pass may
    hand it the six INFERENCE_FIELDS and nothing else."""
    from adrec_still.evaluation.scoring import INFERENCE_FIELDS

    seen: list[set[str]] = []

    class _RecordingEncoder(GRU4RecH):
        def encode(self, batch):
            seen.append(set(batch))
            return super().encode(batch)

    fixture = write_synthetic_benchmark(tmp_path / "tables")
    model = _RecordingEncoder(
        fixture["number_of_pois"], embedding_dim=8, hidden_dim=8, num_layers=1, dropout=0.0
    )
    model.is_fitted = True
    result = selection_pass(model, fixture["landmark_path"])
    assert result["ndcg@10"] is not None
    assert seen, "the selection pass must have built a scorer"
    for keys in seen:
        assert keys == set(INFERENCE_FIELDS)


def test_a_model_with_only_predict_is_scored_densely_and_says_so(tmp_path):
    """Deterministic floors publish normalized log-probabilities through predict()
    only; they are scored through the dense adapter under the same tie policy, and
    the result records that source rather than pretending to a raw surface."""

    from adrec_still.models.baselines.harness import build_fitted_model

    fixture = write_synthetic_benchmark(tmp_path / "tables")
    model = build_fitted_model({"name": "global_pop"}, Path(fixture["landmark_path"]).parent)
    assert not hasattr(model, "encode")
    result = selection_pass(model, fixture["landmark_path"])
    assert result["score_source"] == "dense-predict"
    assert result["score_kind"] == "log_probabilities"
    assert result["ndcg@10"] is not None


def test_the_survival_profile_needs_a_model_adapter_and_says_so_before_training(tmp_path):
    fixture = write_synthetic_benchmark(tmp_path / "tables")
    model = make_model(fixture["number_of_pois"])
    plan = make_plan(
        fixture, tmp_path, max_epochs=1, training_profile="still-train-admin-censor-v1"
    )
    plan.max_train_batches = 1
    with pytest.raises(TypeError, match="per_row_survival_training_losses"):
        train_model(model, plan)


def test_joint_nll_selection_needs_a_joint_scorer(tmp_path):
    fixture = write_synthetic_benchmark(tmp_path / "tables")
    model = make_model(fixture["number_of_pois"])
    plan = make_plan(fixture, tmp_path, max_epochs=1, early_stopping_metric="joint_nll",
                     early_stopping_mode="min")
    with pytest.raises(TypeError, match="score_joint"):
        train_model(model, plan)


def test_fixture_smoke_selects_on_the_real_24h_rows_via_the_exact_adapter(tmp_path):
    """No stub: end-to-end selection on the fixture's 24h risk set. The recorded value
    reproduces through an independent pass of the encoder scorer, and (the trained
    scores being untied) agrees with the dense v0.2 block to rounding."""
    from adrec_still.data.collate import collate_canonical_samples, select_batch_rows
    from adrec_still.evaluation.exact import exact_ranking_contributions
    from adrec_still.evaluation.scoring import INFERENCE_FIELDS, encoder_candidate_factory
    from adrec_still.models.baselines.harness import evaluate_split as harness_evaluate_split

    fixture = write_synthetic_benchmark(tmp_path / "tables")
    model = make_model(fixture["number_of_pois"])
    plan = make_plan(fixture, tmp_path, max_epochs=2, batch_size=16, negative_samples=4)
    summary = train_model(model, plan)
    assert summary["best"]["epoch"] >= 1

    dataset = LandmarkDataset(fixture["landmark_path"], split="validation")
    batch = collate_canonical_samples(
        [dataset[i] for i in range(len(dataset))], max_sequence_length=8
    )
    rows = batch["query_s_sec"].eq(86400)
    sub_batch = select_batch_rows(batch, rows)
    features = {
        key: (value.clone() if isinstance(value := sub_batch[key], torch.Tensor)
              else list(value))
        for key in INFERENCE_FIELDS if key in sub_batch
    }
    contributions = exact_ranking_contributions(
        encoder_candidate_factory(model)(features),
        sub_batch["next_poi"],
        sub_batch["target_in_train_catalog"],
        sub_batch["observed_mask"],
        model.number_of_pois,
        score_kind="logits",
    )
    eligible = contributions["eligible_mask"]
    expected = float(contributions["values"]["ndcg@10"][eligible].sum()) / int(eligible.sum())
    assert summary["best"]["value"] == pytest.approx(expected, abs=1e-12)

    validation = harness_evaluate_split(
        model, fixture["landmark_path"], split="validation", batch_size=8,
        max_sequence_length=8,
    )
    block = validation["by_horizon_seconds"]["86400"]["ranking"]
    assert block["sample_count"] > 0
    assert summary["best"]["value"] == pytest.approx(block["ndcg@10"], abs=1e-9)


def test_without_a_validation_split_selection_falls_back_to_train_loss_and_says_so(tmp_path):
    fixture = write_synthetic_benchmark(tmp_path / "tables")
    model = make_model(fixture["number_of_pois"])
    plan = make_plan(fixture, tmp_path, max_epochs=2)
    plan.validation_landmark_path = None
    plan.max_train_batches = 1
    summary = train_model(model, plan)
    assert summary["selection"]["metric"] == "train_loss"
    assert summary["selection"]["requested_metric"] == "ndcg@10"
    assert "no validation split" in summary["selection"]["note"]
    assert summary["best"]["metric"] == "train_loss"
    assert summary["best"]["value"] == min(
        json.loads(line)["train_loss"]
        for line in (tmp_path / "artifacts" / "metrics.jsonl").read_text().splitlines()
    )


def test_summary_records_execution_settings_and_actual_update_counts(tmp_path):
    fixture = write_synthetic_benchmark(tmp_path / "tables")
    model = make_model(fixture["number_of_pois"])
    plan = make_plan(fixture, tmp_path, max_epochs=2, batch_size=8, negative_samples=4)
    plan.max_train_batches = 3
    summary = train_model(model, plan)
    assert summary["optimizer_updates"] == 3 * summary["epochs_run"]
    assert summary["batches_per_epoch"] == -(-summary["train_row_count"] // 8)
    execution = summary["execution_settings"]
    assert execution["amp"] is False
    assert execution["tf32_requested"] is False
    assert execution["neural_dtype"] == "torch.float32"
    assert execution["time_math_dtype"] == "torch.float64"
    assert execution["torch_version"] == torch.__version__
    assert isinstance(execution["deterministic_algorithms"], bool)
    for key, fixed in FIXED_SCHEDULE_SETTINGS.items():
        assert summary["training"][key] == fixed
    assert summary["training"]["max_epochs"] == 2
    assert summary["training"]["early_stopping_patience"] == 5
    curve = [json.loads(line)
             for line in (tmp_path / "artifacts" / "metrics.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in curve] == [1, 2]
    assert all(row["learning_rate"] == 0.001 for row in curve)
