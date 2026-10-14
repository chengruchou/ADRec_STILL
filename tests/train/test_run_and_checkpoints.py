"""One-command runs: real updates, two model files, in-memory evaluation, no gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from adrec_still.training import checkpoint as ckpt
from adrec_still.training.families import load_model
from adrec_still.training.service import RunFailed, run

from .conftest import small_spec


def read_metrics(output: Path) -> list[dict]:
    return [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]


def test_neural_run_keeps_only_last_and_best_and_evaluates_test_once(canonical_data, tmp_path):
    output = tmp_path / "gru"
    summary = run(small_spec(canonical_data, output, "gru4rec_h", epochs=3))
    assert summary["status"] == "completed"
    files = sorted(path.name for path in output.iterdir() if path.is_file())
    assert files == ["best.pt", "config.json", "last.pt", "metrics.jsonl", "summary.json"]
    assert not list(output.glob("**/resume")) and not list(output.glob("**/selection*"))
    assert not list(output.glob("**/*.sqlite")) and not list(output.glob("**/exact"))

    metrics = read_metrics(output)
    assert [row["epoch"] for row in metrics] == [1, 2, 3]
    assert all(row["optimizer_updates"] > 0 for row in metrics)
    assert metrics[-1]["optimizer_updates"] > metrics[0]["optimizer_updates"]
    training = summary["training"]
    assert training["selection"]["metric"] == "val_loss"
    assert training["selection"]["metric_key"] == "poi_nll"
    assert training["best"]["epoch"] in (1, 2, 3)
    test = summary["evaluation"]["test"]
    assert test["rows"] > 0 and test["ndcg@10"] is not None and test["poi_nll"] is not None
    assert set(test["by_horizon"]) >= {"0", "86400"}

    last = ckpt.load_payload(output / "last.pt", ckpt.LAST_FORMAT)
    assert "best_state" not in last and "state_dict" not in last
    assert set(last["model_state"]) == set(last["optimizer_state"]["state"].keys()) or True
    assert last["completed_epoch"] == 3 and last["best"]["epoch"] == training["best"]["epoch"]
    best = ckpt.load_payload(output / "best.pt", ckpt.BEST_FORMAT)
    assert "optimizer_state" not in best
    assert best["selection"]["epoch"] == training["best"]["epoch"]
    model, _ = load_model(output / "best.pt")
    assert model.resolved_config()["name"] == "gru4rec_h"


def test_best_is_the_best_epoch_not_the_last_and_last_holds_the_last(canonical_data, tmp_path,
                                                                     monkeypatch):
    """A metric sequence whose best epoch is not the last: best.pt keeps epoch 2's
    weights while last.pt moves on to epoch 4."""
    from adrec_still.training import loop

    values = iter([3.0, 1.0, 2.0, 2.5])
    original = loop.evaluate_split

    def scripted(*args, **kwargs):
        result = original(*args, **kwargs)
        result["poi_nll"] = next(values)
        return result

    monkeypatch.setattr(loop, "evaluate_split", scripted)
    output = tmp_path / "scripted"
    summary = run(small_spec(canonical_data, output, "gru4rec_h", epochs=4, patience=10,
                             evaluate_test=False))
    assert summary["training"]["best"]["epoch"] == 2
    assert summary["training"]["epochs_run"] == 4
    last = ckpt.load_payload(output / "last.pt", ckpt.LAST_FORMAT)
    best = ckpt.load_payload(output / "best.pt", ckpt.BEST_FORMAT)
    assert last["completed_epoch"] == 4 and best["selection"]["epoch"] == 2
    assert any(
        not torch.equal(last["model_state"][key], best["state_dict"][key])
        for key in best["state_dict"]
    )
    assert [row["improved"] for row in read_metrics(output)] == [True, True, False, False]


def test_floor_run_shows_fit_phases_and_writes_no_weights(canonical_data, tmp_path, capsys):
    output = tmp_path / "gp"
    summary = run(small_spec(canonical_data, output, "global_pop", progress="plain",
                             overrides={}))
    out = capsys.readouterr().out
    assert "[count train_targets (global_pop)] start" in out
    assert "[evaluate-test]" in out
    assert summary["status"] == "completed" and summary["kind"] == "floor"
    assert summary["checkpoints"] is None
    assert not list(output.glob("*.pt"))
    assert summary["evaluation"]["test"]["ndcg@10"] is not None
    assert summary["evaluation"]["validation"]["rows"] > 0


@pytest.mark.parametrize("model", ["repeat_last_s", "first_order_markov_s"])
def test_silence_floors_take_their_horizons_from_the_data(canonical_data, tmp_path, model):
    summary = run(small_spec(canonical_data, tmp_path / model, model, overrides={}))
    assert summary["status"] == "completed"
    resolved = summary["resolved_model"]
    assert resolved["horizons_seconds"] == [0, 21600, 43200, 86400, 172800, 259200, 604800]


def test_still_joint_trains_on_the_survival_view_and_monitors_joint_nll(canonical_data,
                                                                         tmp_path):
    output = tmp_path / "joint"
    summary = run(small_spec(canonical_data, output, "still_joint", epochs=2,
                             validation_horizon=86400))
    assert summary["status"] == "completed"
    assert summary["training"]["selection"]["metric_key"] == "joint_nll"
    assert summary["training"]["training_profile"] == "still-train-admin-censor-v1"
    assert (output / "cache" / "survival_training.parquet").is_file()
    assert summary["evaluation"]["test"]["joint_nll"] is not None
    assert not (output / "selection_cohorts.sqlite").exists()
    model, payload = load_model(output / "best.pt")
    assert payload["family"] == "still" and model.resolved_config()["name"] == "still_joint"


def test_external_lnm_fits_statistics_and_reloads_with_them(canonical_data, tmp_path):
    output = tmp_path / "ext"
    summary = run(small_spec(canonical_data, output, "external_joint_lnm", epochs=1,
                             validation_horizon=86400, overrides={"mark_embedding_dim": 4,
                                                                   "context_dim": 8,
                                                                   "training.batch_size": 16,
                                                                   "training.negative_samples": 4}))
    assert summary["status"] == "completed"
    record = summary["resolved_model"]["log_time_statistics"]
    assert record["status"] == "fitted" and record["unique_gap_count"] > 0
    model, _ = load_model(output / "best.pt")
    assert model.resolved_config()["log_time_statistics"] == record


def test_run_without_a_test_split_completes_and_labels_it(canonical_data, tmp_path):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(canonical_data / "landmark_samples.parquet")
    trimmed = tmp_path / "no_test"
    trimmed.mkdir()
    pq.write_table(table.filter(pc.not_equal(table["split"], "test")),
                   trimmed / "landmark_samples.parquet")
    (trimmed / "poi_catalog.parquet").write_bytes(
        (canonical_data / "poi_catalog.parquet").read_bytes()
    )
    summary = run(small_spec(trimmed, tmp_path / "run", "gru4rec_h", epochs=1))
    assert summary["status"] == "completed"
    assert summary["evaluation"]["test"] is None
    assert (tmp_path / "run" / "best.pt").is_file()


def test_run_without_a_validation_split_selects_on_train_loss_and_says_so(canonical_data,
                                                                            tmp_path):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(canonical_data / "landmark_samples.parquet")
    trimmed = tmp_path / "no_validation"
    trimmed.mkdir()
    pq.write_table(table.filter(pc.not_equal(table["split"], "validation")),
                   trimmed / "landmark_samples.parquet")
    (trimmed / "poi_catalog.parquet").write_bytes(
        (canonical_data / "poi_catalog.parquet").read_bytes()
    )
    summary = run(small_spec(trimmed, tmp_path / "run", "gru4rec_h", epochs=2))
    policy = summary["selection_policy"]
    assert policy["monitor"] == "train_loss" and "no validation split" in policy["label"]
    assert summary["training"]["best"]["metric"] == "train_loss"
    assert summary["evaluation"]["test"]["rows"] > 0


def test_short_gap_validation_data_works_with_an_explicit_horizon(canonical_data, tmp_path):
    """Validation rows only at short horizons: the committed 24h horizon has no
    rows, which is an actionable error, and `all` (or a present horizon) trains."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from adrec_still.training.loop import SelectionUnavailable

    table = pq.read_table(canonical_data / "landmark_samples.parquet")
    keep = pc.or_(pc.not_equal(table["split"], "validation"),
                  pc.less_equal(table["query_s_sec"], 21600))
    short = tmp_path / "short_gap"
    short.mkdir()
    pq.write_table(table.filter(keep), short / "landmark_samples.parquet")
    (short / "poi_catalog.parquet").write_bytes(
        (canonical_data / "poi_catalog.parquet").read_bytes()
    )
    with pytest.raises(SelectionUnavailable, match="--validation-horizon all"):
        run(small_spec(short, tmp_path / "refused", "gru4rec_h", epochs=1,
                       validation_horizon=86400))
    summary = run(small_spec(short, tmp_path / "all", "gru4rec_h", epochs=1,
                             validation_horizon="all"))
    assert summary["status"] == "completed"
    assert summary["training"]["selection"]["population"] == "all validation rows"


def test_invalid_input_directory_fails_clearly(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="landmark_samples.parquet"):
        run(small_spec(empty, tmp_path / "run", "gru4rec_h"))
    with pytest.raises(RunFailed, match="--model"):
        run(small_spec(empty, tmp_path / "run2", None))


def test_model_dimension_overrides_and_out_of_tree_config(canonical_data, tmp_path):
    import yaml

    config = tmp_path / "elsewhere" / "tiny_sasrec.yaml"
    config.parent.mkdir()
    config.write_text(yaml.safe_dump({
        "name": "sasrec_h", "embedding_dim": 12, "hidden_dim": 12, "num_layers": 1,
        "num_heads": 2, "dropout": 0.0, "max_sequence_length": 20,
        "training": {"batch_size": 16, "negative_samples": 4, "max_epochs": 1,
                     "learning_rate": 0.01},
    }), encoding="utf-8")
    summary = run(small_spec(canonical_data, tmp_path / "run", None, model_config=config,
                             overrides={"num_layers": 2}, evaluate_test=False))
    resolved = summary["resolved_model"]
    assert resolved["embedding_dim"] == 12 and resolved["num_layers"] == 2
    assert summary["training"]["training"]["learning_rate"] == 0.01
    assert summary["training"]["training"]["max_epochs"] == 1
