"""Regressions from the #102 review: derived-cache staleness, inherited checkpoints,
loading across data statistics, resume freedom, batch caps, config paths, batches."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from adrec_still.training import checkpoint as ckpt
from adrec_still.training.evaluate import EvaluationOptions, evaluate_split
from adrec_still.training.families import survival_view
from adrec_still.training.inputs import resolve_data
from adrec_still.training.service import RunSpec, config_base, resolve_config_path, run

from .conftest import small_spec

REPOSITORY_ROOT = Path(__file__).parents[2]


def _shifted_copy(raw_file: Path, destination: Path, years: int) -> Path:
    """The same check-ins (same POIs, same users) moved ``years`` later."""
    lines = []
    for line in raw_file.read_text(encoding="utf-8").splitlines():
        user, moment, latitude, longitude, poi = line.split("\t")
        moment = str(int(moment[:4]) + years) + moment[4:]
        lines.append(f"{user}\t{moment}\t{latitude}\t{longitude}\t{poi}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


# ------------------------------------------------------------------ P1: survival view cache


def test_survival_view_is_rebuilt_when_the_cached_data_changes(raw_gowalla_file, tmp_path):
    cache = tmp_path / "shared_cache"
    data_a = resolve_data(raw_gowalla_file, cache_dir=cache)
    view_a, manifest_a = survival_view(data_a, cache)
    first_a = pq.read_table(view_a, columns=["last_timestamp"])["last_timestamp"][0].as_py()

    raw_b = _shifted_copy(raw_gowalla_file, tmp_path / "loc-gowalla_totalCheckins.txt", 1)
    data_b = resolve_data(raw_b, cache_dir=cache)
    assert data_b.identity == data_a.identity, "same POI mapping, later timestamps"
    view_b, manifest_b = survival_view(data_b, cache)
    first_b = pq.read_table(view_b, columns=["last_timestamp"])["last_timestamp"][0].as_py()
    assert manifest_b["built_from"] != manifest_a["built_from"]
    assert first_b == first_a + 366 * 86400, "the view was rebuilt from B's events"
    # Unchanged inputs are reused.
    _, manifest_again = survival_view(data_b, cache)
    assert manifest_again["built_from"] == manifest_b["built_from"]


# ------------------------------------------------------------ P1: inherited best on resume


def test_cross_directory_resume_saves_the_inherited_best(canonical_data, tmp_path, monkeypatch):
    from adrec_still.training import loop

    values = iter([1.0, 2.0, 3.0])
    original = loop.evaluate_split

    def scripted(*args, **kwargs):
        result = original(*args, **kwargs)
        result["poi_nll"] = next(values)
        return result

    monkeypatch.setattr(loop, "evaluate_split", scripted)
    first = run(small_spec(canonical_data, tmp_path / "first", "gru4rec_h", epochs=1,
                           evaluate_test=False))
    assert first["training"]["best"]["epoch"] == 1
    resumed = run(small_spec(canonical_data, tmp_path / "second", "gru4rec_h", epochs=3,
                             resume=tmp_path / "first" / "last.pt", evaluate_test=False))
    training = resumed["training"]
    assert training["best"]["epoch"] == 1, "nothing after the resume improved"
    assert training["checkpoints"]["best"] == str(tmp_path / "second" / "best.pt")
    assert (tmp_path / "second" / "best.pt").is_file()
    assert (tmp_path / "second" / "last.pt").is_file()
    inherited = ckpt.load_payload(tmp_path / "second" / "best.pt", ckpt.BEST_FORMAT)
    original_best = ckpt.load_payload(tmp_path / "first" / "best.pt", ckpt.BEST_FORMAT)
    assert inherited["selection"]["epoch"] == 1
    for key, value in original_best["state_dict"].items():
        assert torch.equal(value, inherited["state_dict"][key])
    # A third segment from the second directory still finds the historical best.
    third = run(small_spec(canonical_data, tmp_path / "third", "gru4rec_h", epochs=3,
                           resume=tmp_path / "second" / "last.pt", evaluate_test=False))
    assert third["training"]["best"]["epoch"] == 1
    assert third["training"]["historical_best"] is None


def test_resuming_a_finished_run_reports_only_files_that_exist(canonical_data, tmp_path):
    run(small_spec(canonical_data, tmp_path / "done", "gru4rec_h", epochs=1, evaluate_test=False))
    again = run(small_spec(canonical_data, tmp_path / "again", "gru4rec_h", epochs=1,
                           resume=tmp_path / "done" / "last.pt", evaluate_test=False))
    checkpoints = again["training"]["checkpoints"]
    assert again["training"]["epochs_run"] == 1
    for name in ("last", "best"):
        assert checkpoints[name] is not None and Path(checkpoints[name]).is_file()


# ----------------------------------------------------- P1: external LNM statistics on load


def test_external_lnm_loads_across_data_with_different_statistics(raw_gowalla_file, tmp_path):
    overrides = {"mark_embedding_dim": 4, "context_dim": 8, "training.batch_size": 16,
                 "training.negative_samples": 4}
    data_a = resolve_data(raw_gowalla_file, cache_dir=tmp_path / "a")
    first = run(small_spec(data_a.processed_root, tmp_path / "first", "external_joint_lnm",
                           epochs=1, validation_horizon=86400, overrides=overrides,
                           evaluate_test=False))
    recorded = first["resolved_model"]["log_time_statistics"]

    raw_b = _shifted_copy(raw_gowalla_file, tmp_path / "loc-gowalla_totalCheckins.txt", 1)
    data_b = resolve_data(raw_b, cache_dir=tmp_path / "b")
    resumed = run(small_spec(data_b.processed_root, tmp_path / "resumed", "external_joint_lnm",
                             epochs=2, validation_horizon=86400, overrides=overrides,
                             resume=tmp_path / "first" / "last.pt", evaluate_test=False))
    assert resumed["training"]["epochs_run"] == 2
    config = json.loads((tmp_path / "resumed" / "config.json").read_text(encoding="utf-8"))
    restored = config["training_source"]["log_time_statistics"]
    assert restored["restored_from"].endswith("last.pt")
    assert restored["log_time_mean"] == recorded["log_time_mean"]
    warm = run(small_spec(data_b.processed_root, tmp_path / "warm", "external_joint_lnm",
                          epochs=1, validation_horizon=86400, overrides=overrides,
                          weights=tmp_path / "first" / "best.pt", evaluate_test=False))
    assert warm["resolved_model"]["log_time_statistics"]["log_time_std"] == recorded["log_time_std"]


# ------------------------------------------------------------- P2: resume freedom


@pytest.mark.skipif(
    os.name == "nt",
    reason=(
        "pre-existing donor gap on Windows: num_workers=2 spawns DataLoader workers "
        "by pickling, and the dataset's pyarrow internals have no __reduce__, so "
        "worker start fails (donor CI is ubuntu-latest, where fork avoids pickling; "
        "it fails identically in the donor repository on this machine)"
    ),
)
def test_resume_allows_workers_batch_size_and_monitor_changes(canonical_data, tmp_path):
    first = run(small_spec(canonical_data, tmp_path / "first", "gru4rec_h", epochs=1,
                           evaluate_test=False))
    changed = run(small_spec(
        canonical_data, tmp_path / "changed", "gru4rec_h", epochs=2,
        resume=tmp_path / "first" / "last.pt", evaluate_test=False, num_workers=2,
        batch_size=8, learning_rate=0.002, monitor="ndcg@10",
    ))
    training = changed["training"]
    assert training["epochs_run"] == 2
    assert "dataloader_workers" in training["resumed_from"]["rng"]["skipped"]
    changes = training["resumed_from"]["setting_changes"]
    assert any("training.batch_size" in change for change in changes)
    assert any("training.learning_rate" in change for change in changes)
    # The monitor changed, so best selection restarted at the resumed epoch and says so.
    assert training["best"]["metric"] == "ndcg@10"
    assert training["best_scope_start_epoch"] == 2
    assert training["historical_best"]["metric"] == "val_loss"
    assert first["training"]["optimizer_updates"] < training["optimizer_updates"]
    # The optimizer type is the one setting the restored state binds; the training
    # contract refuses a different optimizer before the checkpoint is even compared.
    with pytest.raises(ValueError, match="optimizer"):
        run(small_spec(canonical_data, tmp_path / "sgd", "gru4rec_h", epochs=2,
                       resume=tmp_path / "first" / "last.pt",
                       overrides={"training.optimizer": "sgd", "embedding_dim": 8,
                                  "hidden_dim": 8, "training.negative_samples": 4,
                                  "training.batch_size": 16}))
    problems = ckpt.recipe_differences({"optimizer": "adam"}, {"optimizer": "sgd"})
    assert problems and "training.optimizer" in problems[0]


# ------------------------------------------------------------- P2: eval batch cap


def test_buffered_horizon_rows_never_exceed_the_eval_batch_size(canonical_data):
    from adrec_still.models.baselines.gru4rec import GRU4RecH

    number_of_pois = ckpt.number_of_pois(canonical_data / "poi_catalog.parquet")
    model = GRU4RecH(number_of_pois, embedding_dim=8, hidden_dim=8, num_layers=1, dropout=0.0)
    model.is_fitted = True
    seen: list[int] = []
    original = model.encode

    def recording(batch):
        seen.append(int(batch["query_s_sec"].shape[0]))
        return original(batch)

    model.encode = recording
    landmarks = canonical_data / "landmark_samples.parquet"
    table = pq.read_table(landmarks, columns=["split", "query_s_sec"])
    at_zero = int(pc.sum(pc.and_(pc.equal(table["split"], "validation"),
                                 pc.equal(table["query_s_sec"], 0))).as_py())
    for size in (4, 7):
        seen.clear()
        result = evaluate_split(model, landmarks, split="validation", horizon=0,
                                options=EvaluationOptions(batch_size=size, max_sequence_length=8))
        assert result["rows"] == at_zero
        assert max(seen) <= size
        assert sum(seen) == at_zero
        assert seen[:-1] == [size] * (len(seen) - 1), "full batches first, remainder last"


# ------------------------------------------------------------- P2: config paths


def test_config_relative_paths_do_not_depend_on_the_working_directory(canonical_data, tmp_path,
                                                                     monkeypatch):
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    model = {"name": "gru4rec_h", "embedding_dim": 12, "hidden_dim": 8, "num_layers": 1,
             "dropout": 0.0, "training": {"batch_size": 16, "negative_samples": 4,
                                          "max_epochs": 1}}
    (project / "configs" / "model.yaml").write_text(yaml.safe_dump(model), encoding="utf-8")
    dataset = {"name": "gowalla", "processed_root": "data/processed"}
    (project / "configs" / "dataset.yaml").write_text(yaml.safe_dump(dataset), encoding="utf-8")
    shutil.copytree(canonical_data, project / "data" / "processed")
    experiment = {"name": "x", "model_config": "configs/model.yaml",
                  "dataset_config": "configs/dataset.yaml", "seeds": [41],
                  "output_dir": "runs/x", "evaluation": {"batch_size": 16}}
    experiment_path = project / "configs" / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")

    elsewhere = tmp_path / "elsewhere" / "configs"
    elsewhere.mkdir(parents=True)
    decoy = dict(model, embedding_dim=64)
    (elsewhere / "model.yaml").write_text(yaml.safe_dump(decoy), encoding="utf-8")
    monkeypatch.chdir(elsewhere.parent)

    assert config_base(experiment_path) == project.resolve()
    assert resolve_config_path("configs/model.yaml", experiment_path) == project.resolve() / "configs/model.yaml"
    summary = run(RunSpec(data=None, output=tmp_path / "run", config=experiment_path,
                          device="cpu", progress="off", evaluate_test=False))
    assert summary["resolved_model"]["embedding_dim"] == 12
    config = json.loads((tmp_path / "run" / "config.json").read_text(encoding="utf-8"))
    assert config["data"]["processed_root"] == str(project.resolve() / "data" / "processed")


def test_committed_experiment_config_resolves_from_outside_the_repo(tmp_path, monkeypatch):
    from adrec_still.training.service import experiment_defaults

    monkeypatch.chdir(tmp_path)
    defaults = experiment_defaults(REPOSITORY_ROOT / "tests/fixtures/configs/experiment/gowalla_sasrec_h.yaml")
    # The vendored configs live under tests/fixtures (anchored by its pyproject.toml
    # marker), so repo-relative paths in them resolve against that root.
    assert (
        defaults["processed_root"]
        == (REPOSITORY_ROOT / "tests" / "fixtures").resolve() / "data/processed/gowalla"
    )


# ------------------------------------------------------------- P2: batch skip rule


def test_batch_does_not_count_another_datasets_run_as_completed(raw_gowalla_file, tmp_path,
                                                                 capsys):
    from adrec_still.training.cli import main

    root = tmp_path / "batch"
    common = ["batch", "--models", "global_pop", "--seeds", "41", "--device", "cpu",
              "--progress", "off", "--no-test", "--output-root", str(root)]
    assert main(common + ["--data", str(raw_gowalla_file)]) == 0
    first = json.loads((root / "global_pop" / "seed41" / "summary.json").read_text())
    assert first["status"] == "completed"
    # The same run again is the same experiment: skipped.
    assert main(common + ["--data", str(raw_gowalla_file)]) == 0
    assert "already completed, skipped" in capsys.readouterr().out

    raw_b = _shifted_copy(raw_gowalla_file, tmp_path / "other.txt", 1)
    (tmp_path / "b").mkdir()
    # Different POIs: a different item-ID mapping.
    lines = raw_b.read_text(encoding="utf-8").splitlines()
    raw_c = tmp_path / "b" / "loc-gowalla_totalCheckins.txt"
    raw_c.write_text("\n".join(line.replace("\tp0", "\tq0") for line in lines) + "\n",
                     encoding="utf-8")
    code = main(common + ["--data", str(raw_c), "--cache-dir", str(tmp_path / "cache_c")])
    out = capsys.readouterr()
    assert code == 1
    assert "different dataset" in out.err
    ledger = json.loads((root / "batch.json").read_text(encoding="utf-8"))
    assert ledger["runs"][-1]["status"] == "failed"


# The donor's test_collect_runs_reads_requested_metrics_and_legacy_risk_rows is not
# ported: it exercises scripts/collect_runs.py, a repository script that was not
# vendored into adrec_still.


# ------------------------------------------------- round 2: optimizer options take effect


def test_resume_applies_the_new_optimizer_options_to_the_param_groups(canonical_data, tmp_path):
    first = run(small_spec(canonical_data, tmp_path / "first", "gru4rec_h", epochs=1,
                           evaluate_test=False))
    source = ckpt.load_payload(tmp_path / "first" / "last.pt", ckpt.LAST_FORMAT)
    assert source["optimizer_state"]["param_groups"][0]["lr"] == 0.001
    resumed = run(small_spec(canonical_data, tmp_path / "resumed", "gru4rec_h", epochs=2,
                             resume=tmp_path / "first" / "last.pt", evaluate_test=False,
                             learning_rate=0.2, overrides={
                                 "embedding_dim": 8, "hidden_dim": 8,
                                 "training.negative_samples": 4, "training.batch_size": 16,
                                 "training.weight_decay": 0.01, "training.betas": [0.8, 0.9],
                                 "training.eps": 1e-6,
                             }))
    training = resumed["training"]
    assert training["optimizer_updates"] > first["training"]["optimizer_updates"]
    effective = training["effective_optimizer"]
    assert effective == {"learning_rate": 0.2, "weight_decay": 0.01, "betas": [0.8, 0.9],
                         "eps": 1e-6, "amsgrad": False}
    saved = ckpt.load_payload(tmp_path / "resumed" / "last.pt", ckpt.LAST_FORMAT)
    group = saved["optimizer_state"]["param_groups"][0]
    assert group["lr"] == 0.2 and group["weight_decay"] == 0.01
    assert tuple(group["betas"]) == (0.8, 0.9) and group["eps"] == 1e-6
    # The accumulated Adam state came along (step counts continue, not restart).
    steps = [int(state["step"]) for state in saved["optimizer_state"]["state"].values()]
    assert steps and min(steps) == training["optimizer_updates"]
    # And the update actually used the large rate: weights moved far more than a
    # fresh 0.001 segment of the same length would move them.
    run(small_spec(canonical_data, tmp_path / "baseline", "gru4rec_h", epochs=2,
                   resume=tmp_path / "first" / "last.pt", evaluate_test=False))
    fast = ckpt.load_payload(tmp_path / "resumed" / "last.pt", ckpt.LAST_FORMAT)["model_state"]
    slow = ckpt.load_payload(tmp_path / "baseline" / "last.pt", ckpt.LAST_FORMAT)["model_state"]
    start = source["model_state"]
    key = "item_embedding.weight"
    assert (fast[key] - start[key]).abs().sum() > 10 * (slow[key] - start[key]).abs().sum()


# ------------------------------------------- round 2: early-stopped source, new monitor


def test_resuming_an_early_stopped_run_with_a_new_monitor_or_patience_trains_again(
    canonical_data, tmp_path, monkeypatch
):
    from adrec_still.training import loop

    values = iter([1.0, 2.0])
    original = loop.evaluate_split

    def worsening(*args, **kwargs):
        result = original(*args, **kwargs)
        try:
            result["poi_nll"] = next(values)
        except StopIteration:
            pass
        return result

    monkeypatch.setattr(loop, "evaluate_split", worsening)
    stopped = run(small_spec(canonical_data, tmp_path / "stopped", "gru4rec_h", epochs=5,
                             patience=1, evaluate_test=False))
    assert stopped["training"]["early_stopped"] is True
    assert stopped["training"]["epochs_run"] == 2
    monkeypatch.setattr(loop, "evaluate_split", original)

    # New monitor: the best scope restarts and the old stop no longer applies.
    new_monitor = run(small_spec(canonical_data, tmp_path / "new_monitor", "gru4rec_h",
                                 epochs=5, patience=10, monitor="ndcg@10",
                                 resume=tmp_path / "stopped" / "last.pt", evaluate_test=False))
    training = new_monitor["training"]
    assert training["resumed_from"]["stop_state"] == {
        "source_early_stopped": True, "early_stopped": False, "training_complete": False,
    }
    assert training["epochs_run"] == 5
    assert training["optimizer_updates"] > stopped["training"]["optimizer_updates"]
    assert training["best"] is not None and training["best"]["metric"] == "ndcg@10"
    assert training["best"]["epoch"] >= 3
    assert (tmp_path / "new_monitor" / "best.pt").is_file()
    assert ckpt.load_payload(tmp_path / "new_monitor" / "best.pt",
                             ckpt.BEST_FORMAT)["selection"]["epoch"] >= 3

    # Same monitor, larger patience: the stop is re-decided (2 - 1 < 10) and training goes on.
    more_patience = run(small_spec(canonical_data, tmp_path / "more_patience", "gru4rec_h",
                                   epochs=4, patience=10,
                                   resume=tmp_path / "stopped" / "last.pt",
                                   evaluate_test=False))
    assert more_patience["training"]["epochs_run"] == 4
    assert more_patience["training"]["resumed_from"]["stop_state"]["early_stopped"] is False

    # Same monitor, same patience: the stop still holds and nothing new is trained.
    unchanged = run(small_spec(canonical_data, tmp_path / "unchanged", "gru4rec_h", epochs=5,
                               patience=1, resume=tmp_path / "stopped" / "last.pt",
                               evaluate_test=False))
    assert unchanged["training"]["epochs_run"] == 2
    assert unchanged["training"]["optimizer_updates"] == stopped["training"]["optimizer_updates"]


# ------------------------------------- round 2: batch, same mapping, other observations


def test_batch_treats_same_mapping_other_observations_as_another_experiment(raw_gowalla_file,
                                                                            tmp_path, capsys):
    from adrec_still.training.cli import main

    root = tmp_path / "batch"
    common = ["batch", "--models", "global_pop", "--seeds", "41", "--device", "cpu",
              "--progress", "off", "--no-test", "--output-root", str(root)]
    assert main(common + ["--data", str(raw_gowalla_file)]) == 0
    raw_b = _shifted_copy(raw_gowalla_file, tmp_path / "loc-gowalla_totalCheckins.txt", 1)
    code = main(common + ["--data", str(raw_b)])
    out = capsys.readouterr()
    assert code == 1
    assert "other observations" in out.err
    assert "skipped" not in out.out
    ledger = json.loads((root / "batch.json").read_text(encoding="utf-8"))
    assert ledger["runs"][-1]["status"] == "failed"
    # A's result is untouched, and re-running A is still recognised as the same run.
    assert main(common + ["--data", str(raw_gowalla_file)]) == 0
    assert "already completed, skipped" in capsys.readouterr().out



# ------------------------------------------------------ round 3: AMSGrad state, batch reuse


def test_turning_amsgrad_on_over_a_plain_adam_checkpoint_steps_and_resumes_again(
    canonical_data, tmp_path,
):
    first = run(small_spec(canonical_data, tmp_path / "first", "gru4rec_h", epochs=1,
                           evaluate_test=False))
    source = ckpt.load_payload(tmp_path / "first" / "last.pt", ckpt.LAST_FORMAT)
    assert source["optimizer_state"]["param_groups"][0]["amsgrad"] is False
    assert all("max_exp_avg_sq" not in state
               for state in source["optimizer_state"]["state"].values())

    amsgrad = dict(small_spec(canonical_data, tmp_path / "x", "gru4rec_h").overrides,
                   **{"training.amsgrad": True})
    # The first optimizer step of the resumed segment is what used to raise KeyError.
    resumed = run(small_spec(canonical_data, tmp_path / "amsgrad", "gru4rec_h", epochs=2,
                             resume=tmp_path / "first" / "last.pt", evaluate_test=False,
                             overrides=amsgrad))
    training = resumed["training"]
    assert training["optimizer_updates"] > first["training"]["optimizer_updates"]
    assert training["effective_optimizer"]["amsgrad"] is True
    assert "amsgrad" in " ".join(training["resumed_from"]["setting_changes"])
    saved = ckpt.load_payload(tmp_path / "amsgrad" / "last.pt", ckpt.LAST_FORMAT)
    assert saved["optimizer_state"]["param_groups"][0]["amsgrad"] is True
    states = saved["optimizer_state"]["state"]
    assert states and all(
        "max_exp_avg_sq" in state and state["max_exp_avg_sq"].shape == state["exp_avg_sq"].shape
        for state in states.values()
    )
    # The accumulated state came along, and the running maximum kept growing from it.
    assert min(int(state["step"]) for state in states.values()) == training["optimizer_updates"]
    assert all(bool((state["max_exp_avg_sq"] >= state["exp_avg_sq"] - 1e-12).all())
               for state in states.values())

    # The new last.pt is a complete AMSGrad checkpoint: it resumes and steps again.
    again = run(small_spec(canonical_data, tmp_path / "again", "gru4rec_h", epochs=3,
                           resume=tmp_path / "amsgrad" / "last.pt", evaluate_test=False,
                           overrides=amsgrad))
    assert again["training"]["optimizer_updates"] > training["optimizer_updates"]
    assert again["training"]["epochs_run"] == 3
    # And switching AMSGrad off again over that checkpoint also steps.
    off = run(small_spec(canonical_data, tmp_path / "off", "gru4rec_h", epochs=3,
                         resume=tmp_path / "amsgrad" / "last.pt", evaluate_test=False))
    assert off["training"]["optimizer_updates"] > training["optimizer_updates"]
    assert off["training"]["effective_optimizer"]["amsgrad"] is False


def test_batch_does_not_reuse_a_run_whose_settings_changed(canonical_data, tmp_path, capsys):
    from adrec_still.training.cli import main

    root = tmp_path / "batch"
    model_yaml = tmp_path / "global_pop.yaml"
    model_yaml.write_text(yaml.safe_dump({"name": "global_pop", "smoothing": 1}),
                          encoding="utf-8")
    common = ["batch", "--data", str(canonical_data), "--models", "global_pop", "--seeds", "41",
              "--device", "cpu", "--progress", "off", "--no-test", "--output-root", str(root),
              "--model-config", str(model_yaml)]
    assert main(common) == 0
    recorded = json.loads((root / "global_pop" / "seed41" / "config.json").read_text("utf-8"))
    assert recorded["model_config"]["smoothing"] == 1

    # The same path, other contents: not the completed experiment.
    model_yaml.write_text(yaml.safe_dump({"name": "global_pop", "smoothing": 2}),
                          encoding="utf-8")
    code = main(common)
    out = capsys.readouterr()
    assert code == 1
    assert "smoothing: 1 vs 2" in out.err and "skipped" not in out.out
    ledger = json.loads((root / "batch.json").read_text(encoding="utf-8"))
    assert ledger["runs"][-1]["status"] == "failed"
    recorded = json.loads((root / "global_pop" / "seed41" / "config.json").read_text("utf-8"))
    assert recorded["model_config"]["smoothing"] == 1  # the saved run is untouched
    # A --set override differing from the record is the same kind of change.
    model_yaml.write_text(yaml.safe_dump({"name": "global_pop", "smoothing": 1}),
                          encoding="utf-8")
    assert main(common + ["--set", "smoothing=3"]) == 1
    assert "smoothing: 1 vs 3" in capsys.readouterr().err
    # Back to the recorded settings: recognised and skipped.
    assert main(common) == 0
    assert "already completed, skipped" in capsys.readouterr().out

    # Neural runs: command-line training settings resolve into the compared block.
    neural = ["batch", "--data", str(canonical_data), "--models", "gru4rec_h", "--seeds", "41",
              "--device", "cpu", "--progress", "off", "--no-test", "--epochs", "1",
              "--output-root", str(root), "--set", "embedding_dim=8", "--set", "hidden_dim=8",
              "--set", "training.negative_samples=4", "--set", "training.batch_size=16"]
    assert main(neural) == 0
    assert main(neural + ["--learning-rate", "0.01"]) == 1
    assert "training.learning_rate: 0.001 vs 0.01" in capsys.readouterr().err
    two_epochs = neural[:neural.index("--epochs") + 1] + ["2"] + neural[neural.index("--epochs") + 2:]
    assert main(two_epochs) == 1
    assert "training.max_epochs: 1 vs 2" in capsys.readouterr().err
    assert main(neural) == 0
    assert "already completed, skipped" in capsys.readouterr().out


def test_batch_does_not_reuse_a_run_that_lacks_the_requested_evaluation(canonical_data,
                                                                        tmp_path, capsys):
    from adrec_still.training.cli import main

    root = tmp_path / "batch"
    common = ["batch", "--data", str(canonical_data), "--models", "global_pop", "--seeds", "41",
              "--device", "cpu", "--progress", "off", "--output-root", str(root)]
    assert main(common + ["--no-test"]) == 0
    summary_path = root / "global_pop" / "seed41" / "summary.json"
    assert "test" not in json.loads(summary_path.read_text("utf-8"))["evaluation"]

    code = main(common)  # the test evaluation is wanted now
    out = capsys.readouterr()
    assert code == 1
    assert "without the requested test evaluation" in out.err and "skipped" not in out.out
    assert "test" not in json.loads(summary_path.read_text("utf-8"))["evaluation"]
    assert json.loads((root / "batch.json").read_text("utf-8"))["runs"][-1]["status"] == "failed"
    # Asking for the same (or less) than what was done is a skip.
    assert main(common + ["--no-test"]) == 0
    assert "already completed, skipped" in capsys.readouterr().out
    # The same for a validation evaluation a neural run was completed without
    # (floors always evaluate validation, so the check only bites on neural runs).
    neural = ["batch", "--data", str(canonical_data), "--models", "gru4rec_h", "--seeds", "41",
              "--device", "cpu", "--progress", "off", "--no-test", "--epochs", "1",
              "--output-root", str(root), "--set", "embedding_dim=8", "--set", "hidden_dim=8",
              "--set", "training.negative_samples=4", "--set", "training.batch_size=16"]
    assert main(neural) == 0
    assert main(neural + ["--eval-validation"]) == 1
    assert "without the requested validation evaluation" in capsys.readouterr().err
    assert main(neural) == 0
    assert "already completed, skipped" in capsys.readouterr().out
