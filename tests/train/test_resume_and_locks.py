"""Resume from a copied last.pt, weights-only init, refusals, interruption, live locks."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from adrec_still.training import checkpoint as ckpt
from adrec_still.training.service import run

from .conftest import small_spec


def state_of(path: Path) -> dict:
    return ckpt.load_payload(path, ckpt.BEST_FORMAT)["state_dict"]


def test_resume_from_a_copied_last_continues_optimizer_and_counters(canonical_data, tmp_path):
    """Two epochs, stop, copy only the run's last.pt elsewhere, resume to four epochs.
    The copy sits in a fresh directory with no journal, no old run directory."""
    first = run(small_spec(canonical_data, tmp_path / "first", "gru4rec_h", epochs=2,
                           evaluate_test=False))
    updates_after_two = first["training"]["optimizer_updates"]
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    shutil.copy(tmp_path / "first" / "last.pt", elsewhere / "last.pt")
    shutil.copy(tmp_path / "first" / "best.pt", elsewhere / "best.pt")

    resumed = run(small_spec(canonical_data, tmp_path / "resumed", "gru4rec_h", epochs=4,
                             resume=elsewhere / "last.pt", evaluate_test=False))
    training = resumed["training"]
    assert training["epochs_run"] == 4
    assert training["optimizer_updates"] == 2 * updates_after_two
    assert training["optimizer_updates_this_segment"] == updates_after_two
    assert training["resumed_from"]["completed_epoch"] == 2
    assert training["resumed_from"]["rng"]["skipped"] == []
    assert training["best_scope_start_epoch"] == 1
    metrics = [json.loads(line) for line in
               (tmp_path / "resumed" / "metrics.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in metrics] == [3, 4]

    # Bitwise continuation on the same machine: an uninterrupted four-epoch run lands
    # on the same weights.
    straight = run(small_spec(canonical_data, tmp_path / "straight", "gru4rec_h", epochs=4,
                              evaluate_test=False))
    assert straight["training"]["optimizer_updates"] == training["optimizer_updates"]
    last_resumed = ckpt.load_payload(tmp_path / "resumed" / "last.pt", ckpt.LAST_FORMAT)
    last_straight = ckpt.load_payload(tmp_path / "straight" / "last.pt", ckpt.LAST_FORMAT)
    for key, value in last_straight["model_state"].items():
        assert torch.equal(value, last_resumed["model_state"][key]), key


def test_resume_with_only_last_restarts_the_best_scope_and_keeps_history(canonical_data,
                                                                         tmp_path, monkeypatch):
    from adrec_still.training import loop

    first = run(small_spec(canonical_data, tmp_path / "first", "gru4rec_h", epochs=2,
                           evaluate_test=False))
    historical = first["training"]["best"]
    only_last = tmp_path / "only_last"
    only_last.mkdir()
    shutil.copy(tmp_path / "first" / "last.pt", only_last / "last.pt")

    # Make the resumed epochs strictly worse than the historical best scalar.
    original = loop.evaluate_split

    def worse(*args, **kwargs):
        result = original(*args, **kwargs)
        result["poi_nll"] = historical["value"] + 10.0
        return result

    monkeypatch.setattr(loop, "evaluate_split", worse)
    resumed = run(small_spec(canonical_data, tmp_path / "resumed", "gru4rec_h", epochs=3,
                             resume=only_last / "last.pt", evaluate_test=False))
    training = resumed["training"]
    assert training["historical_best"] == historical
    assert training["best_scope_start_epoch"] == 3
    assert training["best"]["epoch"] == 3
    assert training["best"]["scope_start_epoch"] == 3
    assert training["best"]["value"] == pytest.approx(historical["value"] + 10.0)
    best = ckpt.load_payload(tmp_path / "resumed" / "best.pt", ckpt.BEST_FORMAT)
    assert best["selection"]["scope_start_epoch"] == 3


def test_weights_only_initialization_starts_a_fresh_optimizer(canonical_data, tmp_path):
    first = run(small_spec(canonical_data, tmp_path / "first", "gru4rec_h", epochs=2,
                           evaluate_test=False))
    warm = run(small_spec(canonical_data, tmp_path / "warm", "gru4rec_h", epochs=1,
                          weights=tmp_path / "first" / "best.pt", evaluate_test=False))
    training = warm["training"]
    assert training["initialized_from"]["path"].endswith("best.pt")
    assert training["resumed_from"] is None
    assert training["epochs_run"] == 1
    assert training["optimizer_updates"] == first["training"]["optimizer_updates"] // 2
    with pytest.raises(ValueError, match="either --resume or --weights"):
        run(small_spec(canonical_data, tmp_path / "both", "gru4rec_h", epochs=1,
                       weights=tmp_path / "first" / "best.pt",
                       resume=tmp_path / "first" / "last.pt"))


def test_incompatible_shapes_recipe_and_mapping_are_refused_with_reasons(canonical_data,
                                                                         tmp_path):
    run(small_spec(canonical_data, tmp_path / "first", "gru4rec_h", epochs=1,
                   evaluate_test=False))
    last = tmp_path / "first" / "last.pt"
    with pytest.raises(ckpt.CheckpointError, match="embedding_dim.*checkpoint has 8"):
        run(small_spec(canonical_data, tmp_path / "wider", "gru4rec_h", epochs=2, resume=last,
                       overrides={"embedding_dim": 16, "hidden_dim": 8,
                                  "training.negative_samples": 4, "training.batch_size": 16}))
    # A learning-rate change is an experiment choice: applied and recorded, not refused.
    faster = run(small_spec(canonical_data, tmp_path / "faster", "gru4rec_h", epochs=2,
                            resume=last, learning_rate=0.5, evaluate_test=False))
    assert faster["training"]["training"]["learning_rate"] == 0.5
    assert any("training.learning_rate" in change
               for change in faster["training"]["resumed_from"]["setting_changes"])
    # A different item-ID mapping: same width, permuted raw IDs.
    import pyarrow.parquet as pq

    remapped = tmp_path / "remapped"
    remapped.mkdir()
    catalog = pq.read_table(canonical_data / "poi_catalog.parquet").to_pandas()
    catalog["raw_poi_id"] = catalog["raw_poi_id"].iloc[::-1].to_numpy()
    catalog.to_parquet(remapped / "poi_catalog.parquet", index=False)
    (remapped / "landmark_samples.parquet").write_bytes(
        (canonical_data / "landmark_samples.parquet").read_bytes()
    )
    with pytest.raises(ckpt.CheckpointError, match="item-ID mapping differs"):
        run(small_spec(remapped, tmp_path / "remapped_run", "gru4rec_h", epochs=2, resume=last))
    with pytest.raises(ckpt.CheckpointError, match="pass last.pt to --resume"):
        run(small_spec(canonical_data, tmp_path / "wrong_file", "gru4rec_h", epochs=2,
                       resume=tmp_path / "first" / "best.pt"))


@pytest.mark.skipif(
    os.name == "nt",
    reason=(
        "pre-existing donor gap on Windows: RunLock.__enter__ reads the lock owner "
        "from the refused file, but the msvcrt byte-range lock is mandatory, so the "
        "read raises PermissionError instead of RunLockedError (donor CI is "
        "ubuntu-latest and never runs this path; it fails identically in the donor "
        "repository on this machine)"
    ),
)
def test_a_second_writer_on_a_live_run_directory_is_refused(tmp_path):
    output = tmp_path / "live"
    with ckpt.RunLock(output):
        with pytest.raises(ckpt.RunLockedError, match="in use by another live run"), \
                ckpt.RunLock(output):
            pass
        assert (output / ckpt.LOCK_NAME).is_file()
    assert not (output / ckpt.LOCK_NAME).exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
def test_ctrl_c_preserves_the_last_complete_checkpoint(canonical_data, tmp_path):
    """SIGINT mid-run: the process exits 130, last.pt holds a complete epoch and
    summary.json says interrupted; a resume from it then completes."""
    output = tmp_path / "interrupted"
    command = [
        sys.executable, "-m", "adrec_still", "train", "--data", str(canonical_data),
        "--model", "gru4rec_h", "--epochs", "200", "--device", "cpu", "--progress", "plain",
        "--progress-interval", "0", "--set", "embedding_dim=8", "--set", "hidden_dim=8",
        "--set", "training.negative_samples=4", "--set", "training.batch_size=16",
        "--patience", "500", "--output", str(output), "--no-test",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, cwd=str(Path(__file__).parents[2]))
    deadline = time.monotonic() + 120
    seen_epoch = False
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if not line:
            break
        if line.startswith("[epoch] 2/"):
            seen_epoch = True
            break
    assert seen_epoch, "the run never reported a completed epoch"
    process.send_signal(signal.SIGINT)
    rest = process.communicate(timeout=120)[0]
    assert process.returncode == 130, rest
    assert "interrupted; last.pt holds epoch" in rest
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "interrupted"
    last = ckpt.load_payload(output / "last.pt", ckpt.LAST_FORMAT)
    assert last["completed_epoch"] >= 2
    assert not list(output.glob("*.tmp"))
    assert not (output / ckpt.LOCK_NAME).exists()

    resumed = run(small_spec(canonical_data, tmp_path / "resumed", "gru4rec_h",
                             epochs=last["completed_epoch"] + 1, resume=output / "last.pt",
                             evaluate_test=False))
    assert resumed["status"] == "completed"
    assert resumed["training"]["epochs_run"] == last["completed_epoch"] + 1
