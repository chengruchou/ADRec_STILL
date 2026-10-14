"""The command line, the sequential batch entry point and terminal progress."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from adrec_still.training.cli import main
from adrec_still.training.progress import TerminalReporter

REPOSITORY_ROOT = Path(__file__).parents[2]
SMALL = ["--set", "embedding_dim=8", "--set", "hidden_dim=8",
         "--set", "training.negative_samples=4", "--set", "training.batch_size=16"]


def test_models_lists_all_eighteen_backends(capsys):
    assert main(["models"]) == 0
    lines = [line.split() for line in capsys.readouterr().out.strip().splitlines()]
    assert len(lines) == 18
    assert sum(1 for line in lines if line[2] == "floor") == 8
    assert sum(1 for line in lines if line[2] == "neural") == 10


def test_train_from_a_raw_file_in_a_subprocess_with_real_progress(raw_gowalla_file, tmp_path):
    """Install-and-run shape: a subprocess with stdout redirected sees bounded plain
    progress lines and real numbers; no clean Git, manifest or registration exists."""
    output = tmp_path / "run"
    command = [sys.executable, "-m", "adrec_still", "train", "--data", str(raw_gowalla_file),
               "--model", "sasrec_h", "--epochs", "2", "--device", "cpu",
               "--output", str(output), "--progress-interval", "0", *SMALL]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=str(tmp_path),
                               timeout=600, check=False)
    assert completed.returncode == 0, completed.stderr
    out = completed.stdout
    assert "[preprocess] start" in out
    assert "[train] epoch 1/2" in out and "loss" in out and "eta" in out
    assert "[validation] epoch 1" in out and "poi_nll" in out
    assert "[test]" in out
    assert (output / "cache" / "processed" / "landmark_samples.parquet").is_file()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["git"]["commit"] is None or isinstance(config["git"]["commit"], str)


def test_train_cli_exit_codes(canonical_data, tmp_path, capsys):
    assert main(["train", "--data", str(canonical_data), "--model", "gru4rec_h", "--epochs", "1",
                 "--device", "cpu", "--progress", "off", "--no-test",
                 "--output", str(tmp_path / "ok"), *SMALL]) == 0
    assert main(["train", "--data", str(tmp_path / "missing"), "--model", "gru4rec_h",
                 "--output", str(tmp_path / "bad")]) == 1
    assert "error:" in capsys.readouterr().err
    assert main(["train", "--data", str(canonical_data), "--model", "gru4rec_h", "--epochs", "1",
                 "--device", "cpu", "--progress", "off", "--monitor", "brier",
                 "--output", str(tmp_path / "bad2")]) == 1


def test_batch_runs_in_order_skips_completed_and_stops_on_failure(canonical_data, tmp_path,
                                                                    capsys):
    root = tmp_path / "batch"
    common = ["batch", "--data", str(canonical_data), "--seeds", "41", "--epochs", "1",
              "--device", "cpu", "--progress", "off", "--no-test", "--output-root", str(root),
              *SMALL]
    # user_history_pop's frequency-first guard refuses the default history length on
    # this tiny catalog: a real failure that must stop the sequence.
    code = main(common + ["--models", "global_pop,gru4rec_h,user_history_pop,repeat_last"])
    out = capsys.readouterr()
    assert code == 1
    ledger = json.loads((root / "batch.json").read_text(encoding="utf-8"))
    statuses = [(entry["model"], entry["status"]) for entry in ledger["runs"]]
    assert statuses == [("global_pop", "completed"), ("gru4rec_h", "completed"),
                        ("user_history_pop", "failed")]
    assert "FAILED" in out.err and "stopping the batch" in out.out
    assert not (root / "repeat_last").exists()

    # Restart: completed runs are skipped, the failed one runs again and fails again
    # unless told to continue, in which case the last model still runs.
    code = main(common + ["--models", "global_pop,gru4rec_h,user_history_pop,repeat_last",
                          "--continue-on-failure"])
    out = capsys.readouterr().out
    assert code == 1
    assert "already completed, skipped" in out
    assert (root / "repeat_last" / "seed41" / "summary.json").is_file()


def test_batch_output_layout_is_model_then_seed(canonical_data, tmp_path):
    root = tmp_path / "batch"
    code = main(["batch", "--data", str(canonical_data), "--models", "global_pop",
                 "--seeds", "41,42", "--device", "cpu", "--progress", "off", "--no-test",
                 "--output-root", str(root)])
    assert code == 0
    assert (root / "global_pop" / "seed41" / "summary.json").is_file()
    assert (root / "global_pop" / "seed42" / "summary.json").is_file()


@pytest.mark.parametrize("mode", ["bar", "plain"])
def test_terminal_reporter_renders_both_modes(mode):
    stream = io.StringIO()
    reporter = TerminalReporter(stream, mode=mode, interval=0.0)
    reporter.epoch_start(1, 3, 4)
    for batch in range(1, 5):
        reporter.step(batch, 4, loss=1.0 / batch, lr=0.001)
    reporter.validation(1, poi_nll=0.5, ndcg=0.1)
    reporter.epoch_end(1, train_loss=0.25)
    with reporter.phase("evaluate-test"):
        pass
    text = stream.getvalue()
    assert "[train] epoch 1/3" in text and "4/4" in text and "eta" in text
    assert "[validation] epoch 1 poi_nll 0.5000" in text
    assert "[epoch] 1/3 train_loss 0.2500" in text
    assert "[evaluate-test] start" in text and "[evaluate-test] done" in text
    if mode == "bar":
        assert "\r" in text and "|####" in text
        # Every rewritten line is finished with a newline before the next block.
        assert text.count("\n") >= 4
    else:
        assert "\r" not in text
        assert text.count("[train] epoch 1/3") == 4


def test_plain_progress_is_bounded_by_the_interval(monkeypatch):
    # The monotonic clock starts at boot, so on a freshly booted machine it reads
    # less than the interval; the first update must still be shown.
    monkeypatch.setattr("adrec_still.training.progress.time.monotonic", lambda: 10.0)
    stream = io.StringIO()
    reporter = TerminalReporter(stream, mode="plain", interval=3600.0)
    reporter.epoch_start(1, 1, 1000)
    for batch in range(1, 1001):
        reporter.step(batch, 1000, loss=1.0)
    lines = [line for line in stream.getvalue().splitlines() if "[train]" in line]
    assert len(lines) == 2  # the first step and the forced last step
