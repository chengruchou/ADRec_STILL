"""Native reader handles must not crash atomic status/boundary publishers."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from adrec_still.common import atomic_files
from adrec_still.training.checkpoint import atomic_json as progress_json

boundary_json = progress_json


@pytest.mark.skipif(os.name != "nt", reason="native Windows delete-sharing semantics")
@pytest.mark.parametrize("publish", [progress_json, boundary_json])
def test_atomic_publisher_waits_for_real_concurrent_reader_without_truncating(tmp_path, publish):
    target = tmp_path / "snapshot.json"
    publish(target, {"generation": 1})
    with ThreadPoolExecutor(max_workers=1) as pool:
        with target.open("rb") as reader:
            future = pool.submit(publish, target, {"generation": 2})
            time.sleep(0.1)
            assert not future.done(), "publisher bypassed the active Windows reader"
            assert json.loads(reader.read()) == {"generation": 1}
            assert json.loads(target.read_bytes()) == {"generation": 1}
        future.result(timeout=4)
    assert json.loads(target.read_bytes()) == {"generation": 2}


@pytest.mark.skipif(os.name != "nt", reason="native Windows delete-sharing semantics")
def test_persistent_reader_refuses_without_destroying_committed_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(atomic_files, "REPLACE_RETRY_SECONDS", 0.03)
    target = tmp_path / "snapshot.json"
    progress_json(target, {"generation": 1})
    with target.open("rb"), pytest.raises(PermissionError):
        progress_json(target, {"generation": 2})
    assert json.loads(target.read_bytes()) == {"generation": 1}
    # The donor test was written against the old training_operations publisher, which
    # retained the sibling temporary on failure; the checkpoint publisher it was later
    # rerouted to (donor commit 88b1aec) unlinks its unique temporary in a finally
    # block, so a refused replace leaves no *.tmp behind. The donor assertion
    # (== 1) is stale against the donor's own code and only runs on Windows.
    assert len(list(tmp_path.glob("*.tmp"))) == 0


def test_missing_source_is_not_retried_or_hidden(tmp_path):
    target = tmp_path / "committed"
    target.write_bytes(b"old")
    with pytest.raises(FileNotFoundError):
        atomic_files.replace_file(tmp_path / "missing", target)
    assert target.read_bytes() == b"old"
