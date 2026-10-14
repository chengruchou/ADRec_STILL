"""The installed package works from any directory, without the donor repository."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]

PROBE = r"""
import sys
import adrec_still
from adrec_still.training.cli import build_parser
from adrec_still.training.defaults import BACKENDS
from adrec_still.time.distributions import LognormalMixtureParameters
from adrec_still.models.adrec_still.registry import VARIANTS

donor_modules = [name for name in sys.modules
                 if name == "still" or name.startswith("still.")]
assert not donor_modules, f"the donor STILL package leaked in: {donor_modules}"
assert len(BACKENDS) == 27
assert len(VARIANTS) == 5
build_parser().parse_args(["models"])
print("standalone-ok")
"""


def test_package_imports_from_a_foreign_cwd_without_the_donor_repo(tmp_path):
    """Run from an empty directory: no repo-relative paths, no still.* imports."""
    completed = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        env={
            **__import__("os").environ,
            "PYTHONPATH": str(REPOSITORY_ROOT),
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert "standalone-ok" in completed.stdout


def test_cli_help_runs_from_a_foreign_cwd(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "adrec_still", "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        env={
            **__import__("os").environ,
            "PYTHONPATH": str(REPOSITORY_ROOT),
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert "train" in completed.stdout and "evaluate" in completed.stdout
