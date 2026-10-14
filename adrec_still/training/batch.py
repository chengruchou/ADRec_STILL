"""Sequential model/seed batches: start the next run when the previous one completed.

No registration, ledger or operator identity: each (model, seed) pair trains into
its own directory ``<root>/<model>/seed<N>/``; a directory whose ``summary.json``
already says ``completed`` is skipped when it is this batch's run of the same
experiment (same inputs, same settings as this batch resolves them now, requested
evaluations present), so a batch can be restarted after a crash or a reboot.
Progress is printed per run, and a small ``batch.json`` under the root records
what happened for later reading.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from adrec_still.training import checkpoint as ckpt
from adrec_still.training.cli import EXIT_FAILED, EXIT_INTERRUPTED, EXIT_OK, spec_from_args
from adrec_still.training.loop import STATUS_COMPLETED, STATUS_INTERRUPTED


def _completed(output: Path) -> bool:
    summary = output / "summary.json"
    if not summary.is_file():
        return False
    try:
        return json.loads(summary.read_text(encoding="utf-8")).get("status") == STATUS_COMPLETED
    except ValueError:
        return False


def _same_experiment(output: Path, data: Any, spec: Any) -> tuple[bool, str]:
    """Whether a completed run in ``output`` is this batch's run of the same experiment.

    Compared: the actual input files (path, size, mtime) and the item-ID mapping, the
    model and the seed, and the settings this batch resolves *now* (the model config
    as its YAML currently reads plus overrides, the training block after the command
    line's epochs/batch size/rate/patience/monitor/horizon, the evaluation history
    length, the training-batch cap) against what ``config.json`` recorded. Then the
    evaluations this batch asks for must be in ``summary.json``: a run finished with
    ``--no-test`` is not a finished run when the test split is wanted. Item-ID
    compatibility alone is not sameness: the same catalog with other observations
    is another experiment. Anything else already in that directory is reported,
    never silently counted as done.
    """
    from adrec_still.training.defaults import backend
    from adrec_still.training.service import (
        RunFailed,
        resolve_evaluation,
        resolve_model_config,
        resolve_training,
    )

    try:
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "no readable config.json and summary.json"
    recorded_data = config.get("data") or {}
    if recorded_data.get("identity") != data.identity:
        return False, "a different dataset (item-ID mapping or catalog width differs)"
    if recorded_data.get("inputs") != data.input_stamp():
        return False, "different input data (the same item-ID mapping, other observations)"
    if config.get("model") != spec.model or config.get("seed") != spec.seed:
        return False, "a different model or seed"
    try:
        resolved, source = resolve_model_config(spec)
    except RunFailed as error:
        return False, f"settings this batch cannot resolve ({error})"
    changed = _differences(config.get("model_config"), _as_recorded(resolved))
    if changed:
        return False, f"a different model configuration ({'; '.join(changed)})"
    if backend(resolved["name"])["kind"] != "floor":
        try:
            training, _ = resolve_training(spec, resolved, data)
        except RunFailed as error:
            return False, f"settings this batch cannot resolve ({error})"
        changed = _differences(config.get("training"), _as_recorded(training), "training")
        if changed:
            return False, f"different training settings ({'; '.join(changed)})"
    evaluation = _as_recorded(resolve_evaluation(spec, source, data))
    recorded_evaluation = config.get("evaluation") or {}
    if recorded_evaluation.get("max_sequence_length") != evaluation["max_sequence_length"]:
        return False, (
            "a different evaluation history length (max_sequence_length: "
            f"{recorded_evaluation.get('max_sequence_length')!r} vs "
            f"{evaluation['max_sequence_length']!r})"
        )
    recorded_spec = config.get("spec") or {}
    if recorded_spec.get("max_train_batches") != spec.max_train_batches:
        return False, (
            f"a different training-batch cap (max_train_batches: "
            f"{recorded_spec.get('max_train_batches')!r} vs {spec.max_train_batches!r})"
        )
    done = summary.get("evaluation") or {}
    for split, wanted in (("validation", spec.evaluate_validation),
                          ("test", spec.evaluate_test)):
        if wanted and data.has_split(split) and split not in done:
            return False, f"the same experiment without the requested {split} evaluation"
    return True, ""


def _as_recorded(value: Any) -> Any:
    """``value`` as ``config.json`` holds it (Paths as strings, tuples as lists)."""
    from dataclasses import asdict, is_dataclass

    from adrec_still.training.service import jsonable

    if is_dataclass(value):
        value = asdict(value)
    return json.loads(json.dumps(jsonable(value), sort_keys=True))


def _differences(recorded: Any, current: Any, prefix: str = "") -> list[str]:
    """Dotted keys whose value differs, ``recorded vs current``; the whole value when not mappings."""
    if isinstance(recorded, dict) and isinstance(current, dict):
        out: list[str] = []
        for key in sorted(set(recorded) | set(current)):
            path = f"{prefix}.{key}" if prefix else str(key)
            out.extend(_differences(recorded.get(key), current.get(key), path))
        return out
    if recorded != current:
        return [f"{prefix or 'value'}: {recorded!r} vs {current!r}"]
    return []


def run_batch(args: argparse.Namespace, models: list[str], seeds: list[int], *,
              stop_on_failure: bool = True) -> int:
    from adrec_still.training.inputs import resolve_data
    from adrec_still.training.progress import TerminalReporter
    from adrec_still.training.service import run

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    if args.cache_dir is None:
        # One preprocessing of a raw file serves every run of the batch.
        args.cache_dir = root / "cache"
    ledger_path = root / "batch.json"
    runs: list[dict[str, Any]] = []
    plan = [(model, seed) for model in models for seed in seeds]
    print(f"batch: {len(plan)} runs under {root}", flush=True)
    data = resolve_data(
        args.data, adapter=args.adapter, poi_file=args.poi_file, cache_dir=args.cache_dir,
        workers=args.workers, row_limit=args.row_limit,
        reporter=TerminalReporter(mode="plain" if args.progress == "auto" else args.progress)
        if args.progress != "off" else None,
    )
    exit_code = EXIT_OK
    for index, (model, seed) in enumerate(plan, start=1):
        output = root / model / f"seed{seed}"
        entry: dict[str, Any] = {"model": model, "seed": seed, "output": str(output)}
        spec = spec_from_args(args, model=model, seed=seed, output=output)
        if _completed(output):
            same, why = _same_experiment(output, data, spec)
            if same:
                entry["status"] = "skipped-completed"
                print(f"[{index}/{len(plan)}] {model} seed {seed}: already completed, skipped",
                      flush=True)
                runs.append(entry)
                _write_ledger(ledger_path, runs)
                continue
            entry["status"] = "failed"
            entry["error"] = (
                f"{output} holds a completed run of {why}; use another --output-root "
                "or remove that run"
            )
            print(f"[{index}/{len(plan)}] {model} seed {seed}: FAILED - {entry['error']}",
                  file=sys.stderr, flush=True)
            exit_code = EXIT_FAILED
            runs.append(entry)
            _write_ledger(ledger_path, runs)
            if stop_on_failure:
                print("stopping the batch after a failure (use --continue-on-failure to go on)",
                      flush=True)
                return exit_code
            continue
        print(f"[{index}/{len(plan)}] {model} seed {seed}: starting -> {output}", flush=True)
        started = time.time()
        try:
            summary = run(spec)
            entry["status"] = summary["status"]
            entry["seconds"] = time.time() - started
            if summary["status"] == STATUS_INTERRUPTED:
                print(f"[{index}/{len(plan)}] {model} seed {seed}: interrupted; stopping",
                      flush=True)
                runs.append(entry)
                _write_ledger(ledger_path, runs)
                return EXIT_INTERRUPTED
            print(f"[{index}/{len(plan)}] {model} seed {seed}: {summary['status']} in "
                  f"{entry['seconds']:.0f}s", flush=True)
        except KeyboardInterrupt:
            entry["status"] = STATUS_INTERRUPTED
            runs.append(entry)
            _write_ledger(ledger_path, runs)
            print("batch interrupted", flush=True)
            return EXIT_INTERRUPTED
        except Exception as error:  # noqa: BLE001 - a failed run is reported, not hidden
            entry["status"] = "failed"
            entry["error"] = f"{type(error).__name__}: {error}"
            entry["seconds"] = time.time() - started
            print(f"[{index}/{len(plan)}] {model} seed {seed}: FAILED - {entry['error']}",
                  file=sys.stderr, flush=True)
            exit_code = EXIT_FAILED
            runs.append(entry)
            _write_ledger(ledger_path, runs)
            if stop_on_failure:
                print("stopping the batch after a failure (use --continue-on-failure to go on)",
                      flush=True)
                return exit_code
            continue
        runs.append(entry)
        _write_ledger(ledger_path, runs)
    done = sum(1 for r in runs if r["status"] in (STATUS_COMPLETED, "skipped-completed"))
    print(f"batch finished: {done}/{len(plan)} completed", flush=True)
    return exit_code


def _write_ledger(path: Path, runs: list[dict[str, Any]]) -> None:
    ckpt.atomic_json(path, {"runs": runs, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                        time.gmtime())})
