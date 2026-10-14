"""Tabulate finished runs from their ``summary.json`` files (``collect`` command).

Nothing is recomputed: the table reads the numbers each run wrote. Output
formats: an aligned table, CSV (paste-ready columns), or JSON.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_METRICS = ("ndcg@10", "recall@10", "mrr@10", "poi_nll")


def _rows_from_summary(path: Path, split: str, horizon: str,
                       metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    evaluation = (summary.get("evaluation") or {}).get(split)
    if not evaluation:
        return []
    block = evaluation if horizon == "all" else (evaluation.get("by_horizon") or {}).get(horizon)
    if not block:
        return []
    training = summary.get("training") or {}
    best = training.get("best") or {}
    silence = (summary.get("silence") or {}).get(split) or {}
    repeat = silence.get("repeat") or {}
    return [{
        "run": str(path.parent),
        "model": summary.get("model"),
        "seed": summary.get("seed"),
        "status": summary.get("status"),
        "split": split,
        "horizon": horizon,
        "rows": block.get("rows"),
        "eligible_rows": block.get("eligible_rows"),
        "best_epoch": best.get("epoch"),
        "epochs_run": training.get("epochs_run"),
        "repeat_rate_known": repeat.get("event_rate_known_catalog"),
        **{name: block.get(name) for name in metrics},
    }]


def collect(roots: list[Path], *, split: str, horizon: str,
            metrics: tuple[str, ...] = DEFAULT_METRICS) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        for path in sorted(Path(root).rglob("summary.json")):
            if path.parent.name == "cache":
                continue
            rows.extend(_rows_from_summary(path, split, horizon, metrics))
    return rows


def _format(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def run_collect(args) -> int:
    metrics = tuple(name.strip() for name in args.metrics.split(",") if name.strip())
    rows = collect(list(args.root), split=args.split, horizon=args.horizon, metrics=metrics)
    if not rows:
        print("no summaries found", file=sys.stderr)
        return 1
    columns = ["model", "seed", "status", "split", "horizon", "rows", "eligible_rows",
               "best_epoch", "repeat_rate_known", *metrics, "run"]
    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    if args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return 0
    table = [[_format(row.get(column)) for column in columns] for row in rows]
    widths = [max(len(column), *(len(line[i]) for line in table))
              for i, column in enumerate(columns)]
    print("  ".join(column.ljust(widths[i]) for i, column in enumerate(columns)))
    for line in table:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)))
    return 0
