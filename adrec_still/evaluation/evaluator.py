"""Stable shared evaluation result schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from adrec_still.evaluation.ranking import ranking_metrics
from adrec_still.time.protocol import (
    LogNormalPrediction,
    ResidualTimeDistribution,
    lognormal_time_metrics,
    residual_time_distribution_metrics,
)

EVALUATION_SCHEMA_VERSION = "still-evaluation-v0.2"


def evaluate(
    poi_logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    time_prediction: LogNormalPrediction | None = None,
    split: str | None = None,
    cohorts: list[str] | None = None,
    *,
    time_distribution: ResidualTimeDistribution | None = None,
) -> dict[str, Any]:
    """Evaluate a shared model-output/batch pair.

    The first five parameters keep the published order and semantics, so existing
    positional calls such as ``evaluate(logits, batch, None, "validation")`` and
    ``evaluate(logits, batch, None, None, cohorts)`` stay valid. ``time_prediction``
    is the accepted single-LogNormal path; ``time_distribution`` is the proposed
    generic residual-time adapter (pending #39 review) and is keyword-only so it can
    never capture a positional ``split``/``cohorts`` argument. Passing both time
    adapters is ambiguous and rejected.
    """
    if time_prediction is not None and time_distribution is not None:
        raise ValueError("pass either time_prediction or time_distribution, not both")
    result: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "split": split,
        "ranking": ranking_metrics(
            poi_logits,
            batch["next_poi"],
            target_in_train_catalog=batch["target_in_train_catalog"],
            observed_mask=batch["observed_mask"],
        ),
    }
    if time_prediction is not None:
        result["time"] = lognormal_time_metrics(
            time_prediction,
            batch["residual_time_sec"],
            batch["censor_time_sec"],
            batch["observed_mask"],
        )
    elif time_distribution is not None:
        result["time"] = residual_time_distribution_metrics(
            time_distribution,
            batch["residual_time_sec"],
            batch["censor_time_sec"],
            batch["observed_mask"],
        )
    horizons, counts = torch.unique(batch["query_s_sec"], return_counts=True)
    result["sample_counts_by_horizon_seconds"] = {
        str(int(horizon)): int(count) for horizon, count in zip(horizons, counts, strict=True)
    }
    if cohorts is not None:
        if len(cohorts) != len(batch["query_s_sec"]):
            raise ValueError("cohorts must contain one label per sample")
        cohort_counts: dict[str, int] = {}
        for cohort in cohorts:
            cohort_counts[cohort] = cohort_counts.get(cohort, 0) + 1
        result["sample_counts_by_cohort"] = cohort_counts
    else:
        result["sample_counts_by_cohort"] = {
            "observed": int(batch["observed_mask"].sum()),
            "censored": int((~batch["observed_mask"].bool()).sum()),
        }
    return result


def write_evaluation_json(result: dict[str, Any], output_path: str | Path) -> None:
    """Create parent directories and persist a strict JSON result."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
