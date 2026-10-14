"""Batched in-memory evaluation of one split; no index, no export tree.

One streaming pass over a canonical split. Ranking metrics and the full-catalog
``poi_nll`` come from the exact adapter (:mod:`adrec_still.evaluation.exact`: chunked
float64 normalization, score-descending / POI-ID-ascending ties), scored through
the model's registered surface (:mod:`adrec_still.evaluation.scoring`). Models with a
``score_joint`` adapter also report ``joint_nll`` on observed exact-known rows, and
models whose ``predict`` exposes a residual-time distribution report ``time_nll``
(observed rows: ``-log_density(residual)``; censored rows: ``-log_survival(censor)``)
computed here from that distribution alone. CRPS is not computed anywhere on this
path: no ``crps`` method or quadrature is called, and no value is filled in for it.
Every mean is over its own eligible population with the count reported next to it.

``horizon=<seconds>`` restricts the pass to one landmark horizon, which is what
per-epoch selection uses; ``horizon=None`` scores every row and, with
``breakdown=True``, also reports each horizon separately.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from adrec_still.data.collate import (
    collate_canonical_samples,
    select_batch_rows,
    slice_batch_rows,
)
from adrec_still.data.dataset import LandmarkDataset
from adrec_still.evaluation.exact import (
    EXACT_VERSION,
    TIE_POLICY,
    dense_exact_contributions,
    exact_ranking_contributions,
)
from adrec_still.evaluation.scoring import (
    INFERENCE_FIELDS,
    exact_model_candidate_factory,
    scorer_surface,
)

RANKING_METRICS: tuple[str, ...] = (
    "recall@5", "recall@10", "recall@20",
    "ndcg@5", "ndcg@10", "ndcg@20",
    "mrr@5", "mrr@10", "mrr@20",
    "poi_nll",
)
JOINT_METRICS: tuple[str, ...] = ("joint_nll",)
TIME_METRICS: tuple[str, ...] = ("time_nll",)
#: Metrics a run may monitor for checkpoint selection, with their direction.
MONITOR_DIRECTIONS: dict[str, str] = {
    **{name: "max" for name in RANKING_METRICS if name != "poi_nll"},
    "poi_nll": "min",
    "joint_nll": "min",
    "time_nll": "min",
    "val_loss": "min",
    "train_loss": "min",
}


@dataclass
class EvaluationOptions:
    batch_size: int = 128
    max_sequence_length: int | None = 100
    num_workers: int = 0
    chunk_size: int = 16384
    max_batches: int | None = None

    def validate(self) -> None:
        if self.batch_size <= 0 or self.chunk_size <= 0:
            raise ValueError("evaluation batch_size and chunk_size must be positive")
        if self.max_sequence_length is not None and self.max_sequence_length <= 0:
            raise ValueError("evaluation max_sequence_length must be positive")
        if self.num_workers < 0:
            raise ValueError("evaluation num_workers must be non-negative")
        if self.max_batches is not None and self.max_batches <= 0:
            raise ValueError("evaluation max_batches must be positive when set")


class _Sums:
    """Running sums of per-row values over their eligible rows."""

    def __init__(self) -> None:
        self.rows = 0
        self.counts: dict[str, int] = {}
        self.totals: dict[str, float] = {}

    def add(self, values: dict[str, torch.Tensor], eligible: torch.Tensor, rows: int) -> None:
        self.rows += rows
        count = int(eligible.sum())
        for name, tensor in values.items():
            self.counts[name] = self.counts.get(name, 0) + count
            if count:
                self.totals[name] = self.totals.get(name, 0.0) + float(
                    tensor.to(torch.float64)[eligible].sum()
                )

    def result(self) -> dict[str, Any]:
        out: dict[str, Any] = {"rows": self.rows}
        for name, count in self.counts.items():
            mean = self.totals.get(name, 0.0) / count if count else None
            if mean is not None and not math.isfinite(mean):
                mean = None
            out[name] = mean
            out[f"{name}_rows"] = count
        return out


def split_horizons(landmark_path: str | Path, split: str) -> list[int]:
    """Distinct ``query_s_sec`` values present in ``split``, ascending."""
    table = pq.read_table(Path(landmark_path), columns=["split", "query_s_sec"])
    rows = table.filter(pc.equal(table["split"], split))
    if not rows.num_rows:
        return []
    return sorted(int(value) for value in pc.unique(rows["query_s_sec"]).to_pylist())


def split_present(landmark_path: str | Path, split: str) -> bool:
    table = pq.read_table(Path(landmark_path), columns=["split"])
    return bool(pc.any(pc.equal(table["split"], split)).as_py())


def _eval_mode(model):
    class _Guard:
        def __enter__(self):
            self.was_training = getattr(model, "training", False)
            if hasattr(model, "eval"):
                model.eval()
            return self

        def __exit__(self, *exc: object) -> None:
            if hasattr(model, "train"):
                model.train(self.was_training)

    return _Guard()


def evaluate_split(
    model,
    landmark_path: str | Path,
    *,
    split: str,
    horizon: int | None = None,
    breakdown: bool = False,
    options: EvaluationOptions | None = None,
    time_nll: bool = True,
    stop_condition: Any = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Score ``split`` (or its rows at ``horizon``) and return means with counts.

    ``time_nll`` adds the return-time NLL for models whose ``predict`` exposes a
    residual-time distribution; it costs one dense ``predict`` per batch, so the
    per-epoch selection pass asks for it only when it is the monitored metric.
    ``progress(done, total)`` is called after every batch when given.

    Raises ``ValueError`` when the split has no rows, and reports ``rows: 0`` (with
    every metric ``None``) when it has rows but none at the requested horizon; the
    caller decides whether that is an error for its purpose.
    """
    options = options or EvaluationOptions()
    options.validate()
    started = time.perf_counter()
    dataset = LandmarkDataset(landmark_path, split=split)
    if not len(dataset):
        raise ValueError(f"split has no landmark samples: {split}")
    loader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=False,
        num_workers=options.num_workers,
        collate_fn=partial(
            collate_canonical_samples, max_sequence_length=options.max_sequence_length
        ),
    )
    source, score_kind = scorer_surface(model)
    if source is not None:
        scorer_factory = exact_model_candidate_factory(model, chunk_size=options.chunk_size)
    elif callable(getattr(model, "predict", None)):
        # Deterministic floors publish normalized log-probabilities through predict()
        # only; the dense adapter applies the same tie policy to that matrix.
        source, score_kind, scorer_factory = "dense-predict", "log_probabilities", None
    else:
        raise TypeError(
            f"{type(model).__name__} exposes neither a catalog scorer surface nor predict(), "
            "so it cannot be scored over the catalog"
        )
    joint_capable = callable(getattr(model, "score_joint", None))
    # The models with a residual-time distribution are the joint-capable ones; a plain
    # encoder never pays a dense predict here and its scorer sees forecast fields only.
    time_capable = bool(time_nll) and joint_capable and callable(getattr(model, "predict", None))
    pooled = _Sums()
    per_horizon: dict[int, _Sums] = {}
    stopped = False
    total_batches = len(loader)
    if options.max_batches is not None:
        total_batches = min(total_batches, options.max_batches)
    # With a horizon filter most loader batches keep only a few rows; scoring them one
    # loader batch at a time would pay the adapter's per-chunk overhead hundreds of
    # times over. Buffer the kept rows and score them in full-size batches instead
    # (per-row contributions are independent, so the numbers do not change).
    buffer: list[dict[str, torch.Tensor]] = []
    buffered_rows = 0

    def flush(*, final: bool = False):
        """Full batches of exactly ``options.batch_size`` rows; the remainder stays
        buffered until the end, so the model never sees more rows than configured."""
        nonlocal buffer, buffered_rows
        if not buffer:
            return []
        merged = _merge_batches(buffer) if len(buffer) > 1 else buffer[0]
        size = options.batch_size
        total = int(merged["query_s_sec"].shape[0])
        ready = []
        start = 0
        while total - start >= size:
            ready.append(slice_batch_rows(merged, start, start + size))
            start += size
        if start < total:
            rest = slice_batch_rows(merged, start, None)
            if final:
                ready.append(rest)
                buffer, buffered_rows = [], 0
            else:
                buffer, buffered_rows = [rest], total - start
        else:
            buffer, buffered_rows = [], 0
        return ready

    with _eval_mode(model), torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if options.max_batches is not None and batch_index >= options.max_batches:
                break
            if stop_condition is not None and batch_index > 0 and stop_condition():
                stopped = True
                break
            if progress is not None:
                progress(batch_index + 1, total_batches)
            if horizon is not None:
                rows = batch["query_s_sec"].eq(int(horizon))
                if not bool(rows.any()):
                    continue
                batch = select_batch_rows(batch, rows)
                buffer.append(batch)
                buffered_rows += int(rows.sum())
                if buffered_rows < options.batch_size:
                    continue
                ready = flush()
            else:
                ready = [batch]
            for batch in ready:
                _score_groups(batch, breakdown, pooled, per_horizon, model, scorer_factory,
                              score_kind, options, joint_capable, time_capable)
        for batch in flush(final=True):
            _score_groups(batch, breakdown, pooled, per_horizon, model, scorer_factory,
                          score_kind, options, joint_capable, time_capable)
    result = {
        "split": split,
        "horizon": horizon,
        "evaluator_version": EXACT_VERSION,
        "tie_policy": TIE_POLICY,
        "score_source": source,
        "score_kind": score_kind,
        "candidate_count": int(model.number_of_pois) - 2,
        **_finalize(pooled),
        "stopped": stopped,
        "seconds": time.perf_counter() - started,
    }
    if breakdown:
        result["by_horizon"] = {
            str(h): _finalize(sums) for h, sums in sorted(per_horizon.items())
        }
    return result


def _merge_batches(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Concatenate collated batches; histories are left-padded to the widest part."""
    width = max(int(part["history_poi"].shape[1]) for part in parts)
    merged: dict[str, Any] = {}
    for key in parts[0]:
        if not isinstance(parts[0][key], torch.Tensor):
            merged[key] = [item for part in parts for item in part[key]]
            continue
        pieces = []
        for part in parts:
            tensor = part[key]
            if tensor.ndim == 2 and tensor.shape[1] < width:
                pad = torch.zeros(
                    (tensor.shape[0], width - tensor.shape[1]), dtype=tensor.dtype
                )
                tensor = torch.cat([pad, tensor], dim=1)
            pieces.append(tensor)
        merged[key] = torch.cat(pieces)
    return merged


def _score_groups(batch, breakdown, pooled, per_horizon, model, scorer_factory, score_kind,
                  options, joint_capable, time_capable) -> None:
    """Score the batch once; the per-horizon breakdown buckets the same per-row values.

    Contributions are per row, so a horizon's figures are the pooled figures restricted
    to its rows. Scoring each horizon's rows as its own sub-batch would pay the full
    catalog pass once per horizon present in the batch (seven times on Gowalla) for
    the same numbers.
    """
    values, eligible = _score_batch(
        model, batch, scorer_factory, score_kind, options.chunk_size,
        joint_capable=joint_capable, time_capable=time_capable,
    )
    horizons = batch["query_s_sec"]
    rows = int(horizons.shape[0])
    ranking = {name: values[name] for name in RANKING_METRICS if name in values}
    pooled.add(ranking, eligible["ranking"], rows)
    _add_extra(pooled, values, eligible)
    if not breakdown:
        return
    for group_horizon in torch.unique(horizons).tolist():
        mask = horizons.eq(group_horizon)
        bucket = per_horizon.setdefault(int(group_horizon), _Sums())
        bucket.add({name: tensor[mask] for name, tensor in ranking.items()},
                   eligible["ranking"][mask], int(mask.sum()))
        _add_extra(bucket, {name: tensor[mask] for name, tensor in values.items()},
                   {name: tensor[mask] for name, tensor in eligible.items()})


def _score_batch(model, sub_batch, scorer_factory, score_kind, chunk_size, *,
                 joint_capable, time_capable):
    prediction = None
    if scorer_factory is None or time_capable:
        prediction = model.predict(sub_batch)
    if scorer_factory is None:
        contributions = dense_exact_contributions(
            prediction.poi_logits.detach(),
            sub_batch["next_poi"],
            sub_batch["target_in_train_catalog"],
            sub_batch["observed_mask"],
            score_kind=score_kind,
        )
    else:
        features = {key: sub_batch[key].clone() for key in INFERENCE_FIELDS}
        contributions = exact_ranking_contributions(
            scorer_factory(features),
            sub_batch["next_poi"],
            sub_batch["target_in_train_catalog"],
            sub_batch["observed_mask"],
            model.number_of_pois,
            score_kind=score_kind,
            chunk_size=chunk_size,
        )
    values = {name: tensor for name, tensor in contributions["values"].items()
              if name in RANKING_METRICS
              and (name != "poi_nll" or contributions.get("poi_nll_supported", True))}
    eligible = {"ranking": contributions["eligible_mask"]}
    if joint_capable:
        joint = model.score_joint(sub_batch)
        values["joint_nll"] = -joint["log_joint"].detach().cpu().to(torch.float64)
        eligible["joint_nll"] = joint["eligible"].detach().cpu().bool()
    if prediction is not None and time_capable:
        timed = time_nll_rows(prediction, sub_batch)
        if timed is not None:
            values["time_nll"], eligible["time_nll"] = timed
    return values, eligible


def time_nll_rows(prediction, batch) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Per-row return-time NLL from a prediction's residual-time distribution.

    Observed rows with a positive residual contribute ``-log_density(residual)``;
    censored rows with positive remaining exposure contribute ``-log_survival(censor)``.
    A single-LogNormal prediction is evaluated in closed form. ``None`` when the
    prediction carries no time distribution. Nothing here calls ``crps``.
    """
    observed = batch["observed_mask"].detach().cpu().bool()
    residual = batch["residual_time_sec"].detach().cpu().to(torch.float64)
    censor = batch["censor_time_sec"].detach().cpu().to(torch.float64)
    eligible = (observed & residual.gt(0)) | (~observed & censor.gt(0))
    safe_residual = torch.where(observed & eligible, residual, torch.ones_like(residual))
    safe_censor = torch.where(~observed & eligible, censor, torch.zeros_like(censor))
    distribution = getattr(prediction, "time_distribution", None)
    lognormal = getattr(prediction, "time_prediction", None)
    if distribution is not None:
        density = distribution.log_density(safe_residual).to(torch.float64)
        survival = distribution.log_survival(safe_censor).to(torch.float64)
    elif lognormal is not None:
        mu = lognormal.mu.detach().cpu().to(torch.float64)
        sigma = lognormal.sigma.detach().cpu().to(torch.float64)
        z_residual = (safe_residual.log() - mu) / sigma
        density = (
            -safe_residual.log() - sigma.log() - 0.5 * math.log(2 * math.pi)
            - 0.5 * z_residual * z_residual
        )
        positive_censor = safe_censor.clamp_min(torch.finfo(torch.float64).tiny)
        z_censor = (positive_censor.log() - mu) / sigma
        survival = torch.where(
            safe_censor > 0, torch.special.log_ndtr(-z_censor), torch.zeros_like(safe_censor)
        )
    else:
        return None
    nll = torch.where(
        observed & eligible, -density,
        torch.where(~observed & eligible, -survival, torch.zeros_like(residual)),
    )
    return nll, eligible


def _add_extra(sums: _Sums, values: dict[str, torch.Tensor], eligible: dict[str, torch.Tensor]) -> None:
    """Joint/time metrics have their own eligibility; add them without recounting rows."""
    for name in (*JOINT_METRICS, *TIME_METRICS):
        if name in values:
            mask = eligible[name]
            count = int(mask.sum())
            sums.counts[name] = sums.counts.get(name, 0) + count
            if count:
                sums.totals[name] = sums.totals.get(name, 0.0) + float(
                    values[name].to(torch.float64)[mask].sum()
                )


def _finalize(sums: _Sums) -> dict[str, Any]:
    out = sums.result()
    # Ranking metrics share one eligible population; report its size once.
    out["eligible_rows"] = out.pop("ndcg@10_rows", 0)
    for name in RANKING_METRICS:
        out.pop(f"{name}_rows", None)
    return out


def monitor_value(evaluation: dict[str, Any], monitor: str, *, survival_profile: bool) -> float | None:
    """The scalar a run selects on, or ``None`` when the population was empty."""
    key = monitor
    if monitor == "val_loss":
        key = "joint_nll" if survival_profile else "poi_nll"
    return evaluation.get(key)


def monitor_key(monitor: str, *, survival_profile: bool) -> str:
    if monitor == "val_loss":
        return "joint_nll" if survival_profile else "poi_nll"
    return monitor
