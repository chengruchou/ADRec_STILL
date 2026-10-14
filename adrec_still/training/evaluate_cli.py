"""Standalone evaluation of a saved checkpoint (``evaluate`` command).

Loads a ``best.pt`` (shared-trainer or staged format), rebuilds the recorded
model, checks the data identity (same catalog width and item-ID fingerprint —
a same-shape checkpoint from a different mapping is refused, never silently
matched), and runs the exact ranking evaluation plus, when the canonical
directory carries ``events.parquet``, the silence-aware metrics (repeat splits,
repeat-probability quality, time NLL, silence cells with their comparisons).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from adrec_still.training import checkpoint as ckpt
from adrec_still.training.evaluate import EvaluationOptions, evaluate_split
from adrec_still.training.inputs import resolve_data
from adrec_still.training.progress import Reporter, TerminalReporter
from adrec_still.training.stages import STAGED_BEST_FORMAT


def load_checkpoint_model(checkpoint_path: Path, data, *, device: torch.device):
    """Rebuild the model a best.pt holds, verifying the data identity."""
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "format" not in payload:
        raise ckpt.CheckpointError(f"{checkpoint_path} is not a recognized checkpoint")
    data_identity = {
        "number_of_pois": data.number_of_pois,
        "catalog_fingerprint": (data.identity or {}).get("catalog_fingerprint"),
    }
    problems = ckpt.data_differences(payload.get("data") or {}, data_identity)
    ckpt.require_compatible("evaluate", checkpoint_path, problems)
    if payload["format"] == STAGED_BEST_FORMAT:
        from adrec_still.models.upstream.wrappers import build_upstream_model

        resolved = dict(payload["model"])
        resolved.setdefault("hidden_size", resolved.get("embedding_dim"))
        resolved.setdefault("max_len", resolved.get("max_sequence_length"))
        model = build_upstream_model(resolved, int(resolved["number_of_pois"]))
        model.load_state_dict(payload["state_dict"])
        model.is_fitted = True
        model.to(device)
        return model, payload
    if payload["format"] == ckpt.BEST_FORMAT:
        from adrec_still.training.families import load_model

        model, payload = load_model(checkpoint_path, device=device)
        return model, payload
    raise ckpt.CheckpointError(
        f"{checkpoint_path} holds format {payload['format']!r}; pass a best.pt "
        f"({ckpt.BEST_FORMAT} or {STAGED_BEST_FORMAT})"
    )


def run_evaluate(args) -> int:
    from adrec_still.models.baselines.trainer import resolve_device

    reporter: Reporter = (
        Reporter() if args.progress == "off" else TerminalReporter(mode=args.progress)
    )
    started = time.time()
    device = resolve_device(args.device)
    data = resolve_data(
        args.data, adapter=args.adapter, poi_file=args.poi_file,
        cache_dir=args.cache_dir, workers=args.workers, reporter=reporter,
    )
    model, payload = load_checkpoint_model(Path(args.checkpoint), data, device=device)
    if getattr(model, "family", None) == "adrec_still":
        from adrec_still.training.service import attach_full_history

        attach_full_history(model, data, reporter=reporter)
        for key, value in (("time_samples", args.time_samples),
                           ("diffusion_samples", args.diffusion_samples),
                           ("inference_steps", args.inference_steps)):
            if value is not None:
                setattr(model, key, value)
        if args.eval_seed is not None:
            model.eval_noise_seed = int(args.eval_seed)
    options = EvaluationOptions(
        batch_size=args.batch_size, max_sequence_length=args.max_sequence_length,
        num_workers=args.num_workers, chunk_size=args.chunk_size,
        max_batches=args.max_batches,
    )
    horizons_seconds = (
        [hours * 3600 for hours in args.horizons_hours] if args.horizons_hours else None
    )
    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "model": (payload.get("model") or {}).get("name"),
        "selection": payload.get("selection"),
        "data": data.record(),
        "splits": {},
    }
    splits = [name.strip() for name in args.splits.split(",") if name.strip()]
    for split in splits:
        if not data.has_split(split):
            result["splits"][split] = {"available": False, "reason": "split absent"}
            continue
        with reporter.phase(f"evaluate-{split}"):
            block: dict[str, Any] = {
                "ranking": evaluate_split(
                    model, data.landmark_path, split=split, horizon=None,
                    breakdown=True, options=options,
                    progress=lambda done, total: reporter.progress(
                        "evaluate", done, total
                    ),
                )
            }
        if not args.no_silence:
            if data.events_path is None:
                block["silence"] = {
                    "available": False,
                    "reason": "events.parquet is absent; repeat/cell semantics need it",
                }
            else:
                from adrec_still.data.history import (
                    FullHistoryIndex,
                    train_cutoff_from_split_manifest,
                )
                from adrec_still.evaluation.silence_eval import (
                    evaluate_silence_split,
                    fit_cell_baselines,
                )

                cutoff = (
                    train_cutoff_from_split_manifest(data.split_manifest_path)
                    if data.split_manifest_path is not None else None
                )
                with reporter.phase("full-history-index"):
                    index = FullHistoryIndex(
                        data.events_path, data.catalog_path, train_cutoff_s=cutoff
                    )
                baselines = None
                if not args.no_cell_baselines and index.has_coordinates:
                    with reporter.phase("cell-baselines"):
                        baselines = fit_cell_baselines(
                            index, data.landmark_path,
                            cell_radius_km=args.cell_radius_km,
                            cell_time_threshold_hours=args.cell_time_threshold_hours,
                        )
                with reporter.phase(f"silence-{split}"):
                    block["silence"] = evaluate_silence_split(
                        model, data.landmark_path, split, index=index,
                        options=options, cell_radius_km=args.cell_radius_km,
                        cell_time_threshold_hours=args.cell_time_threshold_hours,
                        horizons_seconds=horizons_seconds,
                        cell_baselines=baselines, max_batches=args.max_batches,
                        progress=lambda done, total: reporter.progress(
                            "silence", done, total
                        ),
                    )
        result["splits"][split] = block
        headline = block["ranking"]
        reporter.message(
            f"[{split}] rows {headline.get('rows')} ndcg@10 "
            f"{headline.get('ndcg@10')} recall@10 {headline.get('recall@10')}"
        )
    result["seconds"] = time.time() - started
    output = Path(args.output) if args.output else Path(args.checkpoint).with_name(
        "evaluation.json"
    )
    from adrec_still.training.service import jsonable

    ckpt.atomic_json(output, jsonable(result))
    reporter.message(f"wrote {output}")
    return 0
