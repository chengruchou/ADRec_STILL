"""``python -m still`` command line."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from adrec_still.training.defaults import BACKENDS

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130


def _parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("overrides look like key=value (e.g. embedding_dim=32)")
    key, raw = text.split("=", 1)
    import yaml

    return key.strip(), yaml.safe_load(raw)


def _horizon(text: str) -> int | str:
    if text.lower() == "all":
        return "all"
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("a horizon is 'all' or a non-negative number of seconds")
    return value


def add_run_arguments(parser: argparse.ArgumentParser, *, batch: bool = False) -> None:
    data = parser.add_argument_group("data")
    data.add_argument("--data", type=Path, required=batch,
                      help="canonical directory (landmark_samples.parquet + poi_catalog.parquet) "
                           "or a raw check-in file (default: the --config's dataset_config)")
    data.add_argument("--adapter", choices=("gowalla", "foursquare_subset", "foursquare_global"),
                      help="raw-file format; inferred from the file name when omitted")
    data.add_argument("--poi-file", type=Path, help="POI table for the foursquare_global adapter")
    data.add_argument("--cache-dir", type=Path,
                      help="where preprocessed raw data and derived views go "
                           "(default: <output>/cache)")
    data.add_argument("--workers", type=int, help="preprocessing worker processes")
    data.add_argument("--row-limit", type=int, help="preprocess only the first N raw rows")

    model = parser.add_argument_group("model")
    if batch:
        model.add_argument("--models", required=True,
                           help="comma-separated backend IDs, e.g. sasrec_h,gru4rec_h,global_pop")
        model.add_argument("--seeds", default="41", help="comma-separated seeds (default 41)")
    else:
        model.add_argument("--model", choices=sorted(BACKENDS), metavar="NAME",
                           help="backend ID; see `python -m still models`")
        model.add_argument("--config", type=Path,
                           help="legacy experiment YAML (its model_config is used; its "
                                "dataset/manifest/selection fields are ignored)")
        model.add_argument("--seed", type=int, default=41)
    model.add_argument("--model-config", type=Path, help="model YAML overriding the defaults")
    model.add_argument("--set", dest="overrides", type=_parse_override, action="append",
                       default=[], metavar="KEY=VALUE",
                       help="override a model/training setting, e.g. --set embedding_dim=32 "
                            "--set training.negative_samples=16")

    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=int)
    training.add_argument("--batch-size", type=int)
    training.add_argument("--learning-rate", type=float)
    training.add_argument("--patience", type=int, help="early-stopping patience in epochs")
    training.add_argument("--monitor",
                          help="selection metric: val_loss (default), ndcg@10, recall@10, "
                               "mrr@10, poi_nll, joint_nll, time_nll, train_loss, ...")
    training.add_argument("--validation-horizon", type=_horizon, metavar="SEC|all",
                          help="score validation rows at this query_s_sec only "
                               "(default: the model config's value, else all rows)")
    training.add_argument("--validation-max-batches", type=int,
                          help="cap the per-epoch validation pass at N batches")
    training.add_argument("--max-train-batches", type=int,
                          help="cap every training epoch at N batches (smoke runs)")
    training.add_argument("--device", help="cpu, cuda, cuda:N or auto (default: auto)")
    training.add_argument("--num-workers", type=int, help="DataLoader workers (default 0)")
    training.add_argument("--max-sequence-length", type=int)
    if not batch:
        training.add_argument("--resume", type=Path, metavar="LAST_PT",
                              help="continue from a last.pt (optimizer, counters, RNG)")
        training.add_argument("--weights", type=Path, metavar="BEST_PT",
                              help="initialize weights from a best.pt; fresh optimizer")

    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument("--eval-batch-size", type=int)
    evaluation.add_argument("--chunk-size", type=int, help="candidate IDs per scoring block")
    evaluation.add_argument("--no-test", action="store_true",
                            help="skip the final test evaluation")
    evaluation.add_argument("--eval-validation", action="store_true",
                            help="also evaluate the full validation split with best weights")

    out = parser.add_argument_group("output")
    if batch:
        out.add_argument("--output-root", type=Path, required=True,
                         help="runs go to <output-root>/<model>/seed<N>/")
    else:
        out.add_argument("--output", type=Path, required=True, help="run directory")
    out.add_argument("--progress", choices=("auto", "bar", "plain", "off"), default="auto")
    out.add_argument("--progress-interval", type=float, default=5.0,
                     help="seconds between plain-text progress lines")


def spec_from_args(args: argparse.Namespace, *, model: str | None, seed: int, output: Path):
    from adrec_still.training.service import RunSpec

    return RunSpec(
        data=args.data, output=output, model=model,
        config=getattr(args, "config", None), model_config=args.model_config,
        overrides=dict(args.overrides), adapter=args.adapter, poi_file=args.poi_file,
        cache_dir=args.cache_dir, workers=args.workers, row_limit=args.row_limit,
        seed=seed, device=args.device, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, patience=args.patience, monitor=args.monitor,
        validation_horizon=args.validation_horizon,
        validation_max_batches=args.validation_max_batches,
        max_train_batches=args.max_train_batches, num_workers=args.num_workers,
        eval_batch_size=args.eval_batch_size, max_sequence_length=args.max_sequence_length,
        chunk_size=args.chunk_size, resume=getattr(args, "resume", None),
        weights=getattr(args, "weights", None), evaluate_test=not args.no_test,
        evaluate_validation=args.eval_validation,
        progress=args.progress, progress_interval=args.progress_interval,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m still",
        description="Train and evaluate STILL models on a dataset.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser(
        "train", help="train one model on one dataset",
        description="Train one model. Example: python -m still train --data data/processed/"
                    "foursquare_nyc --model sasrec_h --epochs 3 --device cpu --output runs/nyc",
    )
    add_run_arguments(train)

    batch = commands.add_parser(
        "batch", help="train a list of models/seeds one after another",
        description="Run every (model, seed) pair in order; the next run starts only "
                    "after the previous one completed. A failure stops the sequence.",
    )
    add_run_arguments(batch, batch=True)
    batch.add_argument("--continue-on-failure", action="store_true",
                       help="keep going after a failed run instead of stopping")

    models = commands.add_parser("models", help="list the implemented backends")
    models.set_defaults(func=_list_models)
    return parser


def _list_models(args: argparse.Namespace) -> int:
    width = max(len(name) for name in BACKENDS)
    for name in sorted(BACKENDS):
        info = BACKENDS[name]
        print(f"{name:<{width}}  {info['family']:<9} {info['kind']}")
    return EXIT_OK


def _hide_gpus_for_cpu_runs(device: str | None) -> None:
    """A ``--device cpu`` process never touches the GPU.

    torch's optimizer step initializes the accelerator even for CPU tensors, which
    would leave a CUDA context on a shared GPU for every CPU run. Hiding the devices
    before the first CUDA call keeps the process where it was asked to run.
    """
    if device == "cpu" and not torch_cuda_initialized():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def torch_cuda_initialized() -> bool:
    import torch

    return torch.cuda.is_initialized()


def _run_train(args: argparse.Namespace) -> int:
    from adrec_still.training import checkpoint as ckpt
    from adrec_still.training.loop import STATUS_INTERRUPTED, SelectionUnavailable
    from adrec_still.training.service import RunFailed, run

    spec = spec_from_args(args, model=args.model, seed=args.seed, output=args.output)
    _hide_gpus_for_cpu_runs(args.device)
    try:
        summary = run(spec)
    except KeyboardInterrupt:
        print("interrupted before training started; nothing was written", file=sys.stderr)
        return EXIT_INTERRUPTED
    except (RunFailed, ckpt.CheckpointError, ckpt.RunLockedError, SelectionUnavailable,
            FileNotFoundError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_FAILED
    if summary["status"] == STATUS_INTERRUPTED:
        return EXIT_INTERRUPTED
    return EXIT_OK


def _run_batch(args: argparse.Namespace) -> int:
    from adrec_still.training.batch import run_batch

    models = [name.strip() for name in args.models.split(",") if name.strip()]
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    unknown = [name for name in models if name not in BACKENDS]
    if unknown:
        print(f"error: unknown models {unknown}; see `python -m still models`", file=sys.stderr)
        return EXIT_USAGE
    _hide_gpus_for_cpu_runs(args.device)
    return run_batch(args, models, seeds, stop_on_failure=not args.continue_on_failure)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        return _run_train(args)
    if args.command == "batch":
        return _run_batch(args)
    return args.func(args)
