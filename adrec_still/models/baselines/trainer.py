"""Shared training contract for the gradient-trained models (Issue #15).

One controlled training contract for GRU4Rec-H, SASRec-H, the Issue #40 matched
time-conditioning arms, the event-only control, the STILL arms and the external
LNM: the same canonical dataset and collator, training-target policy, uniform
negative sampler with sampled-softmax loss, optimizer settings and the per-gap
landmark weighting delivered by Issue #13. The history encoder -- or, among the
Issue #40 arms, the set of visible query-time features -- is the only intended
difference within those arms.

Training targets follow the shared policy: censored rows never enter the POI
target loss, and observed rows whose exact target is outside the training
vocabulary (cold-start / unknown targets) are excluded as well. The primary loss
is the weight-normalized mean ``sum_i(w_i * loss_i) / sum_i(w_i)`` with ``w`` from
:func:`adrec_still.data.weighting.landmark_weights`; a plain ``(w * loss).mean()`` is
never used because its scale depends on the batch weight sum.

The event-only arm preserves SASRec-H's entire architecture but uses only each
observed known-target gap's existing positive-duration s=0 row. The survival
profile (``still-train-admin-censor-v1``) gives every row of the administrative
censor training view exactly one loss term through the model's own adapter.

This module holds the contract: the model registry, the settings resolver, the
loss and the single optimizer step. The epoch loop, validation, checkpoint
retention and resume live in :mod:`adrec_still.training.loop`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, RandomSampler, Sampler, Subset

from adrec_still.data.collate import move_batch, select_batch_rows
from adrec_still.data.dataset import LandmarkDataset
from adrec_still.data.training_views import (
    EVENT_ONLY_TRAINING_PROFILE,
    EVENT_ONLY_WEIGHTING_CONTRACT,
    GAP_CENSOR_TRAINING_PROFILE,
    GAP_CENSOR_WEIGHTING_CONTRACT,
    event_only_sample_eligible,
    event_only_training_mask,
)
from adrec_still.data.weighting import (
    SURVIVAL_WEIGHTING_CONTRACT_VERSION,
    WEIGHTING_CONTRACT_VERSION,
    landmark_weights,
    survival_landmark_weights,
)
from adrec_still.models.baselines.base import BaselineModel
from adrec_still.models.baselines.gru4rec import GRU4RecH
from adrec_still.models.baselines.sasrec import SASRecH
from adrec_still.models.baselines.sasrec_event_only import SASRecEventOnly
from adrec_still.models.baselines.sasrec_time import (
    SASRecClock,
    SASRecElapsed,
    SASRecHPM,
    SASRecS,
)
from adrec_still.models.still.losses import (
    SURVIVAL_TRAINING_PROFILE,
    TRAINING_PROFILE,
    survival_training_masks,
)

TRAINER_VERSION = "still-trainer-v1.0"

#: Gradient-trained baselines sharing this training contract.
TRAINABLE_BASELINES: dict[str, type[BaselineModel]] = {
    GRU4RecH.name: GRU4RecH,
    SASRecH.name: SASRecH,
    SASRecEventOnly.name: SASRecEventOnly,
    # Issue #40 matched time-conditioning controls. They share SASRec-H's encoder,
    # item table and this training contract; only their visible time features differ.
    SASRecHPM.name: SASRecHPM,
    SASRecClock.name: SASRecClock,
    SASRecS.name: SASRecS,
    SASRecElapsed.name: SASRecElapsed,
}

#: Constructor keys copied from a model config, and from a checkpoint's recorded
#: model config, when present. Each model consumes only the keys it declares, so
#: the attention-only settings stay absent from GRU4Rec-H's resolved config.
MODEL_CONSTRUCTOR_KEYS = (
    "embedding_dim",
    "hidden_dim",
    "num_layers",
    "num_heads",
    "dropout",
    "max_sequence_length",
    # Declared only by the Issue #40 time controls. A constructor width a checkpoint
    # could not restore would rebuild at the default and then fail to load its own
    # state dict, so every settable width has to be listed here.
    "time_hidden_dim",
)

REQUIRED_TRAINING_KEYS = (
    "landmark_weighting",
    "optimizer",
    "learning_rate",
    "batch_size",
    "max_epochs",
    "early_stopping_patience",
    "early_stopping_metric",
    "negative_samples",
)

#: Metrics a run may select checkpoints on, and the direction that improves each.
#: ``val_loss`` is the profile's own validation loss (full-catalog ``poi_nll`` for the
#: observed-known profiles, ``joint_nll`` for the survival profile); ``train_loss``
#: is the labeled fallback when no validation split exists.
SELECTION_METRICS: dict[str, str] = {
    "val_loss": "min",
    "train_loss": "min",
    "poi_nll": "min",
    "joint_nll": "min",
    "time_nll": "min",
    **{f"{metric}@{k}": "max" for metric in ("recall", "ndcg", "mrr") for k in (5, 10, 20)},
}

#: Optimizer settings and their resolved defaults (the torch Adam defaults, which are
#: also the #37 v1 values). Every one is passed to ``torch.optim.Adam`` explicitly.
OPTIMIZER_DEFAULTS: dict[str, Any] = {
    "weight_decay": 0.0,
    "betas": (0.9, 0.999),
    "eps": 1.0e-8,
    "amsgrad": False,
}

#: Schedule/stabilization settings this trainer does *not* implement, with the only
#: value each may take. A config that requests anything else is refused rather than
#: silently trained without the requested behavior.
FIXED_SCHEDULE_SETTINGS: dict[str, Any] = {
    "scheduler": None,
    "gradient_clipping": None,
    "amp": False,
    "gradient_accumulation": 1,
    "drop_last": False,
    "min_delta": 0.0,
}

OPTIONAL_TRAINING_KEYS = (
    "early_stopping_mode",
    "validation_max_batches",
    "selection_horizon_sec",
    "tf32",
    "device",
    "training_profile",
)

#: Keys the trainer reads. Anything else in a ``training`` block is reported in the
#: summary as ``unconsumed_training_keys`` so a requested setting is never lost.
KNOWN_TRAINING_KEYS: frozenset[str] = frozenset(
    (
        *REQUIRED_TRAINING_KEYS,
        *OPTIMIZER_DEFAULTS,
        *FIXED_SCHEDULE_SETTINGS,
        *OPTIONAL_TRAINING_KEYS,
    )
)

#: Contract labels recorded in run summaries.
WEIGHTING_CONTRACTS: dict[str, str] = {
    EVENT_ONLY_TRAINING_PROFILE: EVENT_ONLY_WEIGHTING_CONTRACT,
    TRAINING_PROFILE: WEIGHTING_CONTRACT_VERSION,
    SURVIVAL_TRAINING_PROFILE: SURVIVAL_WEIGHTING_CONTRACT_VERSION,
    GAP_CENSOR_TRAINING_PROFILE: GAP_CENSOR_WEIGHTING_CONTRACT,
}

#: Profiles whose rows carry the three-way survival loss partition (observed-known,
#: observed-unknown, censored). The gap-censor profile is the s=0-only member.
SURVIVAL_LOSS_PROFILES = (SURVIVAL_TRAINING_PROFILE, GAP_CENSOR_TRAINING_PROFILE)

#: First non-reserved POI ID (padding and unknown are never candidates).
FIRST_CATALOG_POI_ID = 2


class RowGroupLocalShuffleSampler(Sampler[int]):
    """Seeded shuffle compatible with ``LandmarkDataset``'s one-row-group cache.

    A plain per-row shuffle makes almost every index cross a Parquet row-group
    boundary, and each crossing re-reads and re-decodes an entire group (the
    canonical writer buffers 100,000 rows per group). This sampler re-draws the
    row-group order and the row order inside each group from its generator on
    every epoch, but always emits one group's indices consecutively: an epoch
    performs at most one group read per non-empty group while still visiting
    every training row exactly once. The trade-off is that rows sharing a
    group stay in the same stretch of an epoch; order within and across groups
    remains random and seeded.
    """

    def __init__(self, row_group_lengths: list[int], seed: int) -> None:
        if any(int(length) < 0 for length in row_group_lengths):
            raise ValueError("row-group lengths must be non-negative")
        self._lengths = [int(length) for length in row_group_lengths]
        self._offsets = []
        offset = 0
        for length in self._lengths:
            self._offsets.append(offset)
            offset += length
        self._total = offset
        self._generator = torch.Generator().manual_seed(int(seed))

    def __len__(self) -> int:
        return self._total

    def __iter__(self):
        group_order = torch.randperm(len(self._lengths), generator=self._generator)
        for group in group_order.tolist():
            length = self._lengths[group]
            if not length:
                continue
            within = torch.randperm(length, generator=self._generator)
            yield from (self._offsets[group] + within).tolist()


def build_training_sampler(train_source: Dataset, seed: int) -> Sampler[int]:
    """Seeded epoch shuffling; row-group-local for the canonical Parquet dataset."""
    if isinstance(train_source, LandmarkDataset):
        return RowGroupLocalShuffleSampler(train_source.row_group_lengths(), seed)
    return RandomSampler(train_source, generator=torch.Generator().manual_seed(int(seed)))


def resolve_device(setting: str | torch.device | None) -> torch.device:
    """Resolve ``cpu`` / ``cuda`` / ``cuda:N`` / ``auto`` / ``None`` to a device.

    A CUDA device is made current here, so later CUDA calls that assume an
    initialized context (peak-memory statistics, RNG capture) work on a cold
    process without the caller first running a warm-up.
    """
    if isinstance(setting, torch.device):
        device = setting
    elif setting in (None, "auto"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(setting)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device {device} requested but CUDA is not available; pass --device cpu"
            )
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"device cuda:{index} requested but only {torch.cuda.device_count()} "
                "CUDA device(s) are visible"
            )
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
    return device


def validate_training_config(training: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_TRAINING_KEYS if key not in training]
    if missing:
        raise KeyError(f"training config is missing required keys: {sorted(missing)}")
    if training["optimizer"] != "adam":
        raise ValueError("the shared training contract fixes optimizer: adam")
    if training["landmark_weighting"] not in {"per_gap", "per_landmark"}:
        raise ValueError("training.landmark_weighting must be per_gap or per_landmark")
    if (
        training.get("training_profile") == EVENT_ONLY_TRAINING_PROFILE
        and training["landmark_weighting"] != "per_gap"
    ):
        raise ValueError("event-only training requires per_gap: one unit-weight row per gap")
    if int(training["negative_samples"]) <= 0:
        raise ValueError("training.negative_samples must be positive")
    # A run with no epochs would silently checkpoint an untrained model.
    if int(training["max_epochs"]) < 1:
        raise ValueError("training.max_epochs must be at least 1")
    if int(training["early_stopping_patience"]) < 1:
        raise ValueError("training.early_stopping_patience must be at least 1")
    metric = training["early_stopping_metric"]
    if metric not in SELECTION_METRICS:
        raise ValueError(
            f"training.early_stopping_metric {metric!r} is not a selectable metric; choose "
            f"one of {sorted(SELECTION_METRICS)}"
        )
    # Selection direction follows the metric; a config may state it but never contradict it.
    mode = training.get("early_stopping_mode")
    if mode is not None and mode != SELECTION_METRICS[metric]:
        raise ValueError(
            f"training.early_stopping_metric {metric!r} improves in direction "
            f"{SELECTION_METRICS[metric]!r}; early_stopping_mode {mode!r} contradicts it"
        )
    horizon = training.get("selection_horizon_sec")
    if horizon is not None and (
        isinstance(horizon, bool) or int(horizon) != horizon or int(horizon) < 0
    ):
        raise ValueError("training.selection_horizon_sec must be null or a non-negative integer")
    cap = training.get("validation_max_batches")
    if cap is not None and (isinstance(cap, bool) or int(cap) <= 0):
        raise ValueError("training.validation_max_batches must be null or a positive integer")
    if float(training.get("weight_decay", OPTIMIZER_DEFAULTS["weight_decay"])) < 0:
        raise ValueError("training.weight_decay must be non-negative")
    betas = training.get("betas", OPTIMIZER_DEFAULTS["betas"])
    if (
        isinstance(betas, (str, bytes))
        or len(betas) != 2
        or not all(0.0 <= float(beta) < 1.0 for beta in betas)
    ):
        raise ValueError("training.betas must be two floats in [0, 1)")
    if float(training.get("eps", OPTIMIZER_DEFAULTS["eps"])) <= 0:
        raise ValueError("training.eps must be positive")
    if not isinstance(training.get("amsgrad", OPTIMIZER_DEFAULTS["amsgrad"]), bool):
        raise TypeError("training.amsgrad must be a boolean")
    if not isinstance(training.get("tf32", False), bool):
        raise TypeError("training.tf32 must be a boolean")
    # Settings the trainer does not implement may only carry their fixed value. A
    # config that asks for a scheduler, clipping, AMP, accumulation, drop_last or a
    # nonzero min_delta is refused: training without the requested behavior while
    # recording the request would be a silent recipe change.
    for key, fixed in FIXED_SCHEDULE_SETTINGS.items():
        if key in training and training[key] != fixed:
            raise ValueError(
                f"training.{key}={training[key]!r} is not implemented by the shared trainer "
                f"({TRAINER_VERSION}); the only supported value is {fixed!r}"
            )


def resolve_training_settings(training: dict[str, Any]) -> dict[str, Any]:
    """Validate and resolve every setting the trainer will actually run with.

    Defaults are filled explicitly (optimizer state, fixed schedule settings,
    selection direction/horizon/cap) so the resolved mapping is the complete
    executed recipe; ``unconsumed_training_keys`` lists keys the trainer never reads
    so a requested setting cannot be dropped without a trace.
    """
    validate_training_config(training)
    resolved: dict[str, Any] = {key: training[key] for key in REQUIRED_TRAINING_KEYS}
    resolved["learning_rate"] = float(training["learning_rate"])
    resolved["batch_size"] = int(training["batch_size"])
    resolved["max_epochs"] = int(training["max_epochs"])
    resolved["early_stopping_patience"] = int(training["early_stopping_patience"])
    resolved["negative_samples"] = int(training["negative_samples"])
    for key, default in OPTIMIZER_DEFAULTS.items():
        resolved[key] = training.get(key, default)
    resolved["weight_decay"] = float(resolved["weight_decay"])
    resolved["betas"] = tuple(float(beta) for beta in resolved["betas"])
    resolved["eps"] = float(resolved["eps"])
    resolved["amsgrad"] = bool(resolved["amsgrad"])
    for key, fixed in FIXED_SCHEDULE_SETTINGS.items():
        resolved[key] = fixed
    resolved["early_stopping_mode"] = SELECTION_METRICS[training["early_stopping_metric"]]
    cap = training.get("validation_max_batches")
    resolved["validation_max_batches"] = None if cap is None else int(cap)
    horizon = training.get("selection_horizon_sec")
    resolved["selection_horizon_sec"] = None if horizon is None else int(horizon)
    resolved["tf32"] = bool(training.get("tf32", False))
    resolved["training_profile"] = training.get("training_profile", TRAINING_PROFILE)
    if "device" in training:
        resolved["device"] = training["device"]
    resolved["unconsumed_training_keys"] = sorted(set(training) - KNOWN_TRAINING_KEYS)
    return resolved


def apply_precision_policy(settings: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Apply the recipe's TF32 setting on CUDA and record the actual execution settings.

    Neural parameters and the main neural operations run in float32 (the module
    default dtype); the time/survival math is float64 by construction of the
    distribution modules. AMP is refused by validation, so no gradient scaler exists.
    """
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = settings["tf32"]
        torch.backends.cudnn.allow_tf32 = settings["tf32"]
    record: dict[str, Any] = {
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "neural_dtype": str(torch.get_default_dtype()),
        "time_math_dtype": "torch.float64",
        "amp": False,
        "tf32_requested": settings["tf32"],
        "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn_allowed": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "tf32_effective": device.type == "cuda",
    }
    return record


def select_training_rows(
    batch: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], int]:
    """Keep rows eligible for the POI target loss under the shared policy.

    Censored rows and observed rows with cold-start/unknown exact targets are
    dropped; the returned count is the number of excluded rows.
    """
    keep = batch["observed_mask"].bool() & batch["target_in_train_catalog"].bool()
    kept = select_batch_rows(batch, keep)
    return kept, int((~keep).sum())


def validate_event_only_profile(model: BaselineModel, training: dict[str, Any]) -> bool:
    """Refuse a model/profile mismatch before selection, allocation or gradients."""
    event_only = training.get("training_profile") == EVENT_ONLY_TRAINING_PROFILE
    capable = isinstance(model, SASRecEventOnly) or bool(
        getattr(model, "trains_event_only", False)
    )
    if event_only != capable:
        raise ValueError(
            f"training_profile={EVENT_ONLY_TRAINING_PROFILE!r} and the model's "
            "event-only capability must agree; this profile is reserved for models "
            "that train once per gap at s=0 with known targets"
        )
    if event_only and training["landmark_weighting"] != "per_gap":
        raise ValueError("event-only training requires per_gap: one unit-weight row per gap")
    return event_only


def event_only_training_source(source: Dataset) -> Dataset:
    """Select eligibility once before batching, preserving Parquet row-group locality."""
    if isinstance(source, LandmarkDataset):
        if source.training_view == EVENT_ONLY_TRAINING_PROFILE:
            return source
        return LandmarkDataset(source.path, split=source.split,
                               training_view=EVENT_ONLY_TRAINING_PROFILE)
    return Subset(source, [index for index in range(len(source))
                           if event_only_sample_eligible(source[index])])


def sample_negative_pois(
    row_count: int,
    negative_samples: int,
    number_of_pois: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Uniform catalog negatives, excluding the reserved padding/unknown IDs."""
    if number_of_pois <= FIRST_CATALOG_POI_ID:
        raise ValueError("negative sampling requires at least one non-reserved catalog POI")
    return torch.randint(
        FIRST_CATALOG_POI_ID,
        number_of_pois,
        (row_count, negative_samples),
        generator=generator,
    )


def build_sampled_candidates(
    targets: torch.Tensor,
    negative_samples: int,
    number_of_pois: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared sampled-softmax candidate policy: positive-first candidate IDs plus the
    accidental-hit penalty that pushes a sampled negative equal to the row's positive to
    the masked floor. Every model training under this contract must consume this builder
    so the sampler/correction policy stays identical across arms."""
    negatives = sample_negative_pois(
        int(targets.shape[0]), negative_samples, number_of_pois, generator
    ).to(targets.device)
    candidates = torch.cat([targets.unsqueeze(1), negatives], dim=1)
    accidental_hit_penalty = torch.zeros(
        candidates.shape, dtype=torch.float32, device=targets.device
    )
    accidental_hit_penalty[:, 1:] = negatives.eq(targets.unsqueeze(1)).float() * 1.0e9
    return candidates, accidental_hit_penalty


def per_row_sampled_losses(
    model: BaselineModel,
    kept: dict[str, torch.Tensor],
    negative_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Per-kept-row training losses under the shared sampled-candidate policy.

    A model may provide ``per_row_sampled_training_losses(kept, negative_samples,
    generator)`` to add model-specific terms (for example a time likelihood) while
    consuming :func:`build_sampled_candidates` for its mark loss; otherwise the default
    POI-only sampled-softmax cross-entropy runs.
    """
    if hasattr(model, "per_row_sampled_training_losses"):
        per_row = model.per_row_sampled_training_losses(kept, negative_samples, generator)
        if per_row.shape != kept["next_poi"].shape:
            raise ValueError("model-provided training losses must be one value per kept row")
        return per_row
    targets = kept["next_poi"]
    candidates, accidental_hit_penalty = build_sampled_candidates(
        targets, negative_samples, model.number_of_pois, generator
    )
    logits = model.score_candidates(model.encode(kept), candidates)
    logits = logits - accidental_hit_penalty
    positive_index = torch.zeros(targets.shape[0], dtype=torch.long, device=logits.device)
    return torch.nn.functional.cross_entropy(logits, positive_index, reduction="none")


def weighted_mean_loss(per_row: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """The contract loss ``sum(w * loss) / sum(w)``; never ``(w * loss).mean()``."""
    if per_row.shape != weights.shape:
        raise ValueError("per-row losses and weights must align")
    weight_sum = weights.sum()
    if float(weight_sum) <= 0.0:
        raise ValueError("training weights must have a positive sum")
    return (per_row * weights).sum() / weight_sum


def training_step(
    model: BaselineModel,
    batch: dict[str, torch.Tensor],
    training: dict[str, Any],
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, Any]:
    """One forward pass; returns the loss tensor with its weights and row counts."""
    event_only = validate_event_only_profile(model, training)
    batch = move_batch(batch, device)
    survival_profile = training.get("training_profile") in SURVIVAL_LOSS_PROFILES
    if event_only:
        keep = event_only_training_mask(batch, model.number_of_pois)
        kept = select_batch_rows(batch, keep)
        excluded_rows = int((~keep).sum())
    elif survival_profile:
        kept, excluded_rows = batch, 0
    else:
        kept, excluded_rows = select_training_rows(batch)
    row_count = int(kept["next_poi"].shape[0])
    if row_count == 0:
        return {"loss": None, "used_rows": 0, "excluded_rows": excluded_rows}
    if survival_profile:
        if not hasattr(model, "per_row_survival_training_losses"):
            raise TypeError("the survival profile requires a model survival-loss adapter")
        if training.get("training_profile") == GAP_CENSOR_TRAINING_PROFILE:
            if (kept["query_s_sec"] != 0).any():
                raise ValueError(
                    "the gap-censor profile trains on s=0 rows only; the training "
                    "source delivered a nonzero query_s_sec row"
                )
            # One row per gap: unit weights, no landmark-grid reweighting.
            weights = torch.ones_like(kept["residual_time_sec"], dtype=torch.float64)
        else:
            weights = survival_landmark_weights(kept, mode=training["landmark_weighting"])
        per_row = model.per_row_survival_training_losses(
            kept, int(training["negative_samples"]), generator
        )
        masks = survival_training_masks(kept)
        contribution_counts = {key: int(value.sum()) for key, value in masks.items()}
    else:
        # The retained s=0 row represents its entire gap. Reusing 1/m from the
        # landmark-expanded population would underweight long gaps here.
        weights = (
            torch.ones_like(kept["residual_time_sec"], dtype=torch.float64)
            if event_only else landmark_weights(kept, mode=training["landmark_weighting"])
        )
        per_row = per_row_sampled_losses(
            model, kept, int(training["negative_samples"]), generator
        )
        contribution_counts = {"observed_known": row_count}
    weights = weights.to(per_row.dtype)
    loss = weighted_mean_loss(per_row, weights)
    return {
        "loss": loss,
        "per_row_losses": per_row,
        "weights": weights,
        "used_rows": row_count,
        "excluded_rows": excluded_rows,
        "contribution_counts": contribution_counts,
    }


def build_baseline_model(model_config: dict[str, Any], number_of_pois: int) -> BaselineModel:
    """Instantiate one trainable baseline from its config and the catalog width."""
    name = model_config["name"]
    model_class = TRAINABLE_BASELINES.get(name)
    if model_class is None:
        raise NotImplementedError(f"no shared-trainer support for model: {name}")
    constructor = {
        key: model_config[key] for key in MODEL_CONSTRUCTOR_KEYS if key in model_config
    }
    return model_class(number_of_pois=number_of_pois, **constructor)


def load_checkpoint(path: str | Path) -> BaselineModel:
    """Rebuild a baseline from a checkpoint's recorded model config and weights.

    Reads both the current ``best.pt`` layout (``state_dict`` + ``model``) and the
    historical ``checkpoint.pt`` payload, which used the same two keys.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    resolved = payload["model"]
    model = build_baseline_model(resolved, resolved["number_of_pois"])
    model.load_state_dict(payload["state_dict"])
    model.is_fitted = True
    return model

