"""Built-in defaults so training works from an installed package, without ``configs/``.

The per-model constructor defaults transcribe the committed model YAMLs; the
shared training block is the recipe every neural backend has trained under so
far. A ``--model-config`` file or a legacy experiment config overrides any of it.
"""

from __future__ import annotations

from typing import Any

#: Every implemented training/fit backend, by public ID, with its family and kind.
BACKENDS: dict[str, dict[str, str]] = {
    # deterministic floors
    "global_pop": {"family": "baseline", "kind": "floor"},
    "user_history_pop": {"family": "baseline", "kind": "floor"},
    "repeat_last": {"family": "baseline", "kind": "floor"},
    "first_order_markov": {"family": "baseline", "kind": "floor"},
    "global_pop_s": {"family": "baseline", "kind": "floor"},
    "user_history_pop_s": {"family": "baseline", "kind": "floor"},
    "repeat_last_s": {"family": "baseline", "kind": "floor"},
    "first_order_markov_s": {"family": "baseline", "kind": "floor"},
    # neural baselines
    "gru4rec_h": {"family": "baseline", "kind": "neural"},
    "sasrec_h": {"family": "baseline", "kind": "neural"},
    "sasrec_h_pm": {"family": "baseline", "kind": "neural"},
    "sasrec_clock": {"family": "baseline", "kind": "neural"},
    "sasrec_s": {"family": "baseline", "kind": "neural"},
    "sasrec_event_only": {"family": "baseline", "kind": "neural"},
    "sasrec_elapsed": {"family": "baseline", "kind": "neural"},
    # STILL
    "still_independent": {"family": "still", "kind": "neural"},
    "still_joint": {"family": "still", "kind": "neural"},
    # external
    "external_joint_lnm": {"family": "external", "kind": "neural"},
    # upstream ADRec fork (event-only models; staged trainer)
    "adrec": {"family": "upstream", "kind": "neural"},
    "diffurec": {"family": "upstream", "kind": "neural"},
    "dreamrec": {"family": "upstream", "kind": "neural"},
    "sasrec_upstream": {"family": "upstream", "kind": "neural"},
    # ADRec-STILL extension family (full repeat memory; silence conditioning)
    "adrec_repeat": {"family": "adrec_still", "kind": "neural"},
    "adrec_elapsed": {"family": "adrec_still", "kind": "neural"},
    "adrec_repeat_elapsed": {"family": "adrec_still", "kind": "neural"},
    "adrec_still": {"family": "adrec_still", "kind": "neural"},
    "still_direct": {"family": "adrec_still", "kind": "neural"},
}

_SASREC = {
    "embedding_dim": 128,
    "hidden_dim": 128,
    "num_layers": 2,
    "num_heads": 2,
    "dropout": 0.2,
    "max_sequence_length": 100,
}
_SASREC_TIME = {**_SASREC, "time_hidden_dim": 64}
_STILL = {
    "embedding_dim": 128,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "mixture_components": 2,
}
_UPSTREAM = {
    "hidden_size": 128,
    "max_len": 50,
    "dropout": 0.1,
    "emb_dropout": 0.3,
    "ce_chunk_tokens": 4096,
}
_UPSTREAM_DIFFUSION = {
    **_UPSTREAM,
    "diffusion_steps": 32,
    "noise_schedule": "trunc_lin",
    "schedule_sampler_name": "uniform",
    "lambda_uncertainty": 0.001,
    "rescale_timesteps": True,
    "cfg_scale": 1.0,
}
_ADREC_STILL = {
    "hidden_size": 128,
    "dropout": 0.1,
    "emb_dropout": 0.3,
    "mixture_components": 2,
    "diffusion_steps": 32,
    "noise_schedule": "trunc_lin",
    "schedule_sampler_name": "uniform",
    "lambda_uncertainty": 0.001,
    "rescale_timesteps": True,
    "cfg_scale": 1.0,
    "loss_scale": 1.0,
    "ce_weight": 1.0,
    # Engineering starting points for the silence-cell axes; research-calibrated
    # values are open (docs/MODELING.md), never claim these as confirmed settings.
    "cell_time_threshold_hours": 24.0,
    "time_samples": 2,
    "diffusion_samples": 1,
    "inference_steps": None,
}

#: Constructor defaults per backend (the committed YAML values).
MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "global_pop": {"count_source": "train_targets", "smoothing": 1.0},
    "user_history_pop": {
        "count_source": "train_targets", "smoothing": 1.0,
        "history_weight": 0.9, "max_history_length": 100,
    },
    "repeat_last": {"count_source": "train_targets", "smoothing": 1.0, "repeat_weight": 0.5},
    "first_order_markov": {
        "count_source": "train_targets", "smoothing": 1.0, "transition_weight": 0.9,
    },
    "global_pop_s": {"count_source": "train_targets", "smoothing": 1.0, "silence_alpha": 1.0},
    "user_history_pop_s": {
        "count_source": "train_targets", "smoothing": 1.0,
        "history_weight": 0.9, "max_history_length": 100, "silence_alpha": 1.0,
    },
    "repeat_last_s": {
        "count_source": "train_targets", "smoothing": 1.0, "repeat_weight": 0.5,
        "silence_alpha": 1.0,
    },
    "first_order_markov_s": {
        "count_source": "train_targets", "smoothing": 1.0, "transition_weight": 0.9,
        "silence_alpha": 1.0,
    },
    "gru4rec_h": {"embedding_dim": 128, "hidden_dim": 128, "num_layers": 2, "dropout": 0.2},
    "sasrec_h": dict(_SASREC),
    "sasrec_h_pm": dict(_SASREC_TIME),
    "sasrec_clock": dict(_SASREC_TIME),
    "sasrec_s": dict(_SASREC_TIME),
    "sasrec_event_only": dict(_SASREC),
    "sasrec_elapsed": dict(_SASREC_TIME),
    "still_independent": dict(_STILL),
    "still_joint": {**_STILL, "variant": "survivor_reweighted"},
    "external_joint_lnm": {
        "mark_embedding_dim": 16, "context_dim": 32, "num_layers": 1, "dropout": 0.0,
        "mixture_components": 2, "log_time_statistics": "fit_train_unique_gaps",
    },
    "adrec": {
        **_UPSTREAM_DIFFUSION, "independent": True, "geodesic": False,
        "dif_decoder": "att", "is_causal": True, "loss_scale": 1.0,
        "embedding_pretrain": True, "inference_steps": None,
    },
    "diffurec": dict(_UPSTREAM_DIFFUSION),
    "dreamrec": dict(_UPSTREAM_DIFFUSION),
    "sasrec_upstream": dict(_UPSTREAM),
    "adrec_repeat": dict(_ADREC_STILL),
    "adrec_elapsed": dict(_ADREC_STILL),
    "adrec_repeat_elapsed": dict(_ADREC_STILL),
    "adrec_still": dict(_ADREC_STILL),
    "still_direct": dict(_ADREC_STILL),
}

#: The shared neural training recipe. ``early_stopping_metric`` is absent on
#: purpose: the service fills it from the profile (validation loss) unless a config
#: or the CLI names a metric.
TRAINING_DEFAULTS: dict[str, Any] = {
    "landmark_weighting": "per_gap",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "betas": [0.9, 0.999],
    "eps": 1.0e-8,
    "amsgrad": False,
    "batch_size": 256,
    "max_epochs": 20,
    "early_stopping_patience": 5,
    "negative_samples": 128,
    "selection_horizon_sec": None,
    "validation_max_batches": None,
    "tf32": False,
}

#: The upstream family's staged training recipe (fork values where they exist;
#: every stage budget is adjustable, down to one epoch per stage for smoke runs).
UPSTREAM_TRAINING_DEFAULTS: dict[str, Any] = {
    "optimizer": "adam",
    "learning_rate": 0.001,
    "weight_decay": 1.0e-5,
    "batch_size": 256,
    "max_epochs": 20,
    "early_stopping_patience": 5,
    "eval_interval": 1,
    "pretrain_epochs": 10,
    "warmup_epochs": 5,
    "scheduler_t_max": 500,
    "pcgrad": False,
    "early_stopping_metric": "ndcg@10",
    "selection_horizon_sec": None,
    "validation_max_batches": None,
}

#: Training profile each backend trains under by default.
TRAINING_PROFILE_DEFAULTS: dict[str, str] = {
    "sasrec_event_only": "still-event-only-observed-known-v1",
    "still_independent": "still-train-admin-censor-v1",
    "still_joint": "still-train-admin-censor-v1",
    "external_joint_lnm": "still-train-admin-censor-v1",
    # ADRec-STILL: once-per-gap base-distribution training (s=0, censors kept)
    "adrec_still": "adrec-still-gap-censor-v1",
    "still_direct": "adrec-still-gap-censor-v1",
    # the repeat-memory ablation is event-only (no time model, no censors)
    "adrec_repeat": "still-event-only-observed-known-v1",
    # the elapsed ablations train on the full landmark grid (observed-known rows)
}

#: Selection default per backend where it differs from the family default.
SELECTION_METRIC_DEFAULTS: dict[str, str] = {
    name: "ndcg@10"
    for name in ("adrec_repeat", "adrec_elapsed", "adrec_repeat_elapsed",
                 "adrec_still", "still_direct")
}

#: Evaluation defaults shared by every backend.
EVALUATION_DEFAULTS: dict[str, Any] = {
    "batch_size": 128,
    "max_sequence_length": 100,
    "num_workers": 0,
    "chunk_size": 16384,
}


def backend(name: str) -> dict[str, str]:
    try:
        return BACKENDS[name]
    except KeyError:
        known = ", ".join(sorted(BACKENDS))
        raise ValueError(f"unknown model {name!r}; implemented backends: {known}") from None


def default_model_config(name: str) -> dict[str, Any]:
    info = backend(name)
    config: dict[str, Any] = {"name": name, "family": info["family"], **MODEL_DEFAULTS[name]}
    if info["family"] == "upstream":
        config["training"] = dict(UPSTREAM_TRAINING_DEFAULTS)
    elif info["kind"] == "neural":
        training = dict(TRAINING_DEFAULTS)
        if name in TRAINING_PROFILE_DEFAULTS:
            training["training_profile"] = TRAINING_PROFILE_DEFAULTS[name]
        if name in SELECTION_METRIC_DEFAULTS:
            training["early_stopping_metric"] = SELECTION_METRIC_DEFAULTS[name]
        config["training"] = training
    return config
