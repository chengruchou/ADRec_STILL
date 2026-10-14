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

#: Training profile each backend trains under by default.
TRAINING_PROFILE_DEFAULTS: dict[str, str] = {
    "sasrec_event_only": "still-event-only-observed-known-v1",
    "still_independent": "still-train-admin-censor-v1",
    "still_joint": "still-train-admin-censor-v1",
    "external_joint_lnm": "still-train-admin-censor-v1",
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
    if info["kind"] == "neural":
        training = dict(TRAINING_DEFAULTS)
        if name in TRAINING_PROFILE_DEFAULTS:
            training["training_profile"] = TRAINING_PROFILE_DEFAULTS[name]
        config["training"] = training
    return config
