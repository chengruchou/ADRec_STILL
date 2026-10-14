"""Matched SASRec time-conditioning controls (Issue #40).

Three campaign arms answer one question: does query-time information help beyond
the history the encoder already sees? An honest answer needs the arms to differ in
*information*, not in capacity, so all of them share one SASRec encoder, one item
table, one training contract and one conditioning module of identical shape. The
only difference between them is which columns of a fixed-width query-time feature
vector the conditioning module is allowed to read.

| Arm (`name`) | class | `elapsed` | query clock |
| --- | --- | --- | --- |
| ``sasrec_h_pm`` | :class:`SASRecHPM` | null | null |
| ``sasrec_clock`` | :class:`SASRecClock` | null | visible |
| ``sasrec_s`` | :class:`SASRecS` | visible | visible |
| ``sasrec_elapsed`` | :class:`SASRecElapsed` | visible | null |

``sasrec_elapsed`` is the optional Elapsed-only arm of Issue #40. Its code exists
here so the arm can be enabled by the frozen protocol *before* any test inspection,
but it ships no committed experiment config: it is not part of the seven-configuration
first-stage campaign. The deterministic shuffled-time control is **not** implemented
here; it permutes an input across a split and therefore belongs to the dataset layer,
which is the only place that knows split boundaries.

Masked, not removed
-------------------
Every arm builds the same ``[batch, 5]`` feature vector and the same
``5 -> time_hidden_dim -> 2 * embedding_dim`` module. A masked feature is replaced
by :data:`TIME_FEATURE_NULL` with :func:`torch.where`, never dropped from the layout
and never multiplied by zero: ``0 * inf`` is ``NaN``, so a multiplicative mask would
let a masked column poison a row it is supposed to have no influence over. The
substitution is unconditional, so a masked column contributes the null value whatever
the raw value was.

Two consequences are load-bearing and are asserted by the tests:

* **Parameter counts match exactly.** The arms have identical module structure and
  identical ``state_dict`` keys and shapes; ``time_feature_mask`` is a *non-persistent*
  buffer, so it never enters a checkpoint and a foreign checkpoint can never silently
  redefine an arm's visible features. The arm's identity lives in the recorded model
  ``name``, which is how :func:`adrec_still.models.baselines.trainer.load_checkpoint`
  chooses the class to rebuild.
* **A masked column receives exactly zero gradient.** The null is ``0.0``, so the
  input layer's weight column for a masked feature has no signal to learn from. That
  is the leakage gate: if a masked column ever shows a nonzero gradient, the arm read
  a feature it declared it could not see.

Where the conditioning enters
-----------------------------
Late fusion at the exported query context, exactly as
``docs/SASREC_H_CONTEXT_CONTRACT.md`` froze it: the arms "add time inputs *around*
this encoder; they do not change what H exports". The time features describe the
*query moment*, not the history events, so they modulate the context the query is
scored with rather than being injected per history position. :meth:`encode_sequence`
is inherited unchanged and stays the pure history sequence;
:meth:`SASRecTimeControl.encode_history` exposes the unconditioned SASRec-H context.

The combination is FiLM, ``(1 + tanh(scale)) * history + shift``, with both terms
produced by the conditioning module. A shift alone could only add a query-dependent
global POI bias, which cannot reorder a row by any interaction between its history and
the time; the scale is what lets a conditioned arm express one. The output layer is
initialized at ``INITIALIZER_STD`` with a zero bias, so every arm starts within a small
perturbation of SASRec-H and the arms are matched at initialization as well as in
capacity.

The gate is bounded by ``tanh`` rather than applied raw. The history context is
post-layer-norm, so an unbounded multiplier is an unbounded logit temperature on a
tied-embedding output, and under the shared contract's Adam budget that is not a
theoretical worry: with a raw ``(1 + scale)`` gate, ``sasrec_h_pm`` -- whose modulation
is a single constant -- finished the 96-pair learning panel (at that fixture's lr 0.05, 200 epochs) at a
train loss of 0.64 against SASRec-H's 0.08 on the same panel and seed, purely from oscillation in that
one factor. Bounding the multiplier to ``(0, 2)`` brings it to 0.002. The bound costs no
expressiveness that matters here: a conditioned arm that wants a larger effect scales
the context's own magnitude through the encoder.

The hidden nonlinearity is ``tanh``, not the encoder's ReLU, for one specific reason:
under an all-null input the hidden pre-activation is the first layer's bias alone.
ReLU would zero every unit whose bias is negative and, at exactly zero, has a zero
derivative -- so ``sasrec_h_pm``'s module could be born dead and its matched capacity
would be a parameter count rather than trainable capacity. ``tanh`` keeps every unit's
gradient finite and nonzero, so the parameter-matched arm genuinely trains its constant.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from adrec_still.models.baselines.sasrec import (
    INITIALIZER_STD,
    SASREC_H_CONTEXT_CONTRACT_VERSION,
    SASRecH,
)

#: Identifier of the conditioned-context contract these arms export.
#: Distinct from :data:`SASREC_H_CONTEXT_CONTRACT_VERSION`, which describes the
#: unconditioned history context ``encode_history`` still returns unchanged.
SASREC_TIME_CONTROL_CONTRACT_VERSION = "still-sasrec-time-controls-v0.1"

#: Fixed query-time feature layout. Every arm builds all five columns; an arm that
#: cannot see a column receives :data:`TIME_FEATURE_NULL` in its place.
TIME_FEATURE_NAMES = (
    "elapsed_log1p_hours",
    "query_hour_sin",
    "query_hour_cos",
    "query_weekday_sin",
    "query_weekday_cos",
)

#: The four query-clock columns, masked together by the arms that exclude the clock.
CLOCK_FEATURE_NAMES = TIME_FEATURE_NAMES[1:]

#: The elapsed column: ``log1p(query_s_sec / 3600)``.
ELAPSED_FEATURE_NAME = TIME_FEATURE_NAMES[0]

#: Value substituted for a masked feature. Zero is *not* a point on the unit circle,
#: which is deliberate: a null clock means "no information", not "midnight Monday".
TIME_FEATURE_NULL = 0.0

SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0
DAYS_PER_WEEK = 7.0

#: 1970-01-01 (day index 0) was a Thursday, so adding 3 puts Monday at 0.
EPOCH_WEEKDAY_OFFSET = 3.0

#: Default width of the conditioning module's hidden layer.
DEFAULT_TIME_HIDDEN_DIM = 64


def query_time_features(
    batch: dict[str, torch.Tensor],
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the unmasked ``[batch, 5]`` query-time features in the frozen layout.

    The query moment is ``last_timestamp + query_s_sec`` interpreted as UTC epoch
    seconds, matching Issue #40's "compute clock from last_timestamp + query_s_sec
    in UTC". Both cyclic features come from that absolute moment, so they are
    invariant to *how* the query moment was reached; only ``elapsed`` depends on the
    horizon itself.

    The arithmetic runs in float64 and is cast down only at the end. A 2020-era epoch
    second is about ``1.6e9``; float32 carries roughly seven significant digits there,
    so a one-second -- or even one-hour -- change would be rounded away before the
    modulo ever ran. The narrowing happens after the values are reduced to a day and a
    week, where float32 is exact enough.
    """
    for field in ("last_timestamp", "query_s_sec"):
        if field not in batch:
            raise KeyError(f"query-time features require the canonical field {field!r}")
    target = device if device is not None else batch["query_s_sec"].device
    query_seconds = batch["query_s_sec"].to(device=target, dtype=torch.float64)
    last_timestamp = batch["last_timestamp"].to(device=target, dtype=torch.float64)
    if bool((query_seconds < 0).any()):
        raise ValueError("query_s_sec must be non-negative; log1p of the horizon is undefined")

    elapsed = torch.log1p(query_seconds / SECONDS_PER_HOUR)

    query_moment = last_timestamp + query_seconds
    day_angle = (
        2.0 * math.pi * torch.remainder(query_moment, SECONDS_PER_DAY) / SECONDS_PER_DAY
    )
    day_index = torch.floor(query_moment / SECONDS_PER_DAY)
    weekday = torch.remainder(day_index + EPOCH_WEEKDAY_OFFSET, DAYS_PER_WEEK)
    week_angle = 2.0 * math.pi * weekday / DAYS_PER_WEEK

    features = torch.stack(
        [
            elapsed,
            torch.sin(day_angle),
            torch.cos(day_angle),
            torch.sin(week_angle),
            torch.cos(week_angle),
        ],
        dim=1,
    )
    if features.shape[1] != len(TIME_FEATURE_NAMES):
        raise AssertionError("query-time feature layout does not match TIME_FEATURE_NAMES")
    return features.to(dtype)


class SASRecTimeControl(SASRecH):
    """Shared implementation of the Issue #40 arms; subclasses only declare a mask.

    Not registered or configured itself: an arm is one of the concrete subclasses
    below, whose ``name`` is what a config selects and what a checkpoint records.
    """

    #: Short arm label recorded in ``resolved_config``; the concrete subclasses set it.
    arm: str = ""

    #: Feature names this arm may read. Everything else is nulled.
    visible_time_features: tuple[str, ...] = ()

    def __init__(
        self,
        number_of_pois: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        max_sequence_length: int = 100,
        time_hidden_dim: int = DEFAULT_TIME_HIDDEN_DIM,
    ) -> None:
        super().__init__(
            number_of_pois,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_sequence_length=max_sequence_length,
        )
        if not self.arm or not type(self).__dict__.get("name"):
            raise TypeError("SASRecTimeControl subclasses must declare a name and an arm")
        unknown = set(self.visible_time_features) - set(TIME_FEATURE_NAMES)
        if unknown:
            raise ValueError(f"arm declares features outside the frozen layout: {sorted(unknown)}")
        if int(time_hidden_dim) <= 0:
            raise ValueError("time_hidden_dim must be positive")
        self.time_hidden_dim = int(time_hidden_dim)

        # Non-persistent: the mask is decided by the class, so it must never travel in a
        # state_dict where another arm's checkpoint could redefine this arm's inputs.
        self.register_buffer(
            "time_feature_mask",
            torch.tensor(
                [name in self.visible_time_features for name in TIME_FEATURE_NAMES],
                dtype=torch.bool,
            ),
            persistent=False,
        )
        self.time_input = nn.Linear(len(TIME_FEATURE_NAMES), self.time_hidden_dim)
        self.time_activation = nn.Tanh()
        # Emits the FiLM scale and shift concatenated, in that order.
        self.time_film = nn.Linear(self.time_hidden_dim, 2 * self.embedding_dim)
        nn.init.normal_(self.time_film.weight, mean=0.0, std=INITIALIZER_STD)
        nn.init.zeros_(self.time_film.bias)

    @property
    def masked_time_features(self) -> tuple[str, ...]:
        """Feature names this arm is held to a fixed null on."""
        return tuple(
            name for name in TIME_FEATURE_NAMES if name not in self.visible_time_features
        )

    def time_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return this arm's ``[batch, 5]`` features, masked columns held at the null.

        ``torch.where`` rather than a multiplicative mask: substitution is
        unconditional, so a masked column contributes exactly the null value even if
        the raw value were nonfinite, whereas ``0 * inf`` would be ``NaN``.
        """
        raw = query_time_features(batch, device=self.device, dtype=self.item_embedding.weight.dtype)
        masked = torch.where(
            self.time_feature_mask.unsqueeze(0),
            raw,
            torch.full_like(raw, TIME_FEATURE_NULL),
        )
        if not torch.isfinite(masked).all():
            raise ValueError(
                f"{self.name} produced nonfinite query-time features from its visible inputs"
            )
        return masked

    def time_modulation(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the raw FiLM ``(scale, shift)`` pair, each ``[batch, embedding_dim]``.

        ``scale`` is pre-``tanh``: :meth:`encode` applies the bound. Two rows agree here
        exactly when they agree on every feature the arm can see, which is what makes
        this the surface the invariance tests read.
        """
        hidden = self.time_activation(self.time_input(self.time_features(batch)))
        scale, shift = self.time_film(hidden).chunk(2, dim=1)
        return scale, shift

    def encode_history(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the unconditioned SASRec-H context, unchanged from Issue #15.

        This is the surface that still satisfies ``SASREC_H_CONTEXT_CONTRACT_VERSION``:
        history-only, causal, padding-invariant, zero for an evidence-free history.
        """
        return super().encode(batch)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the time-conditioned context that scores the catalog.

        This is what ``predict`` exports and what the shared trainer's sampled loss
        consumes, so an arm's declared features reach the scores through exactly one
        path. ``encode_sequence`` is inherited unchanged and remains unconditioned.
        """
        history = self.encode_history(batch)
        scale, shift = self.time_modulation(batch)
        return history * (1.0 + torch.tanh(scale)) + shift

    def resolved_config(self) -> dict[str, Any]:
        config = super().resolved_config()
        config.update(
            {
                # What ``encode`` exports is a conditioned context, so it carries its own
                # contract; the history contract still describes ``encode_history``.
                "context_contract_version": SASREC_TIME_CONTROL_CONTRACT_VERSION,
                "history_context_contract_version": SASREC_H_CONTEXT_CONTRACT_VERSION,
                "arm": self.arm,
                "time_conditioning": "film_tanh_gate",
                "time_hidden_dim": self.time_hidden_dim,
                "time_feature_names": list(TIME_FEATURE_NAMES),
                "visible_time_features": list(self.visible_time_features),
                "masked_time_features": list(self.masked_time_features),
                "time_feature_null": TIME_FEATURE_NULL,
            }
        )
        return config


class SASRecHPM(SASRecTimeControl):
    """SASRec-H-PM: the parameter-matched control. Every time input is nulled."""

    name = "sasrec_h_pm"
    arm = "h_pm"
    visible_time_features: tuple[str, ...] = ()


class SASRecClock(SASRecTimeControl):
    """Clock: query hour-of-day and day-of-week only; the elapsed horizon is nulled."""

    name = "sasrec_clock"
    arm = "clock"
    visible_time_features = CLOCK_FEATURE_NAMES


class SASRecS(SASRecTimeControl):
    """Full (SASRec-S): ``log1p(query_s_sec / 3600)`` plus the query-clock features.

    Not to be confused with the lead-owned SASRec-S-LN survival baseline: this arm has
    no time head and predicts no residual time. It is the fully informed *ranking* arm
    of the Issue #40 control set.
    """

    name = "sasrec_s"
    arm = "full"
    visible_time_features = TIME_FEATURE_NAMES


class SASRecElapsed(SASRecTimeControl):
    """Optional Elapsed-only arm: the horizon without the clock.

    Implemented ahead of any test inspection so the frozen protocol can enable it
    without a post-hoc code change; it carries no committed experiment config and is
    not one of the first-stage campaign configurations.
    """

    name = "sasrec_elapsed"
    arm = "elapsed"
    visible_time_features = (ELAPSED_FEATURE_NAME,)


#: The Issue #40 arms, in campaign order. ``sasrec_elapsed`` trails as the optional arm.
TIME_CONTROL_ARMS: tuple[type[SASRecTimeControl], ...] = (
    SASRecHPM,
    SASRecClock,
    SASRecS,
    SASRecElapsed,
)
