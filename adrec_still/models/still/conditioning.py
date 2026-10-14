"""Training/query-available conditioning features for the #41 model arms.

Only information that is legitimately available at prediction time is encoded:

- the anchor timestamp (``last_timestamp``, the last observed history event, UTC seconds);
- the elapsed no-event interval ``query_s_sec``.

The query wall clock is the deterministic function ``last_timestamp + query_s_sec``; no
future event, target, completed label or true residual time is touched. Clock features
use continuous day/week phases in UTC (the Unix epoch starts on a Thursday, which only
shifts the week phase by a constant and is irrelevant to a learned head).
"""

from __future__ import annotations

import math

import torch

SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0
SECONDS_PER_WEEK = 604800.0

#: Feature widths exposed for head sizing.
CLOCK_FEATURE_DIM = 4
ELAPSED_FEATURE_DIM = 1


def clock_features(timestamp_sec: torch.Tensor) -> torch.Tensor:
    """sin/cos of the UTC day phase and week phase for a [batch] timestamp tensor."""
    if timestamp_sec.ndim != 1:
        raise ValueError("timestamp_sec must have shape [batch]")
    timestamp = timestamp_sec.to(torch.float64)
    day_angle = 2.0 * math.pi * (timestamp % SECONDS_PER_DAY) / SECONDS_PER_DAY
    week_angle = 2.0 * math.pi * (timestamp % SECONDS_PER_WEEK) / SECONDS_PER_WEEK
    return torch.stack(
        [day_angle.sin(), day_angle.cos(), week_angle.sin(), week_angle.cos()], dim=1
    ).to(torch.float32)


def elapsed_feature(query_s_sec: torch.Tensor) -> torch.Tensor:
    """``log1p(elapsed hours)`` as a [batch, 1] feature."""
    if query_s_sec.ndim != 1:
        raise ValueError("query_s_sec must have shape [batch]")
    elapsed = query_s_sec.to(torch.float64)
    if (elapsed < 0).any():
        raise ValueError("query_s_sec must be non-negative")
    return torch.log1p(elapsed / SECONDS_PER_HOUR).to(torch.float32).unsqueeze(1)


def query_features(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Elapsed + query-clock features for the direct (independent) arm, [batch, 5]."""
    query_timestamp = batch["last_timestamp"] + batch["query_s_sec"]
    return torch.cat(
        [elapsed_feature(batch["query_s_sec"]), clock_features(query_timestamp)], dim=1
    )


def anchor_features(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Anchor-clock features for the coherent arm's fixed base, [batch, 4].

    The base distribution may condition only on history/anchor metadata; every effect of
    waiting must come from survival conditioning of that fixed base, so ``query_s_sec``
    deliberately never enters here.
    """
    return clock_features(batch["last_timestamp"])


QUERY_FEATURE_DIM = ELAPSED_FEATURE_DIM + CLOCK_FEATURE_DIM
ANCHOR_FEATURE_DIM = CLOCK_FEATURE_DIM
