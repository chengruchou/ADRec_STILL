"""Log-inter-event-time affine statistics for the external LNM baseline (#51).

Official implementation (joint_tpp / Shchur lineage): both the RNN input feature
``(log(dt + 1e-8) - mean) / std`` and the output distribution's affine transform
``log T = mean + std * Z`` use the dataset's log-inter-event-time statistics. The
STILL port fits them **only** from legal training data: observed rows with a
positive total gap, deduplicated to one contribution per underlying user gap so
that multiple query landmarks of the same gap cannot reweight the estimate, and
never touching validation/test splits. The fixture smoke uses explicit identity
statistics (mean 0, std 1) — stated, not silently defaulted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

#: Degenerate-spread guard: below this the fitted std is unusable and the
#: statistics fall back to identity with an explicit status.
MINIMUM_STD = 1.0e-6

#: The population the executable fit reads today: observed completed gaps of the
#: canonical v0.1 train split, one contribution per unique gap. It is the interim
#: engineering population, not the #39 ``still-train-admin-censor-v1`` visible
#: training view the frozen specification requires for a formal run (#37 §4.2).
CANONICAL_TRAIN_POPULATION = "canonical-train-observed-completed-gaps-v0.1"
#: The population a formal real-data run must fit on: the visible training rows of
#: the accepted #39 view (``adrec_still.data.survival.build_survival_training_profile``).
#: Produced by :func:`fit_visible_view_log_time_statistics`, which reads that view
#: file only after checking it against the view manifest.
REQUIRED_FORMAL_POPULATION = "still-train-admin-censor-v1-visible-training-view"
#: Recorded for a pre-population-field checkpoint whose source does not identify
#: the fit: an explicit unknown, never a guessed named population.
LEGACY_UNKNOWN_POPULATION = "unknown-legacy-checkpoint-population"

#: The statuses ``fit_log_time_statistics`` can produce. Both fallbacks read the
#: same canonical training rows before falling back to identity parameters, so
#: they still identify the canonical population.
_FIT_STATUSES = frozenset(
    {"fitted", "fallback_identity_insufficient_gaps", "fallback_identity_degenerate_std"}
)


def legacy_population(source: str, status: str) -> str:
    """Conservatively infer the population of a record that predates the field.

    Historical provenance is never upgraded to a named population unless the
    record's own ``(source, status)`` pair — as this module actually writes it —
    establishes it: a complete identity declaration keeps the fixture population,
    a complete executable-fit record maps to the canonical population, and
    everything else, including an internally inconsistent pair this codebase
    could not have written, records :data:`LEGACY_UNKNOWN_POPULATION`.
    """
    if source == IDENTITY_STATISTICS.source and status == IDENTITY_STATISTICS.status:
        return IDENTITY_STATISTICS.population
    if source == "train-unique-gaps" and status in _FIT_STATUSES:
        return CANONICAL_TRAIN_POPULATION
    return LEGACY_UNKNOWN_POPULATION


@dataclass(frozen=True)
class LogTimeStatistics:
    """Affine statistics ``a``/``b`` with their fit provenance.

    ``population`` names the rows the fit actually read; a fit on the canonical
    completed-gap train split is never relabeled as the #39 view.
    """

    mean: float
    std: float
    source: str
    unique_gap_count: int
    status: str
    population: str = CANONICAL_TRAIN_POPULATION
    #: Identity of the training view these values were fitted on, when the fit read
    #: one: the #39 manifest's ``profile_id`` and the ``landmarks_sha256`` of the
    #: exact file that was read. ``None`` for the canonical and fixture fits, which
    #: have no view manifest -- never a copied or assumed identity.
    view_profile_id: str | None = None
    view_landmarks_sha256: str | None = None

    def as_record(self) -> dict[str, object]:
        return {
            "log_time_mean": self.mean,
            "log_time_std": self.std,
            "source": self.source,
            "unique_gap_count": self.unique_gap_count,
            "status": self.status,
            "population": self.population,
            "view_profile_id": self.view_profile_id,
            "view_landmarks_sha256": self.view_landmarks_sha256,
        }


IDENTITY_STATISTICS = LogTimeStatistics(
    mean=0.0,
    std=1.0,
    source="identity",
    unique_gap_count=0,
    status="identity_declared",
    population="identity-declared-fixture",
)


def fit_log_time_statistics(
    landmark_path: str | Path, split: str = "train"
) -> LogTimeStatistics:
    """Fit ``mean``/``std`` of ``log(total gap seconds)`` on unique training gaps.

    Population: rows of ``split`` with ``event_observed`` and a positive total gap
    ``query_s_sec + residual_time_sec``, deduplicated on ``(user_id,
    last_timestamp)`` — every query landmark of one gap sees the same underlying
    total gap, so one gap contributes exactly once. Unknown-target observed rows
    still carry a legal gap time and are included; censored rows are not (their
    residual is not observed). Fewer than two unique gaps or a degenerate spread
    falls back to identity with an explicit status instead of a fabricated scale.
    """
    if split != "train":
        raise ValueError(
            "log-time statistics may only be fitted on the training split; "
            f"got {split!r}"
        )
    table = pq.read_table(
        landmark_path,
        columns=[
            "split",
            "user_id",
            "last_timestamp",
            "query_s_sec",
            "event_observed",
            "residual_time_sec",
        ],
    )
    rows = table.to_pylist()
    gaps: dict[tuple[int, int], float] = {}
    for row in rows:
        if row["split"] != split or not row["event_observed"]:
            continue
        total = float(row["query_s_sec"]) + float(row["residual_time_sec"])
        if total <= 0:
            continue
        key = (int(row["user_id"]), int(row["last_timestamp"]))
        existing = gaps.get(key)
        if existing is not None and existing != total:
            raise ValueError(
                "inconsistent total gap for one (user, anchor): the canonical "
                "landmark table should give every query of a gap the same "
                f"query_s + residual; key={key} saw {existing} and {total}"
            )
        gaps[key] = total
    count = len(gaps)
    if count < 2:
        return LogTimeStatistics(
            mean=0.0,
            std=1.0,
            source=f"{split}-unique-gaps",
            unique_gap_count=count,
            status="fallback_identity_insufficient_gaps",
        )
    logs = [math.log(total) for total in gaps.values()]
    mean = sum(logs) / count
    variance = sum((value - mean) ** 2 for value in logs) / count
    std = math.sqrt(variance)
    if std < MINIMUM_STD:
        return LogTimeStatistics(
            mean=0.0,
            std=1.0,
            source=f"{split}-unique-gaps",
            unique_gap_count=count,
            status="fallback_identity_degenerate_std",
        )
    return LogTimeStatistics(
        mean=mean,
        std=std,
        source=f"{split}-unique-gaps",
        unique_gap_count=count,
        status="fitted",
    )


def fit_visible_view_log_time_statistics(
    view_path: str | Path,
    manifest: Mapping[str, object],
    *,
    expected_profile_id: str | None = None,
) -> LogTimeStatistics:
    """Fit ``mean``/``std`` of ``log(total gap seconds)`` on the #39 visible view.

    The population is the view's own **observed** rows with a positive total gap
    ``query_s_sec + residual_time_sec``, deduplicated on ``(user_id,
    last_timestamp)`` so every query landmark of one underlying completed gap
    contributes exactly once. Observed rows whose mark falls outside the frozen
    catalog are legal gap times and are included; censored rows are excluded,
    because an exposure lower bound is not an observed duration and treating it as
    one would shrink the fitted scale.

    The view is verified before it is read, not assumed: the manifest must declare
    this tree's ``SURVIVAL_PROFILE_VERSION``, the file's SHA256 must equal the
    manifest's ``landmarks_sha256``, and the row counts actually read must match the
    manifest's. ``expected_profile_id`` is for a caller holding an identity from
    somewhere other than this manifest -- a caller that verifies the manifest it just
    built passes none, because comparing that manifest with itself would prove nothing;
    the file-versus-manifest checks above are what verify such a view.

    No older fit is relabelled: the canonical and fixture fits keep their own
    populations, and this function only ever produces
    :data:`REQUIRED_FORMAL_POPULATION`. Validation/test rows and post-cutoff events
    are unreachable here: the view file contains only the disclosed training prefix,
    and this fit reads nothing else.
    """
    from adrec_still.data.audit import sha256_file
    from adrec_still.data.survival import SURVIVAL_PROFILE_VERSION

    path = Path(view_path)
    declared_version = manifest.get("version")
    if declared_version != SURVIVAL_PROFILE_VERSION:
        raise ValueError(
            f"survival view manifest declares version {declared_version!r}; this tree "
            f"implements {SURVIVAL_PROFILE_VERSION!r}"
        )
    if expected_profile_id is not None and manifest.get("profile_id") != expected_profile_id:
        raise ValueError(
            f"survival view manifest profile_id {manifest.get('profile_id')!r} is not "
            f"the view this run built ({expected_profile_id!r})"
        )
    actual_sha = sha256_file(path)
    if actual_sha != manifest.get("landmarks_sha256"):
        raise ValueError(
            f"survival view file {path} does not match its manifest: its bytes hash to "
            f"{actual_sha}, the manifest records {manifest.get('landmarks_sha256')!r}"
        )
    rows = pq.read_table(
        path,
        columns=[
            "split",
            "user_id",
            "last_timestamp",
            "query_s_sec",
            "event_observed",
            "residual_time_sec",
        ],
    ).to_pylist()
    if len(rows) != manifest.get("row_count"):
        raise ValueError(
            f"survival view row count {len(rows)} differs from the manifest's "
            f"{manifest.get('row_count')!r}"
        )
    # The projection reads `split`, so it is checked rather than assumed: every other
    # guard here verifies the bytes actually read, and the population claim should not
    # be the one thing inherited from trust in the producer.
    foreign_splits = sorted({str(row["split"]) for row in rows} - {"train"})
    if foreign_splits:
        raise ValueError(
            f"survival view {path} contains rows outside the training split "
            f"({', '.join(foreign_splits)}); the visible-view fit reads training rows "
            "only"
        )
    observed_rows = sum(1 for row in rows if row["event_observed"])
    if observed_rows != manifest.get("observed_row_count"):
        raise ValueError(
            f"survival view observed rows {observed_rows} differ from the manifest's "
            f"{manifest.get('observed_row_count')!r}"
        )
    gaps: dict[tuple[int, int], float] = {}
    for row in rows:
        if not row["event_observed"]:
            continue
        total = float(row["query_s_sec"]) + float(row["residual_time_sec"])
        if total <= 0:
            continue
        key = (int(row["user_id"]), int(row["last_timestamp"]))
        existing = gaps.get(key)
        if existing is not None and existing != total:
            raise ValueError(
                "inconsistent total gap for one (user, anchor) in the survival view: "
                f"key={key} saw {existing} and {total}"
            )
        gaps[key] = total
    identity = {
        "view_profile_id": manifest.get("profile_id"),
        "view_landmarks_sha256": actual_sha,
    }
    count = len(gaps)
    source = "survival-view-visible-unique-gaps"
    if count < 2:
        return LogTimeStatistics(
            mean=0.0,
            std=1.0,
            source=source,
            unique_gap_count=count,
            status="fallback_identity_insufficient_gaps",
            population=REQUIRED_FORMAL_POPULATION,
            **identity,
        )
    logs = [math.log(total) for total in gaps.values()]
    mean = sum(logs) / count
    variance = sum((value - mean) ** 2 for value in logs) / count
    std = math.sqrt(variance)
    if std < MINIMUM_STD:
        return LogTimeStatistics(
            mean=0.0,
            std=1.0,
            source=source,
            unique_gap_count=count,
            status="fallback_identity_degenerate_std",
            population=REQUIRED_FORMAL_POPULATION,
            **identity,
        )
    return LogTimeStatistics(
        mean=mean,
        std=std,
        source=source,
        unique_gap_count=count,
        status="fitted",
        population=REQUIRED_FORMAL_POPULATION,
        **identity,
    )
