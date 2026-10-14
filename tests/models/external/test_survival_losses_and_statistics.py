"""External LNM: #39 survival losses and visible-view statistics (#51).

Every loss term is checked against an independent hand computation on the same
parameters, and the statistics fit is checked against the leakage and
de-duplication rules the #39 view defines. (The donor also referenced converged
CRPS checks, which live in its unported test_marginal_crps.py: CRPS was removed
from this port.) Fixture-scale evidence for the adapter's correctness — no
Gowalla data, no training result.
"""

import json
import math
from pathlib import Path

import pytest
import torch

from adrec_still.data.survival import SURVIVAL_PROFILE_VERSION, build_survival_training_profile
from adrec_still.models.external.joint_lnm import FIRST_LEGAL_MARK_ID, ExternalJointLNM
from adrec_still.models.external.statistics import (
    CANONICAL_TRAIN_POPULATION,
    REQUIRED_FORMAL_POPULATION,
    fit_log_time_statistics,
    fit_visible_view_log_time_statistics,
)
from adrec_still.models.still.losses import survival_training_masks

LANDMARKS = (0, 6 * 3600, 24 * 3600)


def build_model(number_of_pois: int = 6) -> ExternalJointLNM:
    torch.manual_seed(11)
    model = ExternalJointLNM(
        number_of_pois, mark_embedding_dim=4, context_dim=6, mixture_components=2
    )
    model.is_fitted = True
    model.eval()
    return model


def survival_batch(number_of_pois: int = 6) -> dict[str, torch.Tensor]:
    """One row of each #39 contribution: known, unknown-mark, positive-followup."""
    history = torch.tensor([[2, 3], [3, 2], [2, 2]], dtype=torch.long)
    return {
        "history_poi": history,
        "history_timestamp": torch.tensor(
            [[0, 3600], [0, 7200], [0, 1800]], dtype=torch.long
        ),
        "history_mask": torch.ones(3, 2, dtype=torch.bool),
        "last_timestamp": torch.tensor([7200, 10800, 3600], dtype=torch.long),
        "query_s_sec": torch.tensor([3600, 7200, 1800], dtype=torch.long),
        "next_poi": torch.tensor([4, 1, 0], dtype=torch.long),
        "residual_time_sec": torch.tensor([1800.0, 5400.0, 0.0], dtype=torch.float64),
        "censor_time_sec": torch.tensor([0.0, 0.0, 2700.0], dtype=torch.float64),
        "observed_mask": torch.tensor([True, True, False]),
        "target_in_train_catalog": torch.tensor([True, False, False]),
    }


def test_the_three_survival_terms_match_an_independent_hand_computation():
    model = build_model()
    batch = survival_batch()
    masks = survival_training_masks(batch)
    assert [int(mask.sum()) for mask in masks.values()] == [1, 1, 1]

    generator = torch.Generator().manual_seed(41)
    losses = model.per_row_survival_training_losses(batch, 3, generator)
    assert losses.shape == (3,)
    assert torch.isfinite(losses).all()

    # Independent reference from the same base parameters.
    with torch.no_grad():
        base = model.base_distribution(model.encode(batch))
    elapsed = batch["query_s_sec"].to(torch.float64)
    residual = batch["residual_time_sec"]
    censor = batch["censor_time_sec"]
    log_s_query = base.log_survival(elapsed)

    # Row 0 (observed known): marginal time NLL + sampled mark surrogate.
    total = elapsed + residual
    time_nll = -(base.log_marginal_density(total) - log_s_query)
    # Row 1 (observed unknown): time term only, no mark contribution.
    assert losses[1].item() == pytest.approx(time_nll[1].item(), rel=1e-12)
    # Row 2 (censored): -(log S(s + c) - log S(s)) over the unknown next mark.
    censor_nll = -(base.log_survival(elapsed + censor) - log_s_query)
    assert losses[2].item() == pytest.approx(censor_nll[2].item(), rel=1e-12)
    # Row 0 keeps the time term plus a strictly positive sampled-mark surrogate.
    assert losses[0].item() > time_nll[0].item()

    generator = torch.Generator().manual_seed(41)
    from adrec_still.models.baselines.trainer import build_sampled_candidates

    candidates, penalty = build_sampled_candidates(
        batch["next_poi"][masks["observed_known"]], 3, model.number_of_pois, generator
    )
    weighted = (
        base.log_mark_probabilities[masks["observed_known"]]
        + base.per_mark_log_density(total)[masks["observed_known"]]
    )
    logits = weighted.gather(1, candidates - FIRST_LEGAL_MARK_ID) - penalty.to(torch.float64)
    mark_nll = -torch.log_softmax(logits, dim=1)[:, 0]
    assert losses[0].item() == pytest.approx(
        (time_nll[masks["observed_known"]] + mark_nll)[0].item(), rel=1e-12
    )


def test_the_censor_term_marginalizes_the_unknown_mark_not_the_last_observed_one():
    """Upstream's end-of-sequence term conditions on the last observed mark; this
    population's likelihood does not, and the two differ on an asymmetric fixture."""
    model = build_model()
    batch = survival_batch()
    generator = torch.Generator().manual_seed(7)
    losses = model.per_row_survival_training_losses(batch, 2, generator)
    with torch.no_grad():
        base = model.base_distribution(model.encode(batch))
    elapsed = batch["query_s_sec"].to(torch.float64)
    exposure = elapsed + batch["censor_time_sec"]
    marginal = -(base.log_survival(exposure) - base.log_survival(elapsed))
    last_mark = batch["history_poi"][:, -1]
    per_mark = base.per_mark_log_survival(exposure) - base.per_mark_log_survival(elapsed)
    last_mark_term = -per_mark.gather(
        1, (last_mark - FIRST_LEGAL_MARK_ID).clamp_min(0).unsqueeze(1)
    ).squeeze(1)
    assert losses[2].item() == pytest.approx(marginal[2].item(), rel=1e-12)
    assert abs(losses[2].item() - last_mark_term[2].item()) > 1e-6


def test_the_survival_losses_produce_finite_gradients_and_move_the_parameters():
    model = build_model()
    model.train()
    batch = survival_batch()
    generator = torch.Generator().manual_seed(5)
    losses = model.per_row_survival_training_losses(batch, 3, generator)
    loss = losses.mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "the survival loss must reach the parameters"
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(bool((g != 0).any()) for g in grads)

    before = model.mark_head.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.step()
    assert not torch.allclose(before, model.mark_head.weight.detach())


def test_an_unknown_mark_row_never_supplies_a_mark_target():
    """UNKNOWN is a reserved id: the unknown-mark row contributes time only, and its
    id is never handed to the candidate builder."""
    model = build_model()
    batch = survival_batch()
    seen: list[torch.Tensor] = []
    import adrec_still.models.baselines.trainer as trainer_module

    real_builder = trainer_module.build_sampled_candidates

    def recording(targets, negatives, number_of_pois, generator):
        seen.append(targets.clone())
        return real_builder(targets, negatives, number_of_pois, generator)

    trainer_module.build_sampled_candidates = recording
    try:
        model.per_row_survival_training_losses(batch, 2, torch.Generator().manual_seed(3))
    finally:
        trainer_module.build_sampled_candidates = real_builder
    assert len(seen) == 1
    assert seen[0].tolist() == [4], "only the observed known target enters the sampler"


def write_raw(path: Path, users: int = 6, events: int = 6, shift_hours: int = 0) -> Path:
    from datetime import datetime, timedelta, timezone

    start = datetime(2021, 1, 1, tzinfo=timezone.utc) + timedelta(hours=shift_hours)
    gaps = (5, 30, 7, 50, 9)
    lines = []
    for user in range(users):
        moment = start + timedelta(hours=user)
        for index in range(events):
            lines.append(
                "\t".join(
                    [
                        f"u{user}",
                        moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        f"{25.0 + user * 0.1:.4f}",
                        f"{121.0 + index * 0.1:.4f}",
                        f"p{index % 4:03d}",
                    ]
                )
            )
            moment += timedelta(hours=gaps[index % len(gaps)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_view(tmp_path: Path, name: str = "view", **kwargs) -> tuple[Path, dict]:
    """A real #39 training view from the committed preprocessing pipeline."""
    from adrec_still.data.pipeline import run_preprocessing

    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    raw = write_raw(root / "loc-gowalla_totalCheckins.txt", **kwargs)
    paths = run_preprocessing(
        config_path=Path(__file__).parents[2] / "fixtures" / "configs" / "dataset" / "gowalla.yaml",
        input_path=raw,
        output_dir=root / "processed",
        manifest_path=root / "manifest.json",
        audit_path=root / "audit.json",
    )
    manifest = build_survival_training_profile(
        paths["events"],
        paths["split_manifest"],
        root / "survival_training.parquet",
        root / "survival_training_manifest.json",
        landmark_seconds=LANDMARKS,
        max_history_length=50,
    )
    return root / "survival_training.parquet", manifest


def test_the_visible_view_fit_reads_the_view_and_records_its_identity(tmp_path):
    view_path, manifest = build_view(tmp_path)
    statistics = fit_visible_view_log_time_statistics(view_path, manifest)
    assert statistics.population == REQUIRED_FORMAL_POPULATION
    assert statistics.source == "survival-view-visible-unique-gaps"
    assert statistics.view_profile_id == manifest["profile_id"]
    assert statistics.view_landmarks_sha256 == manifest["landmarks_sha256"]
    assert statistics.unique_gap_count == manifest["completed_gap_count"]
    record = statistics.as_record()
    assert record["population"] == REQUIRED_FORMAL_POPULATION
    assert record["view_profile_id"] == manifest["profile_id"]
    assert statistics.status == "fitted"

    # Pinned against an independent recomputation from the view file itself, using the
    # standard library's population mean/stdev rather than the module's own arithmetic:
    # an offset added to the mean, or a variance that forgot its square root, changes
    # these numbers and would be caught here.
    import statistics as stdlib_statistics

    import pyarrow.parquet as pq

    totals: dict[tuple[int, int], float] = {}
    for row in pq.read_table(view_path).to_pylist():
        if not row["event_observed"]:
            continue
        total = float(row["query_s_sec"]) + float(row["residual_time_sec"])
        if total > 0:
            totals[(int(row["user_id"]), int(row["last_timestamp"]))] = total
    logs = [math.log(value) for value in totals.values()]
    assert statistics.unique_gap_count == len(logs)
    assert statistics.mean == pytest.approx(stdlib_statistics.fmean(logs), rel=1e-12)
    assert statistics.std == pytest.approx(stdlib_statistics.pstdev(logs), rel=1e-12)
    assert statistics.std > 0 and math.isfinite(statistics.mean)

    # The view this fit reads carries only the disclosed training prefix: every row is
    # a train row anchored strictly before the cutoff, so no validation/test row and no
    # post-cutoff event is reachable from here.
    rows = pq.read_table(view_path).to_pylist()
    assert {row["split"] for row in rows} == {"train"}
    assert all(row["last_timestamp"] < manifest["cutoff_time_sec"] for row in rows)


def test_one_contribution_per_underlying_gap_whatever_the_landmark_schedule(tmp_path):
    """More query landmarks of the same gap must not reweight the estimate."""
    sparse_path, sparse = build_view(tmp_path, name="sparse")
    dense_root = tmp_path / "dense"
    dense_root.mkdir()
    from adrec_still.data.pipeline import run_preprocessing

    raw = write_raw(dense_root / "loc-gowalla_totalCheckins.txt")
    paths = run_preprocessing(
        config_path=Path(__file__).parents[2] / "fixtures" / "configs" / "dataset" / "gowalla.yaml",
        input_path=raw,
        output_dir=dense_root / "processed",
        manifest_path=dense_root / "manifest.json",
        audit_path=dense_root / "audit.json",
    )
    dense = build_survival_training_profile(
        paths["events"],
        paths["split_manifest"],
        dense_root / "survival_training.parquet",
        dense_root / "survival_training_manifest.json",
        landmark_seconds=(0, 3600, 6 * 3600, 12 * 3600, 24 * 3600),
        max_history_length=50,
    )
    assert dense["row_count"] > sparse["row_count"], "the dense schedule must add rows"
    first = fit_visible_view_log_time_statistics(sparse_path, sparse)
    second = fit_visible_view_log_time_statistics(
        dense_root / "survival_training.parquet", dense
    )
    assert first.unique_gap_count == second.unique_gap_count
    assert first.mean == pytest.approx(second.mean, rel=1e-12)
    assert first.std == pytest.approx(second.std, rel=1e-12)


def test_a_view_that_does_not_match_its_manifest_is_refused(tmp_path):
    view_path, manifest = build_view(tmp_path)
    fit_visible_view_log_time_statistics(view_path, manifest)  # sanity: accepted

    with pytest.raises(ValueError, match="profile_id"):
        fit_visible_view_log_time_statistics(
            view_path, manifest, expected_profile_id="another-view"
        )
    with pytest.raises(ValueError, match="version"):
        fit_visible_view_log_time_statistics(
            view_path, {**manifest, "version": "still-train-admin-censor-v0"}
        )
    with pytest.raises(ValueError, match="does not match its manifest"):
        fit_visible_view_log_time_statistics(
            view_path, {**manifest, "landmarks_sha256": "0" * 64}
        )
    with pytest.raises(ValueError, match="row count"):
        fit_visible_view_log_time_statistics(
            view_path, {**manifest, "row_count": manifest["row_count"] + 1}
        )
    with pytest.raises(ValueError, match="observed rows"):
        fit_visible_view_log_time_statistics(
            view_path,
            {**manifest, "observed_row_count": manifest["observed_row_count"] + 1},
        )

    # The split is verified, not inherited from trust in the view builder: a file
    # carrying a validation row is refused even when every hash and count still agrees.
    import pyarrow as pa
    import pyarrow.parquet as pq

    from adrec_still.data.audit import sha256_file

    table = pq.read_table(view_path)
    splits = table.column("split").to_pylist()
    splits[0] = "validation"
    tampered_path = view_path.parent / "tampered_split.parquet"
    pq.write_table(table.set_column(
        table.schema.get_field_index("split"), "split", pa.array(splits)
    ), tampered_path)
    tampered_manifest = {
        **manifest,
        "landmarks_sha256": sha256_file(tampered_path),
    }
    with pytest.raises(ValueError, match="outside the training split"):
        fit_visible_view_log_time_statistics(tampered_path, tampered_manifest)
    other_path, _ = build_view(tmp_path, name="other", shift_hours=3)
    with pytest.raises(ValueError, match="does not match its manifest"):
        fit_visible_view_log_time_statistics(other_path, manifest)


def test_the_visible_fit_moves_only_with_the_visible_population(tmp_path):
    """The fit is a function of the visible view and nothing else.

    Positive control: a genuinely larger visible population changes the unique-gap
    count and the fitted values. Negative control: rebuilding the identical raw prefix
    reproduces them exactly, so any difference above is attributable to the visible
    rows rather than to run-to-run noise.
    """
    base_path, base = build_view(tmp_path, name="base")
    baseline = fit_visible_view_log_time_statistics(base_path, base)
    longer_path, longer = build_view(tmp_path, name="longer", events=8)
    extended = fit_visible_view_log_time_statistics(longer_path, longer)
    # Strictly more visible completed gaps, a later cutoff, and different statistics.
    assert extended.unique_gap_count > baseline.unique_gap_count
    assert longer["cutoff_time_sec"] > base["cutoff_time_sec"]
    assert extended.mean != pytest.approx(baseline.mean, rel=1e-9)

    repeat_path, repeat = build_view(tmp_path, name="repeat")
    again = fit_visible_view_log_time_statistics(repeat_path, repeat)
    assert again.mean == pytest.approx(baseline.mean, rel=1e-12)
    assert again.std == pytest.approx(baseline.std, rel=1e-12)
    assert again.unique_gap_count == baseline.unique_gap_count
    assert again.view_profile_id == baseline.view_profile_id


def test_the_canonical_and_fixture_fits_keep_their_own_population(tmp_path):
    """The older fits are not relabelled by the new one."""
    from adrec_still.data.pipeline import run_preprocessing

    root = tmp_path / "canonical"
    root.mkdir()
    raw = write_raw(root / "loc-gowalla_totalCheckins.txt")
    paths = run_preprocessing(
        config_path=Path(__file__).parents[2] / "fixtures" / "configs" / "dataset" / "gowalla.yaml",
        input_path=raw,
        output_dir=root / "processed",
        manifest_path=root / "manifest.json",
        audit_path=root / "audit.json",
    )
    canonical = fit_log_time_statistics(paths["landmarks"])
    assert canonical.population == CANONICAL_TRAIN_POPULATION
    assert canonical.view_profile_id is None and canonical.view_landmarks_sha256 is None


def test_checkpoint_reload_keeps_the_statistics_and_their_provenance(tmp_path):
    from adrec_still.models.external.registry import load_external_checkpoint
    from adrec_still.training.checkpoint import save_weights

    view_path, manifest = build_view(tmp_path, name="ckpt")
    statistics = fit_visible_view_log_time_statistics(view_path, manifest)
    model = build_model()
    model.set_log_time_statistics(statistics)
    checkpoint = tmp_path / "external.pt"
    save_weights(checkpoint, model, family="external")
    reloaded = load_external_checkpoint(checkpoint)
    record = reloaded.log_time_statistics_record
    assert record["population"] == REQUIRED_FORMAL_POPULATION
    assert record["view_profile_id"] == manifest["profile_id"]
    assert record["view_landmarks_sha256"] == manifest["landmarks_sha256"]
    assert record["unique_gap_count"] == statistics.unique_gap_count
    assert record["log_time_mean"] == pytest.approx(statistics.mean, rel=1e-12)
    assert record["log_time_std"] == pytest.approx(statistics.std, rel=1e-12)
    # The manifest on disk is unchanged by any of this.
    stored = json.loads(
        (view_path.parent / "survival_training_manifest.json").read_text(encoding="utf-8")
    )
    assert stored["profile_id"] == manifest["profile_id"]
    assert stored["version"] == SURVIVAL_PROFILE_VERSION
