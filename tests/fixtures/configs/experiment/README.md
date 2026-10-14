# Experiment configs

> Since #101 (2026-09-25) training reads only `model_config`, `dataset_config` (its
> `processed_root`), `evaluation`, `seeds` and `output_dir` from these files;
> `manifest`, `specification_id`, `specification_status`, `frozen` and `device` are
> kept as historical declarations and are not gates. The `selection_manifest` slots
> were removed with the registration tooling. See `docs/TRAINING.md`.

Public experiment names should describe the comparison directly, for example `silence_conditioning`, `future_model`, `semantic_belief`, or `sessionization`.

The September matrix and owners are in
[TWO_WEEK_EXPERIMENT_PLAN.md](../../docs/TWO_WEEK_EXPERIMENT_PLAN.md). The research
specification is frozen: [#37](https://github.com/chengruchou/STILL/issues/37)
(`still-gowalla-core-v1-20260909`, closed 2026-09-09) as landed in
[CORE_RECIPE_V1.md](../../docs/CORE_RECIPE_V1.md). The seven core model configs carry
the v1 budget in their executed `training` block and the requested recipe in
`formal_recipe`; the seven core experiment configs declare `specification_id` and
`selection_manifest` (the #38 registration slot). The execution-contract flag
`frozen: false` is carried by the two Table 2 configs (and the #51 external one),
whose runner gates on it; the five Table 1 cells run through
`scripts/train_baseline.py`, which has no such gate, so they do not declare it. Do not add placeholder configs that look runnable, and
do not change a v1 value without a new recipe version.

Record training/selection budgets, feature masks, all seed roles, exact candidates,
cohort/profile identity, time units and weighting/evaluator versions. Baseline configs
belong to B; `adrec_*`, the `still_joint`/`still_independent` family and the
`external_*` configs belong to the lead. C records and executes frozen configurations
rather than modifying them in response to test results.

Fixture smoke configs (`still_smoke_synthetic.yaml`, `external_joint_lnm_smoke_synthetic.yaml`)
keep their bounded purpose; the external fixture model config
(`configs/model/external_joint_lnm.yaml`) is explicitly `recipe_status: fixture`, and
its deviations from the requested recipe are recorded in every smoke summary. The
Gowalla candidate for #51 is `configs/model/external_joint_lnm_gowalla_v1_proposed.yaml`
(`recipe_status: proposed`), outside the seven core recipes.

## Foursquare cities (#69)

`foursquare_{nyc,tky}_*` mirror the seven Gowalla core configs and the four original
floors, with each city's dataset config, source manifest, 24h selection manifests and
output directories. The core configs point at `configs/model/foursquare/`, which differs
from `configs/model/` only in `formal_recipe.specification_id`. They execute under
`still-foursquare-core-v1-20260910`, registered in `still.common.profiles` as
**proposed**, not frozen, so the formal gate refuses them until the lead records the
decision; they also declare `frozen: false` until the bounded fixture evidence exists.
The #82 silence-aware floors and the event-only control are Gowalla-only and have no
Foursquare configs. Campaign registries: `configs/campaign/foursquare_{nyc,tky}.yaml`.

## SASRec time-conditioning arms (#40)

`gowalla_sasrec_h_pm`, `gowalla_sasrec_clock` and `gowalla_sasrec_s` are matched
controls: they differ from `gowalla_sasrec_h` and from each other in `model_config` and
`output_dir` only, and their model configs differ only in which query-time features the
shared conditioning module may read. The training block, manifest, seed list and
evaluation block are asserted equal to `gowalla_sasrec_h.yaml` by
`tests/models/baselines/test_sasrec_time_controls.py`, so editing one arm's budget or
capacity in isolation fails the suite rather than silently unmatching the comparison.
Feature layout, masks and run-record fields: [`docs/SASREC_TIME_CONTROLS.md`](../../docs/SASREC_TIME_CONTROLS.md).

The optional Elapsed-only arm (`sasrec_elapsed`) is implemented but deliberately has no
config here: enabling it is a protocol decision under #37, made before test inspection.
