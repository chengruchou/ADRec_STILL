# Migration

What the `adrec_still` package was ported from, where each piece landed, what
was deliberately left behind, and every behavior change made against the two
source codebases. Each change below is stated as
*source behavior → ported behavior* and is observable in the named module.

## Sources

1. **STILL donor repository** — https://github.com/chengruchou/STILL.git at
   commit `7ccf05a16615af561a711b92cffe5b661e6067aa`. The local checkout at
   `D:\STILL` is reference-only.
2. **Upstream ADRec KDD'25 fork** — this repository's `src/` tree as of base
   commit `0a30ef764cd85efa44f884a387198e9bf28c7d4c` (the fork of the official
   ADRec implementation this repo started from). `src/` itself is unchanged;
   the ported, fixed cores live under `adrec_still/models/upstream/`.

## Module map (donor → package)

| donor module | ported to |
| --- | --- |
| `still.data` | `adrec_still.data` |
| `still.evaluation` | `adrec_still.evaluation` |
| `still.models.baselines` | `adrec_still.models.baselines` |
| `still.models.still` | `adrec_still.models.still` |
| `still.models.external` | `adrec_still.models.external` |
| `still.train` | `adrec_still.training` |
| `still.models.still.distributions` | `adrec_still.time.distributions` |
| `still.evaluation.time` | `adrec_still.time.protocol` |

## Deliberately not ported

- The **campaign/cohort/registration/certification** layers (`still.campaign`
  and the registration-gated experiment machinery). This package runs plain
  CLI experiments; provenance is recorded per run in `config.json` /
  `summary.json` instead of a central registry.
- **CRPS quadrature and its certification.** Time quality is reported as
  observed density NLL and censored survival NLL with separate masks and
  counts (`adrec_still.time.protocol`,
  `RESIDUAL_TIME_ADAPTER_VERSION = "adrec-still-residual-time-adapter-v1"`).
  The donor's quadrature/certification machinery stays in the STILL
  repository.
- The donor's **diagnostics reporting** layer.

## Upstream fork port (`src/` → `adrec_still/models/upstream/`)

Ported cores: the ADRec diffusion core (`src/adrec.py` → `upstream/adrec.py`),
DiffuRec and DreamRec (`upstream/diffurec.py`, `upstream/dreamrec.py`), the
SASRec+ variant (`upstream/sasrec.py`), the transformer blocks
(`src/common.py` → `upstream/transformer.py`), the beta schedules and timestep
samplers (`src/step_sample.py` → `upstream/schedules.py`), and the PCGrad
wrapper (`src/pcgrad.py` → `upstream/pcgrad.py`). The fork's
`Att_Diffuse_model` level (embedding table, training objectives, scoring) is
re-expressed as the shared-interface wrappers in `upstream/wrappers.py`, and
the fork's training loop as the staged trainer in
`adrec_still/training/stages.py`.

Preserved upstream mechanics: the clean causal history encoder,
token-independent noising with per-token timesteps, direct x0 prediction, the
`rep_item + lambda_uncertainty * (x_t + time_emb)` fusion, DDPM posterior
sampling, inference that regenerates only the last target while
teacher-forcing the clean prefix each reverse step, the CE + scaled-MSE
objective merged through the PCGrad wrapper (projection off by default, as
upstream), and the three-stage pretrain/warmup/finetune recipe.

Also preserved, deliberately: DreamRec's **train/inference conditioning
mismatch**. The fork's `DreamRec.forward` encodes the history with
`ag_encoder` during training, but its `denoise_sample` conditions the reverse
loop on the raw normalized embeddings — the encoder never runs at inference.
The port reproduces that shipped behavior rather than repairing it; a repaired
variant would be a differently-named experiment.

### Execution fixes (source behavior → ported behavior)

(a) **Attention padding mask** — the fork built the padding mask as
`view(B, 1, L, 1)`, a *query*-side mask, so padded key positions stayed
attendable and left-padding leaked the padding embedding into real queries →
the port masks keys (`[B, 1, 1, L]`), zeroes padded-position outputs after the
final block, and uses a finite `-1e9` floor so an all-masked row's softmax
stays finite (`upstream/transformer.py`).

(b) **One-event evaluation prefixes** — the fork's Val/Test datasets built the
teacher-forcing prefix as `seq[-(len(hist)-1):]`, which is `seq[-0:]` — the
whole list — when a history holds a single event → the shifted prefix is
`hist[1:]`, empty in that case (`adrec_still/data/sequences.py`).

(c) **Item vocabulary** — `item_num_create` hardcoded six dataset sizes
(`src/trainer.py`) → the vocabulary is derived from the data on load; pkl item
IDs are shifted by +1 so PAD=0 and UNK=1 stay reserved
(`adrec_still/data/sequences.py`).

(d) **Paths** — the fork read `../datasets/data/<name>/dataset.pkl` relative
to the working directory and loaded `saved/pretrain/<dataset>/pretrain.pth`
→ data comes from an explicit `--data`; the pretrained embedding is handed
over in memory between stages (no file needed), and reusing a fork
`pretrain.pth` requires an explicit `pretrained_embedding_path` whose table is
remapped by raw ID into this package's vocabulary with shape checks — a table
sized for a different item universe is refused
(`adrec_still/training/stages.py::load_upstream_pretrain_pth`).

(e) **Warm-up budget** — the fork unfroze the embedding at the hardcoded
`epoch_temp == 5` → every stage budget is a plain setting
(`pretrain_epochs` / `warmup_epochs` / `max_epochs`), each adjustable down to
one epoch (`adrec_still/training/stages.py::stage_schedule`).

(f) **Selection crash** — the fork initialized `best_model = None`, evaluated
only every `eval_interval` epochs (never at epoch 0), and saved/evaluated
`best_model` unconditionally — a run where no evaluation improved, or with
fewer epochs than the interval, crashed → the first valid selection
establishes the best; the final evaluation loads the best state and runs in
eval mode; a validation pass that returns no value for the monitor is an
explicit error (`adrec_still/training/stages.py`).

(g) **Metric aggregation** — fork evaluation averaged per-batch means without
weighting, so a ragged final batch skewed results → every metric is
aggregated count-weighted (sums divided by row counts) in the package
evaluators (`adrec_still/evaluation/sequence_eval.py`,
`adrec_still/evaluation/silence_eval.py`).

(h) **Frozen embeddings during warmup** — the fork's PCGrad wrapper set
gradients on all parameters and the single Adam optimizer's weight decay kept
shrinking the frozen embedding table during warmup → each stage builds its
optimizer over the parameters trainable *in that stage*, so a frozen table is
genuinely untouched (`adrec_still/training/stages.py::_stage_optimizer`).

(i) **Shortened reverse schedules** — a shortened schedule is built by
re-spacing `alphas_cumprod` over the retained original timesteps, with a
timestep map so the timestep embedding sees the retained original step index —
never by simply dropping the last steps
(`adrec_still/models/upstream/adrec.py`, `schedules.py::space_timesteps`).

(j) **Evaluation noise** — reverse-process noise at evaluation is seeded from
a fixed `eval_noise_seed` combined with a content hash of the batch's
history, so a fixed seed plus an unchanged history reproduces the forecast
regardless of batch composition, and candidate chunks never redraw noise
(`adrec_still/models/upstream/wrappers.py`,
`adrec_still/models/adrec_still/model.py::_forecast_generator`).

(k) **Lognormal censored survival numerics** *(donor-port fix:
`still.evaluation.time` → `adrec_still.time.protocol`)* — the donor computed
censored survival as `0.5 * erfc(z / sqrt(2))` clamped at `1e-15` before the
log → the port computes `torch.special.log_ndtr(-z)` directly in log space
(also used throughout `adrec_still.time.distributions`), so deep-tail
censoring stays finite without a clamp.

(l) **Dependencies** — the fork hard-imports `einops` (`src/common.py`,
`src/model.py`) and `scipy` (`src/step_sample.py`), and pins `torchtune` in
its requirements → the port uses neither `einops` nor `torchtune`, and the
`scipy` import is deferred into the one non-default `beta` noise schedule that
needs it (`adrec_still/models/upstream/schedules.py`).

## Timezone policy

- **Gowalla**: raw timestamps are UTC and are documented and stored as such
  (`event_time_utc_s`, UTC seconds).
- **Foursquare** (TSMC2014 subset and TIST2015 global): the canonical event
  time is parsed from the file's own UTC timestamp field (the fixed English
  format with its explicit offset). The separate timezone-offset column is
  **audited only** — invalid offsets are counted in the cleaning audit
  (`invalid_timezone_rows`) but never gate a row's validity and never shift a
  timestamp. Nothing in the package claims local wall-clock semantics for any
  dataset.
