# Training

How to install, feed, train, evaluate, and collect runs of the `adrec_still`
package with `python -m adrec_still`. The package lives beside the upstream
ADRec fork (`src/`, `datasets/`), which keeps its own entry point
(`src/main.py`); everything below concerns the package CLI only.

Commands in this guide are labeled:

- **Verified** — executed in this repository as bounded CPU smoke runs
  (the fixture and `ml-100k` runs listed in each section).
- **Provided, not yet executed** — full-scale command shapes that follow the
  same code paths but have not been run end to end on the full datasets.

## Environment

- Python >= 3.10.
- Dependencies: `torch`, `numpy`, `pandas`, `pyarrow`, `pyyaml`
  (see `pyproject.toml`; `pytest` and `ruff` are the dev extras).
- Windows PowerShell is sufficient; no bash, tmux, or job scheduler is
  required. `--num-workers` defaults to 0, so DataLoaders never spawn worker
  processes unless you ask for them.

```powershell
# from the repository root
pip install -e .
python -m adrec_still models   # lists the implemented backends
```

Training writes only into the `--output` directory you name (plus the
`--cache-dir` for raw-file preprocessing, which defaults to `<output>/cache`).
Nothing writes into the repository or a global cache.

## Input data

`--data` accepts three kinds of input.

### 1. A raw check-in file

The adapter is inferred from the file name (`gowalla` in the name → the
Gowalla adapter, `tsmc2014` → `foursquare_subset`, `tist2015`/`global` →
`foursquare_global`); pass `--adapter` explicitly when the name does not give
it away.

- **Gowalla** (`loc-gowalla_totalCheckins.txt`): tab-separated
  `user, ISO-8601 timestamp, latitude, longitude, poi`. The timestamps are
  UTC and are stored as UTC seconds (`event_time_utc_s`).
- **Foursquare NYC/TKY** (`dataset_TSMC2014_NYC.txt` /
  `dataset_TSMC2014_TKY.txt`): one self-contained TSMC2014 file through the
  `foursquare_subset` adapter. The canonical event time is parsed from the
  file's own UTC timestamp field; the separate timezone-offset column is
  audited (invalid offsets are counted) but never shifts a timestamp and never
  drops a row. No local-clock semantics are claimed anywhere.
- **Foursquare global** (TIST2015) needs `--poi-file` for the POI table.

Preprocessing runs once into `--cache-dir` (default `<output>/cache`) and is
reused on later runs against the same file (a source stamp of path, size,
mtime and preprocessing parameters decides reuse). `--row-limit N` preprocesses
only the first N raw rows for smoke runs; `--horizons-hours` sets the query
landmark grid (default `0,6,12,24,48,72,168`); `--workers` parallelizes
preprocessing.

### 2. A canonical processed directory

A directory holding at least `landmark_samples.parquet` and
`poi_catalog.parquet`. When `events.parquet` and `split_manifest.parquet` are
present too, the survival/gap-censor training profiles and the full-history
repeat semantics become available. A directory produced by the old STILL
repository's preprocessing uses the same canonical schema and is accepted
as-is.

Models of the `adrec_still` family (and the `still`/`external` families)
refuse a directory without `events.parquet` with a named error, for example:

```
model family adrec_still needs full repeat semantics, which require
events.parquet beside <...>/landmark_samples.parquet; preprocess from the raw
check-in file or point --data at a complete canonical directory
```

### 3. An upstream `dataset.pkl`

The fork's bundled datasets (`datasets/data/<name>/dataset.pkl`, e.g.
`ml-100k`, `baby`, `beauty`, `sports`, `toys`, `yelp`) hold per-user
train/val/test item orderings and **no timestamps**. Only the upstream
event-only backends (`adrec`, `diffurec`, `dreamrec`, `sasrec_upstream`) train
on them; every time/silence/cell request is refused with a named error rather
than fabricated:

```
model adrec_still needs event timestamps, and dataset.pkl files carry none
(only per-user item orderings); only the upstream event-only models (adrec,
diffurec, dreamrec, sasrec_upstream) train on dataset.pkl
```

The item vocabulary is derived from the data on load (the fork's hardcoded
per-dataset item counts are gone), with raw IDs shifted by +1 so PAD=0 and
UNK=1 stay reserved package-wide.

## Commands

All five subcommands print full flag listings with `--help`; every flag below
exists in the current CLI.

### `python -m adrec_still models`

Lists the implemented backends with their family (`baseline`, `still`,
`external`, `upstream`, `adrec_still`) and kind (`floor`/`neural`).
**Verified.**

### `python -m adrec_still train`

One model, one dataset, one seed, one output directory.

**Verified** — bounded CPU smoke runs of these shapes were executed in this
repository (all five `adrec_still`-family variants on the bundled fixture, and
all four upstream backends on `ml-100k`):

```powershell
# e.g. the full adrec_still variant on the fixture:
python -m adrec_still train --model adrec_still --data tests\fixtures\gowalla_small.txt --device cpu --epochs 2 --output runs\smoke\adrec_still

# The upstream ADRec three-stage path (pretrain -> warmup -> finetune) on the
# bundled ml-100k pkl, one epoch per stage:
python -m adrec_still train --model adrec --data datasets\data\ml-100k\dataset.pkl --device cpu --set training.pretrain_epochs=1 --set training.warmup_epochs=1 --epochs 1 --max-train-batches 4 --output runs\smoke\adrec

# diffurec / dreamrec / sasrec_upstream train the same way (single stage):
python -m adrec_still train --model diffurec --data datasets\data\ml-100k\dataset.pkl --device cpu --epochs 1 --max-train-batches 4 --output runs\smoke\diffurec
```

**Provided, not yet executed** (full-scale shapes; see the
bounded-validation section below for the first real-data numbers):

```powershell
# Full Gowalla on a GPU
python -m adrec_still train --model adrec_still --data D:\data\loc-gowalla_totalCheckins.txt --device cuda --output runs\gowalla\adrec_still --cache-dir D:\cache\gowalla

# Foursquare NYC (TSMC2014; adapter inferred from the file name)
python -m adrec_still train --model adrec_still --data D:\data\dataset_TSMC2014_NYC.txt --device cuda --output runs\nyc\adrec_still

# A canonical directory produced earlier (this repo's cache or the old STILL repo)
python -m adrec_still train --model sasrec_h --data D:\cache\gowalla\processed --device cuda --output runs\gowalla\sasrec_h
```

Useful switches:

- `--seed N` (default 41); `--epochs`, `--batch-size`, `--learning-rate`,
  `--patience` override the training block.
- `--set KEY=VALUE` overrides any model or training setting by dotted path,
  e.g. `--set embedding_dim=32 --set training.negative_samples=16`.
- `--monitor` picks the selection metric (`val_loss`, `ndcg@10`, `recall@10`,
  `mrr@10`, `poi_nll`, `joint_nll`, `time_nll`, `train_loss`, and the
  `@5/@10/@20` ranking variants); `--validation-horizon SEC|all` restricts the
  per-epoch validation pass to one landmark horizon.
- `--max-train-batches` and `--validation-max-batches` bound smoke runs.
- `--device cpu|cuda|cuda:N|auto` (default auto). A `--device cpu` run hides
  the GPUs before torch initializes CUDA, so CPU runs never leave a CUDA
  context on a shared GPU.
- `--model-config FILE` loads a model YAML over the built-in defaults;
  `--config FILE` accepts a legacy experiment YAML (only its `model_config`
  is used).

Exit codes: 0 completed, 1 failed (message on stderr), 2 usage error,
130 interrupted.

### The automatic pipeline for `--model adrec_still`

One `train` invocation against a raw file runs the whole chain; no separate
preprocessing or indexing commands exist or are needed:

1. **Preprocess** the raw file into the canonical directory (cached; skipped
   when `--data` already is one).
2. **Survival view**: the administrative-censor training view is built once
   per run from `events.parquet` + `split_manifest.parquet` and cached.
3. **Full-history index**: the complete per-user event memory (raw-ID repeat
   labels, seen sets, home coordinates) is built from `events.parquet` and
   attached to the model in memory.
4. **Train** under the gap-censor profile (`adrec-still-gap-censor-v1`): the
   survival view filtered to its `s=0` rows, so every actual gap trains the
   base distribution exactly once (observed rows) and every terminal exposure
   contributes one censor row; weights are per-row unit weights.
5. **Validation selection** each epoch on the canonical validation landmarks
   (default monitor for the family: `ndcg@10`); `best.pt` tracks the best
   epoch.
6. **Final test evaluation** with the selected weights (skip with
   `--no-test`; add `--eval-validation` for a full validation pass too).

### The ADRec three-stage path (upstream family)

`--model adrec` trains in three stages: **pretrain** (a position-free SASRec
trains the item-embedding table), **warmup** (the diffusion model trains
around the frozen table), **finetune** (the table unfreezes). Each budget is a
plain setting — `training.pretrain_epochs` (default 10),
`training.warmup_epochs` (default 5), `max_epochs`/`--epochs` for finetune
(default 20) — and each may be lowered to a single epoch for smoke runs.

- The pretrained embedding is handed over **in memory**; no `pretrain.pth`
  file is required for a new dataset. To reuse a fork-produced `pretrain.pth`,
  set `pretrained_embedding_path` in a model config; the table is remapped
  into this package's vocabulary (raw ID `i` → row `i+1`; PAD/UNK keep fresh
  initialization) and a table whose size disagrees with the target vocabulary
  is refused rather than silently matched by shape.
- Selection is scoped **per stage** and reset at each stage switch: `best.pt`
  always holds the best checkpoint of the current stage's scope, so a pretrain
  or warmup metric can never masquerade as the finetune best. Earlier stages'
  bests are kept in the summary as `stage_history`.
- During warmup, each stage builds its optimizer over the parameters trainable
  in that stage, so the frozen embedding is genuinely untouched (no Adam
  weight decay on frozen tables).

`diffurec`, `dreamrec`, and `sasrec_upstream` run a single `train` stage under
the same staged trainer.

### `python -m adrec_still batch`

Runs every (model, seed) pair in order; the next run starts only after the
previous one completed, and a failure stops the sequence unless
`--continue-on-failure` is passed. Output layout is
`<output-root>/<model>/seed<N>/`.

**Provided, not yet executed** at full scale:

```powershell
python -m adrec_still batch --data D:\cache\gowalla\processed --models global_pop,sasrec_h,adrec_still --seeds 41,42,43 --device cuda --output-root runs\gowalla
```

### `python -m adrec_still evaluate`

Evaluates a saved `best.pt` (shared-trainer or staged format) on a dataset:
exact full-catalog ranking plus, when the canonical directory carries
`events.parquet`, the silence-aware metrics (repeat/explore ranking splits,
repeat-probability quality, censor-aware time NLL, and the four-silence-cell
metrics with their simple comparison classifiers). The checkpoint's recorded
data identity (catalog width and item-ID fingerprint) must match the supplied
data; a same-shape checkpoint from a different mapping is refused.

**Verified** (bounded CPU smoke run on the fixture):

```powershell
python -m adrec_still evaluate --data tests\fixtures\gowalla_small.txt --checkpoint runs\smoke\adrec_still\best.pt --device cpu
```

**Provided, not yet executed** at full scale:

```powershell
python -m adrec_still evaluate --data D:\cache\gowalla\processed --checkpoint runs\gowalla\adrec_still\best.pt --device cuda --horizons-hours 0,6,12,24,48,72,168 --output runs\gowalla\adrec_still\evaluation.json
```

Silence-specific switches: `--cell-radius-km` (default 10) and
`--cell-time-threshold-hours` (default 24) set the cell axes (engineering
starting points, see [MODELING.md](MODELING.md)); `--time-samples`,
`--diffusion-samples`, `--inference-steps`, and `--eval-seed` control the
Monte-Carlo forecast of the `adrec_still` family; `--no-silence` and
`--no-cell-baselines` skip those passes; `--max-batches` bounds smoke runs.
The result JSON defaults to `evaluation.json` beside the checkpoint, and the
raw-file preprocessing cache defaults to `cache/` beside the checkpoint.

### `python -m adrec_still collect`

Tabulates finished runs from their `summary.json` files; nothing is
recomputed.

```powershell
python -m adrec_still collect --root runs --split test --metrics ndcg@10,recall@10,mrr@10,poi_nll --format table
python -m adrec_still collect --root runs --format csv > results.csv
```

`--horizon` selects one landmark horizon's row (seconds) or `all` for the
pooled row; `--root` is repeatable. **Verified** as part of the smoke runs.

## What a run writes

```
<output>/config.json      what the run resolved to (data, model, training, device)
<output>/metrics.jsonl    one row per completed epoch (neural runs)
<output>/summary.json     status, best epoch, evaluation, timings
<output>/last.pt          resumable state after the last completed epoch (neural)
<output>/best.pt          selected weights (neural)
<output>/cache/           preprocessed raw data and derived views (when needed)
```

Nothing else. Checkpoints and JSON files are written atomically
(write-to-temp, then replace), so an interrupted write never leaves a corrupt
file. Deterministic floors write no checkpoints (`checkpoints: null`).
`last.pt` is replaced after every completed epoch and carries weights,
optimizer state, counters, RNG state, and the best-so-far; `best.pt` is
replaced only when the monitored metric improves. A `run.lock` file guards the
output directory against two live processes.

**Ctrl+C** keeps the most recent complete checkpoint in `last.pt`, writes a
summary with `status: interrupted`, and exits with code 130. An interrupt
before training started writes nothing.

## Resume and weights

```powershell
# continue a run exactly where it stopped (optimizer, counters, RNG restored)
python -m adrec_still train --model adrec_still --data <same-data> --resume runs\gowalla\adrec_still\last.pt --output runs\gowalla\adrec_still

# start a fresh run from selected weights (fresh optimizer and schedule)
python -m adrec_still train --model adrec_still --data <same-data> --weights runs\gowalla\adrec_still\best.pt --output runs\gowalla\adrec_still_ft --learning-rate 0.0003
```

A resume checks what actually has to match: the model configuration, parameter
shapes, the item-ID mapping the embedding table was trained against, and the
optimizer type. Everything else — epoch budgets, batch size, learning rate,
patience, monitor, device, workers — is an experiment choice: changes are
applied and recorded in the summary. Host, interpreter, library versions and
paths are recorded as information and never compared, so a checkpoint moves
freely between machines. Staged (upstream/`adrec`) runs resume mid-stage,
including mid-pretrain.

## CPU smoke vs full GPU runs

A bounded CPU smoke run of any neural backend looks like:

```powershell
python -m adrec_still train --model sasrec_h --data tests\fixtures\gowalla_small.txt --device cpu --epochs 1 --max-train-batches 2 --validation-max-batches 2 --output runs\smoke\sasrec_h
```

(**Verified** shape — the five `adrec_still`-family variants and the four
upstream backends were exercised exactly this way.) Full GPU runs drop the
caps and set `--device cuda`; they are **provided, not yet executed** in this
repository.

## Bounded validation on real data

Executed on 2026-10-14, on CPU, against a **pre-declared time subset** of the
real Gowalla check-in file: all check-ins in `[2010-06-01, 2010-06-15)` UTC
(264,545 rows, 27,622 users, 159,417 raw POIs → a 117,144-entry frozen training
catalog after the chronological 70/10/20 split). The subset was selected by its
time window alone, before preprocessing; no labels were inspected. These runs
validate that the pipeline executes end to end on real data — they are
**bounded smoke results**, not tuned experiment numbers.

| run | bounds | result (test split) |
|---|---|---|
| `global_pop` | none (floors are cheap) | full test: 168,155 rows / 46,454 ranking-eligible; NDCG@10 0.0063, Recall@10 0.0152, poi_nll 11.22 |
| `sasrec_h` | 2 epochs, 80 train batches/epoch, dim 64, selection on 20 validation batches at horizon 0 | full test: NDCG@10 0.0091, Recall@10 0.0184 (641 s total) |
| `adrec_still` | 2 epochs, 60 train batches/epoch, dim 64, 16 diffusion steps, 1 time sample/stratum, `--no-test` + bounded `evaluate --max-batches 6` | 384 test rows: repeat-bucket NDCG@10 0.0219 vs explore 0.0; event repeat rate 0.237 (all targets) / 0.582 (known catalog); observed time NLL 13.17, censored survival NLL 0.547; four-cell metrics + prior/elapsed/history-elapsed comparisons produced with full coverage accounting (166/384 OOV targets, 127 censored) |

Commands (as executed):

```powershell
# the subset file was written by filtering the raw file to the June 1-14 2010 window
python -m adrec_still train --data results\bounded_gowalla\raw\gowalla_20100601_20100614.txt --adapter gowalla --model global_pop --device cpu --output results\bounded_gowalla\global_pop --cache-dir results\bounded_gowalla\cache --workers 4
python -m adrec_still train --data results\bounded_gowalla\cache\processed --model sasrec_h --device cpu --epochs 2 --batch-size 128 --max-train-batches 80 --validation-max-batches 20 --validation-horizon 0 --output results\bounded_gowalla\sasrec_h --set embedding_dim=64 --set hidden_dim=64 --set training.negative_samples=64
python -m adrec_still train --data results\bounded_gowalla\cache\processed --cache-dir results\bounded_gowalla\cache --model adrec_still --device cpu --epochs 2 --batch-size 64 --max-train-batches 60 --validation-max-batches 5 --validation-horizon 0 --no-test --output results\bounded_gowalla\adrec_still --set hidden_size=64 --set diffusion_steps=16 --set time_samples=1 --set training.negative_samples=64
python -m adrec_still evaluate --data results\bounded_gowalla\cache\processed --checkpoint results\bounded_gowalla\adrec_still\best.pt --splits test --device cpu --batch-size 64 --max-batches 6 --output results\bounded_gowalla\adrec_still\evaluation.json
```

The full Gowalla file and the Foursquare NYC/TKY raw files have **not** been
trained here (the TSMC2014 raw files are not present on this machine); the
full-scale commands in this document are provided for those runs, not yet
executed.

## See also

- [MODELING.md](MODELING.md) — model, loss, query and silence-cell
  definitions, and the capability table of every registered backend.
- [MIGRATION.md](MIGRATION.md) — what was ported from the STILL donor
  repository and the upstream ADRec fork, and which behaviors were
  deliberately changed.
