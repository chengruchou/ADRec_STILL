# Configs

Model YAMLs passed to `--model-config`. Every key here overrides the built-in
defaults in `adrec_still/training/defaults.py`; anything omitted keeps its
default, so a config only needs the values an experiment actually changes.
`--set key=value` overrides on top of a file.

- `model/adrec_still_example.yaml` — the coherent extension model with explicit
  silence/cell parameters.
- `model/adrec_upstream_example.yaml` — the upstream ADRec reference with its
  three-stage budgets spelled out.
