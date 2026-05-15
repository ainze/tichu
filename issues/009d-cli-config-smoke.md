---
id: "009d"
title: "train_bc CLI + YAML configs + CI smoke test"
type: AFK
blocked_by: ["009c"]
parent: "009"
---

## Parent

[#009 Multi-head BC training pipeline](009-bc-training-pipeline.md)

## What to build

User-facing surface: a YAML-driven CLI that runs (or resumes) a BC training
job and a smoke test that proves the whole stack works.

**YAML schema.** A flat dict with these keys:

- `dataset`: one of `synthetic` or `parquet`
- `dataset_kwargs`: dict (synthetic: `seed`, `n_per_head`; parquet:
  `shards_dir`, `tch_source_dir`)
- `learning_rate`: float
- `batch_size`: int
- `epochs`: int
- `head_weights`: dict mapping `decision_type` → float
- `checkpoint_dir`: str
- `checkpoint_every`: int (steps)
- `log_path`: str

**Sample configs.**

- `configs/bc_smoke.yaml`: 1 epoch on `SyntheticBCDataset(seed=0, n_per_head=100)`.
- `configs/bc_full.yaml`: scaffold pointing at the real parquet+tch dirs.
  Production hyperparameters; the file is committed but the training
  command runs only when invoked manually.

**CLI.** `train_bc --config <path> [--resume <ckpt>]`:

- Loads YAML, instantiates dataset + model + optimizer.
- Copies the config file into the checkpoint directory at job start.
- Runs `epochs` epochs, saving checkpoints every `checkpoint_every` steps.
- If `--resume` is given, loads the checkpoint (raising `VersionMismatchError` on drift).

**Smoke test.** A test that invokes `main(['--config', 'configs/bc_smoke.yaml'])`
end-to-end and asserts:
  - exit code 0
  - final loss < initial loss
  - a `step.csv` exists in the log directory with > 1 row
  - at least one checkpoint exists in the checkpoint directory

## Acceptance criteria

- [ ] `train_bc --config configs/bc_smoke.yaml` runs to completion on CI
- [ ] Smoke test asserts loss decreases monotonically (or final < initial)
- [ ] YAML config is copied to the checkpoint dir at job start
- [ ] `--resume` continues from a saved checkpoint with no crash
- [ ] `bc_full.yaml` exists and points at the real parquet/tch directories (but is not invoked in CI)

## Blocked by

- [#009c Dataset + training loop](009c-dataset-and-training-loop.md)
