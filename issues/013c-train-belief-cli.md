---
id: "013c"
title: "`train_belief` CLI — config, smoke, separate checkpoint dir"
type: AFK
parent: "013"
blocked_by: ["013b"]
---

## What to build

- `train_belief --config <path> --run-dir <dir>` CLI.
- YAML config: dataset name, dataset_kwargs, model hidden size, learning rate, batch size, epochs, seed.
- Run-dir layout: `step.csv`, `epoch.csv` (accuracy per epoch), `calibration.csv` (final), `checkpoints/belief_final.bin`. Distinct from policy checkpoint dirs.
- Checkpoint uses the existing `Checkpoint` wrapper with `featurizer_version=FEATURIZER_VERSION` and `action_space_version=""` (belief doesn't predict actions). Loaders for this file are NOT wired into the inference service.
- `configs/belief_smoke.yaml` checked in.

## Acceptance criteria

- [ ] CLI smoke runs, exits 0, loss decreases.
- [ ] `step.csv`, `epoch.csv`, `calibration.csv`, and a checkpoint file all exist after a run.
- [ ] Checkpoint is loadable by `Checkpoint.load(..., expected_featurizer_version=FEATURIZER_VERSION)` without action-space pinning.
- [ ] No import of belief modules in `tichu_inference` (separation enforced by the codebase, not just by docs).

## Blocked by

- [#013b Belief training](013b-belief-training.md)
