---
id: "012d"
title: "`train_bc --refine-from <ckpt>` wires AWR end-to-end"
type: AFK
parent: "012"
blocked_by: ["012c"]
---

## What to build

End-to-end CLI pipeline:

1. `--refine-from <bc_checkpoint>` flag (already reserved in `train_bc`) activates AWR mode.
2. Load BC checkpoint into the model (verifies version match via the existing `Checkpoint` contract — this is the hard "checkpoint format is identical to BC" requirement).
3. Build a `ValueBaseline` and fit it on `(features, round_outcome)` from the dataset.
4. Run `awr_refine_epoch` for `epochs` iterations. Pass a held-out subset (last 10% of examples) so the per-epoch `win_rate_proxy` is logged.
5. Save the final model with `save_checkpoint(...)` — same `Checkpoint` wrapper as BC, identical featurizer/action-space versions stamped.

YAML config additions (under a new `awr:` block):
```yaml
awr:
  beta: 1.0
  max_weight: 20.0
  baseline_hidden: 128
  baseline_epochs: 3
  baseline_lr: 1.0e-3
  held_out_fraction: 0.1
```

A new `configs/awr_smoke.yaml` checked in, mirroring `bc_smoke.yaml` plus the `awr:` block.

## Acceptance criteria

- [ ] `train_bc --config awr_smoke.yaml --run-dir <tmp> --refine-from <bc_ckpt>` exits 0 on the synthetic smoke.
- [ ] Final checkpoint loads via the existing `load_checkpoint(path, BCModel)` without modification (BC-format compatibility).
- [ ] `step.csv` contains the `awr_weight_mean` column.
- [ ] `win_rate_proxy` is logged (stdout or its own CSV) each epoch.
- [ ] BC-only smoke (`bc_smoke.yaml`, no `--refine-from`) still passes.

## Blocked by

- [#012c AWR refinement training step](012c-awr-refine-epoch.md)
