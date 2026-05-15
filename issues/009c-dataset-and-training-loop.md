---
id: "009c"
title: "BC dataset + training loop + checkpoint resume"
type: AFK
blocked_by: ["009b", "007"]
parent: "009"
---

## Parent

[#009 Multi-head BC training pipeline](009-bc-training-pipeline.md)

## What to build

Two data sources and the training loop that consumes them.

**Synthetic dataset.** `SyntheticBCDataset(seed, n_per_head)` generates
`PrivateState` instances by rolling the engine forward with the existing
heuristic agents (#003), picks a legal action as the BC target, and yields
`(features, decision_type, action_index, legal_mask, sample_weight,
skill_decile)` tuples. Used by the smoke test.

**Parquet-backed dataset.** `ParquetBCDataset(shards_dir, tch_source_dir, *,
expected_featurizer_version, expected_action_space_version)` reads the
shards produced by #007, validates the version columns, and yields rows by
re-replaying the relevant round from the .tch source up to the decision
point to reconstruct `PrivateState`. Slow (replay-on-load) but correct and
keeps the parquet small; a featurization-cache slice is a deferred follow-up.

Both implement the same iterator protocol so the training loop is dataset-
agnostic.

**Training loop.** `train_one_epoch(model, dataset, optimizer, *,
batch_size, log_path, weights_by_head)`:

- Pulls examples, groups them by `decision_type`, applies the
  per-decision-type head, computes masked cross-entropy, sums weighted
  losses, steps the optimizer.
- Logs `step, loss_total, loss_play, acc_play, ...` to a CSV path every
  step.

**Checkpointing.** `save_checkpoint(model, optimizer, step, path)` writes a
`Checkpoint` (#006c) whose payload is `torch.save({"model": state_dict,
"optimizer": state_dict, "step": int}, BytesIO())`. `load_checkpoint(path,
model, optimizer)` reverses it, raising `VersionMismatchError` if either
pinned version disagrees with the live constants.

## Acceptance criteria

- [ ] `SyntheticBCDataset(seed=42, n_per_head=10).__iter__` yields ≥ 40 items in deterministic order
- [ ] `ParquetBCDataset` raises `VersionMismatchError` when the shard's featurizer_version disagrees
- [ ] `train_one_epoch` on the synthetic dataset reduces total loss from step 0 to step N (asserted by smoke)
- [ ] CSV log contains one row per training step with per-head loss + acc columns
- [ ] `save_checkpoint` + `load_checkpoint` round-trip: post-load model + optimizer state match
- [ ] Resume from a mid-epoch checkpoint produces identical loss trajectory to the original (modulo nondeterminism — seed is honoured)

## Blocked by

- [#009b Multi-head + loss](009b-multi-head-and-loss.md)
- [#007 Parquet records pipeline](007-parquet-training-records.md)
