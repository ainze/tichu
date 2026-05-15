---
id: "010b"
title: "Call-network training: synthetic dataset, CLI, skill-rate logging"
type: AFK
blocked_by: ["010a", "007"]
parent: "010"
---

## Parent

[#010 Tichu & Grand Tichu call networks](010-tichu-call-networks.md)

## What to build

The data path and training loop for both call networks plus the diagnostic
required by the PRD.

**Synthetic dataset.** `SyntheticCallDataset(seed, n_examples,
positive_rate)` yields `(features, target, skill_decile, sample_weight)`
with deterministic features and a configurable class balance. Used by the
smoke test. The same dataset shape is consumed by both call networks (the
hand-size distinction lives inside the featurizer output).

**Parquet path note.** The negative class for Tichu/Grand Tichu calls is
not directly present in the parquet shards from #007 — the parser emits
only positive call events. A parquet adapter that emits negative examples
(one per non-calling player per call opportunity) is a deferred follow-up;
the smoke test uses the synthetic dataset.

**Training loop.** `train_one_call_epoch(network, examples, optimizer, *,
batch_size, log_path)` runs cross-entropy over the 2-class output with
sample-weight scaling, writes a per-step CSV row, and at the end of each
epoch writes a `calling_rate_by_decile.csv` with `decile, n, calling_rate`
(predicted P(call) > 0.5).

**CLI.** `train_calls --config <path> --run-dir <dir> [--only grand|tichu]`:

- Loads YAML config. Same skeleton as `train_bc` (dataset section, model
  section, lr, batch size, epochs, head_weights, checkpoint_every).
- Trains both networks unless `--only` restricts to one.
- Writes checkpoints using the shared `save_checkpoint` from #009c so the
  inference service can hot-swap them.

**Configs.** `configs/calls_smoke.yaml` (CI smoke, synthetic, 2 epochs) and
`configs/calls_full.yaml` (production scaffold, parquet path that currently
errors with NotImplementedError mirroring `bc_full.yaml`).

## Acceptance criteria

- [ ] `SyntheticCallDataset(seed=0, n_examples=200, positive_rate=0.3)` yields exactly 200 examples with mixed labels
- [ ] `train_one_call_epoch` reduces loss across epochs on the synthetic dataset (asserted by the smoke test)
- [ ] Per-step CSV contains `step, loss, accuracy`
- [ ] `calling_rate_by_decile.csv` is written at the end of each epoch with one row per observed decile
- [ ] `train_calls` runs end-to-end on `calls_smoke.yaml`
- [ ] Checkpoints written here load with `load_checkpoint` from #009c (same wrapper, same VersionMismatchError contract)

## Blocked by

- [#010a Call-network model](010a-call-networks-model.md)
- [#007 Parquet records pipeline](007-parquet-training-records.md)
