---
id: "009"
title: "Multi-head BC training pipeline (play/pass/wish/dragon heads)"
type: AFK
blocked_by: ["007", "008"]
stories: [30, 31, 32, 35, 36, 37]
---

## What to build

Implement the behavioral cloning training pipeline for the four in-game decision heads: play, pass (schupfen), wish-rank, and dragon-give. Tichu/Grand Tichu calls are a separate slice (`#010`).

**Model architecture.** Shared trunk (architecture chosen in `#008`) producing an embedding from `(featurize(private_state), skill_embedding)`. Four heads on top of the trunk: play head (over the in-trick action subspace), pass head (over the 3-card schupfen choices), wish head (over rank choices 2–A), dragon-give head (over 2 direction choices). The skill embedding is a learned lookup over the 10 decile buckets plus one neutral bucket for cold-start players.

**Loss.** Masked cross-entropy on each head. Illegal actions (per `legal_actions_mask`) receive a large negative logit before softmax, not a loss contribution. Losses are summed across heads, weighted by `decision_type` frequency in the batch.

**Training loop.** Reads from the Parquet shards produced by `#007`, sampling proportionally across `decision_type` shards. Logs per-head loss and accuracy to disk (CSV) and optionally to a dashboard (MLflow or W&B, configurable). Checkpoints saved every N steps and at the end of each epoch. Training is resumable from any checkpoint. Multi-GPU is not required but the DataLoader should not be a bottleneck on a single H100.

**Skill weighting.** Records are weighted by `sample_weight` (from `#007`). The skill embedding is concatenated to the trunk input, not used to weight the loss — conditioning is the default; hard filtering is a config flag.

**Configuration.** All hyperparameters (learning rate, batch size, head loss weights, skill-bucket count, checkpoint interval, W&B project name) live in a YAML config. The config is copied into the run artifact directory at job start. Two example configs: `configs/bc_smoke.yaml` (tiny dataset, 1 epoch, for CI) and `configs/bc_full.yaml` (full corpus, production run).

**CLI.** `train_bc --config <path>` starts or resumes a run.

**Smoke test.** One epoch of BC on a tiny synthetic dataset (100 decisions per head, generated from the engine). Assert that training loss decreases. This runs in CI.

## Acceptance criteria

- [ ] Shared trunk + four heads train jointly on Parquet shards
- [ ] Masked cross-entropy is used; illegal actions have zero gradient contribution
- [ ] Per-head loss and accuracy logged to disk on every step
- [ ] Checkpoints saved at configurable intervals; training resumes correctly from any checkpoint
- [ ] Skill embedding is a learned lookup; cold-start players receive a neutral embedding
- [ ] YAML config is copied into the run artifact directory
- [ ] `bc_smoke.yaml` config passes CI smoke test (loss decreases, no crash)
- [ ] `train_bc` CLI is functional end-to-end

## Blocked by

- [#007 Parquet training records pipeline](007-parquet-training-records.md)
- [#008 Trunk architecture decision](008-trunk-architecture-decision.md)
