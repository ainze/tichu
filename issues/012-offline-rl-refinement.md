---
id: "012"
title: "Offline RL refinement (AWR)"
type: AFK
blocked_by: ["009", "011"]
stories: [38, 39]
---

## What to build

Implement an optional offline RL refinement stage using Advantage-Weighted Regression (AWR) that improves a BC checkpoint using game outcomes as a reward signal, without online self-play.

**Method: AWR.** AWR is chosen because it is the simplest offline RL method to integrate on top of BC: it is a weighted variant of the same behavioral cloning objective, where the weight on each (state, action) pair is `exp(advantage / beta)`, clipped to avoid extreme weights. The advantage is estimated from round outcomes in the training data. No environment interaction is required.

**Reuse of BC infrastructure.** AWR refinement reuses the same featurizer, action space, model architecture, and checkpoint format as BC. A checkpoint produced by AWR is a drop-in replacement for a BC checkpoint at inference time. This is a hard requirement — if the checkpoint format diverges, the inference service is broken.

**Reward signal.** `round_outcome` (final point delta for the acting player's team) from the Parquet training records is the reward. Normalize per game to reduce variance. Advantage estimation: fit a simple value baseline (linear or small MLP) on the reward signal; advantage = reward − baseline.

**Training loop.** Load a BC checkpoint as initialization. Run AWR weight computation over the training data. Train for N epochs with the weighted BC objective. Log per-head loss, average advantage weight, and win rate (computed on a small fixed eval subset each epoch). Checkpoint as in BC.

**Configuration.** `train_bc --config <path> --refine-from <bc_checkpoint>` triggers AWR refinement. All AWR hyperparameters (`beta`, advantage clip, baseline architecture) are in the YAML config.

**Eval gate.** No AWR checkpoint is considered for serving until it has an eval matrix row (`#011`) showing improvement over the BC checkpoint it was initialized from.

## Acceptance criteria

- [ ] AWR refinement loads a BC checkpoint and trains with weighted BC objective
- [ ] Checkpoint format is identical to BC; inference service loads it without modification
- [ ] Advantage weights are logged per epoch; extreme weights are clipped
- [ ] Win rate on a small fixed eval subset is logged each epoch as a sanity check
- [ ] AWR smoke test: loss decreases, no crash, checkpoint is loadable by inference service
- [ ] YAML config controls all AWR hyperparameters
- [ ] Eval harness (`#011`) is run on the AWR checkpoint before it is marked ready for serving

## Blocked by

- [#009 Multi-head BC training pipeline](009-bc-training-pipeline.md)
- [#011 Evaluation harness](011-eval-harness.md)
