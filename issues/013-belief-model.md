---
id: "013"
title: "Belief model training"
type: AFK
blocked_by: ["007"]
stories: [34]
---

## What to build

Train a belief model that predicts each opponent's remaining cards from public game history. This is a phase-2 enabler: it is not on the inference path in phase 1, but BSW logs provide paired (public history, true hidden hands) data that makes this a straightforward supervised problem. Building it now is cheap at this data scale and provides the bridge piece for phase-2 search-based extensions.

**Task framing.** For each step in a game, the public game history (all played tricks, all public bids, all public scores) is available. The true hidden hands of the three non-acting players are in the BSW log post-game. Train a model to predict the remaining cards of all three opponents from the public history alone, using the true post-game hands as labels.

**Architecture.** A separate model (not the shared trunk from `#008`). Likely a transformer or MLP over the trick history sequence. Input: public game history at a given step. Output: a probability distribution over card assignments for each of the three other players. Multi-label classification (each card is either in a player's hand or not).

**Training data.** Use the Parquet training records from `#007`. Each record's `state` field contains the full public history. Labels are derived from the BSW post-game hand records. A separate `belief_training` Parquet shard is emitted by the parser for this purpose.

**Evaluation.** Held-out prediction accuracy: for a held-out set of game steps, measure fraction of cards correctly assigned to their true holder. Log calibration (predicted probabilities vs empirical frequencies).

**Not on the inference path.** The belief model is trained but not loaded by the inference service in phase 1. Its checkpoint is stored separately from policy checkpoints. It is documented as the input to phase-2 ISMCTS agents.

**CLI.** `train_belief --config <path>`.

## Acceptance criteria

- [ ] Belief model trains on public history inputs with true hidden hands as labels
- [ ] Held-out card-assignment accuracy is logged after each epoch
- [ ] Calibration plot or metric is produced at the end of training
- [ ] Checkpoint is stored separately from policy checkpoints; inference service does not load it
- [ ] Smoke test: one epoch on synthetic data, loss decreases
- [ ] `train_belief` CLI is functional

## Blocked by

- [#007 Parquet training records pipeline](007-parquet-training-records.md)
