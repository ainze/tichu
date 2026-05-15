---
id: "010"
title: "Tichu & Grand Tichu call networks"
type: AFK
blocked_by: ["007", "008"]
stories: [33, 4]
---

## What to build

Implement separate small networks for the two Tichu-call decisions and their training pipelines. These are structurally distinct from in-trick play (different input states, different timing, binary outputs) and are trained independently of the shared-trunk policy.

**Why separate networks.** Tichu and Grand Tichu calls happen before and at the very start of a hand, when the acting player's information set differs structurally from in-trick states: Grand Tichu is called before any cards are seen (8-card hand only), and regular Tichu is called before the first trick is played (full 14-card hand). Routing these through the shared trunk would require the trunk to handle inputs of different shapes and semantics.

**Grand Tichu network.** Input: featurized 8-card hand. Output: binary call/no-call. Separate small MLP (or logistic head). Trained on `call_grand_tichu` decision records from the Parquet shards.

**Regular Tichu network.** Input: featurized 14-card hand (after schupfen has been resolved). Output: binary call/no-call. Separate small MLP. Trained on `call_tichu` decision records.

**Calling aggressiveness.** The PRD identifies Tichu-calling behavior as a known failure mode (PPO agents learn never to call; BC on BSW data inherits human calling rates). The skill-conditioning embedding is concatenated to both call-network inputs — stronger players call more aggressively, and the model should learn this from the data. Calling rates by skill decile should be logged during training as a diagnostic.

**Offline RL guard.** Both networks must be hot-swap compatible with the inference service (same checkpoint format, same featurizer and action-space version pinning). This is required so that offline RL refinement in `#012` can update them without changes to the serving stack.

**Training and config.** `train_calls --config <path>` trains both networks in a single run (or independently via a flag). Smoke test: one epoch on synthetic data, loss decreases.

## Acceptance criteria

- [ ] Grand Tichu network trains on 8-card-hand inputs from `call_grand_tichu` shards
- [ ] Regular Tichu network trains on 14-card-hand inputs from `call_tichu` shards
- [ ] Calling rate by skill decile is logged during training
- [ ] Both networks use the same versioned featurizer and action space as the main policy
- [ ] Checkpoints are compatible with the inference service's checkpoint format
- [ ] Smoke test passes (loss decreases on synthetic data)
- [ ] `train_calls` CLI is functional

## Blocked by

- [#007 Parquet training records pipeline](007-parquet-training-records.md)
- [#008 Trunk architecture decision](008-trunk-architecture-decision.md)
