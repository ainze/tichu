---
id: "010a"
title: "Tichu / Grand Tichu call network models"
type: AFK
blocked_by: ["006", "008"]
parent: "010"
---

## Parent

[#010 Tichu & Grand Tichu call networks](010-tichu-call-networks.md)

## What to build

Two small MLP classifiers, one per call type. Both consume the v1 featurizer
output (the hand size difference between 8-card and 14-card phases shows up
inside the featurizer's `own_hand` multi-hot section) plus the same skill
embedding used by the BC stack.

**Architecture.** `tichu_training.bc.call_model.CallNetwork(feature_dim, *,
skill_buckets=10, skill_dim=64, hidden=256)`:

- `SkillEmbedding` (reused from #009a).
- 2-layer MLP: `Linear -> GELU -> Linear -> GELU -> Linear` projecting from
  `feature_dim + skill_dim` to 2 logits (call / no-call).
- Forward: `forward(features, skill_decile) -> Tensor[B, 2]`.

Two thin subclasses for clarity and downstream registry use:
`GrandTichuCallNetwork` and `TichuCallNetwork`. Both have identical
architecture; the names are load-bearing for the checkpoint registry and
inference service.

## Acceptance criteria

- [ ] `CallNetwork(feature_dim=F)` returns shape `(B, 2)` on a batch
- [ ] Skill embedding row 10 (neutral) is used when `skill_decile == 10`
- [ ] Forward is differentiable; trunk + head parameters are visible in `.parameters()`
- [ ] `GrandTichuCallNetwork` and `TichuCallNetwork` are importable from `tichu_training.bc.call_model`
- [ ] Two networks instantiated with the same seed produce identical outputs (no implicit nondeterminism)

## Blocked by

- [#009a Trunk + skill embedding](009a-trunk-and-skill-embedding.md)
