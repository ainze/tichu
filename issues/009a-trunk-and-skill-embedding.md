---
id: "009a"
title: "BC model: MLP trunk + skill embedding"
type: AFK
blocked_by: ["006", "008"]
parent: "009"
---

## Parent

[#009 Multi-head BC training pipeline](009-bc-training-pipeline.md)

## What to build

The trunk module that downstream tasks (BC heads, Tichu-call heads, belief
model, RL refinement) all sit on top of. Architecture is fixed by
[ADR-001](../docs/adr/001-trunk-architecture.md).

**Trunk.** `tichu_training.bc.model.TichuTrunk(in_dim, *, hidden=1024, depth=4, out_dim=512)`:

- 4 residual blocks (`Linear -> LayerNorm -> GELU -> Linear` with skip).
- Width 1024, GELU + LayerNorm, final projection to 512-dim trunk output.
- Forward: `forward(features: Tensor[B, in_dim]) -> Tensor[B, out_dim]`.

**Skill embedding.** `SkillEmbedding(num_buckets=10, neutral_bucket=True, dim=64)`:

- `nn.Embedding(num_buckets + 1, dim)` — 10 decile buckets + 1 neutral.
- `forward(decile_indices: Tensor[B] (long, in [0..num_buckets]))` — index
  `num_buckets` (the last row) is the cold-start neutral embedding.

**Joint model API.** `BCModelInput`:

- `features: Tensor[B, F]` from `featurize(...)`.
- `skill_decile: Tensor[B]` (long); cold-start rows pass `num_buckets`.

The trunk consumes the concatenation of features and the skill embedding.

## Acceptance criteria

- [ ] `TichuTrunk(in_dim=F, out_dim=512)` produces `(B, 512)` output for any batch
- [ ] Trunk forward is differentiable end-to-end (no detaches)
- [ ] `SkillEmbedding(num_buckets=10)` has `(11, dim)` weight; row 10 is the neutral
- [ ] Concatenated input shape passes through trunk and yields finite, non-NaN output
- [ ] Module is JIT-traceable (`torch.jit.trace`) on a representative batch — proof that
      no Python branching depends on tensor values
- [ ] All weights are torch parameters (visible in `module.parameters()`)

## Blocked by

- [#006 Featurizer + action space](006-featurizer-action-space.md)
- [#008 Trunk architecture decision](008-trunk-architecture-decision.md)
