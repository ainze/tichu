---
id: "009b"
title: "BC model: multi-head wrapper + masked cross-entropy loss"
type: AFK
blocked_by: ["009a"]
parent: "009"
---

## Parent

[#009 Multi-head BC training pipeline](009-bc-training-pipeline.md)

## What to build

The four task heads bolted onto the shared trunk, plus a masked
cross-entropy loss that zeroes the gradient for illegal actions.

**Heads.** Each is a 2-layer MLP `Linear -> GELU -> Linear` on the 512-dim
trunk output, projecting to the appropriate logit space:

- `play_head` — logits over the full action space (1809).
- `pass_head` — logits over the pass-card (schupfen) sub-space (3 directions
  in v1; the head emits one direction-intent per call).
- `wish_head` — 14 logits (ranks 2..A + no-wish).
- `dragon_head` — 2 logits (left / right).

`BCModel.forward(features, skill_decile) -> dict[str, Tensor]` returns all
four logit tensors keyed by `decision_type` ('play', 'pass_card', 'wish_rank',
'dragon_give'). Heads share parameters via the trunk but have independent
final projections.

**Masked cross-entropy loss.** `masked_cross_entropy(logits, target, legal_mask, sample_weight)`:

- `logits[B, K]`, `target[B]` (long, in [0, K)), `legal_mask[B, K]` (bool;
  `True` for legal action), `sample_weight[B]` (float).
- Add `-INF` (e.g. `-1e9`) to logits where `legal_mask == False` so softmax
  zeros those entries. The masked logits go through `log_softmax`.
- Returns the sample-weight-weighted mean negative log-likelihood of the
  target action.
- Gradient through masked entries must be exactly zero (test by checking
  `logits.grad` is zero on masked positions).

## Acceptance criteria

- [ ] `BCModel(...)` exposes the four heads via a single forward call
- [ ] `masked_cross_entropy` returns a scalar tensor with gradient
- [ ] When `legal_mask[i, j] == False`, `logits.grad[i, j] == 0`
- [ ] `sample_weight` correctly scales the loss (constant weight `w` → loss `w × unweighted_loss`)
- [ ] `BCModel.heads` enumerable in deterministic order matching the four `decision_type`s

## Blocked by

- [#009a Trunk + skill embedding](009a-trunk-and-skill-embedding.md)
