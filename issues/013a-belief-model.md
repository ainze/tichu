---
id: "013a"
title: "BeliefModel — MLP + masked BCE loss"
type: AFK
parent: "013"
blocked_by: []
---

## What to build

A small MLP that maps public-history features to per-opponent per-card occupancy logits.

- `BeliefModel(feature_dim: int, *, num_opponents: int = 3, num_cards: int = 56, hidden: int = 256)` — `Linear → GELU → Linear → GELU → Linear` producing `(batch, num_opponents, num_cards)` logits.
- `belief_loss(logits, labels, mask) -> Tensor` — masked binary cross-entropy. `mask` is `(batch, num_opponents, num_cards)` bool; `labels` is the same shape; positions where `mask == False` are excluded from the loss average.
- `belief_accuracy(logits, labels, mask) -> float` — fraction of masked-in positions where `(logits > 0)` matches `labels`.

## Acceptance criteria

- [ ] Forward pass shape: `(batch, 3, 56)` for default hyperparameters.
- [ ] Loss returns a scalar tensor with non-zero gradient on masked-in positions only.
- [ ] Loss is 0 (or near-zero) when all logits are correct and all mask entries are True.
- [ ] Accuracy with a perfect-prediction synthetic input is 1.0.

## Blocked by

- None — can start immediately.
