---
id: "012b"
title: "AWR weight function — clipped exp(advantage / beta)"
type: AFK
parent: "012"
blocked_by: []
---

## What to build

Pure function that converts a vector of advantages into AWR per-sample weights.

- `awr_weights(advantages: np.ndarray, *, beta: float, max_weight: float = 20.0) -> np.ndarray`
  - Computes `w = exp(advantages / beta)`.
  - Clips to `[0, max_weight]` to avoid a small number of huge weights dominating the gradient.
  - Returns a `float32` array of the same shape.

## Acceptance criteria

- [ ] `beta > 0` required (raises on `beta <= 0`).
- [ ] Weights are non-negative and finite even with extreme advantages (`max_weight` clip enforced).
- [ ] Larger advantages produce monotonically larger weights (before clipping).
- [ ] `beta → ∞` collapses all weights toward 1 (uniform-BC limit).
- [ ] Implementation is numerically stable (subtract `max(advantages)` before exp).

## Blocked by

- None — can start immediately.
