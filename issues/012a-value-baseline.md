---
id: "012a"
title: "Value baseline V(state) — small MLP + MSE fit"
type: AFK
parent: "012"
blocked_by: []
---

## What to build

A small value-function module that maps trunk-feature inputs to a scalar prediction of `round_outcome` (team-0 minus team-1 Ergebnis). Trained by MSE on the held-out outcomes column from the parquet training records.

- `ValueBaseline(feature_dim: int, *, hidden: int = 128)` — 2-layer MLP: `Linear(feature_dim, hidden) → GELU → Linear(hidden, 1)`. Returns a `(batch,)` tensor of value predictions.
- `fit_value_baseline(baseline, features, outcomes, *, batch_size, epochs, lr, log_path=None)` — MSE training with Adam, deterministic given a seeded module init. Returns final MSE.

The baseline is a **separate** module from the BC model — it does not share the trunk. Sharing the trunk is a follow-up if/when needed.

## Acceptance criteria

- [ ] `ValueBaseline` outputs a `(batch,)` float tensor.
- [ ] `fit_value_baseline` reduces MSE monotonically on a synthetic linear-target dataset.
- [ ] After fit, predictions on a held-out set are correlated with the true outcomes (Pearson r > 0.5 for the simple synthetic test).
- [ ] No coupling to `BCModel` or `BCExample`; takes raw `np.ndarray` features and outcomes.

## Blocked by

- None — can start immediately.
