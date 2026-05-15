---
id: "012c"
title: "AWR refinement training step — wire weights into the BC objective"
type: AFK
parent: "012"
blocked_by: ["012a", "012b"]
---

## What to build

Extend the BC training surface so a refinement epoch consumes `round_outcome` per example, computes AWR weights, multiplies them into the existing `sample_weight`, and runs the existing weighted-BC loss.

- Extend `BCExample` with an optional `round_outcome: float = 0.0` field. Default keeps current BC tests intact.
- Extend `SyntheticBCDataset` to emit a non-trivial deterministic `round_outcome` per example (a synthetic linear function of the features, plus noise). Smoke runs use this so the baseline has signal to fit.
- New training step:
  ```python
  def awr_refine_epoch(
      model, examples, baseline, optimizer,
      *, beta, max_weight, batch_size, log_path,
      held_out_subset=None,
  ) -> dict[str, float]
  ```
  computes per-example advantages `outcome - baseline(features)`, computes AWR weights via `012b`, multiplies into `example.sample_weight`, then defers to the existing `train_one_epoch` machinery for the gradient step and per-head logging.
- Per-step CSV gains a column `awr_weight_mean` (average effective weight applied at that step).
- Each call returns a dict with `avg_weight`, `loss_total`, and (if a `held_out_subset` is provided) `win_rate_proxy` — top-1 play-head accuracy on the held-out subset.

## Acceptance criteria

- [ ] `BCExample.round_outcome` defaults to 0.0; existing BC tests unaffected.
- [ ] `SyntheticBCDataset` emits a deterministic non-zero `round_outcome` per example.
- [ ] `awr_refine_epoch` produces gradients identical to `train_one_epoch` when `awr_weights ≡ 1.0` (degenerate equivalence check).
- [ ] `awr_refine_epoch` logs `awr_weight_mean` per step in the CSV.
- [ ] When a `held_out_subset` is provided, `win_rate_proxy` (top-1 accuracy on the play head) is returned.

## Blocked by

- [#012a Value baseline](012a-value-baseline.md)
- [#012b AWR weight function](012b-awr-weights.md)
