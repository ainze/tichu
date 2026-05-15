---
id: "013b"
title: "Belief dataset, training loop, calibration table"
type: AFK
parent: "013"
blocked_by: ["013a"]
---

## What to build

- `BeliefExample` dataclass: `features: ndarray`, `labels: ndarray` (`(3, 56)` bool), `mask: ndarray` (`(3, 56)` bool).
- `SyntheticBeliefDataset(seed, n_examples, feature_dim)` — deterministic generator that, for each example:
  - Draws random features.
  - Distributes 56 cards among 4 players (own + 3 opponents) + a "played" pile; constructs `labels` and `mask` so positions for own-hand and played cards are masked out.
- `train_one_belief_epoch(model, examples, optimizer, *, batch_size, log_path)` — runs one pass, writes per-step CSV `(step, loss, accuracy)`, returns final batch loss.
- `calibration_table(model, examples) -> list[dict]` — bin predicted P into 10 equal-width buckets in `[0, 1]`; emit `(bucket_index, p_low, p_high, predicted_mean, empirical_freq, n)`. Only masked-in positions count.
- `write_calibration_csv(table, path)` — same shape, written to disk.

## Acceptance criteria

- [ ] Synthetic dataset is deterministic (same seed ⇒ identical examples).
- [ ] Each example's `labels` sum to 3 × (cards held by opponents); `mask` excludes own-hand + played cards.
- [ ] Training loss decreases over a 5-epoch synthetic run.
- [ ] Calibration CSV has 10 rows with the documented columns.

## Blocked by

- [#013a BeliefModel](013a-belief-model.md)
