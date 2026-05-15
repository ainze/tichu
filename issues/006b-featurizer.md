---
id: "006b"
title: "Featurizer (PrivateState → fixed-shape ndarray)"
type: AFK
blocked_by: ["002"]
parent: "006"
---

## Parent

[#006 Featurizer + action space](006-featurizer-action-space.md)

## What to build

A pure function `featurize(private_state) -> np.ndarray` whose output is a fixed-length `float32` vector pinned to a version string. No I/O, no globals, no time-dependence.

**Feature sections (v1, rich state + history).** Concatenated in order; each section's length is a fixed constant logged at import time:

1. Own hand (56 dims): multi-hot over the 52 normal cards plus Mahjong, Dog, Phoenix, Dragon.
2. Hand sizes (4 dims): each player's hand_size / 14 (normalized).
3. Team scores (2 dims): scores / 1000 (normalized, can exceed 1.0).
4. Round points so far (4 dims): `round_points_by_player[p] / 100` per seat.
5. Out-order one-hot (4 × 4 = 16 dims): one_hot of `out_order[i]` for i in 0..3; zero block if `i >= len(out_order)`.
6. Tichu callers (4 dims): one-hot per seat.
7. Grand Tichu callers (4 dims): one-hot per seat.
8. Mahjong wish (15 dims): one-hot over ranks 2..A (13) plus "no wish active" plus "wish already fulfilled / N/A".
9. Current player (4 dims): one-hot.
10. Phase one-hot (5 dims): {normal-play, schupfen, dragon-give-pending, mahjong-wish-pending, terminal}.
11. Current trick top combo (CANONICAL_ACTIONS-sized): one-hot index into the action space, or zeros if no top combo. Uses the v1 action-space size from `tichu_training.action_space`.
12. Trick passes (4 dims): one-hot which players have passed since the last play.
13. Play history (k=8 last plays in trick × CANONICAL_ACTIONS dims = 8 × N): each slot is one-hot of the action played, oldest→newest, leading zeros padding when fewer than 8 plays. Uses the action space.
14. Schupfen received (3 × 56 = 168 dims): multi-hot of cards received from each direction (next/partner/prev), zero-blocks before schupfen is applied.
15. Phoenix-played flag (1 dim): 0/1.

**Version constant.** `FEATURIZER_VERSION: str = "v1"` from `tichu_training.featurizer`. Output dtype: `np.float32`. Output shape is a single fixed integer `FEATURIZER_OUTPUT_DIM` available as a module-level constant.

**Purity.**

- Two calls in one process on equal `PrivateState`s return arrays whose `.tobytes()` are identical.
- Two calls in *separate* processes on the same input return identical `.tobytes()` (no env-dependent code paths, no hash-seed-sensitive dict ordering, etc.).

## Acceptance criteria

- [ ] `featurize` returns `np.ndarray` of dtype `float32` and shape `(FEATURIZER_OUTPUT_DIM,)` for every legal `PrivateState` (including schupfen and terminal phases)
- [ ] Per-section dim breakdown logged at import time
- [ ] `featurize(s).tobytes() == featurize(s).tobytes()` for every `s` (within-process purity)
- [ ] Cross-process purity: feature bytes match between two subprocess invocations on the same input
- [ ] `FEATURIZER_VERSION == "v1"` and is a frozen module-level constant
- [ ] Featurizer does not mutate the input `PrivateState` or any of its members

## Blocked by

- [#002 Rules engine](002-rules-engine.md)
- [#006a Action space](006a-action-space.md) (the trick-top + history sections use the action-space size)
